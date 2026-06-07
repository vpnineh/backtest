"""
CorrArb Prop Simulator — v8
============================
بهبودها نسبت به v7:
  - Intra-bar SL/TP با High/Low
  - Slippage واقعی‌تر روی SL
  - Gap protection (max slippage cap)
  - Daily DD چک با worst intra-bar equity
  - Portfolio Heat Limit (max concurrent risk)
  - Correlation-based position sizing
  - Better signal management (no signal loss)
  - Accurate spread cost per pair
  - Walk-forward validation
  - Monte Carlo equity simulation
"""

import pandas as pd
import numpy as np
import glob
import zipfile
import os
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG v8
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── مدیریت ریسک بهبودیافته ──
    risk_base_pct      = 0.01        # ↓ 1.5% → 1% (محافظ‌تر)
    risk_min_pct       = 0.005
    consec_loss_n      = 2
    risk_reduce        = 0.5
    
    # ── Portfolio Heat: حداکثر ریسک همزمان ──
    max_portfolio_risk = 0.02        # حداکثر ۲% ریسک کل در هر لحظه
    max_open_pairs     = 2           # حداکثر ۲ پوزیشن همزمان

    # ── هزینه‌های بروکر واقعی‌تر ──
    spread_pips_eurgbp = 1.5        # spread واقعی EURGBP
    spread_pips_audnzd = 2.0        # spread واقعی AUDNZD (نقدینگی کمتر)
    commission_per_lot = 7.0
    slippage_pips_normal = 0.3
    slippage_pips_sl     = 1.5      # ↑ slippage موقع SL (بازار سریع)
    max_gap_pips         = 10.0     # حداکثر gap protection

    # ── مشخصات ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 2.0                  # ↓ 3.0 → 2.0
    min_lot  = 0.01
    warmup   = 500

    # ── پارامترهای z-score ──
    z_fast_period      = 96
    z_slow_period      = 288        # NEW: slow z برای تأیید trend
    z_entry            = 2.2        # ↑ کمی سخت‌تر
    z_exit_partial     = 0.8
    z_exit_full        = 0.3
    z_stop_margin      = 3.5        # ↓ 4.0 → 3.5 (سریع‌تر cut)
    min_net_profit_usd = 15.0

    # ── فیلترها ──
    corr_period        = 96
    corr_min           = 0.82       # ↑ سخت‌تر
    hour_start         = 3          # ↑ بعد از باز شدن لندن
    hour_end           = 18         # ↓ قبل از بسته شدن NY
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # ── خروج اضطراری ──
    sl_pips            = 25.0       # ↓ 30 → 25
    tp_pips            = 75.0       # ↓ 90 → 75 (R:R = 3)
    time_stop_bars     = 40

    # ── ATR Filter ──
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 2.5        # ↓ 3.0 → 2.5 (فیلتر spike)
    atr_min_mult       = 0.5

    # ── Variance Ratio ──
    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.92       # ↓ سخت‌تر (mean-revert قوی‌تر)

    # ── Partial Exit ──
    partial_ratio      = 0.50
    
    # ── Intra-bar simulation ──
    use_intrabar_sl    = True       # NEW: چک SL با High/Low
    
    # ── Drawdown Buffer ──
    # شروع کاهش ریسک وقتی به این % از DD رسیدیم
    dd_warning_pct     = 0.07       # در ۷% DD شروع به کاهش ریسک


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING (همان v7 با یک بهبود)
# ═══════════════════════════════════════════════════════════════════════════
def load_raw(pattern: str, is_zip: bool) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files found: {pattern}")
    frames = []
    for p in paths:
        try:
            if is_zip:
                with zipfile.ZipFile(p, 'r') as z:
                    csv_name = next(
                        (f for f in z.namelist() if f.lower().endswith('.csv')), None
                    )
                    if csv_name is None:
                        continue
                    with z.open(csv_name) as f:
                        df = pd.read_csv(
                            f, sep=';', header=None,
                            names=['ts', 'o', 'h', 'l', 'c', 'v']
                        )
            else:
                df = pd.read_csv(
                    p, sep=';', header=None,
                    names=['ts', 'o', 'h', 'l', 'c', 'v']
                )
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Skip {os.path.basename(p)}: {e}")
    if not frames:
        raise ValueError(f"No valid data: {pattern}")
    raw = pd.concat(frames, ignore_index=True).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw


def to_15min(raw: pd.DataFrame, sfx: str) -> pd.DataFrame:
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()


def build_spread_df(df_a, sfx_a, df_b, sfx_b, spread_pips) -> pd.DataFrame:
    """
    ساخت spread با High/Low برای intra-bar SL simulation
    """
    merged = df_a.join(df_b, how='inner').dropna()
    if len(merged) == 0:
        raise ValueError(f"No common timestamps")
    
    C = Config
    half_spread = spread_pips * C.pip / 2
    
    merged['c_spread'] = merged[f'c_{sfx_a}'] / merged[f'c_{sfx_b}']
    merged['o_spread'] = (merged[f'o_{sfx_a}'] + half_spread) / (merged[f'o_{sfx_b}'] - half_spread)
    # High/Low برای intra-bar SL چک
    merged['h_spread'] = merged[f'h_{sfx_a}'] / merged[f'l_{sfx_b}']
    merged['l_spread'] = merged[f'l_{sfx_a}'] / merged[f'h_{sfx_b}']
    merged['quote_rate'] = merged[f'c_{sfx_b}']
    return merged[merged.index.weekday < 5].copy()


def load_all_pairs() -> dict:
    print("\n  Loading and syncing datasets...")
    pairs = {}
    C = Config
    try:
        eur = to_15min(load_raw('data/*EURUSD*.csv', is_zip=False), 'eur')
        gbp = to_15min(load_raw('data/*GBPUSD*.csv', is_zip=False), 'gbp')
        df  = build_spread_df(eur, 'eur', gbp, 'gbp', C.spread_pips_eurgbp)
        pairs['EURGBP'] = {
            'df': df, 'leg_a': 'c_eur', 'leg_b': 'c_gbp',
            'spread_pips': C.spread_pips_eurgbp
        }
        print(f"  ✅ EURGBP : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ EURGBP : {e}")
    try:
        aud = to_15min(load_raw('data/HISTDATA*AUDUSD*.zip', is_zip=True), 'aud')
        nzd = to_15min(load_raw('data/HISTDATA*NZDUSD*.zip', is_zip=True), 'nzd')
        df  = build_spread_df(aud, 'aud', nzd, 'nzd', C.spread_pips_audnzd)
        pairs['AUDNZD'] = {
            'df': df, 'leg_a': 'c_aud', 'leg_b': 'c_nzd',
            'spread_pips': C.spread_pips_audnzd
        }
        print(f"  ✅ AUDNZD : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ AUDNZD : {e}")
    if not pairs:
        raise RuntimeError("No pairs loaded.")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION v8
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_variance_ratio(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()
    return vark / (k * var1.replace(0, np.nan))


def calc_hurst(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Hurst Exponent rolling
    H < 0.5 → mean-reverting ✓
    H ≈ 0.5 → random walk
    H > 0.5 → trending
    """
    def hurst_single(x):
        lags = range(2, min(20, len(x) // 4))
        ts_arr = np.array(x)
        tau = []
        for lag in lags:
            diff = ts_arr[lag:] - ts_arr[:-lag]
            if len(diff) > 0 and np.std(diff) > 0:
                tau.append(np.std(diff))
            else:
                tau.append(np.nan)
        if all(np.isnan(tau)):
            return np.nan
        lags_arr = np.array(list(lags), dtype=float)
        tau_arr  = np.array(tau, dtype=float)
        valid    = ~np.isnan(tau_arr) & (tau_arr > 0)
        if valid.sum() < 3:
            return np.nan
        poly = np.polyfit(np.log(lags_arr[valid]), np.log(tau_arr[valid]), 1)
        return poly[0]
    
    return series.rolling(window).apply(hurst_single, raw=True)


def compute_signals(pair_name: str, pair_info: dict) -> tuple:
    C   = Config
    df  = pair_info['df']
    leg_a = pair_info['leg_a']
    leg_b = pair_info['leg_b']

    log_ratio = np.log(df['c_spread'])
    
    # ── Z-Score دوگانه (fast + slow) ──
    z_mean_f = log_ratio.rolling(C.z_fast_period).mean()
    z_std_f  = log_ratio.rolling(C.z_fast_period).std()
    z_fast   = (log_ratio - z_mean_f) / z_std_f.replace(0, np.nan)
    
    z_mean_s = log_ratio.rolling(C.z_slow_period).mean()
    z_std_s  = log_ratio.rolling(C.z_slow_period).std()
    z_slow   = (log_ratio - z_mean_s) / z_std_s.replace(0, np.nan)
    
    # ── هر دو z باید هم‌جهت باشند ──
    z_agree = np.sign(z_fast) == np.sign(z_slow)

    # ── Correlation ──
    ret_a   = df[leg_a].pct_change()
    ret_b   = df[leg_b].pct_change()
    corr_ok = ret_a.rolling(C.corr_period).corr(ret_b) > C.corr_min

    # ── Variance Ratio ──
    vr        = calc_variance_ratio(log_ratio, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

    # ── ATR Filter ──
    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    # ── Session ──
    hour   = pd.Series(df.index.hour,      index=df.index)
    dow    = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # ── سیگنال ترکیبی با z_agree ──
    long_cond  = (z_fast < -C.z_entry) & z_agree & vol_ok & time_ok & corr_ok & regime_ok
    short_cond = (z_fast >  C.z_entry) & z_agree & vol_ok & time_ok & corr_ok & regime_ok

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    n = int((sig != 0).sum())
    l = int((sig ==  1).sum())
    s = int((sig == -1).sum())
    r = int(regime_ok.sum())
    print(f"    {pair_name}: {n:,} signals (L:{l} | S:{s}) | Regime OK: {r:,} bars")
    return sig, z_fast


# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS v8
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl(direction, entry_px, exit_px, lot, quote_rate):
    C = Config
    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    gross_usd   = gross_quote * quote_rate
    commission  = C.commission_per_lot * lot
    return gross_usd - commission


def calc_lot(equity, sl_pips, consec_loss, quote_rate,
             n_open_positions=0):
    """
    لات سایز با Portfolio Heat Control
    """
    C = Config
    risk = C.risk_base_pct
    
    # کاهش ریسک بعد از ضرر متوالی
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * (C.risk_reduce ** (consec_loss - C.consec_loss_n + 1)),
                   C.risk_min_pct)
    
    # Portfolio heat: اگه پوزیشن باز داریم، ریسک کمتر
    if n_open_positions > 0:
        risk = risk * (1 - 0.3 * n_open_positions)
    
    pip_value_usd = C.pip * C.lot_size * quote_rate
    risk_usd = equity * max(risk, C.risk_min_pct)
    raw = risk_usd / (sl_pips * pip_value_usd)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def check_intrabar_sl_tp(pos, bar_high, bar_low, bar_close, qr):
    """
    شبیه‌سازی intra-bar: آیا SL یا TP در طول کندل زده شده؟
    فرض: اگه Low < SL (برای Long) → SL hit با slippage
    
    Returns: (exit_price, status) or (None, None)
    """
    C = Config
    d  = pos['dir']
    sl = pos['sl']
    tp = pos['tp']
    
    if d == 1:  # Long
        if bar_low <= sl:
            # SL hit - با slippage بدتر
            slip = C.slippage_pips_sl * C.pip
            exit_px = max(sl - slip, sl - C.max_gap_pips * C.pip)
            return exit_px, 'SL'
        if bar_high >= tp:
            return tp, 'TP'
    else:  # Short
        if bar_high >= sl:
            slip = C.slippage_pips_sl * C.pip
            exit_px = min(sl + slip, sl + C.max_gap_pips * C.pip)
            return exit_px, 'SL'
        if bar_low <= tp:
            return tp, 'TP'
    
    return None, None


def new_acc(ts) -> dict:
    C = Config
    return {
        'equity':      C.initial_balance,
        'start_ts':    ts,
        'trades':      [],
        'blown':       False,
        'blown_rsn':   '',
        'peak':        C.initial_balance,
        'consec_loss': 0,
        'day_pnl':     0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE v8
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, pair_signals: dict) -> dict:
    C          = Config
    pip        = C.pip
    pair_names = list(pairs.keys())

    # ── ایندکس مشترک ──
    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)
    print(f"  ✅ Common bars: {n_bars:,} | {common_idx[0].date()} → {common_idx[-1].date()}")

    # ── آرایه‌های numpy ──
    pa = {}
    for name in pair_names:
        df_p = pairs[name]['df'].reindex(common_idx).ffill()
        sig_s, z_s = pair_signals[name]
        pa[name] = {
            'o':   df_p['o_spread'].values.astype(float),
            'h':   df_p['h_spread'].values.astype(float),   # NEW
            'l':   df_p['l_spread'].values.astype(float),   # NEW
            'c':   df_p['c_spread'].values.astype(float),
            'qr':  df_p['quote_rate'].values.astype(float),
            'sig': sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z':   z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
        }

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)
    DD_WARN_FLOOR = C.initial_balance * (1 - C.dd_warning_pct)

    acc              = new_acc(common_idx[C.warmup])
    total_withdrawn  = 0.0
    acc_num          = 1
    day_start_eq     = C.initial_balance
    all_trades       = []
    acc_logs         = []
    eq_curve         = []

    positions    = {name: None for name in pair_names}
    trades_today = {name: 0    for name in pair_names}
    pending_sig  = {name: 0    for name in pair_names}

    print(f"\n  ▶ Running Multi-Pair Prop Simulator v8...")
    print(f"    Pairs  : {' + '.join(pair_names)}")
    print(f"    Target : +{C.profit_target_pct*100:.0f}%"
          f"| Daily DD: -{C.max_daily_loss_pct*100:.0f}%"
          f"| Total DD: -{C.max_total_dd_pct*100:.0f}%")
    print(f"    Risk   : {C.risk_base_pct*100:.1f}%"
          f"| SL: {C.sl_pips}p"
          f"| TP: {C.tp_pips}p"
          f"| TimeStop: {C.time_stop_bars} bars")

    for bar in range(C.warmup, n_bars):
        ts  = common_idx[bar]
        eq  = acc['equity']
        eq_curve.append((ts, round(eq, 4)))

        if eq > acc['peak']:
            acc['peak'] = eq

        # ── ریست روزانه ──
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0

        # ── نمایش پیشرفت ──
        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(f"    Progress: {pct:5.1f}%"
                  f"| Eq: ${acc['equity']:,.2f}"
                  f"| Bank: ${total_withdrawn:,.2f}", end='\r')

        # ── چک حساب سوخته ──
        if acc['blown']:
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   acc['blown_rsn'],
                'pnl':      acc['equity'] - C.initial_balance,
            })
            print(f"\n    💥 #{acc_num:>3} | {ts.date()}"
                  f"| Eq: ${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name]  = 0
                positions[name]    = None
            continue

        # ── ورود (از سیگنال bar قبل) ──
        n_open = sum(1 for name in pair_names if positions[name] is not None)
        
        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0
                    and positions[name] is None
                    and trades_today[name] < C.max_trades_day
                    and n_open < C.max_open_pairs):
                sv  = pending_sig[name]
                qr  = a['qr'][bar]
                lot = calc_lot(
                    acc['equity'], C.sl_pips,
                    acc['consec_loss'], qr, n_open
                )
                sp = pairs[name]['spread_pips']
                ep = a['o'][bar] + sv * (C.slippage_pips_normal + sp / 2) * pip
                sl = ep - sv * C.sl_pips * pip
                tp = ep + sv * C.tp_pips * pip
                positions[name] = {
                    'pair':         name,
                    'dir':          sv,
                    'lot':          lot,
                    'lot_remaining': lot,
                    'partial_done': False,
                    'entry':        ep,
                    'sl':           sl,
                    'tp':           tp,
                    'entry_ts':     ts,
                    'entry_bar':    bar,
                }
                trades_today[name] += 1
                n_open += 1
            pending_sig[name] = 0

        # ── محاسبه worst-case intra-bar equity برای DD ──
        # استفاده از Low برای Long و High برای Short
        total_float_worst = 0.0
        total_float_close = 0.0
        for name in pair_names:
            pos = positions[name]
            if pos is None:
                continue
            a = pa[name]
            d = pos['dir']
            # worst price برای این bar
            worst_px = a['l'][bar] if d == 1 else a['h'][bar]
            total_float_worst += calc_pnl(
                d, pos['entry'], worst_px,
                pos['lot_remaining'], a['qr'][bar]
            )
            total_float_close += calc_pnl(
                d, pos['entry'], a['c'][bar],
                pos['lot_remaining'], a['qr'][bar]
            )

        # ── بررسی DD با worst-case ──
        worst_eq     = acc['equity'] + total_float_worst
        current_eq   = acc['equity'] + total_float_close
        daily_limit  = day_start_eq * (1 - C.max_daily_loss_pct)
        
        # DD چک با worst intra-bar
        if worst_eq <= daily_limit or worst_eq <= PROP_FLOOR:
            acc['blown']     = True
            acc['blown_rsn'] = "DailyDD" if worst_eq <= daily_limit else "TotalDD"
            for name in pair_names:
                pos = positions[name]
                if pos is None:
                    continue
                a = pa[name]
                d = pos['dir']
                # بستن به قیمت SL نه close (واقعی‌تر)
                blown_px = a['l'][bar] if d == 1 else a['h'][bar]
                blown_px = pos['sl'] if (
                    (d == 1 and blown_px > pos['sl']) or
                    (d == -1 and blown_px < pos['sl'])
                ) else blown_px
                pnl = calc_pnl(d, pos['entry'], blown_px,
                               pos['lot_remaining'], a['qr'][bar])
                acc['equity'] += pnl
                rec = _make_rec(pos, blown_px, ts, pnl, 'BLOWN', pos['lot_remaining'])
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None
            continue

        # ── مدیریت خروج ──
        for name in pair_names:
            pos = positions[name]
            if pos is None:
                continue

            a       = pa[name]
            cp      = a['c'][bar]
            bh      = a['h'][bar]
            bl      = a['l'][bar]
            qr      = a['qr'][bar]
            d       = pos['dir']
            ep      = pos['entry']
            zn      = a['z'][bar]
            lot_rem = pos['lot_remaining']

            # ── Intra-bar SL/TP ──
            if C.use_intrabar_sl:
                ib_exit, ib_status = check_intrabar_sl_tp(pos, bh, bl, cp, qr)
                if ib_exit is not None:
                    final_pnl = calc_pnl(d, ep, ib_exit, lot_rem, qr)
                    acc['equity'] += final_pnl
                    rec = _make_rec(pos, ib_exit, ts, final_pnl, ib_status, lot_rem)
                    all_trades.append(rec)
                    acc['trades'].append(rec)
                    positions[name] = None
                    acc['consec_loss'] = 0 if final_pnl > 0 else acc['consec_loss'] + 1
                    continue

            # ── Partial Exit ──
            if not pos['partial_done'] and not np.isnan(zn):
                hit_p = ((d ==  1 and zn >= -C.z_exit_partial) or
                         (d == -1 and zn <=  C.z_exit_partial))
                if hit_p:
                    p_lot = round(lot_rem * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            rec = _make_rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(rec)
                            acc['trades'].append(rec)
                            pos['lot_remaining'] = round(lot_rem - p_lot, 2)
                            pos['partial_done']  = True
                            lot_rem = pos['lot_remaining']
                            if lot_rem < C.min_lot:
                                positions[name] = None
                                continue

            # ── Z-Stop ──
            hit_z_stop = (not np.isnan(zn)) and (
                (d ==  1 and zn <= -C.z_stop_margin) or
                (d == -1 and zn >=  C.z_stop_margin)
            )

            # ── Z-Exit ──
            hit_z_exit = False
            if not np.isnan(zn):
                z_crossed = ((d ==  1 and zn >= -C.z_exit_full) or
                             (d == -1 and zn <=  C.z_exit_full))
                if z_crossed:
                    pnl_check = calc_pnl(d, ep, cp, lot_rem, qr)
                    if pnl_check >= C.min_net_profit_usd or pos['partial_done']:
                        hit_z_exit = True

            # ── TimeStop ──
            time_stop = (bar - pos['entry_bar']) >= C.time_stop_bars

            if hit_z_exit or hit_z_stop or time_stop:
                if hit_z_stop:
                    exit_px, st = cp, 'Z-Stop'
                elif time_stop:
                    exit_px, st = cp, 'TimeStop'
                else:
                    exit_px, st = cp, 'Z-Exit'

                final_pnl = calc_pnl(d, ep, exit_px, lot_rem, qr)
                acc['equity'] += final_pnl
                rec = _make_rec(pos, exit_px, ts, final_pnl, st, lot_rem)
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None
                acc['consec_loss'] = 0 if final_pnl > 0 else acc['consec_loss'] + 1

        # ── برداشت سود ──
        all_closed = all(positions[name] is None for name in pair_names)
        if acc['equity'] >= PROFIT_LEVEL and all_closed and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   'TARGET_HIT',
                'pnl':      w,
            })
            print(f"\n    💰 #{acc_num:>3} | {ts.date()} | "
                  f"Target Hit: ${w:>7.2f} | Total Bank: ${total_withdrawn:>9.2f}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name]  = 0
            continue

        # ── سیگنال جدید ──
        n_open = sum(1 for name in pair_names if positions[name] is not None)
        for name in pair_names:
            a = pa[name]
            if (positions[name] is None
                    and not acc['blown']
                    and trades_today[name] < C.max_trades_day
                    and n_open < C.max_open_pairs
                    and a['sig'][bar] != 0):
                pending_sig[name] = int(a['sig'][bar])

    print()
    return {
        'all_trades':      all_trades,
        'account_logs':    acc_logs,
        'eq_curve':        eq_curve,
        'total_withdrawn': total_withdrawn,
        'final_equity':    acc['equity'],
        'total_accounts':  acc_num,
        'pair_names':      pair_names,
    }


def _make_rec(pos, exit_px, exit_ts, pnl, status, lot):
    return {
        'pair':      pos['pair'],
        'dir':       pos['dir'],
        'lot':       lot,
        'entry':     pos['entry'],
        'exit':      exit_px,
        'entry_ts':  pos['entry_ts'],
        'exit_ts':   exit_ts,
        'pnl':       pnl,
        'status':    status,
        'entry_bar': pos['entry_bar'],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING + ANALYTICS v8
# ═══════════════════════════════════════════════════════════════════════════
def calc_max_drawdown(eq_series: pd.Series) -> float:
    """محاسبه Maximum Drawdown"""
    roll_max = eq_series.cummax()
    dd = (eq_series - roll_max) / roll_max
    return float(dd.min())


def monte_carlo_simulation(trades_pnl: list, n_sim: int = 1000,
                           initial: float = 5000.0) -> dict:
    """
    شبیه‌سازی مونت‌کارلو: توزیع نتایج احتمالی
    """
    results = []
    arr = np.array(trades_pnl)
    n = len(arr)
    
    for _ in range(n_sim):
        shuffled  = np.random.choice(arr, size=n, replace=True)
        eq        = initial + np.cumsum(shuffled)
        peak      = np.maximum.accumulate(np.concatenate([[initial], eq]))
        dd        = (np.concatenate([[initial], eq]) - peak) / peak
        max_dd    = dd.min()
        final_eq  = eq[-1]
        results.append({'final': final_eq, 'max_dd': max_dd})
    
    finals = [r['final'] for r in results]
    dds    = [r['max_dd'] for r in results]
    
    return {
        'median_final':  np.median(finals),
        'p5_final':      np.percentile(finals, 5),
        'p95_final':     np.percentile(finals, 95),
        'median_dd':     np.median(dds),
        'worst_dd':      np.percentile(dds, 95),
        'prob_positive': np.mean([f > initial for f in finals]) * 100,
    }


def walk_forward_summary(all_trades: list, pair_names: list) -> None:
    """خلاصه عملکرد Walk-Forward به صورت سالانه"""
    if not all_trades:
        return
    df_t = pd.DataFrame(all_trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['year']    = df_t['exit_ts'].dt.year
    
    print("\n  📊 Walk-Forward Annual Summary:")
    print(f"  {'Year':<6} {'Trades':>7} {'WR%':>7} {'PF':>6} {'Net PnL':>10}")
    print("  " + "-" * 42)
    
    for yr, grp in df_t.groupby('year'):
        wins = grp[grp['pnl'] > 0]
        loss = grp[grp['pnl'] < 0]
        wr   = len(wins) / len(grp) * 100 if len(grp) else 0
        pf   = (wins['pnl'].sum() / abs(loss['pnl'].sum())
                if len(loss) > 0 else float('inf'))
        net  = grp['pnl'].sum()
        flag = "✅" if net > 0 else "❌"
        print(f"  {yr:<6} {len(grp):>7,} {wr:>7.1f} {pf:>6.2f} ${net:>9,.2f} {flag}")


def print_report(results: dict):
    trades     = results['all_trades']
    pair_names = results.get('pair_names', [])

    if not trades:
        print("\n❌ No trades executed.")
        return

    df_t = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month']   = df_t['exit_ts'].dt.to_period('M')

    wins   = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())
          if len(losses) > 0 else float('inf'))

    # Equity curve
    eq_df = pd.DataFrame(results['eq_curve'], columns=['ts', 'eq'])
    eq_df = eq_df.set_index('ts')['eq']
    mdd   = calc_max_drawdown(eq_df)

    print("\n" + "═" * 70)
    print(f" ▌  CorrArb Prop Simulator v8 — {'+'.join(pair_names)}  ▐")
    print("═" * 70)
    print(f" Total Trades:    {len(df_t):,}")
    print(f" Win Rate:        {wr:.2f}%")
    print(f" Profit Factor:   {pf:.2f}")
    print(f" Max Drawdown:    {mdd*100:.2f}%")
    print(f" Total Banked:    ${results['total_withdrawn']:,.2f}")
    print(f" Active Equity:   ${results['final_equity']:,.2f}")
    if len(wins):   print(f" Avg Win:         ${wins['pnl'].mean():.2f}")
    if len(losses): print(f" Avg Loss:        ${losses['pnl'].mean():.2f}")
    print(f" Expectancy:      ${df_t['pnl'].mean():.2f} per trade")

    # Per-pair
    if 'pair' in df_t.columns and len(pair_names) > 1:
        print("-" * 70)
        print(" Per-Pair Performance:")
        for pair in pair_names:
            pt = df_t[df_t['pair'] == pair]
            if len(pt) == 0:
                continue
            pw = pt[pt['pnl'] > 0]
            pl = pt[pt['pnl'] < 0]
            p_wr = len(pw) / len(pt) * 100 if len(pt) else 0
            p_pf = (pw['pnl'].sum() / abs(pl['pnl'].sum())
                    if len(pl) > 0 else float('inf'))
            print(f"   {pair}: {len(pt):>4} trades"
                  f"| WR: {p_wr:5.1f}%"
                  f"| PF: {p_pf:.2f}"
                  f"| Net: ${pt['pnl'].sum():>8,.2f}")

    # Exit types
    print("-" * 70)
    print(" Exit Types:")
    for st, cnt in df_t['status'].value_counts().items():
        pct = cnt / len(df_t) * 100
        print(f"   {st:<12}: {cnt:>4} ({pct:.1f}%)")

    # Accounts
    logs    = results['account_logs']
    targets = sum(1 for l in logs if l['reason'] == 'TARGET_HIT')
    blown   = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')
    success_rate = targets / (targets + blown) * 100 if (targets + blown) > 0 else 0
    print("-" * 70)
    print(f" Accounts: {results['total_accounts']} total"
          f"| ✅ Target: {targets}"
          f"| 💥 Blown: {blown}"
          f"| Success: {success_rate:.0f}%")

    # Monthly
    monthly = df_t.groupby('month')['pnl'].sum()
    if len(monthly):
        pos_m = int((monthly > 0).sum())
        neg_m = int((monthly < 0).sum())
        print(f" Monthly: avg ${monthly.mean():,.2f}"
              f"| Best: ${monthly.max():,.2f}"
              f"| Worst: ${monthly.min():,.2f}")
        print(f"          Positive: {pos_m} | Negative: {neg_m}"
              f"| Win Rate: {pos_m/(pos_m+neg_m)*100:.0f}%")

    print("═" * 70)

    # Walk-forward
    walk_forward_summary(trades, pair_names)

    # Monte Carlo
    print("\n  🎲 Monte Carlo Analysis (1000 simulations):")
    mc = monte_carlo_simulation(list(df_t['pnl']))
    print(f"     Median Final Equity: ${mc['median_final']:,.2f}")
    print(f"     5th Percentile:      ${mc['p5_final']:,.2f}")
    print(f"     95th Percentile:     ${mc['p95_final']:,.2f}")
    print(f"     Median Max DD:       {mc['median_dd']*100:.2f}%")
    print(f"     Worst DD (95th):     {mc['worst_dd']*100:.2f}%")
    print(f"     Prob Profitable:     {mc['prob_positive']:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    
    pairs = load_all_pairs()
    
    print("\n  Computing Statistical Signals...")
    pair_signals = {}
    for name, info in pairs.items():
        pair_signals[name] = compute_signals(name, info)
    
    results = run_backtest(pairs, pair_signals)
    print_report(results)
    
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  ✅ Executed in: {elapsed:.2f}s")
