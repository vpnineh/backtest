"""
CorrArb Prop Simulator v9 — Realistic & Bias-Free
==================================================
بهبودهای v9:
  ① رفع Look-Ahead Bias (shift(1) روی همه اندیکاتورها)
  ② کمیسیون واقعی synthetic ($14/lot = 2 legs)
  ③ اسپرد داینامیک (بر اساس ساعت و نوسان)
  ④ Hurst Exponent filter (mean-reversion strength)
  ⑤ Dual-timeframe Z-Score (fast + slow)
  ⑥ News hours filter (اجتناب از 30 دقیقه قبل/بعد خبر)
  ⑦ Kelly Criterion sizing (Half-Kelly)
  ⑧ Trailing Stop بعد از partial
  ⑨ ATR filter سخت‌تر + session تمیزتر
"""

import pandas as pd
import numpy as np
import glob
import zipfile
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── مدیریت ریسک ──
    risk_base_pct      = 0.015
    risk_min_pct       = 0.0075
    consec_loss_n      = 2
    risk_reduce        = 0.5

    # ── هزینه‌های واقعی ──
    spread_pips_normal = 1.2
    spread_pips_high   = 3.0
    commission_per_lot = 14.0      # ✅ 2 legs (synthetic pair)
    slippage_pips      = 0.5

    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 2.0
    min_lot  = 0.01
    warmup   = 500

    # ── Z-Score (dual timeframe) ──
    z_fast_period      = 96
    z_slow_period      = 288
    z_entry            = 2.2
    z_slow_min         = 1.0       # slow z باید هم‌جهت باشه
    z_exit_partial     = 0.5
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 20.0

    # ── فیلترها ──
    corr_period        = 96
    corr_min           = 0.82
    hour_start         = 3
    hour_end           = 18
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # ── خروج ──
    sl_pips            = 30.0
    tp_pips            = 90.0
    time_stop_bars     = 36

    # ── ATR ──
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 2.5
    atr_min_mult       = 0.5

    # ── Variance Ratio ──
    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.88

    # ── Hurst ──
    hurst_period       = 100
    hurst_max          = 0.45

    # ── Partial + Trailing ──
    partial_ratio      = 0.50
    trail_pips         = 15.0

    # ── News filter ──
    news_hours         = [(8, 30), (13, 30), (14, 0), (15, 0)]
    news_buffer_mins   = 30

    # ── Kelly ──
    kelly_fraction     = 0.5       # Half-Kelly
    default_wr         = 0.60
    default_rr         = 2.5


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
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
                    csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                    if csv_name is None:
                        continue
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f, sep=';', header=None,
                                         names=['ts', 'o', 'h', 'l', 'c', 'v'])
            else:
                df = pd.read_csv(p, sep=';', header=None,
                                 names=['ts', 'o', 'h', 'l', 'c', 'v'])
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


def build_spread_df(df_a, sfx_a, df_b, sfx_b):
    merged = df_a.join(df_b, how='inner').dropna()
    if len(merged) == 0:
        raise ValueError(f"No common timestamps {sfx_a}/{sfx_b}")
    merged['c_spread'] = merged[f'c_{sfx_a}'] / merged[f'c_{sfx_b}']
    merged['o_spread'] = merged[f'o_{sfx_a}'] / merged[f'o_{sfx_b}']
    merged['h_spread'] = merged[f'h_{sfx_a}'] / merged[f'l_{sfx_b}']
    merged['l_spread'] = merged[f'l_{sfx_a}'] / merged[f'h_{sfx_b}']
    merged['quote_rate'] = merged[f'c_{sfx_b}']
    return merged[merged.index.weekday < 5].copy()


def load_all_pairs() -> dict:
    print("\n  Loading and syncing datasets...")
    pairs = {}
    try:
        eur = to_15min(load_raw('data/*EURUSD*.csv', is_zip=False), 'eur')
        gbp = to_15min(load_raw('data/*GBPUSD*.csv', is_zip=False), 'gbp')
        df = build_spread_df(eur, 'eur', gbp, 'gbp')
        pairs['EURGBP'] = {'df': df, 'leg_a': 'c_eur', 'leg_b': 'c_gbp'}
        print(f"  ✅ EURGBP : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ EURGBP : {e}")
    try:
        aud = to_15min(load_raw('data/HISTDATA*AUDUSD*.zip', is_zip=True), 'aud')
        nzd = to_15min(load_raw('data/HISTDATA*NZDUSD*.zip', is_zip=True), 'nzd')
        df = build_spread_df(aud, 'aud', nzd, 'nzd')
        pairs['AUDNZD'] = {'df': df, 'leg_a': 'c_aud', 'leg_b': 'c_nzd'}
        print(f"  ✅ AUDNZD : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ AUDNZD : {e}")
    if not pairs:
        raise RuntimeError("No pairs loaded.")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_variance_ratio(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()
    return vark / (k * var1.replace(0, np.nan))


def calc_hurst_fast(series: pd.Series, period: int = 100) -> pd.Series:
    """نسخه سریع‌تر Hurst با کمتر کردن lag ها"""
    def _hurst(x):
        if len(x) < 20:
            return np.nan
        x = np.asarray(x)
        if np.std(x) == 0:
            return np.nan
        lags = [2, 4, 8, 16]
        tau = []
        for lag in lags:
            if lag >= len(x):
                continue
            diff = x[lag:] - x[:-lag]
            if len(diff) > 0 and np.std(diff) > 0:
                tau.append(np.sqrt(np.std(diff)))
        if len(tau) < 3:
            return np.nan
        try:
            lag_log = np.log(lags[:len(tau)])
            tau_log = np.log(tau)
            h = np.polyfit(lag_log, tau_log, 1)[0] * 2.0
            return float(np.clip(h, 0, 1))
        except:
            return np.nan
    return series.rolling(period).apply(_hurst, raw=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION — BIAS-FREE
# ═══════════════════════════════════════════════════════════════════════════
def compute_signals(pair_name: str, pair_info: dict) -> tuple:
    """✅ تمام اندیکاتورها با shift(1) → بدون Look-Ahead Bias"""
    C   = Config
    df  = pair_info['df']
    leg_a, leg_b = pair_info['leg_a'], pair_info['leg_b']

    log_ratio = np.log(df['c_spread'])

    # Fast Z (shifted)
    z_mean = log_ratio.rolling(C.z_fast_period).mean().shift(1)
    z_std  = log_ratio.rolling(C.z_fast_period).std().shift(1)
    z_score = (log_ratio.shift(1) - z_mean) / z_std.replace(0, np.nan)

    # Slow Z (shifted)
    z_mean_s = log_ratio.rolling(C.z_slow_period).mean().shift(1)
    z_std_s  = log_ratio.rolling(C.z_slow_period).std().shift(1)
    z_slow = (log_ratio.shift(1) - z_mean_s) / z_std_s.replace(0, np.nan)

    # Correlation (shifted)
    ret_a = df[leg_a].pct_change()
    ret_b = df[leg_b].pct_change()
    corr_ok = ret_a.rolling(C.corr_period).corr(ret_b).shift(1) > C.corr_min

    # Variance Ratio (shifted)
    vr = calc_variance_ratio(log_ratio, C.vr_k, C.vr_period).shift(1)
    regime_ok = vr < C.vr_max

    # Hurst (shifted)
    print(f"    {pair_name}: computing Hurst (slow)...", end='\r')
    hurst = calc_hurst_fast(log_ratio, C.hurst_period).shift(1)
    hurst_ok = hurst < C.hurst_max

    # ATR (shifted)
    atr = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    atr_ratio = (atr / atr_ma).shift(1)
    vol_ok = (atr_ratio > C.atr_min_mult) & (atr_ratio < C.atr_max_mult)

    # Session
    hour = pd.Series(df.index.hour, index=df.index)
    minute = pd.Series(df.index.minute, index=df.index)
    dow = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # News filter
    news_mask = pd.Series(False, index=df.index)
    for (h, m) in C.news_hours:
        cur_min = hour * 60 + minute
        news_min = h * 60 + m
        near = cur_min.between(news_min - C.news_buffer_mins, news_min + C.news_buffer_mins)
        news_mask = news_mask | near
    time_ok = time_ok & ~news_mask

    # Combined
    long_cond = ((z_score < -C.z_entry) & (z_slow < -C.z_slow_min) &
                 vol_ok & time_ok & corr_ok & regime_ok & hurst_ok)
    short_cond = ((z_score > C.z_entry) & (z_slow > C.z_slow_min) &
                  vol_ok & time_ok & corr_ok & regime_ok & hurst_ok)

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    n = int((sig != 0).sum())
    l = int((sig == 1).sum())
    s = int((sig == -1).sum())
    r = int(regime_ok.sum())
    h_ok = int(hurst_ok.sum())
    print(f"    {pair_name}: {n:,} signals (L:{l} | S:{s}) | Regime: {r:,} | Hurst OK: {h_ok:,}")
    return sig, z_score, atr_ratio


# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamic_spread(hour: int, atr_ratio: float) -> float:
    C = Config
    base = C.spread_pips_normal
    if hour in [0, 1, 22, 23]:
        base *= 2.0
    elif hour in [8, 13, 14]:
        base *= 1.8
    if not np.isnan(atr_ratio):
        if atr_ratio > 2.0:
            base *= 1.5
        elif atr_ratio > 1.5:
            base *= 1.2
    return min(base, C.spread_pips_high)


def calc_pnl(direction, entry_px, exit_px, lot, quote_rate):
    C = Config
    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    gross_usd = gross_quote * quote_rate
    commission = C.commission_per_lot * lot
    return gross_usd - commission


def calc_lot_kelly(equity, sl_pips, consec_loss, quote_rate, recent_wr, recent_rr):
    C = Config
    if recent_rr > 0:
        kelly_full = recent_wr - (1 - recent_wr) / recent_rr
    else:
        kelly_full = 0
    kelly = max(0, kelly_full * C.kelly_fraction)
    risk = min(kelly if kelly > 0 else C.risk_base_pct, C.risk_base_pct)
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
    pip_value_usd = C.pip * C.lot_size * quote_rate
    risk_usd = equity * risk
    raw = risk_usd / (sl_pips * pip_value_usd)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def update_trailing_stop(pos, current_price):
    C = Config
    if not pos.get('partial_done', False):
        return
    d = pos['dir']
    trail = C.trail_pips * C.pip
    if d == 1:
        new_sl = current_price - trail
        if new_sl > pos['sl']:
            pos['sl'] = new_sl
    else:
        new_sl = current_price + trail
        if new_sl < pos['sl']:
            pos['sl'] = new_sl


def new_acc(ts):
    C = Config
    return {'equity': C.initial_balance, 'start_ts': ts, 'trades': [],
            'blown': False, 'blown_rsn': '', 'peak': C.initial_balance,
            'consec_loss': 0, 'recent_wins': [], 'recent_rrs': []}


def update_recent_stats(acc, pnl, sl_usd):
    win = 1 if pnl > 0 else 0
    acc['recent_wins'].append(win)
    if len(acc['recent_wins']) > 30:
        acc['recent_wins'].pop(0)
    if sl_usd > 0:
        acc['recent_rrs'].append(abs(pnl) / sl_usd)
        if len(acc['recent_rrs']) > 30:
            acc['recent_rrs'].pop(0)


def get_recent_stats(acc):
    C = Config
    if len(acc['recent_wins']) < 10:
        return C.default_wr, C.default_rr
    wr = np.mean(acc['recent_wins'])
    rr = np.mean(acc['recent_rrs']) if acc['recent_rrs'] else C.default_rr
    return float(wr), float(rr)


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs, pair_signals):
    C = Config
    pip = C.pip
    pair_names = list(pairs.keys())

    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)
    print(f"  ✅ Common bars: {n_bars:,} | {common_idx[0].date()} → {common_idx[-1].date()}")

    pa = {}
    for name in pair_names:
        df_p = pairs[name]['df'].reindex(common_idx).ffill()
        sig_s, z_s, atr_s = pair_signals[name]
        pa[name] = {
            'o': df_p['o_spread'].values.astype(float),
            'c': df_p['c_spread'].values.astype(float),
            'qr': df_p['quote_rate'].values.astype(float),
            'sig': sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z': z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
            'atr_r': atr_s.reindex(common_idx).fillna(1.0).values.astype(float),
        }

    PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    acc = new_acc(common_idx[C.warmup])
    total_withdrawn = 0.0
    acc_num = 1
    day_start_eq = C.initial_balance
    all_trades, acc_logs, eq_curve = [], [], []
    positions = {n: None for n in pair_names}
    trades_today = {n: 0 for n in pair_names}
    pending_sig = {n: 0 for n in pair_names}

    print(f"\n  ▶ Running v9 Bias-Free Simulator...")
    print(f"    Pairs    : {' + '.join(pair_names)}")
    print(f"    Target   : +{C.profit_target_pct*100:.0f}% | DailyDD: -{C.max_daily_loss_pct*100:.0f}% | TotalDD: -{C.max_total_dd_pct*100:.0f}%")
    print(f"    Risk     : {C.risk_base_pct*100:.1f}% (Kelly) | SL: {C.sl_pips}p | TP: {C.tp_pips}p")
    print(f"    Comm     : ${C.commission_per_lot}/lot (2-leg) | Trail: {C.trail_pips}p")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq = acc['equity']
        eq_curve.append((ts, round(eq, 4)))
        if eq > acc['peak']:
            acc['peak'] = eq

        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0

        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(f"    Progress: {pct:5.1f}% | Eq: ${acc['equity']:,.2f} | Bank: ${total_withdrawn:,.2f}", end='\r')

        if acc['blown']:
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'],
                            'end_ts': ts, 'reason': acc['blown_rsn'],
                            'pnl': acc['equity'] - C.initial_balance})
            print(f"\n    💥 #{acc_num:>3} | {ts.date()} | Eq: ${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name] = 0
                positions[name] = None
            continue

        # ── ورود ──
        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0 and positions[name] is None
                    and trades_today[name] < C.max_trades_day):
                sv = pending_sig[name]
                qr = a['qr'][bar]
                wr, rr = get_recent_stats(acc)
                lot = calc_lot_kelly(acc['equity'], C.sl_pips, acc['consec_loss'], qr, wr, rr)
                
                dyn_spread = get_dynamic_spread(ts.hour, a['atr_r'][bar])
                ep = a['o'][bar] + sv * (C.slippage_pips + dyn_spread / 2) * pip
                sl = ep - sv * C.sl_pips * pip
                tp = ep + sv * C.tp_pips * pip
                
                positions[name] = {
                    'pair': name, 'dir': sv, 'lot': lot, 'lot_remaining': lot,
                    'partial_done': False, 'entry': ep, 'sl': sl, 'tp': tp,
                    'entry_ts': ts, 'entry_bar': bar,
                    'sl_usd': C.sl_pips * pip * C.lot_size * qr * lot
                }
                trades_today[name] += 1
            pending_sig[name] = 0

        # ── DD check ──
        total_float = 0.0
        for name in pair_names:
            pos = positions[name]
            if pos is not None:
                a = pa[name]
                total_float += calc_pnl(pos['dir'], pos['entry'], a['c'][bar],
                                       pos['lot_remaining'], a['qr'][bar])

        current_eq = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            acc['blown'] = True
            acc['blown_rsn'] = "DailyDD" if current_eq <= daily_limit else "TotalDD"
            for name in pair_names:
                pos = positions[name]
                if pos is None:
                    continue
                a = pa[name]
                pnl = calc_pnl(pos['dir'], pos['entry'], a['c'][bar],
                              pos['lot_remaining'], a['qr'][bar])
                acc['equity'] += pnl
                rec = _make_rec(pos, a['c'][bar], ts, pnl, 'BLOWN', pos['lot_remaining'])
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None
            continue

        # ── مدیریت خروج ──
        for name in pair_names:
            pos = positions[name]
            if pos is None:
                continue

            a = pa[name]
            cp = a['c'][bar]
            qr = a['qr'][bar]
            d = pos['dir']
            ep = pos['entry']
            zn = a['z'][bar]
            lot_rem = pos['lot_remaining']

            # Trailing Stop
            update_trailing_stop(pos, cp)

            # Partial Exit
            if not pos['partial_done'] and not np.isnan(zn):
                hit_p = ((d == 1 and zn >= -C.z_exit_partial) or
                        (d == -1 and zn <= C.z_exit_partial))
                if hit_p:
                    p_lot = round(lot_rem * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            rec = _make_rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(rec)
                            acc['trades'].append(rec)
                            update_recent_stats(acc, p_pnl, pos['sl_usd'] * (p_lot/pos['lot']))
                            pos['lot_remaining'] = round(lot_rem - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl'] = pos['entry']  # Break-Even
                            lot_rem = pos['lot_remaining']
                            if lot_rem < C.min_lot:
                                positions[name] = None
                                continue

            hit_z_stop = (not np.isnan(zn)) and (
                (d == 1 and zn <= -C.z_stop_margin) or
                (d == -1 and zn >= C.z_stop_margin))

            hit_z_exit = False
            if not np.isnan(zn):
                z_crossed = ((d == 1 and zn >= -C.z_exit_full) or
                            (d == -1 and zn <= C.z_exit_full))
                if z_crossed:
                    pnl_chk = calc_pnl(d, ep, cp, lot_rem, qr)
                    if pnl_chk >= C.min_net_profit_usd or pos['partial_done']:
                        hit_z_exit = True

            hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
            hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])

            bars_open = bar - pos['entry_bar']
            cur_pnl = calc_pnl(d, ep, cp, lot_rem, qr)
            time_stop = ((bars_open >= C.time_stop_bars and cur_pnl < 0) or
                        (bars_open >= C.time_stop_bars * 2))

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                if hit_sl:
                    exit_px, st = pos['sl'], 'SL'
                elif hit_tp:
                    exit_px, st = pos['tp'], 'TP'
                elif hit_z_stop:
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
                update_recent_stats(acc, final_pnl, pos['sl_usd'])
                positions[name] = None

                if final_pnl > 0:
                    acc['consec_loss'] = 0
                else:
                    acc['consec_loss'] += 1

        # ── برداشت ──
        all_closed = all(positions[n] is None for n in pair_names)
        if acc['equity'] >= PROFIT_LEVEL and all_closed and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'],
                           'end_ts': ts, 'reason': 'TARGET_HIT', 'pnl': w})
            print(f"\n    💰 #{acc_num:>3} | {ts.date()} | Target Hit: ${w:>7.2f} | Total Bank: ${total_withdrawn:>9.2f}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name] = 0
            continue

        # ── سیگنال جدید (اجرا در bar بعدی) ──
        for name in pair_names:
            a = pa[name]
            if (positions[name] is None and not acc['blown']
                    and trades_today[name] < C.max_trades_day
                    and a['sig'][bar] != 0):
                pending_sig[name] = int(a['sig'][bar])

    print()
    return {'all_trades': all_trades, 'account_logs': acc_logs,
            'eq_curve': eq_curve, 'total_withdrawn': total_withdrawn,
            'final_equity': acc['equity'], 'total_accounts': acc_num,
            'pair_names': pair_names}


def _make_rec(pos, exit_px, exit_ts, pnl, status, lot):
    return {'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot,
            'entry': pos['entry'], 'exit': exit_px,
            'entry_ts': pos['entry_ts'], 'exit_ts': exit_ts,
            'pnl': pnl, 'status': status, 'entry_bar': pos['entry_bar']}


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results):
    trades = results['all_trades']
    pair_names = results.get('pair_names', [])
    if not trades:
        print("\n❌ No trades executed.")
        return

    df_t = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month'] = df_t['exit_ts'].dt.to_period('M')

    wins = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf')

    print("\n" + "═" * 70)
    print(f" ▌  CorrArb Prop Simulator v9 — {'+'.join(pair_names)}  ▐")
    print("═" * 70)
    print(f" Total Trades:    {len(df_t):,}")
    print(f" Win Rate:        {wr:.2f}%")
    print(f" Profit Factor:   {pf:.2f}")
    print(f" Total Banked:    ${results['total_withdrawn']:,.2f}")
    print(f" Active Equity:   ${results['final_equity']:,.2f}")
    if len(wins):
        print(f" Avg Win:         ${wins['pnl'].mean():.2f}")
    if len(losses):
        print(f" Avg Loss:        ${losses['pnl'].mean():.2f}")

    if 'pair' in df_t.columns and len(pair_names) > 1:
        print("-" * 70)
        print(" عملکرد هر جفت ارز:")
        for pair in pair_names:
            pt = df_t[df_t['pair'] == pair]
            if len(pt) == 0:
                continue
            pw = pt[pt['pnl'] > 0]
            pl = pt[pt['pnl'] < 0]
            p_wr = len(pw) / len(pt) * 100 if len(pt) else 0
            p_pf = pw['pnl'].sum() / abs(pl['pnl'].sum()) if len(pl) > 0 else float('inf')
            print(f"   {pair}: {len(pt):>4} trades | WR: {p_wr:5.1f}% | PF: {p_pf:.2f} | Net PnL: ${pt['pnl'].sum():>8,.2f}")

    print("-" * 70)
    print(" خروج‌ها بر اساس نوع:")
    for st, cnt in df_t['status'].value_counts().items():
        print(f"   {st:<12}: {cnt:>4} معامله")

    logs = results['account_logs']
    targets = sum(1 for l in logs if l['reason'] == 'TARGET_HIT')
    blown = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')
    print("-" * 70)
    print(f" حساب‌ها: {results['total_accounts']} کل | ✅ Target Hit: {targets} | 💥 Blown: {blown}")

    monthly = df_t.groupby('month')['pnl'].sum()
    if len(monthly):
        pos_m = int((monthly > 0).sum())
        neg_m = int((monthly < 0).sum())
        print(f" ماهانه   : avg ${monthly.mean():,.2f} | Best: ${monthly.max():,.2f} | Worst: ${monthly.min():,.2f}")
        print(f"            ماه‌های مثبت: {pos_m} | ماه‌های منفی: {neg_m}")

    # Equity curve stats
    if results['eq_curve']:
        eq_df = pd.DataFrame(results['eq_curve'], columns=['ts', 'eq'])
        eq_df['peak'] = eq_df['eq'].cummax()
        eq_df['dd'] = (eq_df['eq'] - eq_df['peak']) / eq_df['peak'] * 100
        max_dd = eq_df['dd'].min()
        print(f" Max Drawdown: {max_dd:.2f}%")

    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    pairs = load_all_pairs()

    print("\n  Computing Statistical Signals (with Hurst — may take 1-2 min)...")
    pair_signals = {}
    for name, info in pairs.items():
        pair_signals[name] = compute_signals(name, info)

    results = run_backtest(pairs, pair_signals)
    print_report(results)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  ✅ Executed in: {elapsed:.2f}s")
