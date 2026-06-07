"""
راه‌حل: سیستم دو حالته
  حالت ۱: Mean-Reversion (الان)
  حالت ۲: Trend-Following (وقتی mean-rev کار نمیکنه)

وقتی یکی کار نمیکنه، دیگری کار میکنه
این تنها راه کاهش ماه‌های منفی است
"""

import pandas as pd
import numpy as np
import glob, zipfile, os, warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class Config:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    risk_base_pct      = 0.012
    risk_min_pct       = 0.006
    consec_loss_n      = 2
    risk_reduce        = 0.6
    max_open_pairs     = 2

    dd_shield_pct      = 0.07
    dd_shield_risk     = 0.006

    spread_pips_eurgbp = 1.2
    spread_pips_audnzd = 1.5
    commission_per_lot = 7.0
    slippage_entry_pips = 0.3
    slippage_sl_pips    = 0.8

    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── Mean-Reversion params ──
    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.8
    z_exit_full        = 0.3
    z_stop_margin      = 3.5
    min_net_profit_usd = 20.0

    # ── Trend-Following params (NEW) ──
    ema_fast           = 20
    ema_slow           = 60
    trend_sl_pips      = 35.0
    trend_tp_pips      = 105.0    # R:R = 3:1
    trend_min_atr_mult = 1.0      # حداقل volatility برای trend

    # ── Regime Detection ──
    regime_period      = 100      # پنجره تشخیص رژیم
    # VR < vr_mr  → Mean-Reversion
    # VR > vr_tr  → Trending
    vr_mr_threshold    = 0.90
    vr_tr_threshold    = 1.10

    # ── Hurst ──
    hurst_period       = 100
    # H < 0.45 → mean-reverting
    # H > 0.55 → trending

    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 3
    hour_end           = 18
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    sl_pips            = 30.0
    tp_pips            = 90.0
    time_stop_bars     = 48

    use_trailing       = True
    trail_activate_z   = 0.5
    trail_sl_pips      = 15.0

    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 2.5
    atr_min_mult       = 0.5

    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.95

    partial_ratio      = 0.50


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_raw(pattern: str, is_zip: bool) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files: {pattern}")
    frames = []
    for p in paths:
        try:
            if is_zip:
                with zipfile.ZipFile(p, 'r') as z:
                    csv_name = next(
                        (f for f in z.namelist()
                         if f.lower().endswith('.csv')), None
                    )
                    if not csv_name:
                        continue
                    with z.open(csv_name) as f:
                        df = pd.read_csv(
                            f, sep=';', header=None,
                            names=['ts','o','h','l','c','v']
                        )
            else:
                df = pd.read_csv(
                    p, sep=';', header=None,
                    names=['ts','o','h','l','c','v']
                )
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ {os.path.basename(p)}: {e}")
    if not frames:
        raise ValueError(f"No data: {pattern}")
    raw = pd.concat(frames, ignore_index=True).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def to_15min(raw: pd.DataFrame, sfx: str) -> pd.DataFrame:
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()


def build_spread_df(df_a, sfx_a, df_b, sfx_b) -> pd.DataFrame:
    merged = df_a.join(df_b, how='inner').dropna()
    merged['c_spread']   = merged[f'c_{sfx_a}'] / merged[f'c_{sfx_b}']
    merged['o_spread']   = merged[f'o_{sfx_a}'] / merged[f'o_{sfx_b}']
    merged['h_spread']   = merged[f'h_{sfx_a}'] / merged[f'l_{sfx_b}']
    merged['l_spread']   = merged[f'l_{sfx_a}'] / merged[f'h_{sfx_b}']
    merged['quote_rate'] = merged[f'c_{sfx_b}']
    return merged[merged.index.weekday < 5].copy()


def load_all_pairs() -> dict:
    print("\n  Loading and syncing datasets...")
    pairs = {}
    try:
        eur = to_15min(load_raw('data/*EURUSD*.csv', False), 'eur')
        gbp = to_15min(load_raw('data/*GBPUSD*.csv', False), 'gbp')
        df  = build_spread_df(eur, 'eur', gbp, 'gbp')
        pairs['EURGBP'] = {
            'df': df, 'leg_a': 'c_eur', 'leg_b': 'c_gbp',
            'spread_pips': Config.spread_pips_eurgbp
        }
        print(f"  ✅ EURGBP: {len(df):,} candles")
    except Exception as e:
        print(f"  ❌ EURGBP: {e}")
    try:
        aud = to_15min(load_raw('data/HISTDATA*AUDUSD*.zip', True), 'aud')
        nzd = to_15min(load_raw('data/HISTDATA*NZDUSD*.zip', True), 'nzd')
        df  = build_spread_df(aud, 'aud', nzd, 'nzd')
        pairs['AUDNZD'] = {
            'df': df, 'leg_a': 'c_aud', 'leg_b': 'c_nzd',
            'spread_pips': Config.spread_pips_audnzd
        }
        print(f"  ✅ AUDNZD: {len(df):,} candles")
    except Exception as e:
        print(f"  ❌ AUDNZD: {e}")
    if not pairs:
        raise RuntimeError("No pairs loaded.")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14) -> pd.Series:
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_variance_ratio(series: pd.Series, k: int, window: int) -> pd.Series:
    r1 = series.diff(1)
    rk = series.diff(k)
    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()
    return vark / (k * var1.replace(0, np.nan))


def calc_hurst(series: pd.Series, window: int) -> pd.Series:
    """
    Hurst Exponent rolling
    H < 0.45 → mean-reverting
    H > 0.55 → trending
    """
    def _hurst(x):
        n = len(x)
        if n < 20:
            return np.nan
        lags = [2, 4, 8, 16]
        lags = [l for l in lags if l < n // 2]
        if len(lags) < 2:
            return np.nan
        tau = []
        for lag in lags:
            diff = x[lag:] - x[:-lag]
            std  = np.std(diff)
            tau.append(std if std > 0 else np.nan)
        valid = [(np.log(l), np.log(t))
                 for l, t in zip(lags, tau)
                 if t is not None and not np.isnan(t)]
        if len(valid) < 2:
            return np.nan
        xs, ys = zip(*valid)
        poly = np.polyfit(xs, ys, 1)
        return float(poly[0])

    return series.rolling(window).apply(_hurst, raw=True)


def detect_regime(
    log_spread: pd.Series,
    vr_period: int,
    vr_k: int,
    hurst_period: int
) -> pd.DataFrame:
    """
    تشخیص رژیم بازار:
    ─────────────────────────────────────
    MEAN_REV : VR < 0.90 یا H < 0.45
    TRENDING : VR > 1.10 یا H > 0.55
    NEUTRAL  : بین این دو
    ─────────────────────────────────────
    """
    C  = Config
    vr = calc_variance_ratio(log_spread, vr_k, vr_period)
    H  = calc_hurst(log_spread, hurst_period)

    regime = pd.Series('NEUTRAL', index=log_spread.index)

    # Mean-Reversion: هر دو تأیید کنند
    mr_mask = (vr < C.vr_mr_threshold) | (H < 0.45)
    regime[mr_mask] = 'MEAN_REV'

    # Trending: هر دو تأیید کنند
    tr_mask = (vr > C.vr_tr_threshold) | (H > 0.55)
    regime[tr_mask] = 'TRENDING'

    # اگه هر دو True بودن → NEUTRAL (تناقض)
    both_mask = mr_mask & tr_mask
    regime[both_mask] = 'NEUTRAL'

    return pd.DataFrame({
        'regime': regime,
        'vr':     vr,
        'hurst':  H,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION — دو استراتژی
# ═══════════════════════════════════════════════════════════════════════════
def compute_signals(pair_name: str, pair_info: dict) -> dict:
    C     = Config
    df    = pair_info['df']
    leg_a = pair_info['leg_a']
    leg_b = pair_info['leg_b']

    log_ratio = np.log(df['c_spread'])

    # ── Regime Detection ──
    regime_df = detect_regime(
        log_ratio,
        C.vr_period, C.vr_k,
        C.hurst_period
    )

    # ── Z-Score برای Mean-Reversion ──
    z_mean  = log_ratio.rolling(C.z_fast_period).mean()
    z_std   = log_ratio.rolling(C.z_fast_period).std()
    z_score = (log_ratio - z_mean) / z_std.replace(0, np.nan)

    # ── EMA برای Trend-Following ──
    ema_fast = df['c_spread'].ewm(span=C.ema_fast, adjust=False).mean()
    ema_slow = df['c_spread'].ewm(span=C.ema_slow, adjust=False).mean()
    trend_dir = np.sign(ema_fast - ema_slow)   # +1 یا -1

    # ── ATR ──
    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()

    # ── Correlation ──
    ret_a   = df[leg_a].pct_change()
    ret_b   = df[leg_b].pct_change()
    corr_ok = ret_a.rolling(C.corr_period).corr(ret_b) > C.corr_min

    # ── Session ──
    hour    = pd.Series(df.index.hour,      index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = (
        hour.between(C.hour_start, C.hour_end) &
        dow.isin(C.trade_days)
    )

    # ── فیلتر ATR ──
    vol_ok = (
        (atr > atr_ma * C.atr_min_mult) &
        (atr < atr_ma * C.atr_max_mult)
    )
    # Trend نیاز به حداقل volatility داره
    vol_trend_ok = atr > atr_ma * C.trend_min_atr_mult

    is_mr = regime_df['regime'] == 'MEAN_REV'
    is_tr = regime_df['regime'] == 'TRENDING'

    # ── سیگنال Mean-Reversion ──
    mr_long  = (z_score < -C.z_entry) & is_mr & vol_ok & time_ok & corr_ok
    mr_short = (z_score >  C.z_entry) & is_mr & vol_ok & time_ok & corr_ok

    # ── سیگنال Trend-Following ──
    # Long: EMA fast > slow (uptrend) + قیمت بالای EMA slow
    tr_long  = (
        (trend_dir == 1) &
        (df['c_spread'] > ema_slow) &
        is_tr & vol_trend_ok & time_ok & corr_ok
    )
    tr_short = (
        (trend_dir == -1) &
        (df['c_spread'] < ema_slow) &
        is_tr & vol_trend_ok & time_ok & corr_ok
    )

    # ── ترکیب سیگنال‌ها ──
    sig = pd.Series(0, index=df.index)
    sig_type = pd.Series('NONE', index=df.index)

    # Mean-Rev اولویت داره
    sig[mr_long]  =  1
    sig[mr_short] = -1
    sig_type[mr_long | mr_short] = 'MR'

    # Trend فقط جاهایی که MR نیست
    no_mr = sig == 0
    sig[tr_long  & no_mr] =  1
    sig[tr_short & no_mr] = -1
    sig_type[(tr_long | tr_short) & no_mr] = 'TR'

    # حذف سیگنال‌های تکراری
    sig = sig.where(sig != sig.shift(), 0)

    mr_n = int(((sig != 0) & (sig_type == 'MR')).sum())
    tr_n = int(((sig != 0) & (sig_type == 'TR')).sum())
    mr_pct = int((is_mr).sum() / len(is_mr) * 100)
    tr_pct = int((is_tr).sum() / len(is_tr) * 100)

    print(f"    {pair_name}:"
          f" MR={mr_pct}% of time ({mr_n} signals)"
          f" | TR={tr_pct}% of time ({tr_n} signals)")

    return {
        'sig':      sig,
        'sig_type': sig_type,
        'z_score':  z_score,
        'atr':      atr,
        'atr_ma':   atr_ma,
        'regime':   regime_df['regime'],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl(direction, entry_px, exit_px, lot, quote_rate) -> float:
    C = Config
    gross = direction * (exit_px - entry_px) * lot * C.lot_size
    return gross * quote_rate - C.commission_per_lot * lot


def calc_lot(equity, sl_pips, consec_loss, quote_rate,
             dd_ratio: float = 0.0) -> float:
    C = Config
    risk = C.risk_base_pct
    if dd_ratio >= C.dd_shield_pct:
        risk = C.dd_shield_risk
    elif consec_loss >= C.consec_loss_n:
        f = C.risk_reduce ** (consec_loss - C.consec_loss_n + 1)
        risk = max(risk * f, C.risk_min_pct)
    pip_val = C.pip * C.lot_size * quote_rate
    raw = (equity * max(risk, C.risk_min_pct)) / (sl_pips * pip_val)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def new_acc(ts) -> dict:
    return {
        'equity':      Config.initial_balance,
        'start_ts':    ts,
        'trades':      [],
        'blown':       False,
        'blown_rsn':   '',
        'peak':        Config.initial_balance,
        'consec_loss': 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE v11
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, pair_signals: dict) -> dict:
    C          = Config
    pip        = C.pip
    pair_names = list(pairs.keys())

    # ── ایندکس مشترک ──
    common_idx = None
    for name in pair_names:
        idx        = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars     = len(common_idx)
    print(f"\n  ✅ Common bars: {n_bars:,}"
          f" | {common_idx[0].date()} → {common_idx[-1].date()}")

    # ── آرایه‌های numpy ──
    pa = {}
    for name in pair_names:
        df_p = pairs[name]['df'].reindex(common_idx).ffill()
        ps   = pair_signals[name]
        pa[name] = {
            'o':        df_p['o_spread'].values.astype(float),
            'c':        df_p['c_spread'].values.astype(float),
            'qr':       df_p['quote_rate'].values.astype(float),
            'sig':      ps['sig'].reindex(common_idx).fillna(0).values.astype(int),
            'sig_type': ps['sig_type'].reindex(common_idx).fillna('NONE').values,
            'z':        ps['z_score'].reindex(common_idx).fillna(np.nan).values.astype(float),
            'atr':      ps['atr'].reindex(common_idx).ffill().values.astype(float),
            'atr_ma':   ps['atr_ma'].reindex(common_idx).ffill().values.astype(float),
            'regime':   ps['regime'].reindex(common_idx).fillna('NEUTRAL').values,
        }

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    acc             = new_acc(common_idx[C.warmup])
    total_withdrawn = 0.0
    acc_num         = 1
    day_start_eq    = C.initial_balance
    all_trades      = []
    acc_logs        = []
    eq_curve        = []

    positions     = {n: None for n in pair_names}
    trades_today  = {n: 0    for n in pair_names}
    pending_sig   = {n: 0    for n in pair_names}
    pending_type  = {n: 'MR' for n in pair_names}

    print(f"\n  ▶ Running Dual-Strategy Simulator v11...")
    print(f"    Pairs   : {' + '.join(pair_names)}")
    print(f"    Strategy: Mean-Reversion + Trend-Following")
    print(f"    Target  : +{C.profit_target_pct*100:.0f}%"
          f"  | DD: -{C.max_total_dd_pct*100:.0f}%")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq = acc['equity']
        eq_curve.append((ts, round(eq, 4)))
        if eq > acc['peak']:
            acc['peak'] = eq

        # ── ریست روزانه ──
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0

        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(f"    {pct:5.1f}%"
                  f" | Eq: ${acc['equity']:,.0f}"
                  f" | Bank: ${total_withdrawn:,.0f}",
                  end='\r')

        # ── چک blown ──
        if acc['blown']:
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   acc['blown_rsn'],
                'pnl':      acc['equity'] - C.initial_balance,
            })
            print(f"\n    💥 #{acc_num:>3} | {ts.date()}"
                  f" | ${acc['equity']:,.2f} | {acc['blown_rsn']}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
                pending_sig[n]  = 0
                positions[n]    = None
            continue

        dd_ratio = max(0.0, (acc['peak'] - acc['equity']) / acc['peak'])

        # ── ورود ──
        n_open = sum(1 for n in pair_names if positions[n] is not None)
        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0
                    and positions[name] is None
                    and trades_today[name] < C.max_trades_day
                    and n_open < C.max_open_pairs):

                sv       = pending_sig[name]
                stype    = pending_type[name]
                qr       = a['qr'][bar]

                # SL/TP بسته به نوع استراتژی
                if stype == 'TR':
                    sl_p = C.trend_sl_pips
                    tp_p = C.trend_tp_pips
                else:
                    sl_p = C.sl_pips
                    tp_p = C.tp_pips

                lot = calc_lot(
                    acc['equity'], sl_p,
                    acc['consec_loss'], qr, dd_ratio
                )
                sp  = pairs[name]['spread_pips']
                ep  = a['o'][bar] + sv * (
                    C.slippage_entry_pips + sp / 2
                ) * pip
                sl  = ep - sv * sl_p * pip
                tp  = ep + sv * tp_p * pip

                positions[name] = {
                    'pair':          name,
                    'dir':           sv,
                    'lot':           lot,
                    'lot_remaining': lot,
                    'partial_done':  False,
                    'entry':         ep,
                    'sl':            sl,
                    'tp':            tp,
                    'trail_sl':      None,
                    'trail_active':  False,
                    'entry_ts':      ts,
                    'entry_bar':     bar,
                    'strategy':      stype,   # 'MR' یا 'TR'
                }
                trades_today[name] += 1
                n_open += 1
            pending_sig[name] = 0

        # ── Floating PnL ──
        total_float = sum(
            calc_pnl(positions[n]['dir'], positions[n]['entry'],
                     pa[n]['c'][bar], positions[n]['lot_remaining'],
                     pa[n]['qr'][bar])
            for n in pair_names if positions[n] is not None
        )

        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            acc['blown']     = True
            acc['blown_rsn'] = (
                "DailyDD" if current_eq <= daily_limit else "TotalDD"
            )
            for name in pair_names:
                pos = positions[name]
                if pos is None:
                    continue
                a  = pa[name]
                xp = a['c'][bar] - pos['dir'] * C.slippage_sl_pips * pip
                pnl = calc_pnl(
                    pos['dir'], pos['entry'],
                    xp, pos['lot_remaining'], a['qr'][bar]
                )
                acc['equity'] += pnl
                all_trades.append(
                    _make_rec(pos, xp, ts, pnl, 'BLOWN', pos['lot_remaining'])
                )
                positions[name] = None
            continue

        # ── مدیریت خروج ──
        for name in pair_names:
            pos = positions[name]
            if pos is None:
                continue

            a       = pa[name]
            cp      = a['c'][bar]
            qr      = a['qr'][bar]
            d       = pos['dir']
            ep      = pos['entry']
            zn      = a['z'][bar]
            lot_rem = pos['lot_remaining']
            strat   = pos['strategy']

            # ── Trailing Stop ──
            if C.use_trailing and pos['partial_done']:
                if not pos['trail_active']:
                    activate = False
                    if strat == 'MR' and not np.isnan(zn):
                        activate = (
                            (d ==  1 and zn >= -C.trail_activate_z) or
                            (d == -1 and zn <=  C.trail_activate_z)
                        )
                    elif strat == 'TR':
                        # trend: trail بعد از رسیدن به ۵۰٪ TP
                        half_tp_dist = abs(pos['tp'] - ep) * 0.5
                        activate = (
                            (d ==  1 and cp >= ep + half_tp_dist) or
                            (d == -1 and cp <= ep - half_tp_dist)
                        )
                    if activate:
                        pos['trail_sl']     = cp - d * C.trail_sl_pips * pip
                        pos['trail_active'] = True
                else:
                    new_trail = cp - d * C.trail_sl_pips * pip
                    if d ==  1 and new_trail > pos['trail_sl']:
                        pos['trail_sl'] = new_trail
                    elif d == -1 and new_trail < pos['trail_sl']:
                        pos['trail_sl'] = new_trail

            hit_trail_sl = (
                pos['trail_active'] and pos['trail_sl'] is not None and (
                    (d ==  1 and cp <= pos['trail_sl']) or
                    (d == -1 and cp >= pos['trail_sl'])
                )
            )

            hit_sl = (
                (d ==  1 and cp <= pos['sl']) or
                (d == -1 and cp >= pos['sl'])
            )
            hit_tp = (
                (d ==  1 and cp >= pos['tp']) or
                (d == -1 and cp <= pos['tp'])
            )

            # ── Partial Exit (فقط MR) ──
            if strat == 'MR' and not pos['partial_done'] and not np.isnan(zn):
                hit_p = (
                    (d ==  1 and zn >= -C.z_exit_partial) or
                    (d == -1 and zn <=  C.z_exit_partial)
                )
                if hit_p:
                    p_lot = round(lot_rem * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            all_trades.append(
                                _make_rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            )
                            pos['lot_remaining'] = round(lot_rem - p_lot, 2)
                            pos['partial_done']  = True
                            lot_rem = pos['lot_remaining']
                            if lot_rem < C.min_lot:
                                positions[name] = None
                                continue

            # ── Z-Stop (فقط MR) ──
            hit_z_stop = False
            hit_z_exit = False
            if strat == 'MR' and not np.isnan(zn):
                hit_z_stop = (
                    (d ==  1 and zn <= -C.z_stop_margin) or
                    (d == -1 and zn >=  C.z_stop_margin)
                )
                z_crossed = (
                    (d ==  1 and zn >= -C.z_exit_full) or
                    (d == -1 and zn <=  C.z_exit_full)
                )
                if z_crossed:
                    pnl_chk = calc_pnl(d, ep, cp, lot_rem, qr)
                    if pnl_chk >= C.min_net_profit_usd or pos['partial_done']:
                        hit_z_exit = True

            time_stop = (bar - pos['entry_bar']) >= C.time_stop_bars

            if (hit_sl or hit_tp or hit_trail_sl
                    or hit_z_exit or hit_z_stop or time_stop):
                if hit_sl:
                    xp = pos['sl'] - d * C.slippage_sl_pips * pip
                    st = 'SL'
                elif hit_tp:
                    xp, st = pos['tp'], 'TP'
                elif hit_trail_sl:
                    xp = pos['trail_sl'] - d * 0.3 * pip
                    st = 'TrailSL'
                elif hit_z_stop:
                    xp, st = cp, 'Z-Stop'
                elif time_stop:
                    xp, st = cp, 'TimeStop'
                else:
                    xp, st = cp, 'Z-Exit'

                # ── tag استراتژی به status ──
                st_full   = f"{strat}_{st}"
                final_pnl = calc_pnl(d, ep, xp, lot_rem, qr)
                acc['equity'] += final_pnl
                all_trades.append(
                    _make_rec(pos, xp, ts, final_pnl, st_full, lot_rem)
                )
                positions[name] = None
                acc['consec_loss'] = (
                    0 if final_pnl > 0 else acc['consec_loss'] + 1
                )

        # ── برداشت سود ──
        all_closed = all(positions[n] is None for n in pair_names)
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
            print(f"\n    💰 #{acc_num:>3} | {ts.date()}"
                  f" | +${w:,.2f} | Bank: ${total_withdrawn:,.2f}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
                pending_sig[n]  = 0
            continue

        # ── سیگنال جدید ──
        n_open = sum(1 for n in pair_names if positions[n] is not None)
        for name in pair_names:
            a = pa[name]
            if (positions[name] is None
                    and not acc['blown']
                    and trades_today[name] < C.max_trades_day
                    and n_open < C.max_open_pairs
                    and a['sig'][bar] != 0):
                pending_sig[name]  = int(a['sig'][bar])
                pending_type[name] = str(a['sig_type'][bar])

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


def _make_rec(pos, exit_px, exit_ts, pnl, status, lot) -> dict:
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
        'strategy':  pos.get('strategy', 'MR'),
        'entry_bar': pos['entry_bar'],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════════
def calc_max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def monte_carlo(pnl_list, n=1_000, init=5_000.0) -> dict:
    arr = np.array(pnl_list)
    rng = np.random.default_rng(42)
    finals, dds = [], []
    for _ in range(n):
        s    = rng.choice(arr, size=len(arr), replace=True)
        eq   = np.concatenate([[init], init + np.cumsum(s)])
        peak = np.maximum.accumulate(eq)
        dds.append(((eq - peak) / peak).min())
        finals.append(eq[-1])
    return {
        'med':      float(np.median(finals)),
        'p5':       float(np.percentile(finals, 5)),
        'p95':      float(np.percentile(finals, 95)),
        'dd_med':   float(np.median(dds)),
        'dd_p95':   float(np.percentile(dds, 95)),
        'prob_pos': float(np.mean([f > init for f in finals]) * 100),
    }


def print_report(results: dict):
    trades     = results['all_trades']
    pair_names = results['pair_names']
    if not trades:
        print("❌ No trades.")
        return

    df = pd.DataFrame(trades)
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])
    df['month']   = df['exit_ts'].dt.to_period('M')
    df['year']    = df['exit_ts'].dt.year

    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    wr = len(wins) / len(df) * 100
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else np.inf

    eq_s = pd.DataFrame(results['eq_curve'], columns=['ts','eq']).set_index('ts')['eq']
    mdd  = calc_max_drawdown(eq_s)

    print("\n" + "═"*68)
    print(f"  Dual-Strategy Simulator v11 — {'+'.join(pair_names)}")
    print("═"*68)
    print(f"  Trades: {len(df):,} | WR: {wr:.1f}% | PF: {pf:.2f} | MDD: {mdd*100:.1f}%")
    print(f"  Banked: ${results['total_withdrawn']:,.2f}"
          f" | Equity: ${results['final_equity']:,.2f}")
    print(f"  Avg Win: ${wins['pnl'].mean():.2f}"
          f" | Avg Loss: ${losses['pnl'].mean():.2f}"
          f" | Expect: ${df['pnl'].mean():.2f}")

    # ── Per Strategy ──
    print("─"*68)
    print("  Per Strategy:")
    for strat in ['MR', 'TR']:
        st_df = df[df['strategy'] == strat]
        if len(st_df) == 0:
            continue
        sw = st_df[st_df['pnl'] > 0]
        sl = st_df[st_df['pnl'] < 0]
        s_wr = len(sw)/len(st_df)*100
        s_pf = sw['pnl'].sum()/abs(sl['pnl'].sum()) if len(sl) else np.inf
        label = "Mean-Rev " if strat == 'MR' else "Trend-Fol"
        print(f"    {label}: {len(st_df):>4} trades"
              f" | WR: {s_wr:.1f}%"
              f" | PF: {s_pf:.2f}"
              f" | Net: ${st_df['pnl'].sum():,.2f}")

    # ── Annual ──
    print("─"*68)
    print(f"  {'Year':<6} {'N':>5} {'WR%':>6} {'PF':>5}"
          f" {'MR_PnL':>9} {'TR_PnL':>9} {'Total':>9}")
    print("  " + "─"*58)
    for yr, grp in df.groupby('year'):
        w = grp[grp['pnl'] > 0]
        l = grp[grp['pnl'] < 0]
        y_wr = len(w)/len(grp)*100
        y_pf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l) else np.inf
        mr_pnl = grp[grp['strategy']=='MR']['pnl'].sum()
        tr_pnl = grp[grp['strategy']=='TR']['pnl'].sum()
        tot    = grp['pnl'].sum()
        flag   = "✅" if tot > 0 else "❌"
        print(f"  {yr:<6} {len(grp):>5} {y_wr:>6.1f} {y_pf:>5.2f}"
              f" ${mr_pnl:>8,.0f} ${tr_pnl:>8,.0f}"
              f" ${tot:>8,.0f} {flag}")

    # ── Monthly ──
    monthly = df.groupby('month')['pnl'].sum()
    pos_m   = int((monthly > 0).sum())
    neg_m   = int((monthly < 0).sum())
    mwr     = pos_m/(pos_m+neg_m)*100
    logs    = results['account_logs']
    tgt     = sum(1 for l in logs if l['reason']=='TARGET_HIT')
    blwn    = sum(1 for l in logs if l['reason']!='TARGET_HIT')
    sr      = tgt/(tgt+blwn)*100 if (tgt+blwn) > 0 else 0

    print("─"*68)
    print(f"  Monthly: avg ${monthly.mean():,.2f}"
          f" | Best: ${monthly.max():,.2f}"
          f" | Worst: ${monthly.min():,.2f}"
          f" | WR: {mwr:.0f}%")
    print(f"  Accounts: {results['total_accounts']}"
          f" | ✅ {tgt} | 💥 {blwn} | SR: {sr:.0f}%")

    mc = monte_carlo(list(df['pnl']))
    print("─"*68)
    print(f"  Monte Carlo: Med=${mc['med']:,.0f}"
          f" | P5=${mc['p5']:,.0f}"
          f" | P95=${mc['p95']:,.0f}"
          f" | DD_med={mc['dd_med']*100:.1f}%"
          f" | Prob+={mc['prob_pos']:.1f}%")
    print("═"*68)


if __name__ == "__main__":
    t0    = datetime.now()
    pairs = load_all_pairs()

    print("\n  Computing Dual-Strategy Signals...")
    pair_signals = {}
    for name, info in pairs.items():
        pair_signals[name] = compute_signals(name, info)

    results = run_backtest(pairs, pair_signals)
    print_report(results)
    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
