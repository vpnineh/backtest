"""
CorrArb Prop Simulator — v9
═══════════════════════════════════════════════════════════════════
بهبودهای این نسخه نسبت به v8:
  ① OU Half-Life filter    — فقط وقتی mean reversion سریعه وارد شو
                             TimeStop = 2 × HL (adaptive, نه fixed)
  ② ATR Dynamic SL         — SL = clip(atr_sl_mult × ATR, min, max)
                             کمتر stop می‌خوریم در volatility بالا
  ③ Trailing Stop          — بعد از +1×SL سود: trail peak−0.5×SL
  ④ EURCHF pair            — EURUSD × USDCHF (همبستگی معکوس قوی)
  ⑤ Swap costs             — هزینه rollover واقعی هر شب
  ⑥ Walk-Forward Optimizer — Train 2010-2019 → Test 2020-2025
  ⑦ واقعیت‌سنجی           — spread واقعی‌تر، slippage بیشتر
───────────────────────────────────────────────────────────────────
جفت‌ارزها: EURGBP synthetic, AUDNZD synthetic, EURCHF synthetic
داده: 2010-2025 | M1 → 15min
"""

import pandas as pd
import numpy as np
import glob, zipfile, os, warnings, itertools
from datetime import datetime

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── مدیریت ریسک ──
    risk_base_pct  = 0.015
    risk_min_pct   = 0.005
    consec_loss_n  = 2
    risk_reduce    = 0.5

    # ── هزینه‌های بروکر (واقعی‌تر از v8) ──
    # EURGBP: spread ~1.5pip، AUDNZD: ~2.5pip، EURCHF: ~1.8pip
    pair_spread = {
        'EURGBP': 1.5,
        'AUDNZD': 2.5,
        'EURCHF': 1.8,
    }
    commission_per_lot = 7.0   # round-trip USD
    slippage_pips      = 0.5   # ↑ از 0.3 (واقعی‌تر)

    # ── Swap/Rollover (واقعی — USD per lot per night) ──
    # مثبت = سود، منفی = هزینه
    pair_swap_long  = {'EURGBP': -0.5,  'AUDNZD': -1.2,  'EURCHF': -0.8}
    pair_swap_short = {'EURGBP': -0.3,  'AUDNZD':  0.3,  'EURCHF': -1.5}

    # ── مشخصات ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── پارامترهای z-score ──
    z_fast_period      = 96
    z_entry            = 2.3         # ↑ از 2.1 — کیفیت بالاتر
    z_exit_partial     = 0.5
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0

    # ── فیلترها ──
    corr_period  = 96
    corr_min     = 0.80
    hour_start   = 2
    hour_end     = 19
    trade_days   = [0, 1, 2, 3, 4]
    max_trades_day = 2  # per pair

    # ── ATR Dynamic SL ──
    atr_period    = 14
    atr_ma_period = 96
    atr_sl_mult   = 2.0    # SL = 2 × ATR of spread
    atr_sl_min_pips = 18.0  # حداقل SL
    atr_sl_max_pips = 45.0  # حداکثر SL
    tp_ratio        = 3.0   # TP = tp_ratio × SL

    # ── Trailing Stop ──
    use_trailing   = True
    trail_trigger_r = 1.0   # بعد از +1R سود، trailing فعال می‌شه
    trail_distance_r = 0.5  # trail = peak − 0.5R

    # ── Variance Ratio regime ──
    vr_period = 200
    vr_k      = 4
    vr_max    = 0.88       # ↓ از 0.90 — سخت‌تر

    # ── OU Half-Life regime ──
    ou_window       = 200
    ou_max_hl_bars  = 32   # فقط ورود اگه HL < 32 بار (8 ساعت)

    # ── Partial Exit ──
    partial_ratio = 0.50

    # ── Cooldown بعد از blow ──
    cooldown_days = 7

    # ── Walk-Forward Optimizer ──
    opt_train_end = '2019-12-31'   # Train period end
    opt_test_start = '2020-01-01'  # Test period start

    # ── ATR Volatility Filter ──
    atr_max_mult = 3.0
    atr_min_mult = 0.5


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_raw(pattern: str, is_zip: bool) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files: {pattern}")
    frames = []
    for p in paths:
        try:
            if is_zip:
                with zipfile.ZipFile(p) as z:
                    csv_name = next(
                        (f for f in z.namelist() if f.lower().endswith('.csv')), None)
                    if not csv_name:
                        continue
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f, sep=';', header=None,
                                         names=['ts','o','h','l','c','v'])
            else:
                df = pd.read_csv(p, sep=';', header=None,
                                 names=['ts','o','h','l','c','v'])
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ {os.path.basename(p)}: {e}")
    if not frames:
        raise ValueError(f"No valid data: {pattern}")
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


def build_spread(df_a, sfx_a, df_b, sfx_b, multiply=False):
    """
    multiply=False → spread = a/b  (مثال: EURUSD/GBPUSD = EURGBP)
    multiply=True  → spread = a*b  (مثال: EURUSD*USDCHF = EURCHF)
    """
    m = df_a.join(df_b, how='inner').dropna()
    if multiply:
        m['c_spread']   = m[f'c_{sfx_a}'] * m[f'c_{sfx_b}']
        m['o_spread']   = m[f'o_{sfx_a}'] * m[f'o_{sfx_b}']
        m['h_spread']   = m[f'h_{sfx_a}'] * m[f'h_{sfx_b}']
        m['l_spread']   = m[f'l_{sfx_a}'] * m[f'l_{sfx_b}']
        # quote = CHF → CHF→USD = 1/USDCHF
        m['quote_rate'] = 1.0 / m[f'c_{sfx_b}']
    else:
        m['c_spread']   = m[f'c_{sfx_a}'] / m[f'c_{sfx_b}']
        m['o_spread']   = m[f'o_{sfx_a}'] / m[f'o_{sfx_b}']
        m['h_spread']   = m[f'h_{sfx_a}'] / m[f'l_{sfx_b}']
        m['l_spread']   = m[f'l_{sfx_a}'] / m[f'h_{sfx_b}']
        m['quote_rate'] = m[f'c_{sfx_b}']
    return m[m.index.weekday < 5].copy()


def load_all_pairs() -> dict:
    C = Config
    print("\n  Loading and syncing datasets...")
    pairs = {}

    # ── EURGBP synthetic ──
    try:
        eur = to_15min(load_raw('data/*EURUSD*.csv', False), 'eur')
        gbp = to_15min(load_raw('data/*GBPUSD*.csv', False), 'gbp')
        df  = build_spread(eur, 'eur', gbp, 'gbp')
        pairs['EURGBP'] = {'df': df, 'leg_a': 'c_eur', 'leg_b': 'c_gbp',
                           'spread_pip': C.pair_spread['EURGBP'],
                           'swap_l': C.pair_swap_long['EURGBP'],
                           'swap_s': C.pair_swap_short['EURGBP']}
        print(f"  ✅ EURGBP : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ EURGBP : {e}")

    # ── AUDNZD synthetic ──
    try:
        aud = to_15min(load_raw('data/HISTDATA*AUDUSD*.zip', True), 'aud')
        nzd = to_15min(load_raw('data/HISTDATA*NZDUSD*.zip', True), 'nzd')
        df  = build_spread(aud, 'aud', nzd, 'nzd')
        pairs['AUDNZD'] = {'df': df, 'leg_a': 'c_aud', 'leg_b': 'c_nzd',
                           'spread_pip': C.pair_spread['AUDNZD'],
                           'swap_l': C.pair_swap_long['AUDNZD'],
                           'swap_s': C.pair_swap_short['AUDNZD']}
        print(f"  ✅ AUDNZD : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ AUDNZD : {e}")

    # ── EURCHF synthetic (EURUSD × USDCHF) ──
    try:
        eur_df = to_15min(load_raw('data/*EURUSD*.csv', False), 'eur')
        chf    = to_15min(load_raw('data/HISTDATA*USDCHF*.zip', True), 'chf')
        df     = build_spread(eur_df, 'eur', chf, 'chf', multiply=True)
        pairs['EURCHF'] = {'df': df, 'leg_a': 'c_eur', 'leg_b': 'c_chf',
                           'spread_pip': C.pair_spread['EURCHF'],
                           'swap_l': C.pair_swap_long['EURCHF'],
                           'swap_s': C.pair_swap_short['EURCHF']}
        print(f"  ✅ EURCHF : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ EURCHF : {e}")

    if not pairs:
        raise RuntimeError("No pairs loaded.")
    return pairs


# ═══════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_vr(series, k, window):
    """Variance Ratio — VR < 1: mean-reverting"""
    r1 = series.diff(1)
    rk = series.diff(k)
    return rk.rolling(window).var() / (k * r1.rolling(window).var().replace(0, np.nan))


def calc_ou_halflife(log_spread: pd.Series, window: int) -> pd.Series:
    """
    Ornstein-Uhlenbeck Half-Life via rolling OLS.
    delta(t) = alpha + beta × x(t-1)  → HL = ln(2) / (-beta)
    HL در واحد بار: اگه HL < 32 → reversion در 8 ساعت ← ورود مجاز
    """
    delta = log_spread.diff()
    lagged = log_spread.shift(1)
    # Rolling covariance / variance → beta
    roll_cov = delta.rolling(window).cov(lagged)
    roll_var = lagged.rolling(window).var().replace(0, np.nan)
    beta = roll_cov / roll_var
    # beta باید منفی باشه (mean-reverting)
    hl = np.log(2) / (-beta).clip(lower=1e-6)
    hl[beta >= 0] = np.nan   # trending → no valid HL
    return hl


def compute_signals(pair_name: str, pair_info: dict, cfg: Config) -> tuple:
    C  = cfg
    df = pair_info['df']
    la = pair_info['leg_a']
    lb = pair_info['leg_b']

    log_r = np.log(df['c_spread'])
    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    # ATR of spread
    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    atr_sl = (atr * C.atr_sl_mult / C.pip).clip(C.atr_sl_min_pips, C.atr_sl_max_pips)

    # Correlation guard
    corr_ok = (df[la].pct_change()
               .rolling(C.corr_period)
               .corr(df[lb].pct_change()) > C.corr_min)

    # Variance Ratio regime
    vr        = calc_vr(log_r, C.vr_k, C.vr_period)
    vr_ok     = vr < C.vr_max

    # OU Half-Life regime
    ou_hl     = calc_ou_halflife(log_r, C.ou_window)
    ou_ok     = ou_hl < C.ou_max_hl_bars

    regime_ok = vr_ok & ou_ok

    # Volatility filter
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    # Session filter
    hour    = pd.Series(df.index.hour,      index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # Signals
    lc = (z < -C.z_entry) & vol_ok & time_ok & corr_ok & regime_ok
    sc = (z >  C.z_entry) & vol_ok & time_ok & corr_ok & regime_ok
    sig = pd.Series(0, index=df.index)
    sig[lc] =  1
    sig[sc] = -1
    sig = sig.where(sig != sig.shift(), 0)

    n = int((sig != 0).sum())
    print(f"    {pair_name}: {n:,} sigs (L:{int((sig==1).sum())} S:{int((sig==-1).sum())}) "
          f"| VR: {int(vr_ok.sum()):,} | OU: {int(ou_ok.sum()):,}")

    return sig, z, atr_sl, ou_hl


# ═══════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════
def calc_pnl(direction, entry_px, exit_px, lot, quote_rate):
    C = Config
    gross = direction * (exit_px - entry_px) * lot * C.lot_size * quote_rate
    return gross - C.commission_per_lot * lot


def calc_swap(direction, lot, swap_long, swap_short):
    """هزینه/سود rollover یک شب"""
    rate = swap_long if direction == 1 else swap_short
    return rate * lot  # USD


def calc_lot(equity, sl_pips, consec_loss, quote_rate, cfg, monthly_stressed=False):
    C = cfg
    risk = C.risk_base_pct
    if monthly_stressed:
        risk *= 0.5
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
    pip_val = C.pip * C.lot_size * quote_rate
    raw = (equity * risk) / (sl_pips * pip_val)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def new_acc(ts, cfg):
    return {
        'equity':      cfg.initial_balance,
        'start_ts':    ts,
        'trades':      [],
        'blown':       False,
        'blown_rsn':   '',
        'peak':        cfg.initial_balance,
        'consec_loss': 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, pair_signals: dict, cfg: Config,
                 date_start=None, date_end=None, verbose=True) -> dict:
    C         = cfg
    pip       = C.pip
    pair_names = list(pairs.keys())

    # Common index
    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()

    # Date filter
    if date_start:
        common_idx = common_idx[common_idx >= pd.Timestamp(date_start)]
    if date_end:
        common_idx = common_idx[common_idx <= pd.Timestamp(date_end)]

    n_bars = len(common_idx)
    if n_bars < C.warmup + 100:
        return None

    if verbose:
        print(f"  ✅ Bars: {n_bars:,} | {common_idx[0].date()} → {common_idx[-1].date()}")

    # Numpy arrays per pair
    pa = {}
    for name in pair_names:
        df_p    = pairs[name]['df'].reindex(common_idx).ffill()
        sig_s, z_s, atr_sl_s, ou_hl_s = pair_signals[name]
        pa[name] = {
            'o':      df_p['o_spread'].values.astype(float),
            'c':      df_p['c_spread'].values.astype(float),
            'qr':     df_p['quote_rate'].values.astype(float),
            'sig':    sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z':      z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
            'atr_sl': atr_sl_s.reindex(common_idx).ffill().fillna(C.atr_sl_max_pips).values.astype(float),
            'ou_hl':  ou_hl_s.reindex(common_idx).ffill().fillna(999).values.astype(float),
            'sp_pip': C.pair_spread.get(name, 1.5),
            'sw_l':   C.pair_swap_long.get(name, -0.5),
            'sw_s':   C.pair_swap_short.get(name, -0.5),
        }

    FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    TARGET  = C.initial_balance * (1 + C.profit_target_pct)

    acc           = new_acc(common_idx[C.warmup], C)
    withdrawn     = 0.0
    acc_num       = 1
    day_eq        = C.initial_balance
    month_eq      = C.initial_balance
    all_trades    = []
    acc_logs      = []
    eq_curve      = []
    cooldown_til  = None

    positions    = {n: None for n in pair_names}
    trades_today = {n: 0    for n in pair_names}
    pending_sig  = {n: 0    for n in pair_names}
    prev_date    = None
    prev_month   = None

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        cur_date  = ts.date()
        cur_month = (ts.year, ts.month)

        # Daily / monthly reset
        if cur_date != prev_date:
            day_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
            prev_date = cur_date

        if cur_month != prev_month:
            month_eq  = acc['equity']
            prev_month = cur_month

        if acc['equity'] > acc['peak']:
            acc['peak'] = acc['equity']

        eq_curve.append((ts, round(acc['equity'], 4)))

        in_cd = cooldown_til is not None and ts < cooldown_til

        # ── Blown ──
        if acc['blown']:
            if verbose:
                print(f"\n    💥 #{acc_num:>3} | {ts.date()} | Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            acc_logs.append({
                'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts,
                'reason': acc['blown_rsn'], 'pnl': acc['equity'] - C.initial_balance,
                'n_trades': len(acc['trades']), 'days': (ts - acc['start_ts']).days,
            })
            cooldown_til = ts + pd.Timedelta(days=C.cooldown_days)
            acc_num += 1
            acc = new_acc(ts, C)
            day_eq = month_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
                pending_sig[n]  = 0
                positions[n]    = None
            continue

        if in_cd:
            continue

        monthly_stressed = (acc['equity'] - month_eq) < -C.initial_balance * 0.03

        # ── Execute pending entries ──
        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0 and positions[name] is None
                    and trades_today[name] < C.max_trades_day):
                sv  = pending_sig[name]
                qr  = a['qr'][bar]
                sl_pips_dyn = a['atr_sl'][bar]
                lot = calc_lot(acc['equity'], sl_pips_dyn, acc['consec_loss'],
                               qr, C, monthly_stressed)
                ep  = a['o'][bar] + sv * (C.slippage_pips + a['sp_pip'] / 2) * pip
                sl  = ep - sv * sl_pips_dyn * pip
                tp  = ep + sv * sl_pips_dyn * C.tp_ratio * pip
                ou_hl_now = a['ou_hl'][bar]
                # TimeStop = 2 × OU HL (adaptive) — cap به 96 بار
                ts_bars = int(np.clip(2 * ou_hl_now, 24, 96)) if not np.isnan(ou_hl_now) else 48
                positions[name] = {
                    'pair': name, 'dir': sv,
                    'lot': lot, 'lot_remaining': lot,
                    'partial_done': False,
                    'entry': ep, 'sl': sl, 'tp': tp,
                    'sl_pips': sl_pips_dyn,
                    'ts_bars': ts_bars,
                    'entry_ts': ts, 'entry_bar': bar,
                    'trail_active': False,
                    'trail_sl': sl,
                    'peak_pnl': 0.0,
                    'swap_total': 0.0,
                }
                trades_today[name] += 1
            pending_sig[name] = 0

        # ── Swap cost every midnight ──
        if ts.hour == 0 and ts.minute == 0:
            for name in pair_names:
                pos = positions[name]
                if pos is not None:
                    sw = calc_swap(pos['dir'], pos['lot_remaining'],
                                   pa[name]['sw_l'], pa[name]['sw_s'])
                    acc['equity']    += sw
                    pos['swap_total'] += sw

        # ── Total floating PnL for DD check ──
        total_float = 0.0
        for name in pair_names:
            pos = positions[name]
            if pos is not None:
                a = pa[name]
                total_float += calc_pnl(
                    pos['dir'], pos['entry'],
                    a['c'][bar], pos['lot_remaining'], a['qr'][bar])

        cur_eq    = acc['equity'] + total_float
        daily_lim = day_eq * (1 - C.max_daily_loss_pct)

        # ── Drawdown breach → close all ──
        if cur_eq <= daily_lim or cur_eq <= FLOOR:
            acc['blown']     = True
            acc['blown_rsn'] = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            for name in pair_names:
                pos = positions[name]
                if pos is None:
                    continue
                a   = pa[name]
                pnl = calc_pnl(pos['dir'], pos['entry'],
                               a['c'][bar], pos['lot_remaining'], a['qr'][bar])
                acc['equity'] += pnl
                rec = _rec(pos, a['c'][bar], ts, pnl, 'BLOWN', pos['lot_remaining'])
                all_trades.append(rec); acc['trades'].append(rec)
                positions[name] = None
            continue

        # ── Exit management ──
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
            lr      = pos['lot_remaining']
            sl_pips = pos['sl_pips']

            # Trailing Stop update
            if C.use_trailing and not pos['partial_done']:
                cur_pnl_r = calc_pnl(d, ep, cp, lr, qr) / (sl_pips * C.pip * lr * C.lot_size * qr + 1e-8)
                if cur_pnl_r >= C.trail_trigger_r and not pos['trail_active']:
                    pos['trail_active'] = True
                if pos['trail_active']:
                    trail_new = cp - d * C.trail_distance_r * sl_pips * pip
                    if d == 1:
                        pos['trail_sl'] = max(pos['trail_sl'], trail_new)
                    else:
                        pos['trail_sl'] = min(pos['trail_sl'], trail_new)

            # ── Partial Exit ──
            if not pos['partial_done'] and not np.isnan(zn):
                hit_p = ((d ==  1 and zn >= -C.z_exit_partial) or
                         (d == -1 and zn <=  C.z_exit_partial))
                if hit_p:
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            rec = _rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(rec); acc['trades'].append(rec)
                            pos['lot_remaining'] = round(lr - p_lot, 2)
                            pos['partial_done']  = True
                            pos['sl']            = pos['entry']   # Break-even stop
                            pos['trail_active']  = False
                            lr = pos['lot_remaining']
                            if lr < C.min_lot:
                                positions[name] = None
                                continue

            if pos is None:
                continue
            lr = pos['lot_remaining']

            # Exit conditions
            hit_zs = (not np.isnan(zn) and
                      ((d == 1 and zn <= -C.z_stop_margin) or
                       (d == -1 and zn >=  C.z_stop_margin)))

            hit_ze = False
            if not np.isnan(zn):
                z_cross = ((d == 1 and zn >= -C.z_exit_full) or
                           (d == -1 and zn <=  C.z_exit_full))
                if z_cross:
                    pnl_chk = calc_pnl(d, ep, cp, lr, qr)
                    if pnl_chk >= C.min_net_profit_usd or pos['partial_done']:
                        hit_ze = True

            # SL: use trailing SL if active, else normal SL
            eff_sl  = pos['trail_sl'] if pos['trail_active'] else pos['sl']
            hit_sl  = (d == 1 and cp <= eff_sl) or (d == -1 and cp >= eff_sl)
            hit_tp  = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])

            # Smart TimeStop (adaptive)
            bars_open   = bar - pos['entry_bar']
            pnl_now     = calc_pnl(d, ep, cp, lr, qr)
            time_stop   = (
                (bars_open >= pos['ts_bars'] and pnl_now < 0) or
                (bars_open >= pos['ts_bars'] * 2)
            )

            if hit_ze or hit_zs or hit_sl or hit_tp or time_stop:
                xp = (eff_sl if hit_sl else pos['tp'] if hit_tp else cp)
                if hit_sl:       st = 'SL'
                elif hit_tp:     st = 'TP'
                elif hit_zs:     st = 'Z-Stop'
                elif time_stop:  st = 'TimeStop'
                else:            st = 'Z-Exit'

                fpnl = calc_pnl(d, ep, xp, lr, qr)
                acc['equity'] += fpnl
                rec = _rec(pos, xp, ts, fpnl, st, lr)
                all_trades.append(rec); acc['trades'].append(rec)
                positions[name] = None

                if fpnl > 0:
                    acc['consec_loss'] = 0
                else:
                    acc['consec_loss'] += 1

        # ── Profit target ──
        all_closed = all(positions[n] is None for n in pair_names)
        if acc['equity'] >= TARGET and all_closed:
            w  = acc['equity'] - C.initial_balance
            withdrawn += w
            dt = (ts - acc['start_ts']).days
            nt = len(acc['trades'])
            if verbose:
                print(f"\n    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | "
                      f"Bank:${withdrawn:>9.2f} | {dt}d | {nt}T")
            acc_logs.append({
                'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts,
                'reason': 'TARGET_HIT', 'pnl': w, 'n_trades': nt, 'days': dt,
            })
            acc_num += 1
            acc = new_acc(ts, C)
            day_eq = month_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
                pending_sig[n]  = 0
            continue

        # ── Collect new signals ──
        for name in pair_names:
            a = pa[name]
            if (positions[name] is None and not acc['blown']
                    and not in_cd
                    and trades_today[name] < C.max_trades_day
                    and a['sig'][bar] != 0):
                pending_sig[name] = int(a['sig'][bar])

    print()
    return {
        'all_trades':   all_trades,
        'account_logs': acc_logs,
        'eq_curve':     eq_curve,
        'withdrawn':    withdrawn,
        'final_equity': acc['equity'],
        'total_accounts': acc_num,
        'pair_names':   pair_names,
        'common_idx':   common_idx,
    }


def _rec(pos, exit_px, ts, pnl, status, lot):
    return {
        'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot,
        'entry': pos['entry'], 'exit': exit_px,
        'entry_ts': pos['entry_ts'], 'exit_ts': ts,
        'pnl': pnl, 'status': status,
        'entry_bar': pos['entry_bar'],
    }


# ═══════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════
def print_report(results: dict, title: str = "v9"):
    if not results or not results['all_trades']:
        print("\n❌ No trades.")
        return

    df_t = pd.DataFrame(results['all_trades'])
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month']   = df_t['exit_ts'].dt.to_period('M')
    df_t['year']    = df_t['exit_ts'].dt.year

    wins   = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    wr  = len(wins) / len(df_t) * 100
    pf  = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else float('inf')

    ci = results['common_idx']
    all_months = pd.period_range(
        ci[Config.warmup].to_period('M'), ci[-1].to_period('M'), freq='M')
    monthly = df_t.groupby('month')['pnl'].sum().reindex(all_months, fill_value=0.0)
    pos_m  = int((monthly > 0).sum())
    neg_m  = int((monthly < 0).sum())
    zero_m = int((monthly == 0).sum())

    logs    = results['account_logs']
    targets = sum(1 for l in logs if l['reason'] == 'TARGET_HIT')
    blown   = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')

    print("\n" + "═" * 68)
    print(f" ▌  CorrArb Prop Simulator {title}  ▐")
    print("═" * 68)
    print(f" Total Trades:     {len(df_t):>6,}")
    print(f" Win Rate:         {wr:>9.2f}%")
    print(f" Profit Factor:    {pf:>9.2f}")
    print(f" Avg Win:          ${wins['pnl'].mean():>8.2f}" if len(wins) else " Avg Win: N/A")
    print(f" Avg Loss:         ${losses['pnl'].mean():>8.2f}" if len(losses) else " Avg Loss: N/A")
    print(f" Net PnL:          ${df_t['pnl'].sum():>10,.2f}")
    print("-" * 68)
    print(f" Passed:           {targets:>4}")
    print(f" Blown:            {blown:>4}")
    print(f" Total Banked:     ${results['withdrawn']:>10,.2f}")
    print(f" Active Equity:    ${results['final_equity']:>10,.2f}")
    print("-" * 68)
    print(f" +Months: {pos_m:>3} / {len(monthly)} ({pos_m/len(monthly)*100:.0f}%)"
          f"   -Months: {neg_m:>3}   0-Months: {zero_m:>3}")
    print(f" Monthly Avg: ${monthly.mean():>8,.2f}"
          f" | Best: ${monthly.max():>8,.2f}"
          f" | Worst: ${monthly.min():>8,.2f}")
    print("-" * 68)

    # Per-pair
    if 'pair' in df_t.columns and len(results['pair_names']) > 1:
        print(" Per Pair:")
        for pair in results['pair_names']:
            pt = df_t[df_t['pair'] == pair]
            if not len(pt):
                continue
            pw = pt[pt['pnl'] > 0]
            pl = pt[pt['pnl'] < 0]
            p_pf = pw['pnl'].sum()/abs(pl['pnl'].sum()) if len(pl) else float('inf')
            print(f"   {pair}: {len(pt):>4}T | WR:{len(pw)/len(pt)*100:5.1f}% | "
                  f"PF:{p_pf:.2f} | Net:${pt['pnl'].sum():>8,.2f}")

    print("-" * 68)
    print(" Exit types:")
    g = df_t.groupby('status')['pnl'].agg(['count','mean','sum'])
    for st, row in g.sort_values('sum').iterrows():
        bar_ = "█" * min(int(abs(row['sum']) / 400), 18)
        sign = "+" if row['sum'] >= 0 else "-"
        print(f"   {st:<10} {row['count']:>5}  avg:${row['mean']:>8.2f}  "
              f"total:{sign}${abs(row['sum']):>8,.2f}  {bar_}")

    print("-" * 68)
    print(" Yearly:")
    for yr, g2 in df_t.groupby('year'):
        w2 = g2[g2['pnl'] > 0]
        l2 = g2[g2['pnl'] < 0]
        ypf = w2['pnl'].sum()/abs(l2['pnl'].sum()) if len(l2) else 99
        s   = g2['pnl'].sum()
        sgn = "+" if s >= 0 else "-"
        print(f"   {yr}: {len(g2):>4}T  WR:{len(w2)/len(g2)*100:5.1f}%  "
              f"PF:{ypf:.2f}  {sgn}${abs(s):>7,.2f}")
    print("═" * 68)


# ═══════════════════════════════════════════════════════════════════
#  WALK-FORWARD OPTIMIZER
# ═══════════════════════════════════════════════════════════════════
def run_optimizer(pairs: dict, pair_signals_full: dict):
    """
    Walk-Forward Optimization:
      Train: 2010-2019 → بهترین parameter combo رو پیدا کن
      Test:  2020-2025 → روی داده ندیده اجرا کن

    پارامترها:
      z_entry:      [2.0, 2.1, 2.3, 2.5]
      atr_sl_mult:  [1.5, 2.0, 2.5]
      vr_max:       [0.85, 0.88, 0.92]
      z_exit_partial: [0.3, 0.5, 0.7]
    """
    param_grid = {
        'z_entry':        [2.0, 2.1, 2.3, 2.5],
        'atr_sl_mult':    [1.5, 2.0, 2.5],
        'vr_max':         [0.85, 0.88, 0.92],
        'z_exit_partial': [0.3, 0.5, 0.7],
    }

    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    total  = len(combos)

    print(f"\n  ▶ Walk-Forward Optimizer — {total} combinations")
    print(f"    Train: 2010-2019 | Test: 2020-2025")

    train_end   = Config.opt_train_end
    test_start  = Config.opt_test_start

    best_train_score = -np.inf
    best_params      = None
    train_results    = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg    = Config()
        for k, v in params.items():
            setattr(cfg, k, v)

        # Re-compute signals with new config
        ps = {}
        for name, info in pairs.items():
            try:
                ps[name] = compute_signals(name, info, cfg)
            except:
                ps[name] = pair_signals_full.get(name, (None,) * 4)

        res = run_backtest(pairs, ps, cfg,
                           date_end=train_end, verbose=False)
        if res is None or not res['all_trades']:
            continue

        df_t = pd.DataFrame(res['all_trades'])
        wins = df_t[df_t['pnl'] > 0]
        loss = df_t[df_t['pnl'] < 0]
        if not len(loss):
            continue
        pf   = wins['pnl'].sum() / abs(loss['pnl'].sum())
        bank = res['withdrawn']
        logs = res['account_logs']
        blown = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')

        # Score = weighted combination
        # بیشتر blown = کمتر امتیاز
        score = pf * 0.4 + (bank / 1000) * 0.4 - blown * 0.2

        train_results.append({**params, 'pf': pf, 'bank': bank, 'blown': blown, 'score': score})

        if score > best_train_score:
            best_train_score = score
            best_params = params

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{total} done... best score so far: {best_train_score:.3f}", end='\r')

    print(f"\n\n  ✅ Best params (train 2010-2019):")
    for k, v in best_params.items():
        print(f"    {k} = {v}")

    # ── Test on out-of-sample ──
    print(f"\n  ▶ Testing best params on 2020-2025...")
    cfg_best = Config()
    for k, v in best_params.items():
        setattr(cfg_best, k, v)

    ps_best = {}
    for name, info in pairs.items():
        ps_best[name] = compute_signals(name, info, cfg_best)

    res_test = run_backtest(pairs, ps_best, cfg_best,
                            date_start=test_start, verbose=True)
    if res_test:
        print_report(res_test, title=f"v9 OPTIMIZER — Out-of-Sample 2020-2025")

    # Show top 5 train results
    if train_results:
        df_r = pd.DataFrame(train_results).sort_values('score', ascending=False)
        print("\n  Top 5 parameter sets (train period):")
        print(f"  {'z_entry':>8} {'sl_mult':>8} {'vr_max':>7} {'z_prt':>6} {'PF':>6} {'Bank':>8} {'Blown':>6} {'Score':>7}")
        for _, row in df_r.head(5).iterrows():
            print(f"  {row['z_entry']:>8.1f} {row['atr_sl_mult']:>8.1f} {row['vr_max']:>7.2f} "
                  f"{row['z_exit_partial']:>6.1f} {row['pf']:>6.2f} "
                  f"${row['bank']:>7,.0f} {row['blown']:>6.0f} {row['score']:>7.3f}")

    return best_params


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()

    # ── بارگذاری داده ──
    pairs = load_all_pairs()

    # ── محاسبه سیگنال‌ها با config پیش‌فرض ──
    print("\n  Computing Signals...")
    cfg_default = Config()
    pair_signals = {}
    for name, info in pairs.items():
        pair_signals[name] = compute_signals(name, info, cfg_default)

    # ── اجرای کامل 15 ساله ──
    print("\n  ▶ Full backtest 2010-2025...")
    results = run_backtest(pairs, pair_signals, cfg_default, verbose=True)
    if results:
        print_report(results, title="v9 Full (2010-2025)")

    print("\n" + "─" * 68)

    # ── Walk-Forward Optimizer ──
    run_opt = input("\n  اجرای Optimizer? (y/n) [~5-10 min]: ").strip().lower()
    if run_opt == 'y':
        best = run_optimizer(pairs, pair_signals)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  ✅ Total time: {elapsed:.1f}s")
