"""
CorrArb v10 — Full Rebuild
============================
بازنویسی کامل بر اساس تحلیل v9b

تغییرات کلیدی:
  ✅ FIX: Permanent pause → Probation mode (half risk)
  ✅ FIX: Month/day boundary با تغییر تاریخ واقعی
  ✅ FIX: Intrabar SL/TP با high/low
  ✅ FIX: Monthly report شامل zero-months
  ✅ NEW: Entry on reversion confirmation (نه اولین breach)
  ✅ NEW: ATR-based dynamic stop
  ✅ NEW: Exit redesign — partial دیرتر، trailing بهتر
  ✅ NEW: Session-specific per pair
  ✅ NEW: AUDNZD-only baseline (EURGBP حذف تا اثبات edge)
  ✅ NEW: Rolling performance tracking
  ✅ NEW: Equity curve export
"""

import pandas as pd
import numpy as np
import glob, zipfile, os, warnings, json
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG — Per-Pair
# ═══════════════════════════════════════════════════════════════════════
class GlobalConfig:
    """تنظیمات مشترک پراپ"""
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05        # 5% = $250
    max_daily_loss_pct = 0.05        # 5% = $250
    max_total_dd_pct   = 0.10        # 10% = $500

    commission_per_lot = 7.0
    slippage_pips      = 0.5
    lot_size           = 100_000
    max_lot            = 3.0
    min_lot            = 0.01
    warmup             = 500

    cooldown_days      = 10
    max_trades_day     = 3           # ↑ از 2

    # Risk
    risk_base_pct      = 0.018       # ↑ کمی بیشتر
    risk_min_pct       = 0.006
    risk_probation_mult= 0.50        # ریسک در حالت probation
    consec_loss_n      = 3           # ↑ از 2 — tolerance بیشتر
    consec_loss_reduce = 0.60

    # Monthly stress
    monthly_loss_threshold = -120.0  # اگر ماه بدتر از این بود، risk کم شود

    # Pair Quality — Probation mode
    pq_min_trades      = 25          # حداقل ترید برای ارزیابی
    pq_window_days     = 180         # پنجره ارزیابی
    pq_start_days      = 120         # حداقل روز قبل از اولین ارزیابی
    pq_min_pf          = 1.08        # حداقل PF
    pq_resume_pf       = 1.12        # PF لازم برای خروج از probation
    pq_eval_interval   = 30          # هر 30 روز


class PairConfig:
    """تنظیمات اختصاصی هر pair"""

    AUDNZD = {
        'spread_pip':    2.5,
        'pip_size':      0.0001,

        # Session — آسیا + اوایل لندن
        'hour_start':    21,         # 21:00 UTC = شروع سیدنی
        'hour_end':      10,         # 10:00 UTC = اوایل لندن
        'trade_days':    [0, 1, 2, 3, 4],

        # Z-Score
        'z_period':      72,         # ↓ سریع‌تر — 18 ساعت
        'z_entry':       2.0,        # ↓ بیشتر ترید
        'z_exit_partial':1.2,        # ↑ partial دیرتر
        'z_exit_full':   0.3,        # ↑ full exit کمی قبل‌تر از صفر
        'z_stop':        4.5,

        # Entry confirmation
        'confirm_bars':  3,          # z باید 3 بار زیر threshold بمانه و بعد برگرده

        # ATR Stop
        'atr_period':    14,
        'atr_sl_mult':   2.0,        # SL = 2 * ATR
        'atr_tp_mult':   4.0,        # TP = 4 * ATR (2:1 effective with partial)
        'atr_ma_period': 48,
        'atr_max_vol':   2.5,        # max ATR/MA ratio
        'atr_min_vol':   0.5,

        # Exit
        'partial_ratio':   0.50,
        'trail_after_partial': True,
        'trail_atr_mult':  1.5,      # trailing stop = entry + 1.5*ATR
        'time_stop_bars':  48,       # 12 ساعت
        'time_stop_bars_max': 96,    # 24 ساعت — force exit
        'min_net_profit':  12.0,

        # Correlation
        'corr_period':     72,
        'corr_min':        0.75,     # ↓ کمی loose‌تر

        # Variance Ratio
        'vr_period':  200,
        'vr_k':       4,
        'vr_max':     0.85,          # ↓ سخت‌تر — فقط mean-reverting
    }

    # EURGBP — آماده ولی غیرفعال
    EURGBP = {
        'enabled':       False,      # ← غیرفعال تا اثبات edge
        'spread_pip':    1.0,
        'pip_size':      0.0001,
        'hour_start':    7,
        'hour_end':      16,
        'trade_days':    [0, 1, 2, 3, 4],
        'z_period':      96,
        'z_entry':       2.2,
        'z_exit_partial':1.0,
        'z_exit_full':   0.2,
        'z_stop':        4.0,
        'confirm_bars':  3,
        'atr_period':    14,
        'atr_sl_mult':   1.8,
        'atr_tp_mult':   3.5,
        'atr_ma_period': 48,
        'atr_max_vol':   2.5,
        'atr_min_vol':   0.5,
        'partial_ratio':   0.50,
        'trail_after_partial': True,
        'trail_atr_mult':  1.5,
        'time_stop_bars':  48,
        'time_stop_bars_max': 96,
        'min_net_profit':  12.0,
        'corr_period':     96,
        'corr_min':        0.80,
        'vr_period':  200,
        'vr_k':       4,
        'vr_max':     0.85,
    }

    @classmethod
    def get(cls, name):
        return getattr(cls, name, None)

    @classmethod
    def active_pairs(cls):
        result = {}
        for name in ['AUDNZD', 'EURGBP']:
            cfg = cls.get(name)
            if cfg and cfg.get('enabled', True):
                result[name] = cfg
        return result


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_raw_csv(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV: {pattern}")
    frames = []
    for p in paths:
        frames.append(pd.read_csv(p, sep=';', header=None,
                                  names=['ts', 'o', 'h', 'l', 'c', 'v']))
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw


def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No ZIP: {pattern}")
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p) as z:
                csv_name = next(
                    (f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if not csv_name:
                    continue
                with z.open(csv_name) as f:
                    frames.append(pd.read_csv(f, sep=';', header=None,
                                              names=['ts', 'o', 'h', 'l', 'c', 'v']))
        except Exception as e:
            print(f"  ⚠ {os.path.basename(p)}: {e}")
    if not frames:
        raise ValueError(f"No valid ZIP: {pattern}")
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw


def to_15min(raw, sfx):
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()


def load_pair_data():
    """بارگذاری داده‌ها برای pairهای فعال"""
    print("\n  Loading datasets...")
    active = PairConfig.active_pairs()
    loaded = {}

    if 'AUDNZD' in active:
        try:
            aud = to_15min(load_raw_zip('data/HISTDATA*AUDUSD*.zip'), 'aud')
            nzd = to_15min(load_raw_zip('data/HISTDATA*NZDUSD*.zip'), 'nzd')
            m = aud.join(nzd, how='inner').dropna()
            m['c_spread'] = m['c_aud'] / m['c_nzd']
            m['o_spread'] = m['o_aud'] / m['o_nzd']
            m['h_spread'] = m['h_aud'] / m['l_nzd']
            m['l_spread'] = m['l_aud'] / m['h_nzd']
            m['quote_rate'] = m['c_nzd']
            # legs برای correlation
            m['leg1'] = m['c_aud']
            m['leg2'] = m['c_nzd']
            m = m[m.index.weekday < 5].copy()
            loaded['AUDNZD'] = m
            print(f"  ✅ AUDNZD: {len(m):,} candles (synthetic)")
        except Exception as e:
            print(f"  ❌ AUDNZD: {e}")

    if 'EURGBP' in active:
        try:
            raw = load_raw_zip('data/HISTDATA*EURGBP*.zip')
            df15 = to_15min(raw, 'eg')
            gbp15 = to_15min(load_raw_csv('data/*GBPUSD*.csv'), 'gbp')
            m = df15.join(gbp15[['c_gbp']], how='inner').dropna()
            m['c_spread'] = m['c_eg']
            m['o_spread'] = m['o_eg']
            m['h_spread'] = m['h_eg']
            m['l_spread'] = m['l_eg']
            m['quote_rate'] = m['c_gbp']
            m['leg1'] = m['c_eg']
            m['leg2'] = m['c_gbp']
            m = m[m.index.weekday < 5].copy()
            loaded['EURGBP'] = m
            print(f"  ✅ EURGBP: {len(m):,} candles (direct)")
        except Exception as e:
            print(f"  ❌ EURGBP: {e}")

    if not loaded:
        raise RuntimeError("No pairs loaded!")
    return loaded


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_vr(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    v1 = r1.rolling(window).var()
    vk = rk.rolling(window).var()
    return vk / (k * v1.replace(0, np.nan))


def session_check(hour, cfg):
    """چک session با پشتیبانی از wrap-around (مثلاً 21-10)"""
    hs, he = cfg['hour_start'], cfg['hour_end']
    if hs <= he:
        return hs <= hour <= he
    else:
        return hour >= hs or hour <= he


# ═══════════════════════════════════════════════════════════════════════
#  SIGNAL GENERATION — with reversion confirmation
# ═══════════════════════════════════════════════════════════════════════
def compute_signals(name, df, pcfg):
    """
    Entry on reversion confirmation:
    1. z از threshold عبور کند
    2. حداقل confirm_bars بار زیر/بالای threshold بماند
    3. بعد z شروع به برگشت کند (z بهتر از بار قبل)
    """
    log_r = np.log(df['c_spread'])

    # Z-Score
    z_mean = log_r.rolling(pcfg['z_period']).mean()
    z_std = log_r.rolling(pcfg['z_period']).std()
    z = (log_r - z_mean) / z_std.replace(0, np.nan)

    # Correlation filter
    if 'leg1' in df.columns and 'leg2' in df.columns:
        corr = (df['leg1'].pct_change()
                .rolling(pcfg['corr_period'])
                .corr(df['leg2'].pct_change()))
        corr_ok = corr > pcfg['corr_min']
    else:
        corr_ok = pd.Series(True, index=df.index)

    # Variance Ratio — mean reversion regime
    vr = calc_vr(log_r, pcfg['vr_k'], pcfg['vr_period'])
    regime_ok = vr < pcfg['vr_max']

    # ATR volatility filter
    atr = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'],
                   pcfg['atr_period'])
    atr_ma = atr.rolling(pcfg['atr_ma_period']).mean()
    vol_ok = ((atr > atr_ma * pcfg['atr_min_vol']) &
              (atr < atr_ma * pcfg['atr_max_vol']))

    # Session filter
    hours = pd.Series(df.index.hour, index=df.index)
    dows = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = pd.Series([
        session_check(h, pcfg) and d in pcfg['trade_days']
        for h, d in zip(hours, dows)
    ], index=df.index)

    # ── Reversion Confirmation Logic ──
    z_vals = z.values
    n = len(z_vals)
    sig = np.zeros(n, dtype=int)
    cb = pcfg['confirm_bars']
    ze = pcfg['z_entry']

    # Track breach state
    long_breach_count = 0
    short_breach_count = 0
    long_armed = False
    short_armed = False
    prev_z = np.nan

    for i in range(n):
        zi = z_vals[i]
        if np.isnan(zi):
            long_breach_count = short_breach_count = 0
            long_armed = short_armed = False
            prev_z = zi
            continue

        # Long setup: z < -entry
        if zi < -ze:
            long_breach_count += 1
            short_breach_count = 0
            short_armed = False
            if long_breach_count >= cb:
                long_armed = True
        else:
            if long_armed and not np.isnan(prev_z) and zi > prev_z:
                # z برگشته — سیگنال long
                sig[i] = 1
                long_armed = False
                long_breach_count = 0
            else:
                long_breach_count = 0
                long_armed = False

        # Short setup: z > entry
        if zi > ze:
            short_breach_count += 1
            long_breach_count = 0
            long_armed = False
            if short_breach_count >= cb:
                short_armed = True
        else:
            if short_armed and not np.isnan(prev_z) and zi < prev_z:
                sig[i] = -1
                short_armed = False
                short_breach_count = 0
            else:
                short_breach_count = 0
                short_armed = False

        prev_z = zi

    sig_series = pd.Series(sig, index=df.index)

    # Apply filters
    base_ok = vol_ok & time_ok & corr_ok & regime_ok
    sig_series = sig_series.where(base_ok, 0)

    # Remove consecutive duplicates
    sig_series = sig_series.where(sig_series != sig_series.shift(), 0)

    n_sig = int((sig_series != 0).sum())
    n_regime = int(regime_ok.sum())
    n_time = int(time_ok.sum())
    print(f"    {name}: {n_sig:,} signals | Regime: {n_regime:,} | "
          f"Session: {n_time:,} bars")

    return sig_series, z, atr


# ═══════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════
def calc_pnl(direction, entry, exit_px, lot, qr, pip):
    G = GlobalConfig
    gross = direction * (exit_px - entry) * lot * G.lot_size * qr
    return gross - G.commission_per_lot * lot


def new_account(ts):
    G = GlobalConfig
    return {
        'equity':      G.initial_balance,
        'start_ts':    ts,
        'trades':      [],
        'blown':       False,
        'blown_rsn':   '',
        'peak':        G.initial_balance,
        'consec_loss': 0,
    }


def make_record(pos, exit_px, exit_ts, pnl, status, lot):
    return {
        'pair':     pos['pair'],
        'dir':      pos['dir'],
        'lot':      lot,
        'entry':    pos['entry'],
        'exit':     exit_px,
        'entry_ts': pos['entry_ts'],
        'exit_ts':  exit_ts,
        'pnl':      pnl,
        'status':   status,
    }


def run_backtest(pair_data, pair_signals):
    G = GlobalConfig
    pair_names = list(pair_data.keys())
    active_cfgs = PairConfig.active_pairs()

    # ── Common index ──
    common_idx = None
    for n in pair_names:
        idx = pair_data[n].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)
    start_date = common_idx[G.warmup]
    print(f"  ✅ Common bars: {n_bars:,} | "
          f"{common_idx[0].date()} → {common_idx[-1].date()}")

    # ── Pre-compute arrays ──
    pa = {}
    for n in pair_names:
        pcfg = active_cfgs[n]
        df_p = pair_data[n].reindex(common_idx).ffill()
        sig_s, z_s, atr_s = pair_signals[n]
        pa[n] = {
            'o':   df_p['o_spread'].values.astype(float),
            'h':   df_p['h_spread'].values.astype(float),
            'l':   df_p['l_spread'].values.astype(float),
            'c':   df_p['c_spread'].values.astype(float),
            'qr':  df_p['quote_rate'].values.astype(float),
            'sig': sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z':   z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
            'atr': atr_s.reindex(common_idx).fillna(np.nan).values.astype(float),
            'sp':  pcfg['spread_pip'],
            'pip': pcfg['pip_size'],
            'cfg': pcfg,
        }

    # ── Prop limits ──
    PROP_FLOOR = G.initial_balance * (1 - G.max_total_dd_pct)
    PROFIT_LEVEL = G.initial_balance * (1 + G.profit_target_pct)

    # ── State ──
    acc = new_account(start_date)
    total_withdrawn = 0.0
    acc_num = 1
    day_start_eq = G.initial_balance
    month_start_eq = G.initial_balance
    cooldown_until = None
    all_trades = []
    acc_logs = []
    equity_curve = []

    positions = {n: None for n in pair_names}
    trades_today = {n: 0 for n in pair_names}
    pending_sig = {n: 0 for n in pair_names}

    # Pair Quality
    pair_trades_hist = {n: [] for n in pair_names}
    pair_status = {n: 'ACTIVE' for n in pair_names}  # ACTIVE / PROBATION
    last_eval_date = {n: None for n in pair_names}

    # Day/month tracking
    prev_date = None
    prev_month = None

    print(f"\n  ▶ Running v10...")
    print(f"    Pairs: {' + '.join(pair_names)}")
    print(f"    Entry: Reversion confirmation ({pa[pair_names[0]]['cfg']['confirm_bars']} bars)")
    print(f"    Stop: ATR-based dynamic")
    print(f"    Quality: Probation mode (half risk, not full pause)")

    for bar in range(G.warmup, n_bars):
        ts = common_idx[bar]
        cur_date = ts.date()
        cur_month = (ts.year, ts.month)
        eq = acc['equity']

        # ── Equity curve ──
        if cur_date != prev_date:
            equity_curve.append({
                'date': cur_date,
                'equity': eq + total_withdrawn,
                'account_equity': eq,
                'withdrawn': total_withdrawn,
            })

        # ── Day reset ──
        if cur_date != prev_date:
            day_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
            prev_date = cur_date

        # ── Month reset + Quality eval ──
        if cur_month != prev_month:
            if prev_month is not None:
                month_pnl = acc['equity'] - month_start_eq
                # Quality evaluation
                days_since_start = (ts - start_date).days
                if days_since_start >= G.pq_start_days:
                    for n in pair_names:
                        if (last_eval_date[n] is not None and
                                (ts - last_eval_date[n]).days < G.pq_eval_interval):
                            continue
                        last_eval_date[n] = ts
                        cutoff = ts - timedelta(days=G.pq_window_days)
                        recent = [(t, p) for t, p in pair_trades_hist[n]
                                  if t >= cutoff]

                        if len(recent) < G.pq_min_trades:
                            # نه‌به‌اندازه‌کافی ترید — status تغییر نکند
                            # ولی اگر probation بود، آزاد کن
                            if pair_status[n] == 'PROBATION':
                                pair_status[n] = 'ACTIVE'
                                print(f"    ▶ FREED {n} (too few trades "
                                      f"for eval) | {ts.date()}")
                            continue

                        wins = sum(p for _, p in recent if p > 0)
                        losses_abs = abs(sum(p for _, p in recent if p < 0))
                        pf = wins / losses_abs if losses_abs > 0 else 99.0

                        old_status = pair_status[n]
                        if pair_status[n] == 'ACTIVE':
                            if pf < G.pq_min_pf:
                                pair_status[n] = 'PROBATION'
                        else:  # PROBATION
                            if pf >= G.pq_resume_pf:
                                pair_status[n] = 'ACTIVE'

                        if pair_status[n] != old_status:
                            icon = "⚠ PROBATION" if pair_status[n] == 'PROBATION' \
                                else "▶ RESUMED"
                            print(f"    {icon} {n} | PF({G.pq_window_days}d)"
                                  f"={pf:.2f} | T={len(recent)} | {ts.date()}")

            month_start_eq = acc['equity']
            prev_month = cur_month

        # ── Peak ──
        if eq > acc['peak']:
            acc['peak'] = eq

        in_cooldown = (cooldown_until is not None and ts < cooldown_until)

        # ── Blown handling ──
        if acc['blown']:
            acc_logs.append({
                'account': acc_num,
                'start_ts': acc['start_ts'],
                'end_ts': ts,
                'reason': acc['blown_rsn'],
                'pnl': acc['equity'] - G.initial_balance,
                'n_trades': len(acc['trades']),
                'days': (ts - acc['start_ts']).days,
            })
            print(f"    💥 #{acc_num:>3} | {ts.date()} | "
                  f"Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            cooldown_until = ts + timedelta(days=G.cooldown_days)
            acc_num += 1
            acc = new_account(ts)
            day_start_eq = month_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
                pending_sig[n] = 0
                positions[n] = None
            prev_date = cur_date
            prev_month = cur_month
            continue

        if in_cooldown:
            continue

        # Monthly stress flag
        monthly_pnl = acc['equity'] - month_start_eq
        monthly_stressed = monthly_pnl < G.monthly_loss_threshold

        # ── Entry from pending signals ──
        for n in pair_names:
            a = pa[n]
            pcfg = a['cfg']
            if (pending_sig[n] != 0
                    and positions[n] is None
                    and trades_today[n] < G.max_trades_day):

                sv = pending_sig[n]
                qr = a['qr'][bar]
                pip = a['pip']
                atr_val = a['atr'][bar]

                if np.isnan(atr_val) or atr_val <= 0:
                    pending_sig[n] = 0
                    continue

                # Risk calculation
                risk = G.risk_base_pct
                if pair_status[n] == 'PROBATION':
                    risk *= G.risk_probation_mult
                if monthly_stressed:
                    risk *= 0.6
                if acc['consec_loss'] >= G.consec_loss_n:
                    risk = max(risk * G.consec_loss_reduce, G.risk_min_pct)

                # ATR-based SL
                sl_distance = atr_val * pcfg['atr_sl_mult']
                tp_distance = atr_val * pcfg['atr_tp_mult']

                # Lot sizing based on ATR stop
                pv = pip * G.lot_size * qr if pip * G.lot_size * qr > 0 else 10.0
                sl_pips = sl_distance / pip
                lot = round(float(np.clip(
                    acc['equity'] * risk / (sl_pips * pv),
                    G.min_lot, G.max_lot
                )), 2)

                # Entry price with spread + slippage
                ep = a['o'][bar] + sv * (G.slippage_pips + a['sp'] / 2) * pip
                sl = ep - sv * sl_distance
                tp = ep + sv * tp_distance

                positions[n] = {
                    'pair':           n,
                    'dir':            sv,
                    'lot':            lot,
                    'lot_remaining':  lot,
                    'partial_done':   False,
                    'entry':          ep,
                    'sl':             sl,
                    'tp':             tp,
                    'sl_initial':     sl,
                    'entry_ts':       ts,
                    'entry_bar':      bar,
                    'pip':            pip,
                    'atr_at_entry':   atr_val,
                }
                trades_today[n] += 1
            pending_sig[n] = 0

        # ── Floating PnL check ──
        total_float = 0.0
        for n in pair_names:
            pos = positions[n]
            if pos is None:
                continue
            a = pa[n]
            pnl_f = calc_pnl(pos['dir'], pos['entry'], a['c'][bar],
                             pos['lot_remaining'], a['qr'][bar], pos['pip'])
            total_float += pnl_f

        current_eq = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - G.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            reason = "DailyDD" if current_eq <= daily_limit else "TotalDD"
            acc['blown'] = True
            acc['blown_rsn'] = reason
            for n in pair_names:
                pos = positions[n]
                if pos is None:
                    continue
                a = pa[n]
                pnl = calc_pnl(pos['dir'], pos['entry'], a['c'][bar],
                               pos['lot_remaining'], a['qr'][bar], pos['pip'])
                acc['equity'] += pnl
                r = make_record(pos, a['c'][bar], ts, pnl, 'BLOWN',
                                pos['lot_remaining'])
                all_trades.append(r)
                acc['trades'].append(r)
                pair_trades_hist[n].append((ts, pnl))
                positions[n] = None
            continue

        # ── Exit logic — with intrabar SL/TP ──
        for n in pair_names:
            pos = positions[n]
            if pos is None:
                continue

            a = pa[n]
            pcfg = a['cfg']
            cp = a['c'][bar]
            hp = a['h'][bar]
            lp = a['l'][bar]
            qr = a['qr'][bar]
            d = pos['dir']
            ep = pos['entry']
            zn = a['z'][bar]
            pip = pos['pip']
            lr = pos['lot_remaining']
            bars_open = bar - pos['entry_bar']

            # ── Intrabar SL check ──
            hit_sl = False
            if d == 1 and lp <= pos['sl']:
                hit_sl = True
            elif d == -1 and hp >= pos['sl']:
                hit_sl = True

            # ── Intrabar TP check ──
            hit_tp = False
            if d == 1 and hp >= pos['tp']:
                hit_tp = True
            elif d == -1 and lp <= pos['tp']:
                hit_tp = True

            # ── Z-based exits ──
            hit_z_stop = False
            hit_z_exit = False
            if not np.isnan(zn):
                if ((d == 1 and zn <= -pcfg['z_stop']) or
                        (d == -1 and zn >= pcfg['z_stop'])):
                    hit_z_stop = True
                if ((d == 1 and zn >= -pcfg['z_exit_full']) or
                        (d == -1 and zn <= pcfg['z_exit_full'])):
                    pnl_check = calc_pnl(d, ep, cp, lr, qr, pip)
                    if pnl_check >= pcfg['min_net_profit'] or pos['partial_done']:
                        hit_z_exit = True

            # ── Time stop ──
            time_stop = False
            pnl_now = calc_pnl(d, ep, cp, lr, qr, pip)
            if bars_open >= pcfg['time_stop_bars'] and pnl_now < 0:
                time_stop = True
            elif bars_open >= pcfg['time_stop_bars_max']:
                time_stop = True

            # ── Partial exit (دیرتر — z باید بیشتر برگشته باشه) ──
            if (not pos['partial_done'] and not np.isnan(zn)
                    and not hit_sl and not hit_tp):
                partial_trigger = False
                if d == 1 and zn >= -pcfg['z_exit_partial']:
                    partial_trigger = True
                elif d == -1 and zn <= pcfg['z_exit_partial']:
                    partial_trigger = True

                if partial_trigger:
                    p_lot = round(lr * pcfg['partial_ratio'], 2)
                    if p_lot >= G.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr, pip)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            r = make_record(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(r)
                            acc['trades'].append(r)
                            pair_trades_hist[n].append((ts, p_pnl))
                            pos['lot_remaining'] = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            lr = pos['lot_remaining']

                            # Trailing stop after partial
                            if pcfg['trail_after_partial']:
                                trail_dist = pos['atr_at_entry'] * pcfg['trail_atr_mult']
                                if d == 1:
                                    new_sl = max(pos['sl'], cp - trail_dist)
                                else:
                                    new_sl = min(pos['sl'], cp + trail_dist)
                                pos['sl'] = new_sl

                            if lr < G.min_lot:
                                positions[n] = None
                                continue

            # ── Trailing update (every bar after partial) ──
            if pos['partial_done'] and pcfg['trail_after_partial']:
                trail_dist = pos['atr_at_entry'] * pcfg['trail_atr_mult']
                if d == 1:
                    new_sl = max(pos['sl'], cp - trail_dist)
                    pos['sl'] = new_sl
                else:
                    new_sl = min(pos['sl'], cp + trail_dist)
                    pos['sl'] = new_sl

            # ── Final exit ──
            if hit_sl or hit_tp or hit_z_exit or hit_z_stop or time_stop:
                # Determine exit price
                if hit_sl:
                    exit_px = pos['sl']
                    status = 'SL'
                elif hit_tp:
                    exit_px = pos['tp']
                    status = 'TP'
                elif hit_z_stop:
                    exit_px = cp
                    status = 'Z-Stop'
                elif time_stop:
                    exit_px = cp
                    status = 'TimeStop'
                else:
                    exit_px = cp
                    status = 'Z-Exit'

                fpnl = calc_pnl(d, ep, exit_px, lr, qr, pip)
                acc['equity'] += fpnl
                r = make_record(pos, exit_px, ts, fpnl, status, lr)
                all_trades.append(r)
                acc['trades'].append(r)
                pair_trades_hist[n].append((ts, fpnl))
                positions[n] = None

                if fpnl > 0:
                    acc['consec_loss'] = 0
                else:
                    acc['consec_loss'] += 1

        # ── Target hit ──
        if (acc['equity'] >= PROFIT_LEVEL and
                all(positions[n] is None for n in pair_names)):
            w = acc['equity'] - G.initial_balance
            total_withdrawn += w
            dt = (ts - acc['start_ts']).days
            nt = len(acc['trades'])
            acc_logs.append({
                'account': acc_num,
                'start_ts': acc['start_ts'],
                'end_ts': ts,
                'reason': 'TARGET_HIT',
                'pnl': w,
                'n_trades': nt,
                'days': dt,
            })
            print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | "
                  f"Bank:${total_withdrawn:>9.2f} | {dt}d | {nt}T")
            acc_num += 1
            acc = new_account(ts)
            day_start_eq = month_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
                pending_sig[n] = 0
            prev_date = cur_date
            prev_month = cur_month
            continue

        # ── New signals ──
        for n in pair_names:
            a = pa[n]
            if (positions[n] is None
                    and not acc['blown']
                    and not in_cooldown
                    and trades_today[n] < G.max_trades_day
                    and a['sig'][bar] != 0):
                pending_sig[n] = int(a['sig'][bar])

    return {
        'all_trades':      all_trades,
        'account_logs':    acc_logs,
        'total_withdrawn': total_withdrawn,
        'final_equity':    acc['equity'],
        'total_accounts':  acc_num,
        'pair_names':      pair_names,
        'equity_curve':    equity_curve,
        'common_idx':      common_idx,
        'pair_status':     pair_status,
    }


# ═══════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════
def print_report(results):
    trades = results['all_trades']
    if not trades:
        print("\n❌ No trades.")
        return

    df_t = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['entry_ts'] = pd.to_datetime(df_t['entry_ts'])
    df_t['month'] = df_t['exit_ts'].dt.to_period('M')

    wins = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())
          if len(losses) and losses['pnl'].sum() != 0 else float('inf'))

    logs = results['account_logs']
    df_acc = pd.DataFrame(logs) if logs else pd.DataFrame()
    targets = (df_acc[df_acc['reason'] == 'TARGET_HIT']
               if len(df_acc) else pd.DataFrame())
    blowns = (df_acc[df_acc['reason'] != 'TARGET_HIT']
              if len(df_acc) else pd.DataFrame())

    # ── Full monthly report (including zero months) ──
    ci = results['common_idx']
    first_trade = df_t['exit_ts'].min()
    last_date = ci[-1]
    all_months = pd.period_range(
        start=first_trade.to_period('M'),
        end=pd.Period(last_date, freq='M'),
        freq='M'
    )
    monthly = df_t.groupby('month')['pnl'].sum().reindex(all_months, fill_value=0.0)
    pos_m = int((monthly > 0).sum())
    neg_m = int((monthly < 0).sum())
    zero_m = int((monthly == 0).sum())
    total_m = len(monthly)

    # Rolling 12-month average
    if len(monthly) >= 12:
        roll12 = monthly.rolling(12).mean()
        avg_roll12 = roll12.dropna().mean()
    else:
        avg_roll12 = monthly.mean()

    # Longest losing streak (months)
    losing_streak = 0
    max_losing_streak = 0
    for v in monthly.values:
        if v <= 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    # Average trade duration
    df_t['duration_bars'] = 0
    for i, row in df_t.iterrows():
        dur = (row['exit_ts'] - row['entry_ts']).total_seconds() / 900
        df_t.at[i, 'duration_bars'] = dur
    avg_dur_hours = df_t['duration_bars'].mean() * 0.25

    print("\n" + "═" * 72)
    print(f" ▌  CorrArb v10 — {'+'.join(results['pair_names'])}  ▐")
    print("═" * 72)
    print(f" {'Total Trades:':<30} {len(df_t):,}")
    print(f" {'Win Rate:':<30} {wr:.2f}%")
    print(f" {'Profit Factor:':<30} {pf:.2f}")
    print(f" {'Avg Win:':<30} ${wins['pnl'].mean():.2f}" if len(wins) else "")
    print(f" {'Avg Loss:':<30} ${losses['pnl'].mean():.2f}" if len(losses) else "")
    print(f" {'Avg Trade Duration:':<30} {avg_dur_hours:.1f} ساعت")
    print(f" {'Avg Trade PnL:':<30} ${df_t['pnl'].mean():.2f}")
    print("-" * 72)
    print(f" {'Accounts Passed:':<30} {len(targets)}")
    print(f" {'Accounts Blown:':<30} {len(blowns)}")
    if len(targets) and 'days' in targets.columns:
        print(f" {'Avg Days/Pass:':<30} {targets['days'].mean():.0f} روز")
        print(f" {'Avg Trades/Pass:':<30} {targets['n_trades'].mean():.0f} ترید")
    print("-" * 72)
    print(f" {'Total Banked (15yr):':<30} ${results['total_withdrawn']:,.2f}")
    print(f" {'Active Equity:':<30} ${results['final_equity']:,.2f}")
    yrs = max(1, total_m / 12)
    print(f" {'Annual Avg:':<30} ${results['total_withdrawn'] / yrs:,.2f}/سال")
    print(f" {'Monthly Avg:':<30} "
          f"${results['total_withdrawn'] / max(1, total_m):.2f}/ماه")
    print("-" * 72)
    print(f" {'ماه‌های مثبت:':<30} {pos_m} از {total_m} ({pos_m/total_m*100:.0f}%)")
    print(f" {'ماه‌های منفی:':<30} {neg_m} از {total_m}")
    print(f" {'ماه‌های صفر (بدون ترید):':<30} {zero_m} از {total_m}")
    print(f" {'بلندترین streak منفی:':<30} {max_losing_streak} ماه")
    print(f" {'بهترین ماه:':<30} ${monthly.max():,.2f}")
    print(f" {'بدترین ماه:':<30} ${monthly.min():,.2f}")
    print(f" {'Rolling 12M Avg:':<30} ${avg_roll12:,.2f}/ماه")
    print("-" * 72)
    print(" عملکرد هر جفت ارز:")
    for pair in results['pair_names']:
        pt = df_t[df_t['pair'] == pair] if 'pair' in df_t.columns else pd.DataFrame()
        if not len(pt):
            continue
        pw = pt[pt['pnl'] > 0]
        pl = pt[pt['pnl'] < 0]
        ppf = (pw['pnl'].sum() / abs(pl['pnl'].sum())
               if len(pl) and pl['pnl'].sum() != 0 else float('inf'))
        st = results['pair_status'].get(pair, '?')
        print(f"   {pair}: {len(pt):>5}T | WR:{len(pw)/len(pt)*100:5.1f}% | "
              f"PF:{ppf:.2f} | Net:${pt['pnl'].sum():>9,.2f} | Status:{st}")
    print("-" * 72)
    print(" خروج‌ها:")
    for st, cnt in df_t['status'].value_counts().items():
        print(f"   {st:<16}: {cnt:>5} ({cnt/len(df_t)*100:.1f}%)")

    # ── Yearly breakdown ──
    print("-" * 72)
    print(" عملکرد سالانه:")
    df_t['year'] = df_t['exit_ts'].dt.year
    yearly = df_t.groupby('year').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wr=('pnl', lambda x: (x > 0).mean() * 100),
    )
    for yr, row in yearly.iterrows():
        print(f"   {yr}: {row['trades']:>4}T | "
              f"WR:{row['wr']:5.1f}% | "
              f"Net:${row['pnl']:>8,.2f}")

    # ── Target: 2% monthly check ──
    print("-" * 72)
    target_monthly = GlobalConfig.initial_balance * 0.02
    months_above_target = int((monthly >= target_monthly).sum())
    print(f" 🎯 هدف ماهانه ۲٪ (${target_monthly:.0f}):")
    print(f"    ماه‌های بالای هدف: {months_above_target} از {total_m} "
          f"({months_above_target/total_m*100:.0f}%)")
    print(f"    میانگین ماهانه واقعی: "
          f"${results['total_withdrawn']/max(1,total_m):.2f}")
    print(f"    نسبت به هدف: "
          f"{(results['total_withdrawn']/max(1,total_m))/target_monthly*100:.0f}%")

    print("═" * 72)
    bk = results['total_withdrawn']
    print(f" v8→$7,151 | v9→$0 | v9b→$938 | v10→${bk:,.0f}")
    print("═" * 72)

    # ── Save equity curve ──
    try:
        ec = pd.DataFrame(results['equity_curve'])
        ec.to_csv('equity_curve_v10.csv', index=False)
        print(f"\n  📊 Equity curve saved: equity_curve_v10.csv ({len(ec)} days)")
    except Exception as e:
        print(f"\n  ⚠ Could not save equity curve: {e}")

    # ── Save monthly detail ──
    try:
        monthly.to_csv('monthly_pnl_v10.csv', header=['pnl'])
        print(f"  📊 Monthly PnL saved: monthly_pnl_v10.csv")
    except Exception as e:
        print(f"  ⚠ Could not save monthly PnL: {e}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CorrArb v10 — Full Rebuild with Edge Improvements    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    pair_data = load_pair_data()
    active_cfgs = PairConfig.active_pairs()

    print("\n  Computing signals (reversion confirmation)...")
    pair_signals = {}
    for n in pair_data:
        if n in active_cfgs:
            sig, z, atr = compute_signals(n, pair_data[n], active_cfgs[n])
            pair_signals[n] = (sig, z, atr)

    # فقط pairهای فعال
    active_data = {n: pair_data[n] for n in pair_signals}

    results = run_backtest(active_data, pair_signals)
    print_report(results)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  ✅ Done in {elapsed:.2f}s")
