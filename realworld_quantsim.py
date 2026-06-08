"""
CorrArb v17 — Dual Timeframe (15M + 1H)
==========================================
تغییرات نسبت به v16:

1. Dual TF engine:
   - 15M: همان v16 (z_period=96, vr=200, sl=30pip)
   - 1H:  پارامترها داینامیک کالیبره شده (z_period=24, vr=50, sl=48pip)

2. Cross-TF Validation (مهم‌ترین بخش):
   - 15M signal فقط اگر 1H z هم‌جهت باشد تأیید می‌شود
   - 1H signal مستقل می‌تواند trade بگیرد (سیگنال قوی‌تر)
   - این مانع از trade در ضد-trend TF بالاتر می‌شود

3. اصلاح blow 2012:
   - بعد از هر blow: cooldown 14 روز (از 7)
   - max_total_dd از 8% به 7% (سخت‌گیرتر)
   - بعد از 2 blow پشت سر هم در یک سال: risk_mult = 0.25

4. کالیبراسیون 1H:
   - z_fast_period: 96 → 24 (÷4)
   - vr_period:     200 → 50 (÷4)
   - atr_ma_period: 96 → 24 (÷4)
   - corr_period:   96 → 24 (÷4)
   - sl_pips:       ×1.6 (نوسان بیشتر در 1H)
   - tp_pips:       ×1.6
   - risk_pct:      ×1.2 (سیگنال کمتر → lot بیشتر)
   - warmup:        500 → 125 (÷4)
   - bars_per_day:  96 → 24 (برای swap)

5. بقیه تغییرات v16 حفظ شد:
   - GBP exposure limit
   - spread stress + swap واقع‌بینانه
   - Risk Control (dd_levels)
   - max_trades_day = 2
"""

import pandas as pd
import numpy as np
import glob, zipfile, warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
# TIMEFRAME CONFIGS
# ═══════════════════════════════════════════════════════

TF_15M = '15min'
TF_1H  = '1h'

TF_RATIO = 4   # 1H = 4 × 15M

def make_tf_config(base_tf='15min'):
    """پارامترهای داینامیک بر اساس TF"""
    r = TF_RATIO if base_tf == '1h' else 1
    return {
        'tf':              base_tf,
        'bars_per_day':    24 if base_tf == '1h' else 96,
        'z_fast_period':   max(24, 96 // r),
        'vr_period':       max(50, 200 // r),
        'atr_period':      14,
        'atr_ma_period':   max(24, 96 // r),
        'corr_period':     max(24, 96 // r),
        'warmup':          max(125, 500 // r),
        'risk_mult_tf':    1.2 if base_tf == '1h' else 1.0,
        'sl_mult':         1.6 if base_tf == '1h' else 1.0,
        'swap_min_nights': 2 if base_tf == '1h' else 4,
    }


# ═══════════════════════════════════════════════════════
# GLOBAL CONFIG
# ═══════════════════════════════════════════════════════
class GlobalConfig:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.04
    max_daily_loss_pct = 0.04
    max_total_dd_pct   = 0.07       # v17: 8% → 7%
    commission_per_lot = 7.0
    lot_size           = 100_000
    max_lot            = 3.0
    min_lot            = 0.01
    consec_loss_n      = 2
    risk_reduce        = 0.5
    cooldown_days      = 14         # v17: 7 → 14
    monthly_loss_threshold = -200.0

    hour_start   = 2
    hour_end     = 19
    bad_hours    = {4, 5, 7, 9, 13, 18, 20}
    trade_days   = [0, 1, 2, 3, 4]
    max_trades_day = 2

    z_entry            = 2.2
    z_exit_partial     = 0.50
    z_exit_full        = 0.0
    z_stop_margin      = 3.8
    min_net_profit_usd = 20.0
    partial_ratio      = 0.75
    atr_max_mult       = 2.8
    atr_min_mult       = 0.5
    atr_blowout_mult   = 3.2
    vr_k               = 4

    stress_hours    = {0, 7, 12}
    slippage_normal = 0.7
    slippage_stress = 1.5
    swap_min_nights = 4

    # Cross-TF validation
    cross_tf_confirm  = True        # v17: فعال
    h1_z_confirm_thr  = 0.5        # اگر 1H z بیشتر از این بود → 15M long confirm نیست
    h1_z_reject_thr   = 1.5        # اگر 1H z بیشتر از این بود → 15M long را رد کن

    # Risk Control
    dd_levels = [
        (0.03, 0.80),
        (0.05, 0.60),
        (0.07, 0.35),
    ]
    rolling_pf_n    = 40
    rolling_pf_bad  = 0.85
    rolling_pf_mult = 0.75

    # v17: اگر 2 blow در یک سال → risk کمتر
    annual_blow_limit = 2
    annual_blow_mult  = 0.25


PAIR_CFG = {
    'AUDNZD': {
        'leg1': 'AUDUSD', 'leg2': 'NZDUSD',
        'formula': 'div', 'quote': 'leg2',
        'spread_pip': 2.5, 'spread_pip_stress': 5.0,
        'swap_pips_day': -0.15,
        'pip_size': 0.0001,
        'vr_max': 0.75, 'corr_min': 0.82,
        'risk_pct': 0.015, 'risk_min': 0.005,
        'sl_pips': 30.0, 'tp_pips': 90.0,
    },
    'AUDCAD': {
        'leg1': 'AUDUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote': 'inv_leg2',
        'spread_pip': 2.5, 'spread_pip_stress': 6.0,
        'swap_pips_day': -0.25,
        'pip_size': 0.0001,
        'vr_max': 0.85, 'corr_min': 0.57,
        'risk_pct': 0.010, 'risk_min': 0.004,
        'sl_pips': 25.0, 'tp_pips': 75.0,
    },
    'GBPCAD': {
        'leg1': 'GBPUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote': 'inv_leg2',
        'spread_pip': 4.0, 'spread_pip_stress': 9.0,
        'swap_pips_day': -0.40,
        'pip_size': 0.0001,
        'vr_max': 0.88, 'corr_min': 0.47,
        'risk_pct': 0.008, 'risk_min': 0.003,
        'sl_pips': 28.0, 'tp_pips': 84.0,
        'gbp_pair': True,
    },
    'GBPCHF': {
        'leg1': 'GBPUSD', 'leg2': 'USDCHF',
        'formula': 'div', 'quote': 'inv_leg2',
        'spread_pip': 3.0, 'spread_pip_stress': 8.0,
        'swap_pips_day': -0.30,
        'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.42,
        'risk_pct': 0.007, 'risk_min': 0.003,
        'sl_pips': 35.0, 'tp_pips': 105.0,
        'gbp_pair': True,
    },
}


# ═══════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════
def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths: return None
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p) as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if not csv_name: continue
                with z.open(csv_name) as f:
                    frames.append(pd.read_csv(f, sep=';', header=None,
                        names=['ts','o','h','l','c','v']))
        except Exception: continue
    if not frames: return None
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def load_raw_csv(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths: return None
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p, sep=';', header=None,
                names=['ts','o','h','l','c','v']))
        except Exception: continue
    if not frames: return None
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def load_instrument(name):
    for pat in [f'data/HISTDATA*{name}*.zip', f'data/*{name}*.zip',
                f'data/HISTDATA*{name}*.csv', f'data/*{name}*.csv']:
        raw = (load_raw_zip(pat) if '.zip' in pat else load_raw_csv(pat))
        if raw is not None and len(raw) > 1000:
            return raw
    return None


def resample_pair(raw, tf, sfx):
    rule = tf
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample(rule).first(),
        f'h_{sfx}': raw['h'].resample(rule).max(),
        f'l_{sfx}': raw['l'].resample(rule).min(),
        f'c_{sfx}': raw['c'].resample(rule).last(),
    }).dropna()


def build_pair_tf(pcfg, tf):
    r1 = load_instrument(pcfg['leg1'])
    r2 = load_instrument(pcfg['leg2'])
    if r1 is None or r2 is None: return None
    d1 = resample_pair(r1, tf, 'leg1')
    d2 = resample_pair(r2, tf, 'leg2')
    m  = d1.join(d2, how='inner').dropna()
    if pcfg['formula'] == 'div':
        m['c_spread'] = m['c_leg1'] / m['c_leg2']
        m['o_spread'] = m['o_leg1'] / m['o_leg2']
        m['h_spread'] = m['h_leg1'] / m['l_leg2']
        m['l_spread'] = m['l_leg1'] / m['h_leg2']
    else:
        m['c_spread'] = m['c_leg1'] * m['c_leg2']
        m['o_spread'] = m['o_leg1'] * m['o_leg2']
        m['h_spread'] = m['h_leg1'] * m['h_leg2']
        m['l_spread'] = m['l_leg1'] * m['l_leg2']
    if pcfg['quote'] == 'leg2':
        m['quote_rate'] = m['c_leg2']
    elif pcfg['quote'] == 'inv_leg2':
        m['quote_rate'] = 1.0 / m['c_leg2'].replace(0, np.nan)
    else:
        m['quote_rate'] = 1.0
    return m[m.index.weekday < 5].dropna().copy()


# ═══════════════════════════════════════════════════════
# SIGNALS
# ═══════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_vr(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    v1 = r1.rolling(window).var()
    vk = rk.rolling(window).var()
    return vk / (k * v1.replace(0, np.nan))


def compute_signals(df, pcfg, tfc):
    """
    tfc: dict از make_tf_config
    """
    G = GlobalConfig
    log_r  = np.log(df['c_spread'].replace(0, np.nan))
    z_mean = log_r.rolling(tfc['z_fast_period']).mean()
    z_std  = log_r.rolling(tfc['z_fast_period']).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    corr    = (df['c_leg1'].pct_change()
               .rolling(tfc['corr_period'])
               .corr(df['c_leg2'].pct_change()))
    corr_ok = corr.abs() > pcfg['corr_min']

    vr_slow   = calc_vr(log_r, G.vr_k, tfc['vr_period'])
    regime_ok = vr_slow < pcfg['vr_max']

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], tfc['atr_period'])
    atr_ma = atr.rolling(tfc['atr_ma_period']).mean()
    vol_ok = (
        (atr > atr_ma * G.atr_min_mult) &
        (atr < atr_ma * G.atr_max_mult) &
        (atr < atr_ma * G.atr_blowout_mult)
    )

    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = (
        hour.between(G.hour_start, G.hour_end) &
        (~hour.isin(G.bad_hours)) &
        dow.isin(G.trade_days)
    )

    sig  = pd.Series(0, index=df.index)
    cond = vol_ok & time_ok & corr_ok & regime_ok
    sig[(z < -G.z_entry) & cond] =  1
    sig[(z >  G.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)
    return sig, z


# ═══════════════════════════════════════════════════════
# CROSS-TF VALIDATION
# ═══════════════════════════════════════════════════════
def cross_tf_ok(sig_15m, z_1h, direction):
    """
    آیا 15M signal با وضعیت 1H تأیید می‌شود؟
    direction: +1 (long) یا -1 (short)
    """
    G = GlobalConfig
    if not G.cross_tf_confirm or np.isnan(z_1h):
        return True   # اگر 1H موجود نیست، pass کن

    if direction == 1:   # long: می‌خواهیم 1H هم over-extended نباشد
        if z_1h >= G.h1_z_reject_thr:
            return False   # 1H خیلی بالاست — رد کن
        return True
    else:                # short
        if z_1h <= -G.h1_z_reject_thr:
            return False
        return True


# ═══════════════════════════════════════════════════════
# RISK MULTIPLIER
# ═══════════════════════════════════════════════════════
def get_risk_mult(equity, peak, pnl_hist, month_pnl, month_threshold,
                  annual_blows=0):
    G = GlobalConfig
    mult = 1.0
    if peak > 0:
        dd = (peak - equity) / peak
        for dd_thresh, dd_mult in G.dd_levels:
            if dd >= dd_thresh:
                mult = min(mult, dd_mult)
    if len(pnl_hist) >= G.rolling_pf_n // 2:
        recent = pnl_hist[-G.rolling_pf_n:]
        wins   = sum(p for p in recent if p > 0)
        losses = abs(sum(p for p in recent if p < 0))
        rpf    = wins / losses if losses > 0 else 1.5
        if rpf < G.rolling_pf_bad:
            mult *= G.rolling_pf_mult
    if month_pnl < month_threshold:
        mult *= 0.50
    # v17: annual blow penalty
    if annual_blows >= G.annual_blow_limit:
        mult = min(mult, G.annual_blow_mult)
    return max(mult, 0.15)


# ═══════════════════════════════════════════════════════
# PNL
# ═══════════════════════════════════════════════════════
def pnl_calc(d, entry, xp, lot, qr, pip, swap_pips=0.0,
             bars_held=0, bars_per_day=96, swap_min_nights=4):
    G = GlobalConfig
    gross = d * (xp - entry) * lot * G.lot_size * qr
    nights = bars_held // bars_per_day
    swap_cost = 0.0
    if nights > swap_min_nights:
        swap_cost = abs(swap_pips) * nights * lot * G.lot_size * qr * pip
    return gross - G.commission_per_lot * lot - swap_cost


# ═══════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════
def new_acc(ts):
    G = GlobalConfig
    return {
        'equity': G.initial_balance, 'start_ts': ts, 'trades': [],
        'blown': False, 'blown_rsn': '', 'peak': G.initial_balance,
        'consec_loss': 0,
    }


def run_portfolio(pair_data_15m, pair_data_1h):
    """
    pair_data_15m: dict {name: (df, sig, z, pcfg, tfc)}
    pair_data_1h:  dict {name: (df, sig, z, pcfg, tfc)}  — برای cross-TF z
    """
    G = GlobalConfig

    # index مشترک از 15M (primary)
    cidx = None
    for name, (df, sig, z, pcfg, tfc) in pair_data_15m.items():
        cidx = df.index if cidx is None else cidx.intersection(df.index)
    cidx = cidx.sort_values()
    N = len(cidx)

    # warmup از بزرگ‌ترین TF
    warmup = max(tfc['warmup'] for _, (_, _, _, _, tfc) in pair_data_15m.items())

    # آرایه‌های 15M
    pa15 = {}
    for name, (df, sig, z, pcfg, tfc) in pair_data_15m.items():
        df_r = df.reindex(cidx).ffill()
        pa15[name] = {
            'o':   df_r['o_spread'].values.astype(float),
            'c':   df_r['c_spread'].values.astype(float),
            'qr':  df_r['quote_rate'].values.astype(float),
            'sig': sig.reindex(cidx).fillna(0).values.astype(int),
            'z':   z.reindex(cidx).fillna(np.nan).values.astype(float),
            'cfg': pcfg, 'tfc': tfc,
        }

    # z از 1H — resample به 15M index با ffill
    z1h = {}
    for name, (df1h, sig1h, z_1h, pcfg, tfc) in pair_data_1h.items():
        z_resampled = z_1h.reindex(cidx, method='ffill')
        z1h[name]   = z_resampled.values.astype(float)
        # سیگنال‌های 1H هم در index 15M
        sig1h_rs    = sig1h.reindex(cidx, method='ffill').fillna(0)
        pa15[name]['sig_1h'] = sig1h_rs.values.astype(int)
        pa15[name]['z_1h']   = z_resampled.values.astype(float)

    FLOOR  = G.initial_balance * (1 - G.max_total_dd_pct)
    TARGET = G.initial_balance * (1 + G.profit_target_pct)

    acc          = new_acc(cidx[warmup])
    withdrawn    = 0.0
    acc_num      = 1
    day_eq       = G.initial_balance
    month_eq     = G.initial_balance
    cooldown_til = None
    all_trades   = []
    acc_logs     = []
    eq_curve     = []
    pnl_hist     = []
    annual_blows = {}   # {year: count}

    positions  = {n: None for n in pair_data_15m}
    day_trades = {n: 0    for n in pair_data_15m}
    pending    = {n: 0    for n in pair_data_15m}   # 0=none, 1=15M, 2=1H
    prev_date  = None
    prev_month = None

    print(f"  ▶ Portfolio: {list(pair_data_15m.keys())}")
    print(f"    Bars:{N:,} | {cidx[0].date()} → {cidx[-1].date()}")
    print(f"    Warmup: {warmup} bars")

    for bar in range(warmup, N):
        ts        = cidx[bar]
        cur_date  = ts.date()
        cur_month = (ts.year, ts.month)
        cur_year  = ts.year
        is_stress = ts.hour in G.stress_hours

        if cur_date != prev_date:
            day_eq = acc['equity']
            for n in pair_data_15m: day_trades[n] = 0
            eq_curve.append({
                'date': str(cur_date),
                'equity': acc['equity'] + withdrawn,
                'account_eq': acc['equity']
            })
            prev_date = cur_date

        if cur_month != prev_month:
            month_eq   = acc['equity']
            prev_month = cur_month

        if acc['equity'] > acc['peak']:
            acc['peak'] = acc['equity']

        in_cd = cooldown_til is not None and ts < cooldown_til

        if acc['blown']:
            acc_logs.append({
                'reason': acc['blown_rsn'],
                'pnl': acc['equity'] - G.initial_balance,
                'days': (ts - acc['start_ts']).days,
                'n_trades': len(acc['trades']),
                'end_ts': ts,
            })
            print(f"    💥 #{acc_num:>3} | {ts.date()} | "
                  f"${acc['equity']:.2f} | {acc['blown_rsn']}")
            # annual blow counter
            annual_blows[cur_year] = annual_blows.get(cur_year, 0) + 1
            cooldown_til = ts + pd.Timedelta(days=G.cooldown_days)
            acc_num += 1
            acc = new_acc(ts)
            day_eq = month_eq = acc['equity']
            for n in pair_data_15m:
                day_trades[n] = 0; pending[n] = 0; positions[n] = None
            pnl_hist = []
            prev_date = cur_date; prev_month = cur_month
            continue

        if in_cd:
            continue

        month_pnl = acc['equity'] - month_eq
        yr_blows  = annual_blows.get(cur_year, 0)
        risk_mult = get_risk_mult(
            acc['equity'], acc['peak'], pnl_hist,
            month_pnl, G.monthly_loss_threshold, yr_blows)

        # GBP exposure limit
        gbp_open = sum(
            1 for n in pair_data_15m
            if positions[n] is not None and pa15[n]['cfg'].get('gbp_pair', False)
        )

        # سیگنال‌ها — 15M اول، سپس 1H اگر 15M نداشت
        for name in pair_data_15m:
            a    = pa15[name]
            sig15 = int(a['sig'][bar])
            sig1h = int(a.get('sig_1h', np.zeros(N))[bar])
            z_1h_val = float(a.get('z_1h', np.full(N, np.nan))[bar])

            # انتخاب سیگنال
            chosen_sig = 0
            chosen_tf  = None
            if sig15 != 0:
                # cross-TF validation
                if cross_tf_ok(sig15, z_1h_val, sig15):
                    chosen_sig = sig15
                    chosen_tf  = '15M'
            elif sig1h != 0:
                chosen_sig = sig1h
                chosen_tf  = '1H'

            if chosen_sig != 0 and positions[name] is None:
                pending[name] = chosen_sig

        # open
        for name in pair_data_15m:
            a    = pa15[name]
            pcfg = a['cfg']
            tfc  = a['tfc']
            if (pending[name] != 0 and positions[name] is None
                    and day_trades[name] < G.max_trades_day):

                if pcfg.get('gbp_pair', False) and gbp_open >= 1:
                    pending[name] = 0
                    continue

                sv  = pending[name]
                pip = pcfg['pip_size']
                qr  = a['qr'][bar]
                sp  = pcfg['spread_pip_stress'] if is_stress else pcfg['spread_pip']
                sl  = G.slippage_stress if is_stress else G.slippage_normal

                # SL/TP با تنظیم TF (اگر 1H signal بود، sl بزرگتر)
                sl_pips = pcfg['sl_pips'] * tfc['sl_mult']
                tp_pips = pcfg['tp_pips'] * tfc['sl_mult']

                risk = pcfg['risk_pct'] * risk_mult * tfc['risk_mult_tf']
                if acc['consec_loss'] >= G.consec_loss_n:
                    risk = max(risk * G.risk_reduce, pcfg['risk_min'])

                pv  = pip * G.lot_size * qr
                if pv <= 0: pv = 10.0
                lot = round(float(np.clip(
                    acc['equity'] * risk / (sl_pips * pv),
                    G.min_lot, G.max_lot)), 2)
                ep = a['o'][bar] + sv * (sl + sp / 2) * pip
                positions[name] = {
                    'dir': sv, 'lot': lot, 'lot_rem': lot,
                    'partial_done': False, 'entry': ep,
                    'sl': ep - sv * sl_pips * pip,
                    'tp': ep + sv * tp_pips * pip,
                    'entry_ts': ts, 'entry_bar': bar, 'pip': pip,
                    'swap_pips': pcfg.get('swap_pips_day', 0.0),
                    'bars_per_day': tfc['bars_per_day'],
                    'swap_min_nights': tfc['swap_min_nights'],
                }
                day_trades[name] += 1
                if pcfg.get('gbp_pair', False):
                    gbp_open += 1
            pending[name] = 0

        # float
        total_float = sum(
            (p['dir'] * (pa15[n]['c'][bar] - p['entry'])
             * p['lot_rem'] * G.lot_size * pa15[n]['qr'][bar]
             - G.commission_per_lot * p['lot_rem'])
            for n in pair_data_15m if (p := positions[n]) is not None
        )
        cur_eq    = acc['equity'] + total_float
        daily_lim = day_eq * (1 - G.max_daily_loss_pct)

        if cur_eq <= daily_lim or cur_eq <= FLOOR:
            rsn = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            acc['blown'] = True; acc['blown_rsn'] = rsn
            for name in pair_data_15m:
                pos = positions[name]
                if pos is None: continue
                bh = bar - pos['entry_bar']
                pnl = pnl_calc(pos['dir'], pos['entry'], pa15[name]['c'][bar],
                               pos['lot_rem'], pa15[name]['qr'][bar], pos['pip'],
                               pos['swap_pips'], bh,
                               pos['bars_per_day'], pos['swap_min_nights'])
                acc['equity'] += pnl
                all_trades.append({'pair': name, 'pnl': pnl,
                                   'status': 'BLOWN', 'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                positions[name] = None
            continue

        # exit
        for name in pair_data_15m:
            pos = positions[name]
            if pos is None: continue
            a   = pa15[name]; pcfg = a['cfg']
            cp  = a['c'][bar]; d = pos['dir']
            ep  = pos['entry']; zn = a['z'][bar]
            lr  = pos['lot_rem']; qr = a['qr'][bar]; pip = pos['pip']
            bh  = bar - pos['entry_bar']
            bpd = pos['bars_per_day']
            smn = pos['swap_min_nights']

            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -G.z_exit_partial) or
                        (d == -1 and zn <= G.z_exit_partial)):
                    p_lot = round(lr * G.partial_ratio, 2)
                    if p_lot >= G.min_lot:
                        p_pnl = pnl_calc(d, ep, cp, p_lot, qr, pip,
                                         pos['swap_pips'], bh, bpd, smn)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            pnl_hist.append(p_pnl)
                            all_trades.append({'pair': name, 'pnl': p_pnl,
                                               'status': 'Partial', 'exit_ts': ts})
                            acc['trades'].append(all_trades[-1])
                            pos['lot_rem'] = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl'] = pos['entry']
                            lr = pos['lot_rem']
                            if lr < G.min_lot:
                                positions[name] = None; continue

            if positions[name] is None: continue
            lr      = pos['lot_rem']
            pnl_now = pnl_calc(d, ep, cp, lr, qr, pip,
                               pos['swap_pips'], bh, bpd, smn)

            hit_zs = (not np.isnan(zn) and
                      ((d == 1 and zn <= -G.z_stop_margin) or
                       (d == -1 and zn >= G.z_stop_margin)))
            hit_ze = (not np.isnan(zn) and
                      ((d == 1 and zn >= -G.z_exit_full) or
                       (d == -1 and zn <= G.z_exit_full)))
            if hit_ze and pnl_now < G.min_net_profit_usd and not pos['partial_done']:
                hit_ze = False
            hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
            hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])

            if hit_ze or hit_zs or hit_sl or hit_tp:
                xp   = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                st   = ('SL' if hit_sl else 'TP' if hit_tp else
                        'Z-Stop' if hit_zs else 'Z-Exit')
                fpnl = pnl_calc(d, ep, xp, lr, qr, pip,
                                pos['swap_pips'], bh, bpd, smn)
                acc['equity'] += fpnl
                pnl_hist.append(fpnl)
                all_trades.append({'pair': name, 'pnl': fpnl,
                                   'status': st, 'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                positions[name] = None
                if fpnl > 0: acc['consec_loss'] = 0
                else:        acc['consec_loss'] += 1

        # target
        if acc['equity'] >= TARGET and all(positions[n] is None for n in pair_data_15m):
            w  = acc['equity'] - G.initial_balance
            withdrawn += w
            dt = (ts - acc['start_ts']).days
            nt = len(acc['trades'])
            acc_logs.append({'reason': 'TARGET_HIT', 'pnl': w,
                             'days': dt, 'n_trades': nt, 'end_ts': ts})
            print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:.2f} | "
                  f"Bank:${withdrawn:.2f} | {dt}d | {nt}T")
            acc_num += 1
            acc = new_acc(ts)
            day_eq = month_eq = acc['equity']
            pnl_hist = []
            for n in pair_data_15m: day_trades[n] = 0; pending[n] = 0
            prev_date = cur_date; prev_month = cur_month
            continue

    return {
        'all_trades': all_trades, 'account_logs': acc_logs,
        'withdrawn': withdrawn, 'final_equity': acc['equity'],
        'common_idx': cidx, 'eq_curve': eq_curve,
        'warmup': warmup,
    }


# ═══════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════
def print_report(res, title):
    if not res['all_trades']:
        print("❌ No trades"); return

    df = pd.DataFrame(res['all_trades'])
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])
    df['month']   = df['exit_ts'].dt.to_period('M')
    df['year']    = df['exit_ts'].dt.year

    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    wr     = len(wins) / len(df) * 100
    pf     = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else 99.0

    ci = res['common_idx']
    wm = res['warmup']
    all_months = pd.period_range(
        start=ci[wm].to_period('M'),
        end=ci[-1].to_period('M'), freq='M')
    monthly = df.groupby('month')['pnl'].sum().reindex(all_months, fill_value=0.0)

    pos_m = int((monthly > 0).sum())
    neg_m = int((monthly < 0).sum())
    tot_m = len(monthly)
    ms = cur = 0
    for v in monthly:
        cur = cur + 1 if v < 0 else 0
        ms  = max(ms, cur)

    logs   = pd.DataFrame(res['account_logs']) if res['account_logs'] else pd.DataFrame()
    n_pass = int((logs['reason'] == 'TARGET_HIT').sum()) if len(logs) else 0
    n_blow = int((logs['reason'] != 'TARGET_HIT').sum()) if len(logs) else 0
    neg_yr = int((df.groupby('year')['pnl'].sum() < 0).sum())

    m_std  = monthly.std()
    sharpe = (monthly.mean() / m_std * np.sqrt(12)) if m_std > 0 else 0

    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)
    print(f"  Trades:{len(df):,}  WR:{wr:.1f}%  PF:{pf:.3f}")
    print(f"  AvgWin:${wins['pnl'].mean():.2f}  AvgLoss:${losses['pnl'].mean():.2f}")
    print(f"  Net:${df['pnl'].sum():,.2f}  Banked:${res['withdrawn']:,.2f}"
          f"  Eq:${res['final_equity']:,.2f}")
    print(f"  Pass:{n_pass}  Blown:{n_blow}  NegYr:{neg_yr}")
    print(f"  +Mo:{pos_m}/{tot_m}({pos_m/tot_m*100:.0f}%)  -Mo:{neg_m}  Streak:{ms}mo")
    print(f"  MonthAvg:${monthly.mean():.2f}  Median:${monthly.median():.2f}"
          f"  Sharpe:{sharpe:.2f}")
    print(f"  Best:${monthly.max():,.2f}  Worst:${monthly.min():,.2f}")
    print("-" * 70)
    if 'pair' in df.columns:
        print("  By Pair:")
        for pair, g in df.groupby('pair'):
            w2 = g[g['pnl'] > 0]; l2 = g[g['pnl'] < 0]
            ppf = w2['pnl'].sum() / abs(l2['pnl'].sum()) if len(l2) else 99.0
            print(f"    {pair}: {len(g)}T  WR:{len(w2)/len(g)*100:.0f}%"
                  f"  PF:{ppf:.2f}  Net:${g['pnl'].sum():,.0f}")
    print("-" * 70)
    print("  Yearly:")
    for yr, g in df.groupby('year'):
        w2 = g[g['pnl'] > 0]; l2 = g[g['pnl'] < 0]
        ypf  = w2['pnl'].sum() / abs(l2['pnl'].sum()) if len(l2) else 99.0
        mark = '✅' if g['pnl'].sum() >= 0 else '❌'
        print(f"    {mark} {yr}:{len(g):>4}T  WR:{len(w2)/len(g)*100:5.1f}%"
              f"  PF:{ypf:.2f}  ${g['pnl'].sum():>+8,.2f}")
    print("-" * 70)
    target_mo = GlobalConfig.initial_balance * 0.02
    above = int((monthly >= target_mo).sum())
    print(f"  🎯 هدف $100/ماه: {above}/{tot_m} ({above/tot_m*100:.0f}%)")
    print(f"  📊 میانگین: ${monthly.mean():.2f} → {monthly.mean()/target_mo*100:.0f}% از هدف")
    print("═" * 70)

    monthly.to_csv('monthly_v17.csv', header=['pnl'])
    pd.DataFrame(res['eq_curve']).to_csv('equity_v17.csv', index=False)
    print("  📊 monthly_v17.csv + equity_v17.csv saved")

    print("\n  📈 تکامل نسخه‌ها:")
    rows = [
        ("v13  بدون RiskCtrl",  91, 59, 3, 10, "—"),
        ("v14b با RiskCtrl",    68, 56, 3,  2, "0.94"),
        ("v16  Sweet Spot",     32, 54, 3,  7, "0.61"),
        (f"v17  Dual TF",
         round(monthly.mean()), round(pos_m / tot_m * 100),
         neg_yr, n_blow, f"{sharpe:.2f}"),
    ]
    print(f"  {'Version':<24} {'MonAvg':>8} {'+Mo%':>6} {'NegYr':>7}"
          f" {'Blow':>6} {'Sharpe':>7}")
    print("  " + "─" * 62)
    for v, ma, pm, ny, bl, sh in rows:
        print(f"  {v:<24} ${ma:>6}   {pm:>4}%   {ny:>5}   {bl:>5}   {sh:>6}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CorrArb v17 — Dual Timeframe (15M + 1H)                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  تغییرات v17:")
    print("  • 1H TF اضافه شد (پارامترها داینامیک کالیبره)")
    print("  • Cross-TF validation: 15M فقط اگر 1H تأیید کند")
    print("  • cooldown: 7→14 روز | max_dd: 8%→7%")
    print("  • Annual blow penalty: ≥2 blow در سال → risk×0.25")
    print()

    tfc_15m = make_tf_config('15min')
    tfc_1h  = make_tf_config('1h')

    pair_data_15m = {}
    pair_data_1h  = {}

    for name, pcfg in PAIR_CFG.items():
        print(f"  Loading {name}...")
        df15 = build_pair_tf(pcfg, '15min')
        df1h = build_pair_tf(pcfg, '1h')
        if df15 is None or df1h is None:
            print(f"  ❌ {name}: not found"); continue
        sig15, z15 = compute_signals(df15, pcfg, tfc_15m)
        sig1h, z1h = compute_signals(df1h, pcfg, tfc_1h)
        n15 = int((sig15 != 0).sum())
        n1h = int((sig1h != 0).sum())
        print(f"  ✅ {name}: 15M={len(df15):,}bars/{n15:,}sig  "
              f"1H={len(df1h):,}bars/{n1h:,}sig")
        pair_data_15m[name] = (df15, sig15, z15, pcfg, tfc_15m)
        pair_data_1h[name]  = (df1h, sig1h, z1h, pcfg, tfc_1h)

    print()
    res = run_portfolio(pair_data_15m, pair_data_1h)
    print_report(res, "v17 — Dual TF (15M+1H) Quad Portfolio")
    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
