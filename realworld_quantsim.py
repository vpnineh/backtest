"""
CorrArb v15 — Realistic + Robust
===================================
اصلاحات نسبت به v13/v14b:

1. Spread واقع‌بینانه:
   - spread_stress_mult در ساعات پرنوسان (open/close بازارها)
   - spread استاتیک در ساعات آرام

2. Swap/Rollover cost روزانه

3. Regime filter قوی‌تر:
   - VR دو بازه‌ای (fast + slow)
   - ATR blow-out filter (اگر ATR خیلی بالا رفت → stop)

4. GBP exposure limit:
   - حداکثر یک GBP pair هم‌زمان باز

5. Risk parameters محافظه‌کارانه‌تر:
   - z_entry = 2.3 (از 2.1)
   - max_daily_loss = 4% (از 5%)
   - max_total_dd = 8% (از 10%)
   - max_trades_day = 1 (از 2)
   - min_net_profit = 25 (از 15)

6. Risk Control از v14b حفظ شده (dd_levels)

7. Rolling PF filter بهبودیافته
"""

import pandas as pd
import numpy as np
import glob, zipfile, warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
class GlobalConfig:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.04       # v15: از 0.05 به 0.04 (pass سریع‌تر)
    max_daily_loss_pct = 0.04       # v15: از 0.05 به 0.04 (فاند strict)
    max_total_dd_pct   = 0.08       # v15: از 0.10 به 0.08
    commission_per_lot = 7.0
    lot_size           = 100_000
    max_lot            = 3.0
    min_lot            = 0.01
    warmup             = 500
    consec_loss_n      = 2
    risk_reduce        = 0.5
    cooldown_days      = 7          # v15: از 10 به 7
    monthly_loss_threshold = -200.0 # v15: از -250 به -200

    hour_start   = 3                # v15: از 2 به 3 (اول لندن کمی صبر)
    hour_end     = 18               # v15: از 19 به 18
    bad_hours    = {4, 5, 7, 9, 12, 13, 17, 18, 20}  # v15: ساعت‌های NFP/FOMC اضافه
    trade_days   = [0, 1, 2, 3, 4]

    max_trades_day     = 1          # v15: از 2 به 1
    z_fast_period      = 96
    z_entry            = 2.3        # v15: از 2.1 به 2.3 (سیگنال باکیفیت‌تر)
    z_exit_partial     = 0.50
    z_exit_full        = 0.0
    z_stop_margin      = 3.5        # v15: از 4.0 به 3.5 (زودتر cut)
    min_net_profit_usd = 25.0       # v15: از 15 به 25
    partial_ratio      = 0.75
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 2.5        # v15: از 3.0 به 2.5 (ریژیم volatile بسته‌تر)
    atr_min_mult       = 0.5
    atr_blowout_mult   = 2.8        # v15: جدید — اگر ATR از MA بیشتر از این بود → no trade
    vr_period          = 200
    vr_period_fast     = 60         # v15: جدید — VR سریع
    vr_k               = 4
    corr_period        = 96

    # ── Slippage واقع‌بینانه‌تر ──
    slippage_pips_normal = 0.8      # v15: از 0.5 به 0.8
    slippage_pips_stress = 2.5      # v15: جدید — در ساعات پرنوسان

    # ── Stress hours: باز/بسته شدن بازارها و اعلام‌های مهم ──
    stress_hours = {0, 1, 8, 13, 14, 21, 22, 23}  # Sydney open, London open, NY open/close

    # ── Risk Control (از v14b) ──
    dd_levels = [
        (0.03, 0.80),   # 3% DD → risk * 0.80
        (0.05, 0.60),   # 5% DD → risk * 0.60
        (0.07, 0.35),   # 7% DD → risk * 0.35
    ]
    rolling_pf_n    = 40
    rolling_pf_bad  = 0.85
    rolling_pf_mult = 0.75


PAIR_CFG = {
    'AUDNZD': {
        'leg1': 'AUDUSD', 'leg2': 'NZDUSD',
        'formula': 'div', 'quote': 'leg2',
        'spread_pip': 2.5,          # آرام
        'spread_pip_stress': 6.0,   # v15: در ساعات پرنوسان
        'swap_pips_day': -0.3,      # v15: swap روزانه (تقریبی)
        'pip_size': 0.0001,
        'vr_max': 0.70,             # v15: سخت‌گیرانه‌تر (از 0.75)
        'corr_min': 0.82,           # v15: سخت‌گیرانه‌تر (از 0.80)
        'risk_pct': 0.012,          # v15: کمی کمتر (از 0.015)
        'risk_min': 0.004,
        'sl_pips': 32.0,
        'tp_pips': 96.0,            # نسبت 1:3
    },
    'AUDCAD': {
        'leg1': 'AUDUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote': 'inv_leg2',
        'spread_pip': 2.5,
        'spread_pip_stress': 7.0,
        'swap_pips_day': -0.5,
        'pip_size': 0.0001,
        'vr_max': 0.80,
        'corr_min': 0.58,
        'risk_pct': 0.009,          # v15: از 0.010
        'risk_min': 0.003,
        'sl_pips': 26.0,
        'tp_pips': 78.0,
    },
    'GBPCAD': {
        'leg1': 'GBPUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote': 'inv_leg2',
        'spread_pip': 4.0,
        'spread_pip_stress': 12.0,  # v15: GBPCAD در stress خیلی wide می‌شود
        'swap_pips_day': -0.8,
        'pip_size': 0.0001,
        'vr_max': 0.83,
        'corr_min': 0.48,
        'risk_pct': 0.007,          # v15: از 0.008
        'risk_min': 0.002,
        'sl_pips': 30.0,
        'tp_pips': 90.0,
        'gbp_pair': True,           # v15: تگ برای GBP limit
    },
    'GBPCHF': {
        'leg1': 'GBPUSD', 'leg2': 'USDCHF',
        'formula': 'div', 'quote': 'inv_leg2',
        'spread_pip': 3.0,
        'spread_pip_stress': 10.0,
        'swap_pips_day': -0.6,
        'pip_size': 0.0001,
        'vr_max': 0.85,
        'corr_min': 0.42,
        'risk_pct': 0.006,          # v15: از 0.007
        'risk_min': 0.002,
        'sl_pips': 36.0,
        'tp_pips': 108.0,
        'gbp_pair': True,
    },
}


# ═══════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════
def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p) as z:
                csv_name = next(
                    (f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if not csv_name:
                    continue
                with z.open(csv_name) as f:
                    frames.append(pd.read_csv(
                        f, sep=';', header=None,
                        names=['ts', 'o', 'h', 'l', 'c', 'v']))
        except Exception:
            continue
    if not frames:
        return None
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw


def load_raw_csv(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(
                p, sep=';', header=None,
                names=['ts', 'o', 'h', 'l', 'c', 'v']))
        except Exception:
            continue
    if not frames:
        return None
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw


def load_instrument(name):
    for pat in [
        f'data/HISTDATA*{name}*.zip',
        f'data/*{name}*.zip',
        f'data/HISTDATA*{name}*.csv',
        f'data/*{name}*.csv',
    ]:
        raw = (load_raw_zip(pat) if '.zip' in pat else load_raw_csv(pat))
        if raw is not None and len(raw) > 1000:
            return raw
    return None


def to_15min(raw, sfx):
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()


def build_pair(pcfg):
    r1 = load_instrument(pcfg['leg1'])
    r2 = load_instrument(pcfg['leg2'])
    if r1 is None or r2 is None:
        return None
    d1 = to_15min(r1, 'leg1')
    d2 = to_15min(r2, 'leg2')
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
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def calc_vr(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    v1 = r1.rolling(window).var()
    vk = rk.rolling(window).var()
    return vk / (k * v1.replace(0, np.nan))


def compute_signals(df, pcfg):
    G = GlobalConfig
    log_r  = np.log(df['c_spread'].replace(0, np.nan))
    z_mean = log_r.rolling(G.z_fast_period).mean()
    z_std  = log_r.rolling(G.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    corr    = (df['c_leg1'].pct_change()
               .rolling(G.corr_period)
               .corr(df['c_leg2'].pct_change()))
    corr_ok = corr.abs() > pcfg['corr_min']

    # v15: دو VR (slow + fast) — هر دو باید ok باشند
    vr_slow = calc_vr(log_r, G.vr_k, G.vr_period)
    vr_fast = calc_vr(log_r, G.vr_k, G.vr_period_fast)
    regime_ok = (vr_slow < pcfg['vr_max']) & (vr_fast < pcfg['vr_max'] * 1.15)

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], G.atr_period)
    atr_ma = atr.rolling(G.atr_ma_period).mean()

    # v15: blowout filter — اگر ATR خیلی بالا رفت کلاً بست
    vol_ok = (
        (atr > atr_ma * G.atr_min_mult) &
        (atr < atr_ma * G.atr_max_mult) &
        (atr < atr_ma * G.atr_blowout_mult)   # فیلتر blowout
    )

    hour   = pd.Series(df.index.hour, index=df.index)
    dow    = pd.Series(df.index.dayofweek, index=df.index)
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
    return sig, z, atr, atr_ma


# ═══════════════════════════════════════════════════════
# RISK MULTIPLIER (بهبودیافته از v14b)
# ═══════════════════════════════════════════════════════
def get_risk_mult(equity, peak, pnl_hist, month_pnl, month_threshold):
    G = GlobalConfig
    mult = 1.0

    # DD از peak
    if peak > 0:
        dd = (peak - equity) / peak
        for dd_thresh, dd_mult in G.dd_levels:
            if dd >= dd_thresh:
                mult = min(mult, dd_mult)

    # Rolling PF
    if len(pnl_hist) >= G.rolling_pf_n // 2:
        recent = pnl_hist[-G.rolling_pf_n:]
        wins   = sum(p for p in recent if p > 0)
        losses = abs(sum(p for p in recent if p < 0))
        rpf    = wins / losses if losses > 0 else 1.5
        if rpf < G.rolling_pf_bad:
            mult *= G.rolling_pf_mult

    # Monthly stress
    if month_pnl < month_threshold:
        mult *= 0.50   # v15: از 0.60 به 0.50

    return max(mult, 0.15)   # v15: از 0.20 به 0.15


# ═══════════════════════════════════════════════════════
# PNL (با swap)
# ═══════════════════════════════════════════════════════
def pnl_calc(d, entry, xp, lot, qr, pip, swap_pips=0.0, bars_held=0):
    G = GlobalConfig
    gross = d * (xp - entry) * lot * G.lot_size * qr
    # تعداد شب‌های نگه‌داشته شده (هر 96 بار = 1 روز در 15min)
    nights = bars_held // 96
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


def run_portfolio(pair_data):
    G = GlobalConfig
    cidx = None
    for name, (df, sig, z, atr, atr_ma, pcfg) in pair_data.items():
        cidx = df.index if cidx is None else cidx.intersection(df.index)
    cidx = cidx.sort_values()
    N = len(cidx)

    pa = {}
    for name, (df, sig, z, atr, atr_ma, pcfg) in pair_data.items():
        df_r = df.reindex(cidx).ffill()
        pa[name] = {
            'o':      df_r['o_spread'].values.astype(float),
            'c':      df_r['c_spread'].values.astype(float),
            'qr':     df_r['quote_rate'].values.astype(float),
            'sig':    sig.reindex(cidx).fillna(0).values.astype(int),
            'z':      z.reindex(cidx).fillna(np.nan).values.astype(float),
            'atr':    atr.reindex(cidx).ffill().values.astype(float),
            'atr_ma': atr_ma.reindex(cidx).ffill().values.astype(float),
            'cfg':    pcfg,
        }

    FLOOR  = G.initial_balance * (1 - G.max_total_dd_pct)
    TARGET = G.initial_balance * (1 + G.profit_target_pct)

    acc          = new_acc(cidx[G.warmup])
    withdrawn    = 0.0
    acc_num      = 1
    day_eq       = G.initial_balance
    month_eq     = G.initial_balance
    cooldown_til = None
    all_trades   = []
    acc_logs     = []
    eq_curve     = []
    pnl_hist     = []

    positions  = {n: None for n in pair_data}
    day_trades = {n: 0    for n in pair_data}
    pending    = {n: 0    for n in pair_data}
    prev_date  = None
    prev_month = None

    print(f"  ▶ Portfolio: {list(pair_data.keys())}")
    print(f"    Bars:{N:,} | {cidx[0].date()} → {cidx[-1].date()}")

    for bar in range(G.warmup, N):
        ts        = cidx[bar]
        cur_date  = ts.date()
        cur_month = (ts.year, ts.month)
        cur_hour  = ts.hour
        is_stress = cur_hour in G.stress_hours

        if cur_date != prev_date:
            day_eq = acc['equity']
            for n in pair_data: day_trades[n] = 0
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

        # blown
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
            cooldown_til = ts + pd.Timedelta(days=G.cooldown_days)
            acc_num += 1
            acc = new_acc(ts)
            day_eq = month_eq = acc['equity']
            for n in pair_data:
                day_trades[n] = 0; pending[n] = 0; positions[n] = None
            pnl_hist = []
            prev_date = cur_date; prev_month = cur_month
            continue

        if in_cd:
            continue

        # Risk multiplier
        month_pnl = acc['equity'] - month_eq
        risk_mult = get_risk_mult(
            acc['equity'], acc['peak'], pnl_hist,
            month_pnl, G.monthly_loss_threshold)

        # v15: GBP exposure limit
        gbp_open = sum(
            1 for n in pair_data
            if positions[n] is not None and pair_data[n][5].get('gbp_pair', False)
        )

        # open
        for name in pair_data:
            a    = pa[name]
            pcfg = a['cfg']
            if (pending[name] != 0 and positions[name] is None
                    and day_trades[name] < G.max_trades_day):

                # v15: GBP limit
                if pcfg.get('gbp_pair', False) and gbp_open >= 1:
                    pending[name] = 0
                    continue

                sv  = pending[name]
                pip = pcfg['pip_size']
                qr  = a['qr'][bar]

                # v15: spread واقع‌بینانه
                sp = pcfg['spread_pip_stress'] if is_stress else pcfg['spread_pip']
                sl = G.slippage_pips_stress if is_stress else G.slippage_pips_normal

                risk = pcfg['risk_pct'] * risk_mult
                if acc['consec_loss'] >= G.consec_loss_n:
                    risk = max(risk * G.risk_reduce, pcfg['risk_min'])

                pv  = pip * G.lot_size * qr
                if pv <= 0: pv = 10.0
                lot = round(float(np.clip(
                    acc['equity'] * risk / (pcfg['sl_pips'] * pv),
                    G.min_lot, G.max_lot)), 2)
                ep = a['o'][bar] + sv * (sl + sp / 2) * pip
                positions[name] = {
                    'dir': sv, 'lot': lot, 'lot_rem': lot,
                    'partial_done': False, 'entry': ep,
                    'sl': ep - sv * pcfg['sl_pips'] * pip,
                    'tp': ep + sv * pcfg['tp_pips'] * pip,
                    'entry_ts': ts, 'entry_bar': bar, 'pip': pip,
                    'swap_pips': pcfg.get('swap_pips_day', 0.0),
                }
                day_trades[name] += 1
                if pcfg.get('gbp_pair', False):
                    gbp_open += 1
            pending[name] = 0

        # float (بدون swap در float — فقط در close)
        total_float = sum(
            (p['dir'] * (pa[n]['c'][bar] - p['entry'])
             * p['lot_rem'] * GlobalConfig.lot_size * pa[n]['qr'][bar]
             - GlobalConfig.commission_per_lot * p['lot_rem'])
            for n in pair_data if (p := positions[n]) is not None
        )
        cur_eq    = acc['equity'] + total_float
        daily_lim = day_eq * (1 - G.max_daily_loss_pct)

        if cur_eq <= daily_lim or cur_eq <= FLOOR:
            rsn = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            acc['blown'] = True; acc['blown_rsn'] = rsn
            for name in pair_data:
                pos = positions[name]
                if pos is None: continue
                bars_held = bar - pos['entry_bar']
                pnl = pnl_calc(pos['dir'], pos['entry'], pa[name]['c'][bar],
                               pos['lot_rem'], pa[name]['qr'][bar], pos['pip'],
                               pos['swap_pips'], bars_held)
                acc['equity'] += pnl
                all_trades.append({'pair': name, 'pnl': pnl,
                                   'status': 'BLOWN', 'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                positions[name] = None
            continue

        # exit
        for name in pair_data:
            pos = positions[name]
            if pos is None: continue
            a    = pa[name]; pcfg = a['cfg']
            cp   = a['c'][bar]; d = pos['dir']
            ep   = pos['entry']; zn = a['z'][bar]
            lr   = pos['lot_rem']; qr = a['qr'][bar]; pip = pos['pip']
            bh   = bar - pos['entry_bar']

            # partial exit
            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -G.z_exit_partial) or
                        (d == -1 and zn <= G.z_exit_partial)):
                    p_lot = round(lr * G.partial_ratio, 2)
                    if p_lot >= G.min_lot:
                        p_pnl = pnl_calc(d, ep, cp, p_lot, qr, pip,
                                         pos['swap_pips'], bh)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            pnl_hist.append(p_pnl)
                            all_trades.append({'pair': name, 'pnl': p_pnl,
                                               'status': 'Partial', 'exit_ts': ts})
                            acc['trades'].append(all_trades[-1])
                            pos['lot_rem'] = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl'] = pos['entry']   # breakeven SL
                            lr = pos['lot_rem']
                            if lr < G.min_lot:
                                positions[name] = None; continue

            if positions[name] is None: continue
            lr      = pos['lot_rem']
            pnl_now = pnl_calc(d, ep, cp, lr, qr, pip, pos['swap_pips'], bh)

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
                xp  = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                st  = ('SL' if hit_sl else 'TP' if hit_tp else
                       'Z-Stop' if hit_zs else 'Z-Exit')
                fpnl = pnl_calc(d, ep, xp, lr, qr, pip, pos['swap_pips'], bh)
                acc['equity'] += fpnl
                pnl_hist.append(fpnl)
                all_trades.append({'pair': name, 'pnl': fpnl,
                                   'status': st, 'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                positions[name] = None
                if fpnl > 0: acc['consec_loss'] = 0
                else:        acc['consec_loss'] += 1

        # target
        if acc['equity'] >= TARGET and all(positions[n] is None for n in pair_data):
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
            for n in pair_data: day_trades[n] = 0; pending[n] = 0
            prev_date = cur_date; prev_month = cur_month
            continue

        # signals
        for name in pair_data:
            a = pa[name]
            if (positions[name] is None and not acc['blown'] and not in_cd
                    and day_trades[name] < G.max_trades_day and a['sig'][bar] != 0):
                pending[name] = int(a['sig'][bar])

    return {
        'all_trades': all_trades, 'account_logs': acc_logs,
        'withdrawn': withdrawn, 'final_equity': acc['equity'],
        'common_idx': cidx, 'eq_curve': eq_curve,
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
    all_months = pd.period_range(
        start=ci[GlobalConfig.warmup].to_period('M'),
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
    above     = int((monthly >= target_mo).sum())
    print(f"  🎯 هدف $100/ماه: {above}/{tot_m} ({above/tot_m*100:.0f}%)")
    print(f"  📊 میانگین: ${monthly.mean():.2f} → {monthly.mean()/target_mo*100:.0f}% از هدف")
    print("═" * 70)

    monthly.to_csv('monthly_v15.csv', header=['pnl'])
    pd.DataFrame(res['eq_curve']).to_csv('equity_v15.csv', index=False)
    print("  📊 monthly_v15.csv + equity_v15.csv saved")

    print("\n  📈 مقایسه نسخه‌ها:")
    rows = [
        ("v13 بدون RiskCtrl",  91, 59, 3, 10, "—"),
        ("v14b با RiskCtrl",   68, 56, 3,  2, "0.94"),
        (f"v15 Realistic",
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
    print("║  CorrArb v15 — Realistic + Robust                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("  تغییرات نسبت به v13/v14b:")
    print("  • Spread stress در ساعات پرنوسان")
    print("  • Swap/Rollover cost روزانه")
    print("  • VR filter دوگانه (fast + slow)")
    print("  • ATR blowout filter")
    print("  • GBP exposure: حداکثر 1 pair هم‌زمان")
    print("  • z_entry=2.3 | max_dd=8% | max_daily=4% | max_trades=1")
    print()

    pair_data = {}
    for name, pcfg in PAIR_CFG.items():
        print(f"  Loading {name}...")
        df = build_pair(pcfg)
        if df is None:
            print(f"  ❌ {name}: not found"); continue
        sig, z, atr, atr_ma = compute_signals(df, pcfg)
        n = int((sig != 0).sum())
        print(f"  ✅ {name}: {len(df):,} bars | {n:,} signals")
        pair_data[name] = (df, sig, z, atr, atr_ma, pcfg)

    print()
    res = run_portfolio(pair_data)
    print_report(res, "v15 — Quad Realistic (AUDNZD+AUDCAD+GBPCAD+GBPCHF)")
    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
