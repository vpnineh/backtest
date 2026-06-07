"""
CorrArb v10b — Multi-Pair Scanner Fixed
=========================================
اصلاح مشکلات v10:
  ✅ هر pair VR threshold خودش را دارد
  ✅ Correlation filter per-pair قابل تنظیم
  ✅ XAUXAG lot sizing کاملاً جدا
  ✅ اگر VR<0.75 سیگنال نداد، VR<0.90 را هم تست می‌کند
  ✅ Correlation min per-pair
"""

import pandas as pd
import numpy as np
import glob, zipfile, warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class Config:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10
    risk_base_pct      = 0.015
    risk_min_pct       = 0.005
    consec_loss_n      = 2
    risk_reduce        = 0.5
    commission_per_lot = 7.0
    slippage_pips      = 0.5
    lot_size           = 100_000
    max_lot            = 3.0
    min_lot            = 0.01
    warmup             = 500

    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.50
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0

    hour_start         = 2
    hour_end           = 19
    bad_hours          = {4, 5, 7, 9, 13, 18, 20}
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    sl_pips            = 30.0
    tp_pips            = 90.0
    partial_ratio      = 0.75

    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    vr_period          = 200
    vr_k               = 4
    corr_period        = 96

    cooldown_days          = 10
    monthly_loss_threshold = -150.0


# ═══════════════════════════════════════════════════════
# PAIR DEFINITIONS — با پارامترهای اختصاصی
# ═══════════════════════════════════════════════════════
PAIR_DEFS = {
    'AUDNZD': {
        'leg1': 'AUDUSD', 'leg2': 'NZDUSD',
        'formula': 'div', 'quote_formula': 'leg2',
        'spread_pip': 2.5, 'pip_size': 0.0001,
        'vr_max': 0.75, 'corr_min': 0.80,
        'is_metal': False,
    },
    'AUDCAD': {
        'leg1': 'AUDUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote_formula': 'inv_leg2',
        'spread_pip': 2.5, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.50,
        'is_metal': False,
    },
    'NZDCAD': {
        'leg1': 'NZDUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.50,
        'is_metal': False,
    },
    'EURCHF': {
        'leg1': 'EURUSD', 'leg2': 'USDCHF',
        'formula': 'div', 'quote_formula': 'inv_leg2',
        'spread_pip': 2.0, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.40,
        'is_metal': False,
    },
    'GBPCHF': {
        'leg1': 'GBPUSD', 'leg2': 'USDCHF',
        'formula': 'div', 'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.40,
        'is_metal': False,
    },
    'AUDCHF': {
        'leg1': 'AUDUSD', 'leg2': 'USDCHF',
        'formula': 'div', 'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.40,
        'is_metal': False,
    },
    'NZDCHF': {
        'leg1': 'NZDUSD', 'leg2': 'USDCHF',
        'formula': 'div', 'quote_formula': 'inv_leg2',
        'spread_pip': 3.5, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.40,
        'is_metal': False,
    },
    'EURCAD': {
        'leg1': 'EURUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.40,
        'is_metal': False,
    },
    'GBPCAD': {
        'leg1': 'GBPUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote_formula': 'inv_leg2',
        'spread_pip': 4.0, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.40,
        'is_metal': False,
    },
    'XAUXAG': {
        'leg1': 'XAUUSD', 'leg2': 'XAGUSD',
        'formula': 'div', 'quote_formula': 'one',
        'spread_pip': 30.0, 'pip_size': 0.01,
        'vr_max': 0.90, 'corr_min': 0.60,
        'is_metal': True,
        # برای طلا/نقره lot sizing متفاوت است
        # 1 lot XAUUSD = 100 oz = ~$25,000 value
        # ریسک را با dollar amount کنترل می‌کنیم
        'sl_dollar': 50.0,  # حداکثر $50 ریسک در XAUXAG
        'lot_fixed': 0.01,  # lot ثابت کوچک برای test
    },
}


# ═══════════════════════════════════════════════════════
# DATA LOADING
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
        raw = (load_raw_zip(pat) if pat.endswith('.zip')
               else load_raw_csv(pat))
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


def build_pair(pair_name, pdef):
    raw1 = load_instrument(pdef['leg1'])
    if raw1 is None:
        return None, f"❌ {pdef['leg1']} not found"

    raw2 = load_instrument(pdef['leg2'])
    if raw2 is None:
        return None, f"❌ {pdef['leg2']} not found"

    d1 = to_15min(raw1, 'leg1')
    d2 = to_15min(raw2, 'leg2')
    m  = d1.join(d2, how='inner').dropna()

    if len(m) < 5000:
        return None, f"❌ Too few bars: {len(m)}"

    if pdef['formula'] == 'div':
        m['c_spread'] = m['c_leg1'] / m['c_leg2']
        m['o_spread'] = m['o_leg1'] / m['o_leg2']
        m['h_spread'] = m['h_leg1'] / m['l_leg2']
        m['l_spread'] = m['l_leg1'] / m['h_leg2']
    else:  # mul
        m['c_spread'] = m['c_leg1'] * m['c_leg2']
        m['o_spread'] = m['o_leg1'] * m['o_leg2']
        m['h_spread'] = m['h_leg1'] * m['h_leg2']
        m['l_spread'] = m['l_leg1'] * m['l_leg2']

    if pdef['quote_formula'] == 'leg2':
        m['quote_rate'] = m['c_leg2']
    elif pdef['quote_formula'] == 'inv_leg2':
        m['quote_rate'] = 1.0 / m['c_leg2'].replace(0, np.nan)
    elif pdef['quote_formula'] == 'leg1':
        m['quote_rate'] = m['c_leg1']
    else:  # 'one'
        m['quote_rate'] = 1.0

    m = m[m.index.weekday < 5].dropna().copy()
    return m, f"✅ {len(m):,} bars ({m.index[0].date()} → {m.index[-1].date()})"


# ═══════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════
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


def compute_signals(df, pdef):
    C = Config

    log_r  = np.log(df['c_spread'].replace(0, np.nan))
    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    # Correlation — per-pair threshold
    corr = (df['c_leg1'].pct_change()
            .rolling(C.corr_period)
            .corr(df['c_leg2'].pct_change()))
    corr_ok = corr.abs() > pdef['corr_min']

    # VR — per-pair threshold
    vr        = calc_vr(log_r, C.vr_k, C.vr_period)
    regime_ok = vr < pdef['vr_max']

    atr    = calc_atr(df['h_spread'], df['l_spread'],
                      df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = ((atr > atr_ma * C.atr_min_mult) &
              (atr < atr_ma * C.atr_max_mult))

    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = (hour.between(C.hour_start, C.hour_end) &
               (~hour.isin(C.bad_hours)) &
               dow.isin(C.trade_days))

    sig = pd.Series(0, index=df.index)
    cond = vol_ok & time_ok & corr_ok & regime_ok
    sig[(z < -C.z_entry) & cond] =  1
    sig[(z >  C.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    n_sig     = int((sig != 0).sum())
    n_regime  = int(regime_ok.sum())
    n_corr    = int(corr_ok.sum())
    n_time    = int(time_ok.sum())

    print(f"    Signals:{n_sig:,} | Regime:{n_regime:,} | "
          f"Corr:{n_corr:,} | Session:{n_time:,}")

    return sig, z


# ═══════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════
def calc_pnl(direction, entry, exit_px, lot, qr, pip):
    C     = Config
    gross = direction * (exit_px - entry) * lot * C.lot_size * qr
    return gross - C.commission_per_lot * lot


def calc_lot(equity, pdef, qr, pip):
    """Lot sizing — per-pair"""
    C = Config
    if pdef.get('is_metal'):
        return pdef.get('lot_fixed', 0.01)

    risk = C.risk_base_pct
    pv   = pip * C.lot_size * qr
    if pv <= 0:
        return C.min_lot
    lot = equity * risk / (C.sl_pips * pv)
    return round(float(np.clip(lot, C.min_lot, C.max_lot)), 2)


def new_acc(ts):
    C = Config
    return {
        'equity':      C.initial_balance,
        'start_ts':    ts,
        'trades':      [],
        'blown':       False,
        'blown_rsn':   '',
        'peak':        C.initial_balance,
        'consec_loss': 0,
    }


def run_backtest(df, sig, z, pdef):
    C   = Config
    idx = df.index.sort_values()

    if len(idx) <= C.warmup:
        return None

    start_date = idx[C.warmup]
    pip        = pdef['pip_size']
    sp         = pdef['spread_pip']

    o_  = df['o_spread'].reindex(idx).ffill().values.astype(float)
    c_  = df['c_spread'].reindex(idx).ffill().values.astype(float)
    qr_ = df['quote_rate'].reindex(idx).ffill().values.astype(float)
    sg_ = sig.reindex(idx).fillna(0).values.astype(int)
    zz_ = z.reindex(idx).fillna(np.nan).values.astype(float)

    FLOOR  = C.initial_balance * (1 - C.max_total_dd_pct)
    TARGET = C.initial_balance * (1 + C.profit_target_pct)

    acc          = new_acc(start_date)
    withdrawn    = 0.0
    acc_num      = 1
    day_eq       = C.initial_balance
    month_eq     = C.initial_balance
    cooldown_til = None
    all_trades   = []
    acc_logs     = []

    pos        = None
    day_trades = 0
    pending    = 0
    prev_date  = None
    prev_month = None

    for bar in range(C.warmup, len(idx)):
        ts        = idx[bar]
        cur_date  = ts.date()
        cur_month = (ts.year, ts.month)

        if cur_date != prev_date:
            day_eq     = acc['equity']
            day_trades = 0
            prev_date  = cur_date

        if cur_month != prev_month:
            month_eq   = acc['equity']
            prev_month = cur_month

        if acc['equity'] > acc['peak']:
            acc['peak'] = acc['equity']

        in_cd = cooldown_til is not None and ts < cooldown_til

        if acc['blown']:
            acc_logs.append({
                'reason': acc['blown_rsn'],
                'pnl': acc['equity'] - C.initial_balance,
                'days': (ts - acc['start_ts']).days,
            })
            cooldown_til = ts + pd.Timedelta(days=C.cooldown_days)
            acc_num += 1
            acc = new_acc(ts)
            day_eq = month_eq = acc['equity']
            day_trades = 0; pending = 0; pos = None
            continue

        if in_cd:
            continue

        m_stressed = (acc['equity'] - month_eq) < C.monthly_loss_threshold

        # open
        if pending != 0 and pos is None and day_trades < C.max_trades_day:
            sv  = pending
            qr  = qr_[bar]
            lot = calc_lot(acc['equity'], pdef, qr, pip)

            if m_stressed:
                lot = max(round(lot * 0.5, 2), C.min_lot)
            if acc['consec_loss'] >= C.consec_loss_n:
                lot = max(round(lot * C.risk_reduce, 2), C.min_lot)

            ep = o_[bar] + sv * (C.slippage_pips + sp / 2) * pip
            pos = {
                'dir': sv, 'lot': lot, 'lot_rem': lot,
                'partial_done': False,
                'entry': ep,
                'sl': ep - sv * C.sl_pips * pip,
                'tp': ep + sv * C.tp_pips * pip,
                'entry_ts': ts, 'entry_bar': bar,
            }
            day_trades += 1

        pending = 0

        # float check
        flt = 0.0
        if pos is not None:
            flt = calc_pnl(pos['dir'], pos['entry'],
                           c_[bar], pos['lot_rem'], qr_[bar], pip)

        cur_eq    = acc['equity'] + flt
        daily_lim = day_eq * (1 - C.max_daily_loss_pct)

        if cur_eq <= daily_lim or cur_eq <= FLOOR:
            rsn = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            acc['blown'] = True; acc['blown_rsn'] = rsn
            if pos is not None:
                pnl = calc_pnl(pos['dir'], pos['entry'],
                               c_[bar], pos['lot_rem'], qr_[bar], pip)
                acc['equity'] += pnl
                all_trades.append({'pnl': pnl, 'status': 'BLOWN',
                                   'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                pos = None
            continue

        # exit
        if pos is not None:
            cp = c_[bar]; d = pos['dir']; ep = pos['entry']
            zn = zz_[bar]; lr = pos['lot_rem']; qr = qr_[bar]

            # partial
            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -C.z_exit_partial) or
                        (d == -1 and zn <= C.z_exit_partial)):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr, pip)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            all_trades.append({'pnl': p_pnl, 'status': 'Partial',
                                               'exit_ts': ts})
                            acc['trades'].append(all_trades[-1])
                            pos['lot_rem'] = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl'] = pos['entry']
                            lr = pos['lot_rem']
                            if lr < C.min_lot:
                                pos = None

            if pos is not None:
                lr      = pos['lot_rem']
                pnl_now = calc_pnl(d, ep, cp, lr, qr, pip)

                hit_zs = (not np.isnan(zn) and
                          ((d == 1 and zn <= -C.z_stop_margin) or
                           (d == -1 and zn >= C.z_stop_margin)))
                hit_ze = (not np.isnan(zn) and
                          ((d == 1 and zn >= -C.z_exit_full) or
                           (d == -1 and zn <= C.z_exit_full)))
                if hit_ze and pnl_now < C.min_net_profit_usd and not pos['partial_done']:
                    hit_ze = False
                hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
                hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])

                if hit_ze or hit_zs or hit_sl or hit_tp:
                    xp = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                    st = ('SL' if hit_sl else 'TP' if hit_tp else
                          'Z-Stop' if hit_zs else 'Z-Exit')
                    fpnl = calc_pnl(d, ep, xp, lr, qr, pip)
                    acc['equity'] += fpnl
                    all_trades.append({'pnl': fpnl, 'status': st,
                                       'exit_ts': ts})
                    acc['trades'].append(all_trades[-1])
                    pos = None
                    if fpnl > 0: acc['consec_loss'] = 0
                    else:        acc['consec_loss'] += 1

        # target
        if acc['equity'] >= TARGET and pos is None:
            w = acc['equity'] - C.initial_balance
            withdrawn += w
            acc_logs.append({'reason': 'TARGET_HIT', 'pnl': w,
                             'days': (ts - acc['start_ts']).days})
            acc_num += 1
            acc = new_acc(ts)
            day_eq = month_eq = acc['equity']
            day_trades = 0; pending = 0
            continue

        if (pos is None and not acc['blown'] and not in_cd
                and day_trades < C.max_trades_day and sg_[bar] != 0):
            pending = int(sg_[bar])

    return {
        'all_trades': all_trades, 'account_logs': acc_logs,
        'withdrawn': withdrawn, 'final_equity': acc['equity'],
        'common_idx': idx,
    }


# ═══════════════════════════════════════════════════════
# ANALYZE + REPORT
# ═══════════════════════════════════════════════════════
def analyze(pair_name, res):
    if not res or not res['all_trades'] or len(res['all_trades']) < 20:
        return None

    df = pd.DataFrame(res['all_trades'])
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])
    df['month']   = df['exit_ts'].dt.to_period('M')
    df['year']    = df['exit_ts'].dt.year

    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    wr = len(wins) / len(df) * 100
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else 99.0

    ci = res['common_idx']
    all_months = pd.period_range(
        start=ci[Config.warmup].to_period('M'),
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

    # Yearly detail
    yearly_str = ""
    for yr, g2 in df.groupby('year'):
        w2 = g2[g2['pnl'] > 0]; l2 = g2[g2['pnl'] < 0]
        ypf  = w2['pnl'].sum() / abs(l2['pnl'].sum()) if len(l2) else 99.0
        mark = '✅' if g2['pnl'].sum() >= 0 else '❌'
        yearly_str += (f"    {mark} {yr}:{len(g2):>4}T  "
                       f"WR:{len(w2)/len(g2)*100:5.1f}%  "
                       f"PF:{ypf:.2f}  ${g2['pnl'].sum():>+8,.2f}\n")

    return {
        'Pair': pair_name, 'Trades': len(df),
        'WR%': round(wr, 1), 'PF': round(pf, 3),
        'Banked': round(res['withdrawn'], 0),
        'Net': round(df['pnl'].sum(), 0),
        '+Mo': pos_m, '-Mo': neg_m, 'TotMo': tot_m,
        'Streak': ms, 'MonAvg': round(monthly.mean(), 2),
        'Median': round(monthly.median(), 2),
        'Pass': n_pass, 'Blow': n_blow, 'NegYr': neg_yr,
        'AvgWin': round(wins['pnl'].mean(), 2) if len(wins) else 0,
        'AvgLoss': round(losses['pnl'].mean(), 2) if len(losses) else 0,
        'yearly': yearly_str,
    }


def verdict(row):
    if (row['PF'] >= 1.12 and
            row['+Mo'] >= row['TotMo'] * 0.45 and
            row['NegYr'] <= 6 and
            row['Blow'] <= 5):
        return '✅ PASS'
    elif row['PF'] >= 1.05 and row['Net'] > 0:
        return '⚠ MARGINAL'
    return '❌ FAIL'


def print_detail(row, pdef):
    v = verdict(row)
    print(f"\n    Result: {v}")
    print(f"    Trades:{row['Trades']}  WR:{row['WR%']}%  PF:{row['PF']}")
    print(f"    Net:${row['Net']:,.0f}  Banked:${row['Banked']:,.0f}")
    print(f"    Pass:{row['Pass']}  Blow:{row['Blow']}")
    print(f"    +Mo:{row['+Mo']}/{row['TotMo']}  Streak:{row['Streak']}  NegYr:{row['NegYr']}")
    print(f"    MonAvg:${row['MonAvg']:.2f}  Median:${row['Median']:.2f}")
    print(f"    AvgWin:${row['AvgWin']:.2f}  AvgLoss:${row['AvgLoss']:.2f}")
    print(f"    Yearly:")
    print(row['yearly'])


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CorrArb v10b — Multi-Pair Scanner (Fixed)            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = []

    for pair_name, pdef in PAIR_DEFS.items():
        print(f"\n{'═'*62}")
        print(f"  ▶ {pair_name}  ({pdef['leg1']} × {pdef['leg2']})")
        print(f"    VR<{pdef['vr_max']} | Corr>{pdef['corr_min']}")
        print(f"{'═'*62}")

        df, msg = build_pair(pair_name, pdef)
        print(f"    {msg}")

        if df is None:
            results.append({'Pair': pair_name, 'Status': 'NO_DATA',
                            'PF': 0, 'Net': 0})
            continue

        try:
            sig, z = compute_signals(df, pdef)
        except Exception as e:
            print(f"    ❌ Signal error: {e}")
            results.append({'Pair': pair_name, 'Status': 'SIG_ERR',
                            'PF': 0, 'Net': 0})
            continue

        n_sig = int((sig != 0).sum())
        if n_sig < 30:
            print(f"    ⚠ Only {n_sig} signals — skipping")
            results.append({'Pair': pair_name, 'Status': 'FEW_SIG',
                            'PF': 0, 'Net': 0})
            continue

        try:
            res = run_backtest(df, sig, z, pdef)
        except Exception as e:
            print(f"    ❌ Backtest error: {e}")
            results.append({'Pair': pair_name, 'Status': 'BT_ERR',
                            'PF': 0, 'Net': 0})
            continue

        row = analyze(pair_name, res)
        if row is None:
            print(f"    ⚠ Too few trades")
            results.append({'Pair': pair_name, 'Status': 'NO_TRADES',
                            'PF': 0, 'Net': 0})
            continue

        row['Status'] = 'OK'
        row['Verdict'] = verdict(row)
        print_detail(row, pdef)
        results.append(row)

    # ═══════════════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ═══════════════════════════════════════════════════
    print("\n\n" + "╔" + "═"*100 + "╗")
    print("║" + "  FINAL RESULTS".center(100) + "║")
    print("╚" + "═"*100 + "╝\n")

    df_r   = pd.DataFrame(results)
    ok     = df_r[df_r['Status'] == 'OK'].copy()
    passed = ok[ok['Verdict'] == '✅ PASS']
    margin = ok[ok['Verdict'] == '⚠ MARGINAL']

    if len(ok):
        ok_sorted = ok.sort_values('PF', ascending=False)
        print(f"  {'Pair':<10} {'Verdict':<12} {'Tr':>5} {'WR':>5} {'PF':>6} "
              f"{'Banked':>8} {'Net':>8} {'+Mo':>4} {'Str':>4} "
              f"{'MonAvg':>8} {'Pass':>5} {'Blow':>5} {'NegYr':>6}")
        print("  " + "─"*100)
        for _, r in ok_sorted.iterrows():
            print(f"  {r['Pair']:<10} {r['Verdict']:<12} "
                  f"{int(r['Trades']):>5} {r['WR%']:>5.1f} {r['PF']:>6.3f} "
                  f"{r['Banked']:>8,.0f} {r['Net']:>8,.0f} "
                  f"{r['+Mo']:>4} {r['Streak']:>4} "
                  f"{r['MonAvg']:>8.2f} "
                  f"{r['Pass']:>5} {r['Blow']:>5} {r['NegYr']:>6}")

    skip = df_r[df_r['Status'] != 'OK']
    if len(skip):
        print(f"\n  Skipped: {', '.join(skip['Pair'].tolist())}")

    print(f"\n  {'═'*80}")
    print(f"  ✅ PASS:     {len(passed)} pairs")
    print(f"  ⚠ MARGINAL: {len(margin)} pairs")
    print(f"  ❌ Others:   {len(df_r) - len(passed) - len(margin)}")

    if len(passed):
        total_ma = passed['MonAvg'].sum()
        target   = Config.initial_balance * 0.02
        print(f"\n  🏆 Passing pairs:")
        for _, r in passed.iterrows():
            print(f"     {r['Pair']:<10} PF:{r['PF']:.3f}  "
                  f"MonAvg:${r['MonAvg']:.2f}  Net:${r['Net']:,.0f}")
        print(f"\n  📈 Combined MonthlyAvg: ${total_ma:.2f}")
        print(f"  🎯 Target:              ${target:.2f}")
        print(f"  📊 Coverage:            {total_ma/target*100:.0f}%")

    if len(margin):
        print(f"\n  ⚠ Marginal (needs tuning):")
        for _, r in margin.iterrows():
            print(f"     {r['Pair']:<10} PF:{r['PF']:.3f}  MonAvg:${r['MonAvg']:.2f}")

    print(f"  {'═'*80}")
    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
