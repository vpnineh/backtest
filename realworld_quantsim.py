"""
CorrArb v10 — Multi-Pair Scanner
==================================
هدف: تست همه pairهای synthetic ممکن با baseline برنده

Baseline:
  ✅ No TimeStop
  ✅ SL = 30
  ✅ Partial = 0.75
  ✅ z_exit_partial = 0.50
  ✅ VR < 0.75
  ✅ No Bad Hours (4,5,7,9,13,18,20)

Pairs to test:
  FX Synthetic:
    AUDNZD = AUDUSD / NZDUSD
    AUDCAD = AUDUSD * USDCAD
    NZDCAD = NZDUSD * USDCAD
    EURCHF = EURUSD / USDCHF
    GBPCHF = GBPUSD / USDCHF
    AUDCHF = AUDUSD / USDCHF
    NZDCHF = NZDUSD / USDCHF
    EURCAD = EURUSD * USDCAD
    GBPCAD = GBPUSD * USDCAD

  Metals Ratio:
    XAUXAG = XAUUSD / XAGUSD
"""

import pandas as pd
import numpy as np
import glob, zipfile, warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
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

    # Signal — locked
    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.50
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0

    # Filters — locked best
    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 2
    hour_end           = 19
    bad_hours          = {4, 5, 7, 9, 13, 18, 20}
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # Exit — locked
    sl_pips            = 30.0
    tp_pips            = 90.0
    partial_ratio      = 0.75

    # Regime — locked
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5
    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.75

    cooldown_days          = 10
    monthly_loss_threshold = -150.0


# ═══════════════════════════════════════════════════════
# PAIR DEFINITIONS
# ═══════════════════════════════════════════════════════
PAIR_DEFS = {
    # name: (leg1_pattern, leg2_pattern, formula, quote_formula, spread_pip, pip_size)
    #
    # formula options:
    #   'div' = leg1 / leg2
    #   'mul' = leg1 * leg2
    #
    # quote_formula:
    #   'leg2'     = quote_rate = leg2 close
    #   'inv_leg2' = quote_rate = 1 / leg2 close
    #   'leg1'     = quote_rate = leg1 close (for metals)

    'AUDNZD': {
        'leg1': 'AUDUSD', 'leg2': 'NZDUSD',
        'formula': 'div',
        'quote_formula': 'leg2',
        'spread_pip': 2.5, 'pip_size': 0.0001,
    },
    'AUDCAD': {
        'leg1': 'AUDUSD', 'leg2': 'USDCAD',
        'formula': 'mul',
        'quote_formula': 'inv_leg2',
        'spread_pip': 2.5, 'pip_size': 0.0001,
    },
    'NZDCAD': {
        'leg1': 'NZDUSD', 'leg2': 'USDCAD',
        'formula': 'mul',
        'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
    },
    'EURCHF': {
        'leg1': 'EURUSD', 'leg2': 'USDCHF',
        'formula': 'div',
        'quote_formula': 'inv_leg2',
        'spread_pip': 2.0, 'pip_size': 0.0001,
    },
    'GBPCHF': {
        'leg1': 'GBPUSD', 'leg2': 'USDCHF',
        'formula': 'div',
        'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
    },
    'AUDCHF': {
        'leg1': 'AUDUSD', 'leg2': 'USDCHF',
        'formula': 'div',
        'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
    },
    'NZDCHF': {
        'leg1': 'NZDUSD', 'leg2': 'USDCHF',
        'formula': 'div',
        'quote_formula': 'inv_leg2',
        'spread_pip': 3.5, 'pip_size': 0.0001,
    },
    'EURCAD': {
        'leg1': 'EURUSD', 'leg2': 'USDCAD',
        'formula': 'mul',
        'quote_formula': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
    },
    'GBPCAD': {
        'leg1': 'GBPUSD', 'leg2': 'USDCAD',
        'formula': 'mul',
        'quote_formula': 'inv_leg2',
        'spread_pip': 4.0, 'pip_size': 0.0001,
    },
    'XAUXAG': {
        'leg1': 'XAUUSD', 'leg2': 'XAGUSD',
        'formula': 'div',
        'quote_formula': 'leg2',
        'spread_pip': 5.0, 'pip_size': 0.01,
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
        frames.append(pd.read_csv(
            p, sep=';', header=None,
            names=['ts', 'o', 'h', 'l', 'c', 'v']))
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


def load_instrument(name):
    """بارگذاری یک instrument — اول ZIP بعد CSV"""
    raw = load_raw_zip(f'data/HISTDATA*{name}*.zip')
    if raw is None:
        raw = load_raw_zip(f'data/*{name}*.zip')
    if raw is None:
        raw = load_raw_csv(f'data/*{name}*.csv')
    if raw is None:
        raw = load_raw_csv(f'data/HISTDATA*{name}*.csv')
    return raw


def build_pair(pair_name, pair_def):
    """ساخت synthetic pair از دو leg"""
    leg1_name = pair_def['leg1']
    leg2_name = pair_def['leg2']

    raw1 = load_instrument(leg1_name)
    if raw1 is None:
        return None, f"❌ {leg1_name} not found"

    raw2 = load_instrument(leg2_name)
    if raw2 is None:
        return None, f"❌ {leg2_name} not found"

    d1 = to_15min(raw1, 'leg1')
    d2 = to_15min(raw2, 'leg2')

    m = d1.join(d2, how='inner').dropna()

    if len(m) < 10000:
        return None, f"❌ Too few bars: {len(m)}"

    # Spread calculation
    if pair_def['formula'] == 'div':
        m['c_spread'] = m['c_leg1'] / m['c_leg2']
        m['o_spread'] = m['o_leg1'] / m['o_leg2']
        m['h_spread'] = m['h_leg1'] / m['l_leg2']
        m['l_spread'] = m['l_leg1'] / m['h_leg2']
    else:  # mul
        m['c_spread'] = m['c_leg1'] * m['c_leg2']
        m['o_spread'] = m['o_leg1'] * m['o_leg2']
        m['h_spread'] = m['h_leg1'] * m['h_leg2']
        m['l_spread'] = m['l_leg1'] * m['l_leg2']

    # Quote rate for PnL conversion
    if pair_def['quote_formula'] == 'leg2':
        m['quote_rate'] = m['c_leg2']
    elif pair_def['quote_formula'] == 'inv_leg2':
        m['quote_rate'] = 1.0 / m['c_leg2']
    elif pair_def['quote_formula'] == 'leg1':
        m['quote_rate'] = m['c_leg1']
    else:
        m['quote_rate'] = 1.0

    m = m[m.index.weekday < 5].copy()
    return m, f"✅ {len(m):,} bars ({m.index[0].date()} → {m.index[-1].date()})"


# ═══════════════════════════════════════════════════════
# INDICATORS + SIGNALS
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


def compute_signals(df):
    C = Config

    log_r  = np.log(df['c_spread'])
    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    corr_ok = (df['c_leg1'].pct_change()
               .rolling(C.corr_period)
               .corr(df['c_leg2'].pct_change()) > C.corr_min)

    vr        = calc_vr(log_r, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

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

    sig  = pd.Series(0, index=df.index)
    cond = vol_ok & time_ok & corr_ok & regime_ok
    sig[(z < -C.z_entry) & cond] =  1
    sig[(z >  C.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    return sig, z


# ═══════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════
def calc_pnl(direction, entry, exit_px, lot, qr, pip):
    C     = Config
    gross = direction * (exit_px - entry) * lot * C.lot_size * qr
    return gross - C.commission_per_lot * lot


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


def run_backtest(df, sig, z, spread_pip, pip_size):
    C          = Config
    idx        = df.index.sort_values()
    start_date = idx[C.warmup]
    pip        = pip_size
    sp         = spread_pip

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
                'account': acc_num, 'reason': acc['blown_rsn'],
                'pnl': acc['equity'] - C.initial_balance,
                'days': (ts - acc['start_ts']).days,
            })
            cooldown_til = ts + pd.Timedelta(days=C.cooldown_days)
            acc_num   += 1
            acc        = new_acc(ts)
            day_eq     = month_eq = acc['equity']
            day_trades = 0; pending = 0; pos = None
            continue

        if in_cd:
            continue

        m_stressed = (acc['equity'] - month_eq) < C.monthly_loss_threshold

        # open
        if pending != 0 and pos is None and day_trades < C.max_trades_day:
            sv   = pending
            risk = C.risk_base_pct * (0.5 if m_stressed else 1.0)
            if acc['consec_loss'] >= C.consec_loss_n:
                risk = max(risk * C.risk_reduce, C.risk_min_pct)

            pv  = pip * C.lot_size * qr_[bar]
            if pv <= 0:
                pv = 10.0
            lot = round(float(np.clip(
                acc['equity'] * risk / (C.sl_pips * pv),
                C.min_lot, C.max_lot)), 2)

            ep  = o_[bar] + sv * (C.slippage_pips + sp / 2) * pip
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
                all_trades.append({'pnl': pnl, 'status': 'BLOWN', 'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                pos = None
            continue

        # exit
        if pos is not None:
            cp = c_[bar]; d = pos['dir']; ep = pos['entry']
            zn = zz_[bar]; lr = pos['lot_rem']

            # partial
            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -C.z_exit_partial) or
                        (d == -1 and zn <= C.z_exit_partial)):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr_[bar], pip)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            all_trades.append({'pnl': p_pnl, 'status': 'Partial', 'exit_ts': ts})
                            acc['trades'].append(all_trades[-1])
                            pos['lot_rem'] = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl'] = pos['entry']
                            lr = pos['lot_rem']
                            if lr < C.min_lot:
                                pos = None

            if pos is not None:
                lr      = pos['lot_rem']
                pnl_now = calc_pnl(d, ep, cp, lr, qr_[bar], pip)

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
                    fpnl = calc_pnl(d, ep, xp, lr, qr_[bar], pip)
                    acc['equity'] += fpnl
                    all_trades.append({'pnl': fpnl, 'status': st, 'exit_ts': ts})
                    acc['trades'].append(all_trades[-1])
                    pos = None
                    if fpnl > 0: acc['consec_loss'] = 0
                    else:        acc['consec_loss'] += 1

        # target
        if acc['equity'] >= TARGET and pos is None:
            w = acc['equity'] - C.initial_balance
            withdrawn += w
            acc_logs.append({
                'account': acc_num, 'reason': 'TARGET_HIT',
                'pnl': w, 'days': (ts - acc['start_ts']).days,
            })
            acc_num += 1
            acc = new_acc(ts)
            day_eq = month_eq = acc['equity']
            day_trades = 0; pending = 0
            continue

        # signal
        if (pos is None and not acc['blown'] and not in_cd
                and day_trades < C.max_trades_day and sg_[bar] != 0):
            pending = int(sg_[bar])

    return {
        'all_trades':   all_trades,
        'account_logs': acc_logs,
        'withdrawn':    withdrawn,
        'final_equity': acc['equity'],
        'common_idx':   idx,
    }


# ═══════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════
def analyze_result(pair_name, res):
    """آنالیز نتایج یک pair"""
    trades = res['all_trades']
    if not trades or len(trades) < 10:
        return None

    df = pd.DataFrame(trades)
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
        ms = max(ms, cur)

    logs   = pd.DataFrame(res['account_logs']) if res['account_logs'] else pd.DataFrame()
    n_pass = int((logs['reason'] == 'TARGET_HIT').sum()) if len(logs) else 0
    n_blow = int((logs['reason'] != 'TARGET_HIT').sum()) if len(logs) else 0

    # Yearly neg
    yearly = df.groupby('year')['pnl'].sum()
    neg_yr = int((yearly < 0).sum())

    return {
        'Pair':     pair_name,
        'Trades':   len(df),
        'WR%':      round(wr, 1),
        'PF':       round(pf, 3),
        'Banked':   round(res['withdrawn'], 0),
        'Net':      round(df['pnl'].sum(), 0),
        '+Mo':      pos_m,
        '-Mo':      neg_m,
        'TotMo':    tot_m,
        'Streak':   ms,
        'MonAvg':   round(monthly.mean(), 2),
        'Median':   round(monthly.median(), 2),
        'Pass':     n_pass,
        'Blow':     n_blow,
        'NegYr':    neg_yr,
        'AvgWin':   round(wins['pnl'].mean(), 2) if len(wins) else 0,
        'AvgLoss':  round(losses['pnl'].mean(), 2) if len(losses) else 0,
    }


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CorrArb v10 — Multi-Pair Scanner                     ║")
    print("║   Baseline: VR<0.75 | No Bad Hours | SL30 | P0.75      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = []

    for pair_name, pdef in PAIR_DEFS.items():
        print(f"\n{'─'*60}")
        print(f"  ▶ {pair_name}  ({pdef['leg1']} / {pdef['leg2']})")
        print(f"{'─'*60}")

        # Build pair
        df, msg = build_pair(pair_name, pdef)
        print(f"    {msg}")

        if df is None:
            results.append({'Pair': pair_name, 'Status': 'SKIP', 'PF': 0})
            continue

        # Signals
        try:
            sig, z = compute_signals(df)
            n_sig = int((sig != 0).sum())
            print(f"    Signals: {n_sig:,}")

            if n_sig < 50:
                print(f"    ⚠ Too few signals — skipping")
                results.append({'Pair': pair_name, 'Status': 'FEW_SIG', 'PF': 0})
                continue
        except Exception as e:
            print(f"    ❌ Signal error: {e}")
            results.append({'Pair': pair_name, 'Status': 'ERROR', 'PF': 0})
            continue

        # Backtest
        try:
            res = run_backtest(df, sig, z,
                               pdef['spread_pip'], pdef['pip_size'])
        except Exception as e:
            print(f"    ❌ Backtest error: {e}")
            results.append({'Pair': pair_name, 'Status': 'ERROR', 'PF': 0})
            continue

        # Analyze
        row = analyze_result(pair_name, res)
        if row is None:
            print(f"    ⚠ No trades or too few")
            results.append({'Pair': pair_name, 'Status': 'NO_TRADES', 'PF': 0})
            continue

        row['Status'] = 'OK'

        # Quick verdict
        if row['PF'] >= 1.12 and row['+Mo'] >= row['TotMo'] * 0.45 and row['NegYr'] <= 6:
            verdict = '✅ PASS'
        elif row['PF'] >= 1.05:
            verdict = '⚠ MARGINAL'
        else:
            verdict = '❌ FAIL'

        row['Verdict'] = verdict
        results.append(row)

        print(f"    {verdict}  PF:{row['PF']:.3f}  "
              f"Net:${row['Net']:,.0f}  "
              f"Bank:${row['Banked']:,.0f}  "
              f"+Mo:{row['+Mo']}/{row['TotMo']}  "
              f"Pass:{row['Pass']} Blow:{row['Blow']}  "
              f"NegYr:{row['NegYr']}")

    # ═══════════════════════════════════════════════════════
    # FINAL TABLE
    # ═══════════════════════════════════════════════════════
    print("\n\n" + "╔" + "═"*100 + "╗")
    print("║" + "  MULTI-PAIR SCANNER RESULTS".center(100) + "║")
    print("╚" + "═"*100 + "╝")

    df_res = pd.DataFrame(results)
    ok_rows = df_res[df_res['Status'] == 'OK'].copy()

    if len(ok_rows):
        ok_rows = ok_rows.sort_values('PF', ascending=False)

        print(f"\n  {'Pair':<10} {'Verdict':<12} {'Tr':>5} {'WR':>5} {'PF':>6} "
              f"{'Banked':>8} {'Net':>8} {'+Mo':>4} {'-Mo':>4} {'Str':>4} "
              f"{'MonAvg':>8} {'Med':>7} {'Pass':>5} {'Blow':>5} {'NegYr':>6}")
        print("  " + "─"*115)

        for _, r in ok_rows.iterrows():
            print(f"  {r['Pair']:<10} {r['Verdict']:<12} "
                  f"{int(r['Trades']):>5} {r['WR%']:>5.1f} {r['PF']:>6.3f} "
                  f"{r['Banked']:>8,.0f} {r['Net']:>8,.0f} "
                  f"{r['+Mo']:>4} {r['-Mo']:>4} {r['Streak']:>4} "
                  f"{r['MonAvg']:>8.2f} {r['Median']:>7.2f} "
                  f"{r['Pass']:>5} {r['Blow']:>5} {r['NegYr']:>6}")

    # Skipped
    skip = df_res[df_res['Status'] != 'OK']
    if len(skip):
        print(f"\n  Skipped/Failed:")
        for _, r in skip.iterrows():
            print(f"    {r['Pair']:<10} — {r['Status']}")

    # ── نتیجه‌گیری ──
    passed = ok_rows[ok_rows['Verdict'].str.contains('PASS')] if len(ok_rows) else pd.DataFrame()
    marginal = ok_rows[ok_rows['Verdict'].str.contains('MARGINAL')] if len(ok_rows) else pd.DataFrame()

    print(f"\n  " + "═"*80)
    print(f"  📊 نتیجه‌گیری:")
    print(f"     Pairs tested:  {len(PAIR_DEFS)}")
    print(f"     ✅ PASS:       {len(passed)}")
    print(f"     ⚠ MARGINAL:   {len(marginal)}")
    print(f"     ❌ FAIL/SKIP:  {len(df_res) - len(passed) - len(marginal)}")

    if len(passed):
        print(f"\n  🏆 Pairs worth keeping:")
        for _, r in passed.iterrows():
            print(f"     {r['Pair']:<10} PF:{r['PF']:.3f}  Net:${r['Net']:,.0f}  "
                  f"MonAvg:${r['MonAvg']:.2f}")

        total_monthly = passed['MonAvg'].sum()
        target = Config.initial_balance * 0.02
        print(f"\n  📈 Combined monthly estimate: ${total_monthly:.2f}")
        print(f"  🎯 Target: ${target:.2f}/mo")
        print(f"  📊 Coverage: {total_monthly/target*100:.0f}%")

    if len(marginal):
        print(f"\n  ⚠ Marginal (may improve with tuning):")
        for _, r in marginal.iterrows():
            print(f"     {r['Pair']:<10} PF:{r['PF']:.3f}  Net:${r['Net']:,.0f}")

    print(f"\n  " + "═"*80)
    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
