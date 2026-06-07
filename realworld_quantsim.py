"""
CorrArb v9h Fix — Regime Analysis
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
    PAIR_SPREAD        = {'AUDNZD': 2.5}
    PIP_SIZE           = {'AUDNZD': 0.0001}
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
    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 2
    hour_end           = 19
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
    vr_max             = 0.90
    cooldown_days      = 10
    monthly_loss_threshold = -150.0


# ═══════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════
def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No ZIP: {pattern}")
    frames = []
    for p in paths:
        with zipfile.ZipFile(p) as z:
            csv_name = next(
                (f for f in z.namelist() if f.lower().endswith('.csv')), None)
            if not csv_name:
                continue
            with z.open(csv_name) as f:
                frames.append(pd.read_csv(
                    f, sep=';', header=None,
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


def load_audnzd():
    print("  Loading AUDNZD...")
    aud = to_15min(load_raw_zip('data/HISTDATA*AUDUSD*.zip'), 'aud')
    nzd = to_15min(load_raw_zip('data/HISTDATA*NZDUSD*.zip'), 'nzd')
    m   = aud.join(nzd, how='inner').dropna()
    m['c_spread']   = m['c_aud'] / m['c_nzd']
    m['o_spread']   = m['o_aud'] / m['o_nzd']
    m['h_spread']   = m['h_aud'] / m['l_nzd']
    m['l_spread']   = m['l_aud'] / m['h_nzd']
    m['quote_rate'] = m['c_nzd']
    m = m[m.index.weekday < 5].copy()
    print(f"  ✅ {len(m):,} candles")
    return m


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


def safe_qcut(series, n_quantiles, labels):
    """qcut با handle کردن duplicate edges"""
    try:
        return pd.qcut(series, q=n_quantiles, labels=labels, duplicates='drop')
    except Exception:
        return pd.Series('Unknown', index=series.index)


def safe_cut(series, bins, labels=None):
    """cut با handle کردن خطاهای monotonic"""
    bins = sorted(list(set(bins)))
    if len(bins) < 2:
        return pd.Series('All', index=series.index)
    try:
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    except Exception:
        return pd.Series('Unknown', index=series.index)


# ═══════════════════════════════════════════════════════
# BUILD FEATURES
# ═══════════════════════════════════════════════════════
def build_features(df):
    C     = Config
    log_r = np.log(df['c_spread'])

    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    atr    = calc_atr(df['h_spread'], df['l_spread'],
                      df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    atr_ratio = atr / atr_ma.replace(0, np.nan)

    vr   = calc_vr(log_r, C.vr_k, C.vr_period)
    corr = (df['c_aud'].pct_change()
            .rolling(C.corr_period)
            .corr(df['c_nzd'].pct_change()))

    ma_fast = df['c_spread'].rolling(48).mean()
    ma_slow = df['c_spread'].rolling(200).mean()
    trend   = (ma_fast - ma_slow) / ma_slow.replace(0, np.nan) * 100

    # Realized vol — روزانه
    log_daily = np.log(df['c_spread']).resample('D').last().dropna()
    rvol_30d  = log_daily.diff().rolling(30).std()
    rvol_30d  = rvol_30d.reindex(df.index, method='ffill')

    # Spread distance از mean — normalized
    spread_dist = (log_r - log_r.rolling(C.z_fast_period).mean()).abs()

    # Session
    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    sig  = pd.Series(0, index=df.index)
    cond = ((atr_ratio > C.atr_min_mult) &
            (atr_ratio < C.atr_max_mult) &
            time_ok &
            (corr > C.corr_min) &
            (vr < C.vr_max))
    sig[(z < -C.z_entry) & cond] =  1
    sig[(z >  C.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    features = pd.DataFrame({
        'z':           z,
        'atr':         atr,
        'atr_ratio':   atr_ratio,
        'vr':          vr,
        'corr':        corr,
        'trend':       trend,
        'rvol_30d':    rvol_30d,
        'spread_dist': spread_dist,
        'sig':         sig,
    }, index=df.index)

    return features, sig, z


# ═══════════════════════════════════════════════════════
# BACKTEST با feature ذخیره
# ═══════════════════════════════════════════════════════
def calc_pnl(direction, entry, exit_px, lot, qr):
    C = Config
    gross = direction * (exit_px - entry) * lot * C.lot_size * qr
    return gross - C.commission_per_lot * lot


def run_analysis(df, features, sig, z):
    C          = Config
    idx        = df.index.sort_values()
    pip        = C.PIP_SIZE['AUDNZD']
    spread     = C.PAIR_SPREAD['AUDNZD']

    o_   = df['o_spread'].reindex(idx).ffill().values.astype(float)
    c_   = df['c_spread'].reindex(idx).ffill().values.astype(float)
    qr_  = df['quote_rate'].reindex(idx).ffill().values.astype(float)
    sg_  = sig.reindex(idx).fillna(0).values.astype(int)
    zz_  = z.reindex(idx).fillna(np.nan).values.astype(float)

    feat_z     = features['z'].reindex(idx).fillna(np.nan).values
    feat_atr_r = features['atr_ratio'].reindex(idx).fillna(np.nan).values
    feat_vr    = features['vr'].reindex(idx).fillna(np.nan).values
    feat_corr  = features['corr'].reindex(idx).fillna(np.nan).values
    feat_trend = features['trend'].reindex(idx).fillna(np.nan).values
    feat_rvol  = features['rvol_30d'].reindex(idx).fillna(np.nan).values

    FLOOR  = C.initial_balance * (1 - C.max_total_dd_pct)
    TARGET = C.initial_balance * (1 + C.profit_target_pct)

    def new_acc(ts):
        return {'equity': C.initial_balance, 'start_ts': ts,
                'trades': [], 'blown': False, 'blown_rsn': '',
                'peak': C.initial_balance, 'consec_loss': 0}

    acc          = new_acc(idx[C.warmup])
    withdrawn    = 0.0
    acc_num      = 1
    day_eq       = C.initial_balance
    month_eq     = C.initial_balance
    cooldown_til = None
    records      = []

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
            cooldown_til = ts + pd.Timedelta(days=C.cooldown_days)
            acc_num += 1
            acc      = new_acc(ts)
            day_eq   = month_eq = acc['equity']
            day_trades = pending = 0
            pos = None
            continue

        if in_cd:
            continue

        m_stressed = (acc['equity'] - month_eq) < C.monthly_loss_threshold

        # open trade
        if pending != 0 and pos is None and day_trades < C.max_trades_day:
            sv   = pending
            risk = C.risk_base_pct * (0.5 if m_stressed else 1.0)
            if acc['consec_loss'] >= C.consec_loss_n:
                risk = max(risk * C.risk_reduce, C.risk_min_pct)
            pv  = pip * C.lot_size * qr_[bar]
            lot = round(float(np.clip(
                acc['equity'] * risk / (C.sl_pips * pv),
                C.min_lot, C.max_lot)), 2)
            ep  = o_[bar] + sv * (C.slippage_pips + spread / 2) * pip

            pos = {
                'dir':          sv,
                'lot':          lot,
                'lot_rem':      lot,
                'partial_done': False,
                'entry':        ep,
                'sl':           ep - sv * C.sl_pips * pip,
                'tp':           ep + sv * C.tp_pips * pip,
                'entry_ts':     ts,
                'entry_bar':    bar,
                'partial_pnl':  0.0,
                # features at entry
                'f_z':    feat_z[bar],
                'f_atr':  feat_atr_r[bar],
                'f_vr':   feat_vr[bar],
                'f_corr': feat_corr[bar],
                'f_trend':feat_trend[bar],
                'f_rvol': feat_rvol[bar],
                'f_hour': ts.hour,
                'f_dow':  ts.dayofweek,
                'f_year': ts.year,
                'f_month':ts.month,
            }
            day_trades += 1

        pending = 0

        flt = 0.0
        if pos is not None:
            flt = calc_pnl(pos['dir'], pos['entry'],
                           c_[bar], pos['lot_rem'], qr_[bar])

        cur_eq    = acc['equity'] + flt
        daily_lim = day_eq * (1 - C.max_daily_loss_pct)

        if cur_eq <= daily_lim or cur_eq <= FLOOR:
            acc['blown']     = True
            acc['blown_rsn'] = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            if pos is not None:
                pnl = calc_pnl(pos['dir'], pos['entry'],
                               c_[bar], pos['lot_rem'], qr_[bar])
                acc['equity'] += pnl
                records.append({
                    'year': pos['f_year'], 'month': pos['f_month'],
                    'hour': pos['f_hour'], 'dow': pos['f_dow'],
                    'z_abs': abs(pos['f_z']) if not np.isnan(pos['f_z']) else np.nan,
                    'atr_ratio': pos['f_atr'], 'vr': pos['f_vr'],
                    'corr': pos['f_corr'], 'trend': pos['f_trend'],
                    'rvol': pos['f_rvol'],
                    'total_pnl': pos['partial_pnl'] + pnl,
                    'status': 'BLOWN',
                    'bars': bar - pos['entry_bar'],
                    'exit_ts': ts,
                })
                pos = None
            continue

        if pos is not None:
            cp = c_[bar]
            d  = pos['dir']
            ep = pos['entry']
            zn = zz_[bar]
            lr = pos['lot_rem']

            # partial
            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -C.z_exit_partial) or
                        (d == -1 and zn <=  C.z_exit_partial)):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr_[bar])
                        if p_pnl > 0:
                            acc['equity']      += p_pnl
                            pos['partial_pnl'] += p_pnl
                            pos['lot_rem']      = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl']           = pos['entry']
                            lr = pos['lot_rem']
                            if lr < C.min_lot:
                                records.append({
                                    'year': pos['f_year'], 'month': pos['f_month'],
                                    'hour': pos['f_hour'], 'dow': pos['f_dow'],
                                    'z_abs': abs(pos['f_z']) if not np.isnan(pos['f_z']) else np.nan,
                                    'atr_ratio': pos['f_atr'], 'vr': pos['f_vr'],
                                    'corr': pos['f_corr'], 'trend': pos['f_trend'],
                                    'rvol': pos['f_rvol'],
                                    'total_pnl': pos['partial_pnl'],
                                    'status': 'PartOnly',
                                    'bars': bar - pos['entry_bar'],
                                    'exit_ts': ts,
                                })
                                pos = None

            if pos is not None:
                lr      = pos['lot_rem']
                pnl_now = calc_pnl(d, ep, cp, lr, qr_[bar])
                hit_zs  = (not np.isnan(zn) and
                           ((d==1 and zn<=-C.z_stop_margin) or
                            (d==-1 and zn>=C.z_stop_margin)))
                hit_ze  = (not np.isnan(zn) and
                           ((d==1 and zn>=-C.z_exit_full) or
                            (d==-1 and zn<=C.z_exit_full)))
                if hit_ze and pnl_now < C.min_net_profit_usd and not pos['partial_done']:
                    hit_ze = False
                hit_sl = ((d==1 and cp<=pos['sl']) or (d==-1 and cp>=pos['sl']))
                hit_tp = ((d==1 and cp>=pos['tp']) or (d==-1 and cp<=pos['tp']))

                if hit_ze or hit_zs or hit_sl or hit_tp:
                    xp   = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                    st   = ('SL' if hit_sl else 'TP' if hit_tp else
                            'Z-Stop' if hit_zs else 'Z-Exit')
                    fpnl = calc_pnl(d, ep, xp, lr, qr_[bar])
                    acc['equity'] += fpnl
                    records.append({
                        'year': pos['f_year'], 'month': pos['f_month'],
                        'hour': pos['f_hour'], 'dow': pos['f_dow'],
                        'z_abs': abs(pos['f_z']) if not np.isnan(pos['f_z']) else np.nan,
                        'atr_ratio': pos['f_atr'], 'vr': pos['f_vr'],
                        'corr': pos['f_corr'], 'trend': pos['f_trend'],
                        'rvol': pos['f_rvol'],
                        'total_pnl': pos['partial_pnl'] + fpnl,
                        'status': st,
                        'bars': bar - pos['entry_bar'],
                        'exit_ts': ts,
                    })
                    pos = None
                    if fpnl > 0: acc['consec_loss'] = 0
                    else:        acc['consec_loss'] += 1

        if acc['equity'] >= TARGET and pos is None:
            w = acc['equity'] - C.initial_balance
            withdrawn += w
            acc_num += 1
            acc      = new_acc(ts)
            day_eq   = month_eq = acc['equity']
            day_trades = pending = 0
            continue

        if (pos is None and not acc['blown'] and not in_cd
                and day_trades < C.max_trades_day and sg_[bar] != 0):
            pending = int(sg_[bar])

    return pd.DataFrame(records), withdrawn


# ═══════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════
def group_analysis(df, col, bins, label):
    """تحلیل گروهی با safe binning"""
    df2 = df.dropna(subset=[col]).copy()
    if not len(df2):
        return

    # ساختن bins یکتا و monotonic
    clean_bins = sorted(list(dict.fromkeys(
        [df2[col].min() - 0.001] + list(bins) + [df2[col].max() + 0.001]
    )))
    clean_bins = [b for i, b in enumerate(clean_bins)
                  if i == 0 or b > clean_bins[i-1]]

    if len(clean_bins) < 2:
        return

    try:
        df2['_bin'] = pd.cut(df2[col], bins=clean_bins, include_lowest=True)
    except Exception as e:
        print(f"    ⚠ {label}: {e}")
        return

    g = df2.groupby('_bin', observed=True).agg(
        n=('total_pnl', 'count'),
        wr=('total_pnl', lambda x: (x > 0).mean()),
        avg=('total_pnl', 'mean'),
        total=('total_pnl', 'sum'),
    )

    print(f"\n  {label}")
    for idx_v, row in g.iterrows():
        if row['n'] < 5:
            continue
        mark = '✅' if row['avg'] > 0 else '❌'
        bar_ = '█' * min(int(abs(row['total']) / 300), 18)
        sign = '+' if row['total'] >= 0 else '-'
        print(f"    {mark} {str(idx_v):<22}  n={int(row['n']):>4}  "
              f"WR={row['wr']*100:4.0f}%  "
              f"avg=${row['avg']:>7.2f}  "
              f"{sign}${abs(row['total']):>7,.0f}  {bar_}")

    return g


def analyze_regimes(df):
    if not len(df):
        print("No data")
        return

    df = df.copy()
    df['win'] = df['total_pnl'] > 0
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])

    print("\n" + "═"*72)
    print("  REGIME ANALYSIS")
    print("═"*72)

    # 1. Variance Ratio
    group_analysis(df, 'vr',
                   [0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
                   "1️⃣  Variance Ratio — هرچه کمتر → mean-reversion قوی‌تر")

    # 2. ATR Ratio
    group_analysis(df, 'atr_ratio',
                   [0, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0],
                   "2️⃣  ATR Ratio — نوسان نسبی")

    # 3. Correlation
    group_analysis(df, 'corr',
                   [0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 1.0],
                   "3️⃣  Correlation AUD/NZD")

    # 4. Realized Vol — با qcut به جای cut
    print("\n  4️⃣  Realized Volatility (30d) — quartile")
    df2 = df.dropna(subset=['rvol']).copy()
    if len(df2):
        try:
            df2['rv_q'] = pd.qcut(df2['rvol'], q=4,
                                  labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High'],
                                  duplicates='drop')
            g = df2.groupby('rv_q', observed=True).agg(
                n=('total_pnl', 'count'),
                wr=('total_pnl', lambda x: (x > 0).mean()),
                avg=('total_pnl', 'mean'),
                total=('total_pnl', 'sum'),
                rv_mean=('rvol', 'mean'),
            )
            for idx_v, row in g.iterrows():
                mark = '✅' if row['avg'] > 0 else '❌'
                print(f"    {mark} {str(idx_v):<10}  "
                      f"rv_avg={row['rv_mean']:.5f}  "
                      f"n={int(row['n']):>4}  "
                      f"WR={row['wr']*100:4.0f}%  "
                      f"avg=${row['avg']:>7.2f}  "
                      f"total=${row['total']:>8,.0f}")
        except Exception as e:
            print(f"    ⚠ {e}")

    # 5. Trend strength
    df['trend_abs'] = df['trend'].abs()
    group_analysis(df, 'trend_abs',
                   [0, 0.05, 0.10, 0.20, 0.40, 1.0, 5.0],
                   "5️⃣  Trend Strength |MA48-MA200| — هرچه بیشتر → trend قوی‌تر")

    # 6. Z-score at entry
    group_analysis(df, 'z_abs',
                   [2.0, 2.3, 2.6, 3.0, 3.5, 4.0, 6.0],
                   "6️⃣  |Z| at Entry")

    # 7. Hour
    print("\n  7️⃣  Session Hour (UTC)")
    g = df.groupby('hour').agg(
        n=('total_pnl', 'count'),
        wr=('total_pnl', lambda x: (x > 0).mean()),
        avg=('total_pnl', 'mean'),
        total=('total_pnl', 'sum'),
    )
    for h, row in g.iterrows():
        if row['n'] < 8:
            continue
        mark = '✅' if row['avg'] > 0 else '❌'
        bar_ = '█' * min(int(abs(row['total']) / 150), 20)
        sign = '+' if row['total'] >= 0 else '-'
        print(f"    {mark} H{int(h):02d}: n={int(row['n']):>4}  "
              f"WR={row['wr']*100:4.0f}%  "
              f"avg=${row['avg']:>6.2f}  "
              f"{sign}${abs(row['total']):>7,.0f}  {bar_}")

    # 8. Day of week
    dow_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'}
    print("\n  8️⃣  Day of Week")
    g = df.groupby('dow').agg(
        n=('total_pnl', 'count'),
        wr=('total_pnl', lambda x: (x > 0).mean()),
        avg=('total_pnl', 'mean'),
        total=('total_pnl', 'sum'),
    )
    for d, row in g.iterrows():
        mark = '✅' if row['avg'] > 0 else '❌'
        print(f"    {mark} {dow_names.get(d, d)}: n={int(row['n']):>4}  "
              f"WR={row['wr']*100:4.0f}%  "
              f"avg=${row['avg']:>6.2f}  "
              f"total=${row['total']:>8,.0f}")

    # 9. Yearly با feature میانگین
    print("\n  9️⃣  Yearly + Avg Features")
    g = df.groupby('year').agg(
        n=('total_pnl', 'count'),
        wr=('total_pnl', lambda x: (x > 0).mean()),
        avg=('total_pnl', 'mean'),
        total=('total_pnl', 'sum'),
        avg_vr=('vr', 'mean'),
        avg_rvol=('rvol', 'mean'),
        avg_corr=('corr', 'mean'),
        avg_trend=('trend_abs', 'mean'),
    )
    print(f"    {'Yr':>4}  {'n':>4}  {'WR':>5}  {'Avg':>7}  {'Total':>8}  "
          f"{'VR':>6}  {'RVol':>8}  {'Corr':>6}  {'Trend':>7}")
    print("    " + "─"*75)
    for yr, row in g.iterrows():
        mark = '✅' if row['total'] > 0 else '❌'
        print(f"    {mark}{int(yr)}: {int(row['n']):>4}  "
              f"WR:{row['wr']*100:4.0f}%  "
              f"${row['avg']:>6.2f}  "
              f"${row['total']:>8,.0f}  "
              f"VR:{row['avg_vr']:.3f}  "
              f"RV:{row['avg_rvol']:.5f}  "
              f"Cr:{row['avg_corr']:.3f}  "
              f"Tr:{row['avg_trend']:.3f}")

    # ── نتیجه‌گیری خودکار ──
    print("\n" + "═"*72)
    print("  📊 نتیجه‌گیری خودکار:")

    # بهترین VR range
    df2 = df.dropna(subset=['vr']).copy()
    df2['_vr_bin'] = pd.cut(df2['vr'],
                             bins=[0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
                             include_lowest=True)
    vr_g = df2.groupby('_vr_bin', observed=True)['total_pnl'].mean()
    good_vr = vr_g[vr_g > 0]
    if len(good_vr):
        print(f"  ✅ VR: سودده در {good_vr.index[0].left:.2f} – {good_vr.index[-1].right:.2f}")
        print(f"     → فیلتر پیشنهادی: vr_max = {good_vr.index[-1].right:.2f}")

    # بهترین RVol
    if len(df2):
        try:
            df2['_rvq'] = pd.qcut(df2['rvol'], q=4, duplicates='drop')
            rv_g = df2.groupby('_rvq', observed=True).agg(
                avg=('total_pnl', 'mean'), rv=('rvol', 'mean'))
            good_rv = rv_g[rv_g['avg'] > 0]
            if len(good_rv):
                print(f"  ✅ RVol: سودده در quartile‌های: {list(good_rv.index)}")
                rv_vals = [rv_g.loc[i, 'rv'] for i in good_rv.index]
                print(f"     avg_rvol در quartile‌های خوب: {[f'{v:.5f}' for v in rv_vals]}")
        except Exception:
            pass

    # بهترین ساعت‌ها
    g2 = df.groupby('hour').agg(avg=('total_pnl', 'mean'), n=('total_pnl', 'count'))
    good_h = g2[(g2['avg'] > 0) & (g2['n'] >= 8)].index.tolist()
    if good_h:
        print(f"  ✅ ساعت‌های سودده: {good_h}")
        bad_h = g2[(g2['avg'] < 0) & (g2['n'] >= 8)].index.tolist()
        if bad_h:
            print(f"  ❌ ساعت‌های زیان‌ده: {bad_h}")

    # سال‌های بد — VR چقدر بوده؟
    bad_yrs = g[g['total'] < 0]
    good_yrs = g[g['total'] >= 0]
    if len(bad_yrs) and len(good_yrs):
        print(f"\n  مقایسه سال‌های خوب vs بد:")
        print(f"  {'':10} {'VR':>6}  {'RVol':>8}  {'Corr':>6}  {'Trend':>7}")
        print(f"  {'Good years':10} "
              f"VR:{good_yrs['avg_vr'].mean():.3f}  "
              f"RV:{good_yrs['avg_rvol'].mean():.5f}  "
              f"Cr:{good_yrs['avg_corr'].mean():.3f}  "
              f"Tr:{good_yrs['avg_trend'].mean():.3f}")
        print(f"  {'Bad years':10} "
              f"VR:{bad_yrs['avg_vr'].mean():.3f}  "
              f"RV:{bad_yrs['avg_rvol'].mean():.5f}  "
              f"Cr:{bad_yrs['avg_corr'].mean():.3f}  "
              f"Tr:{bad_yrs['avg_trend'].mean():.3f}")

    print("═"*72)

    # ذخیره
    try:
        df.to_csv('regime_trades.csv', index=False)
        print("  📊 regime_trades.csv saved")
    except Exception as e:
        print(f"  ⚠ {e}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   CorrArb v9h — Regime Analysis (Fixed)             ║")
    print("╚══════════════════════════════════════════════════════╝")

    df               = load_audnzd()
    features, sig, z = build_features(df)
    print(f"  Signals: {(sig != 0).sum():,}")

    df_trades, withdrawn = run_analysis(df, features, sig, z)
    print(f"  Trades: {len(df_trades):,} | Withdrawn: ${withdrawn:,.2f}")

    analyze_regimes(df_trades)

    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
