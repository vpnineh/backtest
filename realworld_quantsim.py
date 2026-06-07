"""
v9h — Regime Analysis
======================
هدف: بفهمیم سال‌های بد چه ویژگی‌ای دارند
تا بتوانیم فیلتر بزنیم
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


def build_features(df):
    """
    ساخت feature های مختلف برای هر بار
    تا بفهمیم در بارهای سودده vs زیان‌ده چه تفاوتی وجود دارد
    """
    C     = Config
    log_r = np.log(df['c_spread'])

    # Z-score
    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    # ATR و نسبت ATR
    atr    = calc_atr(df['h_spread'], df['l_spread'],
                      df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    atr_ratio = atr / atr_ma.replace(0, np.nan)

    # Variance Ratio
    vr = calc_vr(log_r, C.vr_k, C.vr_period)

    # Correlation
    corr = (df['c_aud'].pct_change()
            .rolling(C.corr_period)
            .corr(df['c_nzd'].pct_change()))

    # Trend strength — میانگین متحرک spread
    ma_fast = df['c_spread'].rolling(48).mean()
    ma_slow = df['c_spread'].rolling(200).mean()
    trend   = (ma_fast - ma_slow) / ma_slow.replace(0, np.nan) * 100

    # Realized volatility — std بازده روزانه
    daily_ret = log_r.resample('D').last().diff()
    rvol_30d  = daily_ret.rolling(30).std()
    rvol_30d  = rvol_30d.reindex(df.index, method='ffill')

    # Session
    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # Signal
    sig  = pd.Series(0, index=df.index)
    cond = (atr_ratio > C.atr_min_mult) & (atr_ratio < C.atr_max_mult) & time_ok & (corr > C.corr_min) & (vr < C.vr_max)
    sig[(z < -C.z_entry) & cond] =  1
    sig[(z >  C.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    features = pd.DataFrame({
        'z':        z,
        'atr':      atr,
        'atr_ratio': atr_ratio,
        'vr':       vr,
        'corr':     corr,
        'trend':    trend,
        'rvol_30d': rvol_30d,
        'sig':      sig,
    }, index=df.index)

    return features, sig, z


def calc_pnl(direction, entry, exit_px, lot, qr):
    C     = Config
    gross = direction * (exit_px - entry) * lot * C.lot_size * qr
    return gross - C.commission_per_lot * lot


def run_analysis(df, features, sig, z):
    """
    اجرای backtest کامل و ذخیره feature های هر ترید
    """
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

    acc          = {'equity': C.initial_balance, 'start_ts': idx[C.warmup],
                    'trades': [], 'blown': False, 'blown_rsn': '',
                    'peak': C.initial_balance, 'consec_loss': 0}
    withdrawn    = 0.0
    acc_num      = 1
    day_eq       = C.initial_balance
    month_eq     = C.initial_balance
    cooldown_til = None
    trade_records = []

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
            acc = {'equity': C.initial_balance, 'start_ts': ts,
                   'trades': [], 'blown': False, 'blown_rsn': '',
                   'peak': C.initial_balance, 'consec_loss': 0}
            day_eq = month_eq = acc['equity']
            day_trades = pending = 0
            pos = None
            continue

        if in_cd:
            continue

        m_stressed = (acc['equity'] - month_eq) < C.monthly_loss_threshold

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

            # ذخیره feature های لحظه ورود
            entry_features = {
                'z_entry':    feat_z[bar],
                'atr_ratio':  feat_atr_r[bar],
                'vr_entry':   feat_vr[bar],
                'corr_entry': feat_corr[bar],
                'trend':      feat_trend[bar],
                'rvol_30d':   feat_rvol[bar],
                'hour':       ts.hour,
                'dow':        ts.dayofweek,
                'year':       ts.year,
                'month':      ts.month,
            }
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
                'entry_feat':   entry_features,
                'partial_pnl':  0.0,
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
            acc['blown'] = True
            acc['blown_rsn'] = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            if pos is not None:
                pnl = calc_pnl(pos['dir'], pos['entry'],
                               c_[bar], pos['lot_rem'], qr_[bar])
                acc['equity'] += pnl
                rec = {**pos['entry_feat'],
                       'total_pnl': pos['partial_pnl'] + pnl,
                       'status': 'BLOWN', 'exit_ts': ts,
                       'bars_held': bar - pos['entry_bar']}
                trade_records.append(rec)
                pos = None
            continue

        if pos is not None:
            cp = c_[bar]
            d  = pos['dir']
            ep = pos['entry']
            zn = zz_[bar]
            lr = pos['lot_rem']

            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -C.z_exit_partial) or
                        (d == -1 and zn <=  C.z_exit_partial)):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr_[bar])
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            pos['partial_pnl'] += p_pnl
                            pos['lot_rem']      = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl']           = pos['entry']
                            lr = pos['lot_rem']
                            if lr < C.min_lot:
                                rec = {**pos['entry_feat'],
                                       'total_pnl': pos['partial_pnl'],
                                       'status': 'PartialOnly', 'exit_ts': ts,
                                       'bars_held': bar - pos['entry_bar']}
                                trade_records.append(rec)
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
                    rec = {**pos['entry_feat'],
                           'total_pnl': pos['partial_pnl'] + fpnl,
                           'status': st, 'exit_ts': ts,
                           'bars_held': bar - pos['entry_bar']}
                    trade_records.append(rec)
                    pos = None
                    if fpnl > 0: acc['consec_loss'] = 0
                    else:        acc['consec_loss'] += 1

        if acc['equity'] >= TARGET and pos is None:
            w = acc['equity'] - C.initial_balance
            withdrawn += w
            acc_num += 1
            acc = {'equity': C.initial_balance, 'start_ts': ts,
                   'trades': [], 'blown': False, 'blown_rsn': '',
                   'peak': C.initial_balance, 'consec_loss': 0}
            day_eq = month_eq = acc['equity']
            day_trades = pending = 0
            continue

        if (pos is None and not acc['blown'] and not in_cd
                and day_trades < C.max_trades_day and sg_[bar] != 0):
            pending = int(sg_[bar])

    return pd.DataFrame(trade_records), withdrawn


def analyze_regimes(df_trades):
    """
    تحلیل: در چه شرایطی تریدها سودده/زیان‌ده هستند
    """
    if not len(df_trades):
        print("No trades")
        return

    df = df_trades.copy()
    df['win'] = df['total_pnl'] > 0
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])

    print("\n" + "═"*70)
    print("  REGIME ANALYSIS — چه چیزی سودده/زیان‌ده را تفکیک می‌کند؟")
    print("═"*70)

    # ── 1. تحلیل بر اساس VR ──
    print("\n  1️⃣  Variance Ratio (vr_entry)")
    print("     هرچه VR کمتر → mean-reversion قوی‌تر")
    bins_vr = [0, 0.5, 0.65, 0.75, 0.85, 0.90, 1.0]
    df['vr_bin'] = pd.cut(df['vr_entry'], bins=bins_vr)
    g = df.groupby('vr_bin', observed=True).agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum')
    )
    for idx_v, row in g.iterrows():
        mark = '✅' if row['avg_pnl'] > 0 else '❌'
        print(f"    {mark} VR {idx_v}: n={int(row['count']):>4}  "
              f"WR={row['wr']*100:.0f}%  "
              f"avg=${row['avg_pnl']:>7.2f}  "
              f"total=${row['total']:>8,.0f}")

    # ── 2. تحلیل بر اساس ATR Ratio ──
    print("\n  2️⃣  ATR Ratio (volatility)")
    print("     هرچه بیشتر → بازار پرنوسان‌تر")
    bins_atr = [0, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0]
    df['atr_bin'] = pd.cut(df['atr_ratio'], bins=bins_atr)
    g = df.groupby('atr_bin', observed=True).agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum')
    )
    for idx_v, row in g.iterrows():
        mark = '✅' if row['avg_pnl'] > 0 else '❌'
        print(f"    {mark} ATR {idx_v}: n={int(row['count']):>4}  "
              f"WR={row['wr']*100:.0f}%  "
              f"avg=${row['avg_pnl']:>7.2f}  "
              f"total=${row['total']:>8,.0f}")

    # ── 3. تحلیل بر اساس Correlation ──
    print("\n  3️⃣  Correlation AUD/NZD")
    bins_corr = [0.8, 0.85, 0.90, 0.95, 1.0]
    df['corr_bin'] = pd.cut(df['corr_entry'], bins=bins_corr)
    g = df.groupby('corr_bin', observed=True).agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum')
    )
    for idx_v, row in g.iterrows():
        mark = '✅' if row['avg_pnl'] > 0 else '❌'
        print(f"    {mark} Corr {idx_v}: n={int(row['count']):>4}  "
              f"WR={row['wr']*100:.0f}%  "
              f"avg=${row['avg_pnl']:>7.2f}  "
              f"total=${row['total']:>8,.0f}")

    # ── 4. تحلیل بر اساس Realized Volatility ──
    print("\n  4️⃣  Realized Volatility (30d)")
    print("     نوسان‌پذیری 30 روز اخیر")
    rv_pct = df['rvol_30d'].quantile([0.25, 0.5, 0.75])
    bins_rv = [0, rv_pct[0.25], rv_pct[0.5], rv_pct[0.75], df['rvol_30d'].max() + 0.001]
    df['rv_bin'] = pd.cut(df['rvol_30d'], bins=bins_rv,
                          labels=['Q1-Low', 'Q2', 'Q3', 'Q4-High'])
    g = df.groupby('rv_bin', observed=True).agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum')
    )
    for idx_v, row in g.iterrows():
        mark = '✅' if row['avg_pnl'] > 0 else '❌'
        print(f"    {mark} RVol {idx_v}: n={int(row['count']):>4}  "
              f"WR={row['wr']*100:.0f}%  "
              f"avg=${row['avg_pnl']:>7.2f}  "
              f"total=${row['total']:>8,.0f}")

    # ── 5. تحلیل بر اساس Trend ──
    print("\n  5️⃣  Trend Strength (MA48 vs MA200)")
    print("     هرچه بزرگتر → trend قوی‌تر (mean reversion ضعیف‌تر)")
    tr_pct = df['trend'].abs().quantile([0.33, 0.67])
    df['trend_abs'] = df['trend'].abs()
    df['trend_bin'] = pd.cut(df['trend_abs'],
                             bins=[0, tr_pct[0.33], tr_pct[0.67], df['trend_abs'].max()+0.001],
                             labels=['Low', 'Mid', 'High'])
    g = df.groupby('trend_bin', observed=True).agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum')
    )
    for idx_v, row in g.iterrows():
        mark = '✅' if row['avg_pnl'] > 0 else '❌'
        print(f"    {mark} Trend {idx_v}: n={int(row['count']):>4}  "
              f"WR={row['wr']*100:.0f}%  "
              f"avg=${row['avg_pnl']:>7.2f}  "
              f"total=${row['total']:>8,.0f}")

    # ── 6. تحلیل بر اساس ساعت ──
    print("\n  6️⃣  Session (Hour UTC)")
    g = df.groupby('hour').agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum')
    )
    for h, row in g.iterrows():
        if row['count'] < 10: continue
        mark = '✅' if row['avg_pnl'] > 0 else '❌'
        bar_  = '█' * min(int(abs(row['total']) / 200), 15)
        print(f"    {mark} H{h:02d}: n={int(row['count']):>4}  "
              f"WR={row['wr']*100:.0f}%  "
              f"avg=${row['avg_pnl']:>6.2f}  "
              f"{bar_}")

    # ── 7. تحلیل بر اساس سال ──
    print("\n  7️⃣  Yearly Performance")
    df['year'] = df['exit_ts'].dt.year
    g = df.groupby('year').agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum'),
        avg_vr=('vr_entry', 'mean'),
        avg_rv=('rvol_30d', 'mean'),
        avg_corr=('corr_entry', 'mean'),
    )
    print(f"    {'Year':>4}  {'n':>4}  {'WR':>5}  {'AvgPnL':>7}  {'Total':>8}  {'AvgVR':>6}  {'AvgRV':>8}  {'AvgCorr':>8}")
    print("    " + "─"*70)
    for yr, row in g.iterrows():
        mark = '✅' if row['total'] > 0 else '❌'
        print(f"    {mark}{yr}: {int(row['count']):>4}  "
              f"WR:{row['wr']*100:4.0f}%  "
              f"avg:${row['avg_pnl']:>6.2f}  "
              f"tot:${row['total']:>8,.0f}  "
              f"VR:{row['avg_vr']:.3f}  "
              f"RV:{row['avg_rv']:.5f}  "
              f"Corr:{row['avg_corr']:.3f}")

    # ── 8. Z-score at entry ──
    print("\n  8️⃣  |Z| at entry")
    print("     هرچه بزرگتر → spread بیشتر از mean فاصله داشته")
    df['z_abs'] = df['z_entry'].abs()
    bins_z = [2.1, 2.5, 3.0, 3.5, 4.0, 6.0]
    df['z_bin'] = pd.cut(df['z_abs'], bins=bins_z)
    g = df.groupby('z_bin', observed=True).agg(
        count=('total_pnl', 'count'),
        wr=('win', 'mean'),
        avg_pnl=('total_pnl', 'mean'),
        total=('total_pnl', 'sum')
    )
    for idx_v, row in g.iterrows():
        mark = '✅' if row['avg_pnl'] > 0 else '❌'
        print(f"    {mark} |Z| {idx_v}: n={int(row['count']):>4}  "
              f"WR={row['wr']*100:.0f}%  "
              f"avg=${row['avg_pnl']:>7.2f}  "
              f"total=${row['total']:>8,.0f}")

    print("\n" + "═"*70)
    print("  نتیجه‌گیری:")
    print("  فیلترهای بالقوه بر اساس آنالیز فوق:")

    # خودکار بهترین VR را پیدا کن
    bins_vr2 = [0, 0.5, 0.65, 0.75, 0.85, 0.90, 1.0]
    df['vr_bin2'] = pd.cut(df['vr_entry'], bins=bins_vr2)
    vr_g = df.groupby('vr_bin2', observed=True)['total_pnl'].mean()
    best_vr = vr_g[vr_g > 0]
    if len(best_vr):
        print(f"  ✅ VR بهتر از: {best_vr.index[0].left:.2f} تا {best_vr.index[-1].right:.2f}")

    # بهترین RV
    rv_g = df.groupby('rv_bin', observed=True)['total_pnl'].mean()
    best_rv = rv_g[rv_g > 0]
    if len(best_rv):
        print(f"  ✅ RVol مناسب: {list(best_rv.index)}")

    # بهترین ساعت
    good_hours = g[(g['avg_pnl'] > 0) & (g['count'] >= 10)].index.tolist() if 'avg_pnl' in g.columns else []

    print("═"*70)

    # ذخیره
    df_trades.to_csv('trades_with_features.csv', index=False)
    print("\n  📊 trades_with_features.csv saved")

    return df


if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   CorrArb v9h — Regime Analysis                     ║")
    print("╚══════════════════════════════════════════════════════╝")

    df            = load_audnzd()
    features, sig, z = build_features(df)
    print(f"  Signals: {(sig != 0).sum():,}")

    df_trades, withdrawn = run_analysis(df, features, sig, z)
    print(f"  Total trades: {len(df_trades):,} | Withdrawn: ${withdrawn:,.2f}")

    analyze_regimes(df_trades)

    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
