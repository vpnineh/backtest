"""
CorrArb v9f — Z-Exit-Partial Sweep
=====================================
Baseline:
  No TimeStop | SL=30 | Partial=0.75
Sweep:
  z_exit_partial = 0.25 / 0.50 / 0.75 / 1.00 / 1.25
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
    z_exit_partial     = 0.50   # ← sweep این
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0

    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    sl_pips            = 30.0   # fixed
    tp_pips            = 90.0
    partial_ratio      = 0.75   # fixed from v9e
    use_time_stop      = False  # fixed

    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.90

    cooldown_days          = 10
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
    C      = Config
    log_r  = np.log(df['c_spread'])
    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    corr_ok = (df['c_aud'].pct_change()
               .rolling(C.corr_period)
               .corr(df['c_nzd'].pct_change()) > C.corr_min)

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
def calc_pnl(direction, entry, exit_px, lot, qr):
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


def run_backtest(df, sig, z, verbose=False):
    C          = Config
    idx        = df.index.sort_values()
    start_date = idx[C.warmup]
    pip        = C.PIP_SIZE['AUDNZD']
    spread     = C.PAIR_SPREAD['AUDNZD']

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
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   acc['blown_rsn'],
                'pnl':      acc['equity'] - C.initial_balance,
                'n_trades': len(acc['trades']),
                'days':     (ts - acc['start_ts']).days,
            })
            if verbose:
                print(f"    💥 #{acc_num:>3} | {ts.date()} | "
                      f"Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            cooldown_til = ts + pd.Timedelta(days=C.cooldown_days)
            acc_num   += 1
            acc        = new_acc(ts)
            day_eq     = month_eq = acc['equity']
            day_trades = 0
            pending    = 0
            pos        = None
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
            rsn              = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            acc['blown']     = True
            acc['blown_rsn'] = rsn
            if pos is not None:
                pnl = calc_pnl(pos['dir'], pos['entry'],
                               c_[bar], pos['lot_rem'], qr_[bar])
                acc['equity'] += pnl
                all_trades.append({
                    'dir': pos['dir'], 'lot': pos['lot_rem'],
                    'entry': pos['entry'], 'exit': c_[bar],
                    'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                    'pnl': pnl, 'status': 'BLOWN',
                })
                acc['trades'].append(all_trades[-1])
                pos = None
            continue

        if pos is not None:
            cp = c_[bar]
            d  = pos['dir']
            ep = pos['entry']
            zn = zz_[bar]
            lr = pos['lot_rem']

            # partial exit
            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -C.z_exit_partial) or
                        (d == -1 and zn <=  C.z_exit_partial)):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr_[bar])
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            all_trades.append({
                                'dir': d, 'lot': p_lot,
                                'entry': ep, 'exit': cp,
                                'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                                'pnl': p_pnl, 'status': 'Partial',
                            })
                            acc['trades'].append(all_trades[-1])
                            pos['lot_rem']      = round(lr - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl']           = pos['entry']
                            lr = pos['lot_rem']
                            if lr < C.min_lot:
                                pos = None

            if pos is not None:
                lr      = pos['lot_rem']
                pnl_now = calc_pnl(d, ep, cp, lr, qr_[bar])

                hit_zs = (not np.isnan(zn) and
                          ((d == 1 and zn <= -C.z_stop_margin) or
                           (d == -1 and zn >=  C.z_stop_margin)))

                hit_ze = (not np.isnan(zn) and
                          ((d == 1 and zn >= -C.z_exit_full) or
                           (d == -1 and zn <=  C.z_exit_full)))
                if hit_ze and pnl_now < C.min_net_profit_usd and not pos['partial_done']:
                    hit_ze = False

                hit_sl = ((d == 1 and cp <= pos['sl']) or
                          (d == -1 and cp >= pos['sl']))
                hit_tp = ((d == 1 and cp >= pos['tp']) or
                          (d == -1 and cp <= pos['tp']))

                if hit_ze or hit_zs or hit_sl or hit_tp:
                    xp = (pos['sl'] if hit_sl else
                          pos['tp'] if hit_tp else cp)
                    st = ('SL'     if hit_sl else
                          'TP'     if hit_tp else
                          'Z-Stop' if hit_zs else 'Z-Exit')
                    fpnl = calc_pnl(d, ep, xp, lr, qr_[bar])
                    acc['equity'] += fpnl
                    all_trades.append({
                        'dir': d, 'lot': lr,
                        'entry': ep, 'exit': xp,
                        'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                        'pnl': fpnl, 'status': st,
                    })
                    acc['trades'].append(all_trades[-1])
                    pos = None

                    if fpnl > 0:
                        acc['consec_loss'] = 0
                    else:
                        acc['consec_loss'] += 1

        if acc['equity'] >= TARGET and pos is None:
            w  = acc['equity'] - C.initial_balance
            withdrawn += w
            dt = (ts - acc['start_ts']).days
            nt = len(acc['trades'])
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   'TARGET_HIT',
                'pnl':      w,
                'n_trades': nt,
                'days':     dt,
            })
            if verbose:
                print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | "
                      f"Bank:${withdrawn:>9.2f} | {dt}d | {nt}T")
            acc_num   += 1
            acc        = new_acc(ts)
            day_eq     = month_eq = acc['equity']
            day_trades = 0
            pending    = 0
            continue

        if (pos is None and not acc['blown'] and not in_cd
                and day_trades < C.max_trades_day
                and sg_[bar] != 0):
            pending = int(sg_[bar])

    return {
        'all_trades':   all_trades,
        'account_logs': acc_logs,
        'withdrawn':    withdrawn,
        'final_equity': acc['equity'],
        'common_idx':   idx,
    }


# ═══════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════
def run_sweep(df, sig, z):
    values  = [0.25, 0.50, 0.75, 1.00, 1.25]
    summary = []

    print("\n╔" + "═"*62 + "╗")
    print("║" + "  Z-EXIT-PARTIAL SWEEP | SL=30 | Partial=0.75".center(62) + "║")
    print("╚" + "═"*62 + "╝")

    for zep in values:
        Config.z_exit_partial = zep
        res = run_backtest(df, sig, z, verbose=False)

        dft = pd.DataFrame(res['all_trades'])
        if not len(dft):
            continue

        dft['exit_ts'] = pd.to_datetime(dft['exit_ts'])
        dft['month']   = dft['exit_ts'].dt.to_period('M')

        wins   = dft[dft['pnl'] > 0]['pnl']
        losses = dft[dft['pnl'] < 0]['pnl']
        pf     = wins.sum() / abs(losses.sum()) if len(losses) else 99.0

        ci = res['common_idx']
        all_months = pd.period_range(
            start=ci[Config.warmup].to_period('M'),
            end=ci[-1].to_period('M'), freq='M')
        monthly = (dft.groupby('month')['pnl'].sum()
                   .reindex(all_months, fill_value=0.0))

        logs   = res['account_logs']
        df_acc = pd.DataFrame(logs) if logs else pd.DataFrame()
        n_pass = int((df_acc['reason'] == 'TARGET_HIT').sum()) if len(df_acc) else 0
        n_blow = int((df_acc['reason'] != 'TARGET_HIT').sum()) if len(df_acc) else 0

        p_df  = dft[dft['status'] == 'Partial']
        ze_df = dft[dft['status'] == 'Z-Exit']
        sl_df = dft[dft['status'] == 'SL']

        # سال‌های منفی
        neg_years = int((dft.groupby(dft['exit_ts'].dt.year)['pnl'].sum() < 0).sum())

        summary.append({
            'ZExitPartial': zep,
            'Trades':       len(dft),
            'WR%':          round((dft['pnl'] > 0).mean() * 100, 1),
            'PF':           round(pf, 3),
            'Banked':       round(res['withdrawn'], 0),
            'NetPnL':       round(dft['pnl'].sum(), 0),
            '+Mo':          int((monthly > 0).sum()),
            '-Mo':          int((monthly < 0).sum()),
            'MonAvg':       round(monthly.mean(), 2),
            'Pass':         n_pass,
            'Blow':         n_blow,
            'NegYears':     neg_years,
            'PartCnt':      len(p_df),
            'PartAvg':      round(p_df['pnl'].mean(), 2) if len(p_df) else 0,
            'ZExitAvg':     round(ze_df['pnl'].mean(), 2) if len(ze_df) else 0,
            'SLAvg':        round(sl_df['pnl'].mean(), 2) if len(sl_df) else 0,
        })

        print(f"  z_exit_partial={zep:.2f} | PF={pf:.3f} | "
              f"Bank=${res['withdrawn']:,.0f} | "
              f"+Mo={int((monthly>0).sum())} | "
              f"Pass={n_pass} Blow={n_blow}")

    # ── جدول نهایی ──
    print("\n\n╔" + "═"*78 + "╗")
    print("║" + "  FINAL COMPARISON".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    df_s = pd.DataFrame(summary)
    bp   = df_s['PF'].idxmax()
    bb   = df_s['Banked'].idxmax()
    bm   = df_s['+Mo'].idxmax()
    bn   = df_s['NetPnL'].idxmax()

    print(f"\n {'ZEP':>5}  {'Tr':>5}  {'WR':>5}  {'PF':>5}  "
          f"{'Banked':>8}  {'Net':>8}  {'+Mo':>4}  {'-Mo':>4}  "
          f"{'MonAvg':>7}  {'Pass':>4}  {'Blow':>4}  {'NegYr':>6}  "
          f"{'PrtAvg':>7}  {'ZEAvg':>7}")
    print("  " + "─"*105)

    for i, r in df_s.iterrows():
        flags = []
        if i == bp: flags.append('◀PF')
        if i == bb: flags.append('◀Bank')
        if i == bm: flags.append('◀+Mo')
        if i == bn: flags.append('◀Net')

        print(f" {r['ZExitPartial']:>5.2f}  {int(r['Trades']):>5}  "
              f"{r['WR%']:>5.1f}  {r['PF']:>5.3f}  "
              f"{r['Banked']:>8,.0f}  {r['NetPnL']:>8,.0f}  "
              f"{r['+Mo']:>4}  {r['-Mo']:>4}  "
              f"{r['MonAvg']:>7.2f}  {r['Pass']:>4}  {r['Blow']:>4}  "
              f"{r['NegYears']:>6}  "
              f"{r['PartAvg']:>7.2f}  {r['ZExitAvg']:>7.2f}  "
              f"{'  '.join(flags)}")

    print("\n" + "─"*78)
    print(f" ✅ Best PF:     z_exit_partial = {df_s.loc[bp,'ZExitPartial']:.2f}")
    print(f" ✅ Best Bank:   z_exit_partial = {df_s.loc[bb,'ZExitPartial']:.2f}")
    print(f" ✅ Best +Month: z_exit_partial = {df_s.loc[bm,'ZExitPartial']:.2f}")
    print(f" ✅ Best Net:    z_exit_partial = {df_s.loc[bn,'ZExitPartial']:.2f}")
    print("─"*78)
    print(" توضیح:")
    print(" ZEP    = z_exit_partial (هرچه بزرگتر = partial دیرتر)")
    print(" NegYr  = سال‌های منفی از 16 سال")
    print(" PrtAvg = میانگین سود هر partial exit")
    print(" ZEAvg  = میانگین سود هر Z-Exit")
    print("─"*78)

    return df_s


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  CorrArb v9f — Z-Exit-Partial Sweep                ║")
    print("║  Baseline: No TimeStop | SL=30 | Partial=0.75      ║")
    print("╚══════════════════════════════════════════════════════╝")

    df     = load_audnzd()
    sig, z = compute_signals(df)
    print(f"  Signals: {(sig != 0).sum():,}")

    summary = run_sweep(df, sig, z)

    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
