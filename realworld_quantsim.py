"""
CorrArb v9c Audit
=================
هدف:
  - برگشت به منطق v9b
  - فقط AUDNZD
  - بدون Pair Quality
  - بدون تغییر در edge
  - فقط گزارش و آنالیز بهتر
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

    risk_base_pct      = 0.015
    risk_min_pct       = 0.005
    consec_loss_n      = 2
    risk_reduce        = 0.5

    PAIR_SPREAD = {'AUDNZD': 2.5}
    PIP_SIZE    = {'AUDNZD': 0.0001}

    commission_per_lot = 7.0
    slippage_pips      = 0.5

    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.5
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
    time_stop_bars     = 36
    partial_ratio      = 0.50

    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    vr_period = 200
    vr_k      = 4
    vr_max    = 0.90

    cooldown_days = 10
    monthly_loss_threshold = -150.0


def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No ZIP: {pattern}")
    frames = []
    for p in paths:
        with zipfile.ZipFile(p) as z:
            csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
            if not csv_name:
                continue
            with z.open(csv_name) as f:
                frames.append(pd.read_csv(f, sep=';', header=None,
                                          names=['ts','o','h','l','c','v']))
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def to_15min(raw, sfx):
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()


def load_audnzd():
    print("\n  Loading AUDNZD...")
    aud = to_15min(load_raw_zip('data/HISTDATA*AUDUSD*.zip'), 'aud')
    nzd = to_15min(load_raw_zip('data/HISTDATA*NZDUSD*.zip'), 'nzd')
    m   = aud.join(nzd, how='inner').dropna()
    m['c_spread']   = m['c_aud'] / m['c_nzd']
    m['o_spread']   = m['o_aud'] / m['o_nzd']
    m['h_spread']   = m['h_aud'] / m['l_nzd']
    m['l_spread']   = m['l_aud'] / m['h_nzd']
    m['quote_rate'] = m['c_nzd']
    m = m[m.index.weekday < 5].copy()
    print(f"  ✅ AUDNZD: {len(m):,} candles")
    return m


def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
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

    corr_ok = (df['c_aud'].pct_change()
               .rolling(C.corr_period)
               .corr(df['c_nzd'].pct_change()) > C.corr_min)

    vr        = calc_vr(log_r, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    sig = pd.Series(0, index=df.index)
    cond = vol_ok & time_ok & corr_ok & regime_ok
    sig[(z < -C.z_entry) & cond] =  1
    sig[(z >  C.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    print(f"  Signals: {(sig != 0).sum():,}")
    return sig, z


def calc_pnl(direction, entry, exit_px, lot, qr, pip):
    C = Config
    gross = direction * (exit_px - entry) * lot * C.lot_size * qr
    return gross - C.commission_per_lot * lot


def new_acc(ts):
    C = Config
    return {
        'equity': C.initial_balance,
        'start_ts': ts,
        'trades': [],
        'blown': False,
        'blown_rsn': '',
        'peak': C.initial_balance,
        'consec_loss': 0,
    }


def rec(pos, exit_px, exit_ts, pnl, status, lot):
    return {
        'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot,
        'entry': pos['entry'], 'exit': exit_px,
        'entry_ts': pos['entry_ts'], 'exit_ts': exit_ts,
        'pnl': pnl, 'status': status
    }


def run_backtest(df, sig, z):
    C = Config
    idx = df.index.sort_values()
    start_date = idx[C.warmup]

    o  = df['o_spread'].values.astype(float)
    c  = df['c_spread'].values.astype(float)
    qr = df['quote_rate'].values.astype(float)
    sg = sig.reindex(idx).fillna(0).values.astype(int)
    zz = z.reindex(idx).fillna(np.nan).values.astype(float)

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    acc = new_acc(start_date)
    total_withdrawn = 0.0
    acc_num = 1
    day_start_eq = C.initial_balance
    month_start_eq = C.initial_balance
    cooldown_until = None
    all_trades = []
    acc_logs = []

    position = None
    trades_today = 0
    pending_sig = 0

    prev_date = None
    prev_month = None

    print(f"  ▶ Backtest: {len(idx):,} bars | {idx[0].date()} → {idx[-1].date()}")

    for bar in range(C.warmup, len(idx)):
        ts = idx[bar]
        cur_date = ts.date()
        cur_month = (ts.year, ts.month)

        if cur_date != prev_date:
            day_start_eq = acc['equity']
            trades_today = 0
            prev_date = cur_date

        if cur_month != prev_month:
            month_start_eq = acc['equity']
            prev_month = cur_month

        if acc['equity'] > acc['peak']:
            acc['peak'] = acc['equity']

        in_cooldown = cooldown_until is not None and ts < cooldown_until

        if acc['blown']:
            acc_logs.append({
                'account': acc_num,
                'start_ts': acc['start_ts'],
                'end_ts': ts,
                'reason': acc['blown_rsn'],
                'pnl': acc['equity'] - C.initial_balance,
                'n_trades': len(acc['trades']),
                'days': (ts - acc['start_ts']).days
            })
            print(f"    💥 #{acc_num:>3} | {ts.date()} | Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            cooldown_until = ts + pd.Timedelta(days=C.cooldown_days)
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = month_start_eq = acc['equity']
            trades_today = 0
            pending_sig = 0
            position = None
            continue

        if in_cooldown:
            continue

        monthly_pnl = acc['equity'] - month_start_eq
        monthly_stressed = monthly_pnl < C.monthly_loss_threshold

        if pending_sig != 0 and position is None and trades_today < C.max_trades_day:
            sv  = pending_sig
            pip = C.PIP_SIZE['AUDNZD']
            risk = C.risk_base_pct * (0.5 if monthly_stressed else 1.0)
            if acc['consec_loss'] >= C.consec_loss_n:
                risk = max(risk * C.risk_reduce, C.risk_min_pct)

            pv  = pip * C.lot_size * qr[bar] if pip * C.lot_size * qr[bar] > 0 else 10.0
            lot = round(float(np.clip(acc['equity'] * risk / (C.sl_pips * pv),
                                      C.min_lot, C.max_lot)), 2)

            ep  = o[bar] + sv * (C.slippage_pips + C.PAIR_SPREAD['AUDNZD']/2) * pip
            sl  = ep - sv * C.sl_pips * pip
            tp  = ep + sv * C.tp_pips * pip

            position = {
                'pair': 'AUDNZD', 'dir': sv, 'lot': lot, 'lot_remaining': lot,
                'partial_done': False, 'entry': ep, 'sl': sl, 'tp': tp,
                'entry_ts': ts, 'entry_bar': bar, 'pip': pip
            }
            trades_today += 1

        pending_sig = 0

        total_float = 0.0
        if position is not None:
            total_float = calc_pnl(position['dir'], position['entry'], c[bar],
                                   position['lot_remaining'], qr[bar], position['pip'])

        current_eq = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            reason = "DailyDD" if current_eq <= daily_limit else "TotalDD"
            acc['blown'] = True
            acc['blown_rsn'] = reason
            if position is not None:
                pnl = calc_pnl(position['dir'], position['entry'], c[bar],
                               position['lot_remaining'], qr[bar], position['pip'])
                acc['equity'] += pnl
                r = rec(position, c[bar], ts, pnl, 'BLOWN', position['lot_remaining'])
                all_trades.append(r)
                acc['trades'].append(r)
                position = None
            continue

        if position is not None:
            cp = c[bar]
            d  = position['dir']
            ep = position['entry']
            zn = zz[bar]
            pip = position['pip']
            lr = position['lot_remaining']

            if not position['partial_done'] and not np.isnan(zn):
                if (d == 1 and zn >= -C.z_exit_partial) or (d == -1 and zn <= C.z_exit_partial):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr[bar], pip)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            r = rec(position, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(r)
                            acc['trades'].append(r)
                            position['lot_remaining'] = round(lr - p_lot, 2)
                            position['partial_done'] = True
                            position['sl'] = position['entry']
                            lr = position['lot_remaining']
                            if lr < C.min_lot:
                                position = None

            if position is not None:
                pnl_now = calc_pnl(d, ep, cp, position['lot_remaining'], qr[bar], pip)
                hit_z_stop = not np.isnan(zn) and (
                    (d==1 and zn<=-C.z_stop_margin) or (d==-1 and zn>=C.z_stop_margin))
                hit_z_exit = not np.isnan(zn) and (
                    (d==1 and zn>=-C.z_exit_full) or (d==-1 and zn<=C.z_exit_full))

                if hit_z_exit and pnl_now < C.min_net_profit_usd and not position['partial_done']:
                    hit_z_exit = False

                hit_sl = (d==1 and cp<=position['sl']) or (d==-1 and cp>=position['sl'])
                hit_tp = (d==1 and cp>=position['tp']) or (d==-1 and cp<=position['tp'])
                bars_open = bar - position['entry_bar']
                time_stop = (bars_open >= C.time_stop_bars and pnl_now < 0) or \
                            (bars_open >= C.time_stop_bars * 2)

                if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                    exit_px = position['sl'] if hit_sl else (position['tp'] if hit_tp else cp)
                    st = ('SL' if hit_sl else ('TP' if hit_tp else
                          ('Z-Stop' if hit_z_stop else ('TimeStop' if time_stop else 'Z-Exit'))))
                    fpnl = calc_pnl(d, ep, exit_px, position['lot_remaining'], qr[bar], pip)
                    acc['equity'] += fpnl
                    r = rec(position, exit_px, ts, fpnl, st, position['lot_remaining'])
                    all_trades.append(r)
                    acc['trades'].append(r)
                    position = None
                    if fpnl > 0:
                        acc['consec_loss'] = 0
                    else:
                        acc['consec_loss'] += 1

        if acc['equity'] >= PROFIT_LEVEL and position is None:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            dt = (ts - acc['start_ts']).days
            acc_logs.append({
                'account': acc_num,
                'start_ts': acc['start_ts'],
                'end_ts': ts,
                'reason': 'TARGET_HIT',
                'pnl': w,
                'n_trades': len(acc['trades']),
                'days': dt
            })
            print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | Bank:${total_withdrawn:>9.2f} | {dt}d | {len(acc['trades'])}T")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = month_start_eq = acc['equity']
            trades_today = 0
            pending_sig = 0
            continue

        if position is None and not acc['blown'] and not in_cooldown and trades_today < C.max_trades_day and sg[bar] != 0:
            pending_sig = int(sg[bar])

    return {
        'all_trades': all_trades,
        'account_logs': acc_logs,
        'total_withdrawn': total_withdrawn,
        'final_equity': acc['equity'],
        'common_idx': idx
    }


def print_report(res):
    if not res['all_trades']:
        print("❌ No trades")
        return

    df = pd.DataFrame(res['all_trades'])
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])
    df['entry_ts'] = pd.to_datetime(df['entry_ts'])
    df['month'] = df['exit_ts'].dt.to_period('M')

    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    wr = len(wins) / len(df) * 100
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else float('inf')

    all_months = pd.period_range(
        start=res['common_idx'][Config.warmup].to_period('M'),
        end=res['common_idx'][-1].to_period('M'),
        freq='M'
    )
    monthly = df.groupby('month')['pnl'].sum().reindex(all_months, fill_value=0.0)

    print("\n" + "═"*70)
    print(" CorrArb v9c Audit — AUDNZD only")
    print("═"*70)
    print(f"Total Trades:        {len(df):,}")
    print(f"Win Rate:            {wr:.2f}%")
    print(f"Profit Factor:       {pf:.2f}")
    print(f"Avg Win:             ${wins['pnl'].mean():.2f}")
    print(f"Avg Loss:            ${losses['pnl'].mean():.2f}")
    print(f"Net PnL:             ${df['pnl'].sum():,.2f}")
    print("-"*70)
    print(f"Total Banked:        ${res['total_withdrawn']:,.2f}")
    print(f"Active Equity:       ${res['final_equity']:,.2f}")
    print(f"Monthly Avg:         ${monthly.mean():.2f}")
    print(f"Positive Months:     {(monthly > 0).sum()} / {len(monthly)}")
    print(f"Negative Months:     {(monthly < 0).sum()} / {len(monthly)}")
    print(f"Zero Months:         {(monthly == 0).sum()} / {len(monthly)}")
    print(f"Best Month:          ${monthly.max():,.2f}")
    print(f"Worst Month:         ${monthly.min():,.2f}")
    print("-"*70)
    print("Exit Stats:")
    for st, cnt in df['status'].value_counts().items():
        print(f"  {st:<12} {cnt:>5} ({cnt/len(df)*100:.1f}%)")

    print("-"*70)
    print("By Exit Type:")
    g = df.groupby('status')['pnl'].agg(['count','mean','sum'])
    print(g.sort_values('sum'))
    print("═"*70)


if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════╗")
    print("║      CorrArb v9c Audit Baseline     ║")
    print("╚══════════════════════════════════════╝")
    df = load_audnzd()
    sig, z = compute_signals(df)
    res = run_backtest(df, sig, z)
    print_report(res)
    print(f"\n✅ Done in {(datetime.now()-t0).total_seconds():.2f}s")
