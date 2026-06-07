"""
CorrArb v9g — Final Version
============================
Baseline قفل‌شده:
  ✅ No TimeStop
  ✅ SL = 30 pips
  ✅ Partial = 0.75
  ✅ z_exit_partial = 0.50

بهبودهای جدید (فقط Risk/Prop):
  ✅ Dynamic risk بر اساس rolling PF
  ✅ Max consecutive accounts blown → stop
  ✅ گزارش کامل‌تر برای تصمیم‌گیری واقعی
"""

import pandas as pd
import numpy as np
import glob, zipfile, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class Config:
    # ── Prop ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── Risk — قفل‌شده ──
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

    # ── Signal — قفل‌شده ──
    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.50   # ← قفل
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0

    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # ── Exit — قفل‌شده ──
    sl_pips            = 30.0
    tp_pips            = 90.0
    partial_ratio      = 0.75   # ← قفل
    use_time_stop      = False  # ← قفل

    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.90

    cooldown_days          = 10
    monthly_loss_threshold = -150.0

    # ── Dynamic Risk ──
    # بر اساس rolling PF آخرین N ترید
    dynamic_risk_enabled   = True
    rolling_pf_window      = 30     # آخرین 30 ترید
    rolling_pf_good        = 1.20   # اگر PF > این → risk * 1.2
    rolling_pf_bad         = 0.90   # اگر PF < این → risk * 0.7

    # ── Max Blown در row ──
    max_consec_blown       = 3      # اگر 3 حساب پشت‌سرهم blown → pause 30 روز
    blown_pause_days       = 30


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


def rolling_pf(trades_history, window):
    """PF آخرین N ترید"""
    if len(trades_history) < window // 2:
        return 1.0   # کافی نیست → neutral
    recent = trades_history[-window:]
    wins   = sum(p for p in recent if p > 0)
    losses = abs(sum(p for p in recent if p < 0))
    return wins / losses if losses > 0 else 1.5


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


def run_backtest(df, sig, z):
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

    acc              = new_acc(start_date)
    withdrawn        = 0.0
    acc_num          = 1
    day_eq           = C.initial_balance
    month_eq         = C.initial_balance
    cooldown_til     = None
    all_trades       = []
    acc_logs         = []
    monthly_log      = []    # (year, month, pnl)
    eq_curve         = []    # (date, total_equity)

    pos              = None
    day_trades       = 0
    pending          = 0
    prev_date        = None
    prev_month       = None

    # Dynamic risk tracking
    global_pnl_hist  = []    # همه PnL‌ها برای rolling PF
    consec_blown     = 0     # تعداد blown پشت‌سرهم
    blown_pause_til  = None  # pause بعد از blown‌های متوالی

    print(f"  ▶ Running v9g Final...")

    for bar in range(C.warmup, len(idx)):
        ts        = idx[bar]
        cur_date  = ts.date()
        cur_month = (ts.year, ts.month)

        if cur_date != prev_date:
            day_eq     = acc['equity']
            day_trades = 0
            eq_curve.append({
                'date':   str(cur_date),
                'equity': round(acc['equity'] + withdrawn, 2),
            })
            prev_date = cur_date

        if cur_month != prev_month:
            if prev_month is not None:
                monthly_log.append({
                    'year':  prev_month[0],
                    'month': prev_month[1],
                    'pnl':   round(acc['equity'] - month_eq, 2),
                })
            month_eq   = acc['equity']
            prev_month = cur_month

        if acc['equity'] > acc['peak']:
            acc['peak'] = acc['equity']

        # pause بعد از blown‌های متوالی
        in_blown_pause = (blown_pause_til is not None and
                          ts < blown_pause_til)
        in_cd          = ((cooldown_til is not None and ts < cooldown_til)
                          or in_blown_pause)

        # ── Blown ──
        if acc['blown']:
            acc_logs.append({
                'account':  acc_num,
                'start_ts': str(acc['start_ts'].date()),
                'end_ts':   str(ts.date()),
                'reason':   acc['blown_rsn'],
                'pnl':      round(acc['equity'] - C.initial_balance, 2),
                'n_trades': len(acc['trades']),
                'days':     (ts - acc['start_ts']).days,
            })
            consec_blown += 1
            icon = '💥'
            extra = ''
            if consec_blown >= C.max_consec_blown:
                blown_pause_til = ts + timedelta(days=C.blown_pause_days)
                extra = f' ⏸ PAUSE {C.blown_pause_days}d'
                consec_blown = 0
            print(f"    {icon} #{acc_num:>3} | {ts.date()} | "
                  f"Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}{extra}")
            cooldown_til = ts + timedelta(days=C.cooldown_days)
            acc_num   += 1
            acc        = new_acc(ts)
            day_eq     = month_eq = acc['equity']
            day_trades = 0
            pending    = 0
            pos        = None
            continue

        if in_cd:
            continue

        # reset consec_blown اگر target hit شده باشد
        m_stressed = (acc['equity'] - month_eq) < C.monthly_loss_threshold

        # Dynamic risk multiplier
        if C.dynamic_risk_enabled and len(global_pnl_hist) >= C.rolling_pf_window // 2:
            rpf = rolling_pf(global_pnl_hist, C.rolling_pf_window)
            if rpf >= C.rolling_pf_good:
                risk_mult = 1.2
            elif rpf < C.rolling_pf_bad:
                risk_mult = 0.7
            else:
                risk_mult = 1.0
        else:
            risk_mult = 1.0

        # ── Open pending ──
        if pending != 0 and pos is None and day_trades < C.max_trades_day:
            sv   = pending
            risk = C.risk_base_pct * risk_mult
            if m_stressed:
                risk *= 0.5
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

        # ── Float DD ──
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
                tr = {
                    'dir': pos['dir'], 'lot': pos['lot_rem'],
                    'entry': pos['entry'], 'exit': c_[bar],
                    'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                    'pnl': pnl, 'status': 'BLOWN',
                }
                all_trades.append(tr)
                acc['trades'].append(tr)
                global_pnl_hist.append(pnl)
                pos = None
            continue

        # ── Exit ──
        if pos is not None:
            cp = c_[bar]
            d  = pos['dir']
            ep = pos['entry']
            zn = zz_[bar]
            lr = pos['lot_rem']

            # Partial
            if not pos['partial_done'] and not np.isnan(zn):
                if ((d == 1 and zn >= -C.z_exit_partial) or
                        (d == -1 and zn <=  C.z_exit_partial)):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr_[bar])
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            tr = {
                                'dir': d, 'lot': p_lot,
                                'entry': ep, 'exit': cp,
                                'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                                'pnl': p_pnl, 'status': 'Partial',
                            }
                            all_trades.append(tr)
                            acc['trades'].append(tr)
                            global_pnl_hist.append(p_pnl)
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
                    tr = {
                        'dir': d, 'lot': lr,
                        'entry': ep, 'exit': xp,
                        'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                        'pnl': fpnl, 'status': st,
                    }
                    all_trades.append(tr)
                    acc['trades'].append(tr)
                    global_pnl_hist.append(fpnl)
                    pos = None

                    if fpnl > 0:
                        acc['consec_loss'] = 0
                    else:
                        acc['consec_loss'] += 1

        # ── Target ──
        if acc['equity'] >= TARGET and pos is None:
            w  = acc['equity'] - C.initial_balance
            withdrawn += w
            dt = (ts - acc['start_ts']).days
            nt = len(acc['trades'])
            acc_logs.append({
                'account':  acc_num,
                'start_ts': str(acc['start_ts'].date()),
                'end_ts':   str(ts.date()),
                'reason':   'TARGET_HIT',
                'pnl':      round(w, 2),
                'n_trades': nt,
                'days':     dt,
            })
            print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | "
                  f"Bank:${withdrawn:>9.2f} | {dt}d | {nt}T")
            consec_blown = 0    # reset بعد از موفقیت
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
        'monthly_log':  monthly_log,
        'eq_curve':     eq_curve,
    }


# ═══════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════
def print_report(res):
    trades = res['all_trades']
    if not trades:
        print("❌ No trades")
        return

    df = pd.DataFrame(trades)
    df['exit_ts']  = pd.to_datetime(df['exit_ts'])
    df['entry_ts'] = pd.to_datetime(df['entry_ts'])
    df['month']    = df['exit_ts'].dt.to_period('M')
    df['year']     = df['exit_ts'].dt.year

    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    wr     = len(wins) / len(df) * 100
    pf     = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else 99.0

    # Monthly از monthly_log
    ml = pd.DataFrame(res['monthly_log'])
    if len(ml):
        ml['period'] = ml.apply(
            lambda r: pd.Period(f"{int(r['year'])}-{int(r['month']):02d}", freq='M'), axis=1)
        ml = ml.set_index('period')['pnl']

    ci = res['common_idx']
    all_months = pd.period_range(
        start=ci[Config.warmup].to_period('M'),
        end=ci[-1].to_period('M'), freq='M')
    monthly = ml.reindex(all_months, fill_value=0.0) if len(ml) else pd.Series(0, index=all_months)

    pos_m  = int((monthly > 0).sum())
    neg_m  = int((monthly < 0).sum())
    zero_m = int((monthly == 0).sum())
    tot_m  = len(monthly)

    # Consecutive losing months
    max_consec_neg = cur_neg = 0
    for v in monthly:
        if v < 0:
            cur_neg += 1
            max_consec_neg = max(max_consec_neg, cur_neg)
        else:
            cur_neg = 0

    logs    = res['account_logs']
    df_acc  = pd.DataFrame(logs) if logs else pd.DataFrame()
    targets = df_acc[df_acc['reason'] == 'TARGET_HIT'] if len(df_acc) else pd.DataFrame()
    blowns  = df_acc[df_acc['reason'] != 'TARGET_HIT'] if len(df_acc) else pd.DataFrame()

    # سودآوری سالانه
    yearly = df.groupby('year')['pnl'].sum()
    neg_years = int((yearly < 0).sum())

    print("\n" + "═"*70)
    print("  CorrArb v9g — FINAL | AUDNZD")
    print("  No TimeStop | SL=30 | Partial=0.75 | ZEP=0.50")
    print("═"*70)
    print(f"  Trades:         {len(df):>6,}  |  WR: {wr:.1f}%  |  PF: {pf:.3f}")
    print(f"  Avg Win:       ${wins['pnl'].mean():>7.2f}  |  Avg Loss: ${losses['pnl'].mean():>7.2f}")
    print(f"  Net PnL:       ${df['pnl'].sum():>10,.2f}")
    print("-"*70)
    print(f"  Total Banked:  ${res['withdrawn']:>10,.2f}")
    print(f"  Active Equity: ${res['final_equity']:>10,.2f}")
    print(f"  Accounts Pass: {len(targets):>4}  |  Blown: {len(blowns):>3}")
    if len(targets):
        print(f"  Avg Days/Pass: {targets['days'].mean():.0f}d  |  Avg T/Pass: {targets['n_trades'].mean():.0f}")
    print("-"*70)
    print(f"  +Months: {pos_m:>3} / {tot_m}  ({pos_m/tot_m*100:.0f}%)")
    print(f"  -Months: {neg_m:>3} / {tot_m}")
    print(f"   0Month: {zero_m:>3} / {tot_m}")
    print(f"  Max consec. negative months: {max_consec_neg}")
    print(f"  Monthly avg: ${monthly.mean():.2f}  |  Median: ${monthly.median():.2f}")
    print(f"  Best month:  ${monthly.max():,.2f}  |  Worst: ${monthly.min():,.2f}")
    print(f"  Negative years: {neg_years} / {len(yearly)}")
    print("-"*70)
    print("  By Exit Type:")
    g = df.groupby('status')['pnl'].agg(['count', 'mean', 'sum'])
    for st, row in g.sort_values('sum').iterrows():
        mark = '▲' if row['sum'] >= 0 else '▼'
        print(f"    {st:<10} {int(row['count']):>5}  "
              f"avg:${row['mean']:>8.2f}  "
              f"total:${row['sum']:>10,.2f}  {mark}")
    print("-"*70)
    print("  Yearly:")
    for yr, g2 in df.groupby('year'):
        w2 = g2[g2['pnl'] > 0]
        l2 = g2[g2['pnl'] < 0]
        ypf  = w2['pnl'].sum() / abs(l2['pnl'].sum()) if len(l2) else 99.0
        sign = '+' if g2['pnl'].sum() >= 0 else '-'
        mark = '✅' if g2['pnl'].sum() >= 0 else '❌'
        print(f"    {mark} {yr}: {len(g2):>4}T  "
              f"WR:{len(w2)/len(g2)*100:5.1f}%  "
              f"PF:{ypf:.2f}  "
              f"{sign}${abs(g2['pnl'].sum()):>8,.2f}")
    print("-"*70)

    # هدف ۲٪
    target_m = Config.initial_balance * 0.02
    above_target = int((monthly >= target_m).sum())
    print(f"  🎯 هدف ۲٪/ماه (${target_m:.0f}):")
    print(f"     ماه‌های بالای هدف: {above_target} از {tot_m} ({above_target/tot_m*100:.0f}%)")
    print(f"     میانگین ماهانه:    ${monthly.mean():.2f}")
    print(f"     هدف چقدر محقق شد: {monthly.mean()/target_m*100:.0f}%")
    print("═"*70)

    # ذخیره
    try:
        pd.DataFrame(res['eq_curve']).to_csv('equity_v9g.csv', index=False)
        monthly.to_frame('pnl').to_csv('monthly_v9g.csv')
        print("  📊 equity_v9g.csv + monthly_v9g.csv saved")
    except Exception as e:
        print(f"  ⚠ Save error: {e}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   CorrArb v9g — Final Locked Baseline               ║")
    print("╚══════════════════════════════════════════════════════╝")

    df     = load_audnzd()
    sig, z = compute_signals(df)
    print(f"  Signals: {(sig != 0).sum():,}")

    res = run_backtest(df, sig, z)
    print_report(res)

    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.1f}s")
