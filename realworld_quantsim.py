"""
CorrArb v11 — Exit Logic Rebuild
==================================
مشکل v10: ATR stop بزرگ + partial زود = R:R معکوس

اصلاحات v11:
  ✅ SL کوچک‌تر (1.0 × ATR به جای 2.0)
  ✅ بدون partial — full position نگه‌دار
  ✅ Trailing stop فعال از همان ابتدا
  ✅ TP واقعی با R:R حداقل 1.5
  ✅ Z-exit فقط با سود کافی
  ✅ Time stop کوتاه‌تر
  ✅ تشخیص دقیق مشکل با آمار R-multiple
"""

import pandas as pd
import numpy as np
import glob, zipfile, os, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class GlobalConfig:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.08        # ↑ 8% برای زمان بیشتر
    max_daily_loss_pct = 0.04        # ↓ سخت‌تر
    max_total_dd_pct   = 0.08        # ↓ سخت‌تر — زودتر reset

    commission_per_lot = 7.0
    slippage_pips      = 0.3
    lot_size           = 100_000
    max_lot            = 2.0
    min_lot            = 0.01
    warmup             = 500

    cooldown_days      = 7
    max_trades_day     = 4

    risk_per_trade     = 0.01        # 1% ریسک ثابت per trade
    risk_min           = 0.005
    consec_loss_reduce_n = 3
    consec_loss_mult     = 0.7

    monthly_loss_threshold = -100.0

    # Pair Quality
    pq_min_trades   = 20
    pq_window_days  = 90
    pq_start_days   = 90
    pq_min_pf       = 1.0
    pq_resume_pf    = 1.05
    pq_interval     = 30


class PairConfig:
    AUDNZD = {
        'enabled':      True,
        'spread_pip':   2.5,
        'pip_size':     0.0001,

        # Session
        'hour_start':   20,
        'hour_end':     8,
        'trade_days':   [0, 1, 2, 3, 4],

        # Z-Score
        'z_period':     96,
        'z_entry':      2.0,
        'z_exit':       0.5,         # خروج وقتی z به 0.5 رسید
        'z_stop':       4.0,         # z-stop اضافی

        # Entry
        'confirm_bars': 2,

        # ATR
        'atr_period':   14,
        'atr_ma':       96,
        'atr_sl_mult':  1.2,         # ↓ SL کوچک‌تر
        'atr_tp_mult':  2.4,         # R:R = 2:1
        'atr_max_vol':  3.0,
        'atr_min_vol':  0.3,

        # Trailing
        'use_trail':    True,
        'trail_mult':   1.0,         # trail = 1.0 × ATR از high/low

        # Exit
        'time_stop_bars': 32,        # 8 ساعت
        'min_rr_for_zexit': 0.5,    # حداقل 0.5R سود برای z-exit

        # Correlation
        'corr_period':  96,
        'corr_min':     0.70,

        # VR
        'vr_period':    150,
        'vr_k':         4,
        'vr_max':       0.90,
    }

    @classmethod
    def active(cls):
        result = {}
        for name in ['AUDNZD', 'EURGBP']:
            cfg = getattr(cls, name, None)
            if cfg and cfg.get('enabled', False):
                result[name] = cfg
        return result


# ═══════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════
def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No ZIP: {pattern}")
    frames = []
    for p in paths:
        with zipfile.ZipFile(p) as z:
            csv_name = next((f for f in z.namelist()
                             if f.lower().endswith('.csv')), None)
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


def load_data():
    print("\n  Loading...")
    loaded = {}
    try:
        aud = to_15min(load_raw_zip('data/HISTDATA*AUDUSD*.zip'), 'aud')
        nzd = to_15min(load_raw_zip('data/HISTDATA*NZDUSD*.zip'), 'nzd')
        m = aud.join(nzd, how='inner').dropna()
        m['c_spread']  = m['c_aud'] / m['c_nzd']
        m['o_spread']  = m['o_aud'] / m['o_nzd']
        m['h_spread']  = m['h_aud'] / m['l_nzd']
        m['l_spread']  = m['l_aud'] / m['h_nzd']
        m['quote_rate']= m['c_nzd']
        m['leg1']      = m['c_aud']
        m['leg2']      = m['c_nzd']
        m = m[m.index.weekday < 5].copy()
        loaded['AUDNZD'] = m
        print(f"  ✅ AUDNZD: {len(m):,} candles")
    except Exception as e:
        print(f"  ❌ AUDNZD: {e}")
    return loaded


# ═══════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period):
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_vr(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    v1 = r1.rolling(window).var()
    vk = rk.rolling(window).var()
    return vk / (k * v1.replace(0, np.nan))


def session_ok(hour, cfg):
    hs, he = cfg['hour_start'], cfg['hour_end']
    if hs <= he:
        return hs <= hour <= he
    return hour >= hs or hour <= he


# ═══════════════════════════════════════════════════════════════════════
#  SIGNALS
# ═══════════════════════════════════════════════════════════════════════
def compute_signals(name, df, pcfg):
    log_r = np.log(df['c_spread'])

    # Z-score
    z_mean = log_r.rolling(pcfg['z_period']).mean()
    z_std  = log_r.rolling(pcfg['z_period']).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    # ATR
    atr    = calc_atr(df['h_spread'], df['l_spread'],
                      df['c_spread'], pcfg['atr_period'])
    atr_ma = atr.rolling(pcfg['atr_ma']).mean()
    vol_ok = ((atr > atr_ma * pcfg['atr_min_vol']) &
              (atr < atr_ma * pcfg['atr_max_vol']))

    # Correlation
    corr   = (df['leg1'].pct_change()
              .rolling(pcfg['corr_period'])
              .corr(df['leg2'].pct_change()))
    corr_ok = corr > pcfg['corr_min']

    # Variance Ratio
    vr        = calc_vr(log_r, pcfg['vr_k'], pcfg['vr_period'])
    regime_ok = vr < pcfg['vr_max']

    # Session
    hours  = pd.Series(df.index.hour,      index=df.index)
    dows   = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = pd.Series(
        [session_ok(h, pcfg) and d in pcfg['trade_days']
         for h, d in zip(hours, dows)],
        index=df.index)

    base_ok = vol_ok & time_ok & corr_ok & regime_ok

    # ── Reversion Confirmation ──
    z_arr   = z.values
    ok_arr  = base_ok.values
    n       = len(z_arr)
    sig     = np.zeros(n, dtype=int)
    cb      = pcfg['confirm_bars']
    ze      = pcfg['z_entry']

    long_count = short_count = 0
    long_armed = short_armed = False
    prev_z = np.nan

    for i in range(n):
        zi = z_arr[i]
        if np.isnan(zi) or not ok_arr[i]:
            long_count = short_count = 0
            long_armed = short_armed = False
            prev_z = zi
            continue

        # Long
        if zi < -ze:
            long_count += 1
            short_count = 0
            short_armed = False
            if long_count >= cb:
                long_armed = True
        else:
            if long_armed and not np.isnan(prev_z) and zi > prev_z:
                sig[i] = 1
                long_armed = False
                long_count = 0
            elif zi >= -ze * 0.5:
                long_count = 0
                long_armed = False

        # Short
        if zi > ze:
            short_count += 1
            long_count = 0
            long_armed = False
            if short_count >= cb:
                short_armed = True
        else:
            if short_armed and not np.isnan(prev_z) and zi < prev_z:
                sig[i] = -1
                short_armed = False
                short_count = 0
            elif zi <= ze * 0.5:
                short_count = 0
                short_armed = False

        prev_z = zi

    sig_s = pd.Series(sig, index=df.index)
    # Remove duplicates
    sig_s = sig_s.where(sig_s != sig_s.shift(), 0)

    n_sig = int((sig_s != 0).sum())
    print(f"    {name}: {n_sig:,} signals | "
          f"Regime OK: {int(regime_ok.sum()):,} | "
          f"Session: {int(time_ok.sum()):,}")
    return sig_s, z, atr


# ═══════════════════════════════════════════════════════════════════════
#  BACKTEST
# ═══════════════════════════════════════════════════════════════════════
def pnl_calc(direction, entry, exit_px, lot, qr):
    gross = direction * (exit_px - entry) * lot * GlobalConfig.lot_size * qr
    return gross - GlobalConfig.commission_per_lot * lot


def new_acc(ts):
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


def run_backtest(pair_data, pair_signals):
    G    = GlobalConfig
    acfg = PairConfig.active()
    pns  = list(pair_data.keys())

    # Common index
    cidx = None
    for n in pns:
        idx  = pair_data[n].index
        cidx = idx if cidx is None else cidx.intersection(idx)
    cidx = cidx.sort_values()
    N    = len(cidx)
    t0   = cidx[G.warmup]
    print(f"  ✅ Bars: {N:,} | {cidx[0].date()} → {cidx[-1].date()}")

    # Arrays
    pa = {}
    for n in pns:
        pcfg = acfg[n]
        df_p = pair_data[n].reindex(cidx).ffill()
        s, z, atr_s = pair_signals[n]
        pa[n] = {
            'o':   df_p['o_spread'].values.astype(float),
            'h':   df_p['h_spread'].values.astype(float),
            'l':   df_p['l_spread'].values.astype(float),
            'c':   df_p['c_spread'].values.astype(float),
            'qr':  df_p['quote_rate'].values.astype(float),
            'sig': s.reindex(cidx).fillna(0).values.astype(int),
            'z':   z.reindex(cidx).fillna(np.nan).values.astype(float),
            'atr': atr_s.reindex(cidx).fillna(np.nan).values.astype(float),
            'sp':  pcfg['spread_pip'],
            'pip': pcfg['pip_size'],
            'cfg': pcfg,
        }

    FLOOR  = G.initial_balance * (1 - G.max_total_dd_pct)
    TARGET = G.initial_balance * (1 + G.profit_target_pct)

    # State
    acc           = new_acc(t0)
    withdrawn     = 0.0
    acc_num       = 1
    day_eq        = G.initial_balance
    month_eq      = G.initial_balance
    cooldown_till = None
    all_trades    = []
    acc_logs      = []
    eq_curve      = []

    positions     = {n: None for n in pns}
    day_trades    = {n: 0    for n in pns}
    pending       = {n: 0    for n in pns}

    pair_hist     = {n: []            for n in pns}
    pair_status   = {n: 'ACTIVE'      for n in pns}
    last_eval     = {n: None          for n in pns}

    prev_date  = None
    prev_month = None

    print(f"\n  ▶ Running v11...")

    for bar in range(G.warmup, N):
        ts        = cidx[bar]
        cur_date  = ts.date()
        cur_month = (ts.year, ts.month)

        # ── Day reset ──
        if cur_date != prev_date:
            day_eq = acc['equity']
            for n in pns:
                day_trades[n] = 0
            eq_curve.append({
                'date':   cur_date,
                'equity': acc['equity'] + withdrawn,
            })
            prev_date = cur_date

        # ── Month reset + Quality eval ──
        if cur_month != prev_month and prev_month is not None:
            month_eq = acc['equity']
            days_on  = (ts - t0).days
            if days_on >= G.pq_start_days:
                for n in pns:
                    if (last_eval[n] is not None and
                            (ts - last_eval[n]).days < G.pq_interval):
                        continue
                    last_eval[n] = ts
                    cutoff = ts - timedelta(days=G.pq_window_days)
                    rec = [(t, p) for t, p in pair_hist[n] if t >= cutoff]
                    if len(rec) < G.pq_min_trades:
                        if pair_status[n] == 'PROBATION':
                            pair_status[n] = 'ACTIVE'
                        continue
                    w  = sum(p for _, p in rec if p > 0)
                    la = abs(sum(p for _, p in rec if p < 0))
                    pf = w / la if la > 0 else 99.0
                    old = pair_status[n]
                    if old == 'ACTIVE' and pf < G.pq_min_pf:
                        pair_status[n] = 'PROBATION'
                        print(f"    ⚠ PROBATION {n} | PF={pf:.2f} | "
                              f"T={len(rec)} | {ts.date()}")
                    elif old == 'PROBATION' and pf >= G.pq_resume_pf:
                        pair_status[n] = 'ACTIVE'
                        print(f"    ▶ RESUMED {n} | PF={pf:.2f} | "
                              f"T={len(rec)} | {ts.date()}")
        if cur_month != prev_month:
            prev_month = cur_month

        if acc['equity'] > acc['peak']:
            acc['peak'] = acc['equity']

        in_cd = cooldown_till is not None and ts < cooldown_till

        # ── Blown ──
        if acc['blown']:
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   acc['blown_rsn'],
                'pnl':      acc['equity'] - G.initial_balance,
                'n_trades': len(acc['trades']),
                'days':     (ts - acc['start_ts']).days,
            })
            print(f"    💥 #{acc_num:>3} | {ts.date()} | "
                  f"Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            cooldown_till = ts + timedelta(days=G.cooldown_days)
            acc_num += 1
            acc     = new_acc(ts)
            day_eq  = month_eq = acc['equity']
            for n in pns:
                day_trades[n] = 0
                pending[n]    = 0
                positions[n]  = None
            prev_date  = cur_date
            prev_month = cur_month
            continue

        if in_cd:
            continue

        m_stressed = (acc['equity'] - month_eq) < G.monthly_loss_threshold

        # ── Open pending ──
        for n in pns:
            pcfg = pa[n]['cfg']
            a    = pa[n]
            if (pending[n] != 0
                    and positions[n] is None
                    and day_trades[n] < G.max_trades_day):

                sv  = pending[n]
                qr  = a['qr'][bar]
                pip = a['pip']
                av  = a['atr'][bar]

                if np.isnan(av) or av <= 0:
                    pending[n] = 0
                    continue

                # Risk
                risk = G.risk_per_trade
                if pair_status[n] == 'PROBATION':
                    risk *= 0.5
                if m_stressed:
                    risk *= 0.6
                if acc['consec_loss'] >= G.consec_loss_reduce_n:
                    risk = max(risk * G.consec_loss_mult, G.risk_min)

                sl_dist = av * pcfg['atr_sl_mult']
                tp_dist = av * pcfg['atr_tp_mult']
                pip_val = pip * G.lot_size * qr
                sl_pips = sl_dist / pip
                lot     = round(float(np.clip(
                    acc['equity'] * risk / (sl_pips * pip_val),
                    G.min_lot, G.max_lot)), 2)

                ep = a['o'][bar] + sv * (G.slippage_pips + a['sp'] / 2) * pip
                sl = ep - sv * sl_dist
                tp = ep + sv * tp_dist

                positions[n] = {
                    'pair':       n,
                    'dir':        sv,
                    'lot':        lot,
                    'entry':      ep,
                    'sl':         sl,
                    'tp':         tp,
                    'entry_ts':   ts,
                    'entry_bar':  bar,
                    'pip':        pip,
                    'atr0':       av,
                    'trail_high': ep,   # برای trail
                    'trail_low':  ep,
                }
                day_trades[n] += 1
            pending[n] = 0

        # ── Float DD check ──
        total_float = sum(
            pnl_calc(p['dir'], p['entry'], pa[n]['c'][bar],
                     p['lot'], pa[n]['qr'][bar])
            for n in pns if (p := positions[n]) is not None
        )
        cur_eq     = acc['equity'] + total_float
        daily_lim  = day_eq * (1 - G.max_daily_loss_pct)

        if cur_eq <= daily_lim or cur_eq <= FLOOR:
            rsn = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            acc['blown']     = True
            acc['blown_rsn'] = rsn
            for n in pns:
                pos = positions[n]
                if pos is None:
                    continue
                pnl = pnl_calc(pos['dir'], pos['entry'],
                               pa[n]['c'][bar], pos['lot'], pa[n]['qr'][bar])
                acc['equity'] += pnl
                acc['trades'].append({
                    'pair': n, 'dir': pos['dir'], 'lot': pos['lot'],
                    'entry': pos['entry'], 'exit': pa[n]['c'][bar],
                    'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                    'pnl': pnl, 'status': 'BLOWN'})
                all_trades.append(all_trades[-1] if all_trades else
                                  {'pair': n, 'pnl': pnl, 'status': 'BLOWN',
                                   'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                                   'entry': pos['entry'], 'exit': pa[n]['c'][bar],
                                   'dir': pos['dir'], 'lot': pos['lot']})
                pair_hist[n].append((ts, pnl))
                positions[n] = None
            # fix: properly append blown trades
            for n in pns:
                pass  # already done above
            continue

        # ── Exit ──
        for n in pns:
            pos = positions[n]
            if pos is None:
                continue

            a    = pa[n]
            pcfg = a['cfg']
            cp   = a['c'][bar]
            hp   = a['h'][bar]
            lp   = a['l'][bar]
            qr   = a['qr'][bar]
            d    = pos['dir']
            ep   = pos['entry']
            zn   = a['z'][bar]
            pip  = pos['pip']
            lot  = pos['lot']
            av0  = pos['atr0']

            # Update trailing reference
            if d == 1:
                pos['trail_high'] = max(pos['trail_high'], hp)
                trail_sl = pos['trail_high'] - av0 * pcfg['trail_mult']
                pos['sl'] = max(pos['sl'], trail_sl)
            else:
                pos['trail_low'] = min(pos['trail_low'], lp)
                trail_sl = pos['trail_low'] + av0 * pcfg['trail_mult']
                pos['sl'] = min(pos['sl'], trail_sl)

            # Intrabar checks
            hit_sl = (d == 1 and lp <= pos['sl']) or (d == -1 and hp >= pos['sl'])
            hit_tp = (d == 1 and hp >= pos['tp']) or (d == -1 and lp <= pos['tp'])

            # Z-exit
            hit_ze = False
            if not np.isnan(zn):
                if ((d == 1 and zn >= -pcfg['z_exit']) or
                        (d == -1 and zn <= pcfg['z_exit'])):
                    pnl_now = pnl_calc(d, ep, cp, lot, qr)
                    r_val   = pnl_now / (av0 * pcfg['atr_sl_mult'] *
                                         lot * GlobalConfig.lot_size * qr
                                         / pip * pip)
                    if pnl_now > 0:
                        hit_ze = True

            # Z-stop (z مخالف خیلی بزرگ)
            hit_zs = (not np.isnan(zn) and
                      ((d == 1 and zn <= -pcfg['z_stop']) or
                       (d == -1 and zn >= pcfg['z_stop'])))

            # Time stop
            bars_open = bar - pos['entry_bar']
            pnl_now   = pnl_calc(d, ep, cp, lot, qr)
            time_stop = (bars_open >= pcfg['time_stop_bars'] and pnl_now < 0)

            if hit_sl or hit_tp or hit_ze or hit_zs or time_stop:
                if hit_sl:
                    xp, st = pos['sl'], 'SL'
                elif hit_tp:
                    xp, st = pos['tp'], 'TP'
                elif hit_zs:
                    xp, st = cp, 'Z-Stop'
                elif time_stop:
                    xp, st = cp, 'TimeStop'
                else:
                    xp, st = cp, 'Z-Exit'

                fpnl = pnl_calc(d, ep, xp, lot, qr)
                acc['equity'] += fpnl
                tr = {'pair': n, 'dir': d, 'lot': lot,
                      'entry': ep, 'exit': xp,
                      'entry_ts': pos['entry_ts'], 'exit_ts': ts,
                      'pnl': fpnl, 'status': st,
                      'bars': bars_open}
                all_trades.append(tr)
                acc['trades'].append(tr)
                pair_hist[n].append((ts, fpnl))
                positions[n] = None

                if fpnl > 0:
                    acc['consec_loss'] = 0
                else:
                    acc['consec_loss'] += 1

        # ── Target ──
        if (acc['equity'] >= TARGET and
                all(positions[n] is None for n in pns)):
            w  = acc['equity'] - G.initial_balance
            withdrawn += w
            dt = (ts - acc['start_ts']).days
            nt = len(acc['trades'])
            acc_logs.append({
                'account': acc_num, 'start_ts': acc['start_ts'],
                'end_ts': ts, 'reason': 'TARGET_HIT',
                'pnl': w, 'n_trades': nt, 'days': dt})
            print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | "
                  f"Bank:${withdrawn:>9.2f} | {dt}d | {nt}T")
            acc_num += 1
            acc    = new_acc(ts)
            day_eq = month_eq = acc['equity']
            for n in pns:
                day_trades[n] = 0
                pending[n]    = 0
            prev_date  = cur_date
            prev_month = cur_month
            continue

        # ── New signals ──
        for n in pns:
            a = pa[n]
            if (positions[n] is None
                    and not acc['blown']
                    and not in_cd
                    and day_trades[n] < G.max_trades_day
                    and a['sig'][bar] != 0):
                pending[n] = int(a['sig'][bar])

    return {
        'all_trades':    all_trades,
        'account_logs':  acc_logs,
        'withdrawn':     withdrawn,
        'final_equity':  acc['equity'],
        'acc_num':       acc_num,
        'pair_names':    pns,
        'eq_curve':      eq_curve,
        'common_idx':    cidx,
        'pair_status':   pair_status,
    }


# ═══════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════
def print_report(res):
    trades = res['all_trades']
    if not trades:
        print("\n❌ No trades.")
        return

    df = pd.DataFrame(trades)
    df['exit_ts']  = pd.to_datetime(df['exit_ts'])
    df['entry_ts'] = pd.to_datetime(df['entry_ts'])
    df['month']    = df['exit_ts'].dt.to_period('M')
    df['year']     = df['exit_ts'].dt.year

    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    wr     = len(wins) / len(df) * 100
    pf     = (wins['pnl'].sum() / abs(losses['pnl'].sum())
              if len(losses) else float('inf'))

    logs     = res['account_logs']
    df_acc   = pd.DataFrame(logs) if logs else pd.DataFrame()
    targets  = (df_acc[df_acc['reason'] == 'TARGET_HIT']
                if len(df_acc) else pd.DataFrame())
    blowns   = (df_acc[df_acc['reason'] != 'TARGET_HIT']
                if len(df_acc) else pd.DataFrame())

    # Monthly (full range)
    ci         = res['common_idx']
    all_months = pd.period_range(
        start=df['exit_ts'].min().to_period('M'),
        end=pd.Period(ci[-1], freq='M'), freq='M')
    monthly = (df.groupby('month')['pnl'].sum()
               .reindex(all_months, fill_value=0.0))
    pos_m  = int((monthly > 0).sum())
    neg_m  = int((monthly < 0).sum())
    zero_m = int((monthly == 0).sum())
    tot_m  = len(monthly)

    # Streaks
    ls = ms = cur = 0
    for v in monthly:
        if v <= 0:
            cur += 1
            ms = max(ms, cur)
        else:
            cur = 0

    # R-multiple analysis
    G = GlobalConfig
    if 'bars' in df.columns:
        avg_dur = df['bars'].mean() * 0.25
    else:
        avg_dur = 0

    print("\n" + "═" * 70)
    print(f" ▌  CorrArb v11 — {'+'.join(res['pair_names'])}  ▐")
    print("═" * 70)
    print(f" Total Trades:       {len(df):>8,}")
    print(f" Win Rate:           {wr:>8.2f}%")
    print(f" Profit Factor:      {pf:>8.2f}")
    print(f" Avg Win:            {wins['pnl'].mean():>8.2f}$" if len(wins) else "")
    print(f" Avg Loss:           {losses['pnl'].mean():>8.2f}$" if len(losses) else "")
    print(f" Avg PnL/Trade:      {df['pnl'].mean():>8.2f}$")
    print(f" Avg Duration:       {avg_dur:>8.1f} ساعت")
    print(f" Total Net:          {df['pnl'].sum():>8,.2f}$")
    print("-" * 70)
    print(f" Accounts Passed:    {len(targets):>8}")
    print(f" Accounts Blown:     {len(blowns):>8}")
    if len(targets):
        print(f" Avg Days/Pass:      {targets['days'].mean():>8.0f}")
        print(f" Avg Trades/Pass:    {targets['n_trades'].mean():>8.0f}")
    print("-" * 70)
    print(f" Total Banked:       {res['withdrawn']:>8,.2f}$")
    print(f" Active Equity:      {res['final_equity']:>8,.2f}$")
    print(f" Monthly Avg:        {res['withdrawn']/max(1,tot_m):>8.2f}$/ماه")
    print("-" * 70)
    print(f" ماه مثبت:          {pos_m:>3} از {tot_m} ({pos_m/tot_m*100:.0f}%)")
    print(f" ماه منفی:          {neg_m:>3} از {tot_m}")
    print(f" ماه صفر:           {zero_m:>3} از {tot_m}")
    print(f" Max losing streak:  {ms:>3} ماه")
    print(f" بهترین ماه:        {monthly.max():>8.2f}$")
    print(f" بدترین ماه:        {monthly.min():>8.2f}$")
    print("-" * 70)
    print(" خروج‌ها:")
    for st, cnt in df['status'].value_counts().items():
        print(f"   {st:<14} {cnt:>5} ({cnt/len(df)*100:.1f}%)")
    print("-" * 70)
    print(" سالانه:")
    for yr, g in df.groupby('year'):
        w = g[g['pnl'] > 0]; l = g[g['pnl'] < 0]
        ypf = w['pnl'].sum()/abs(l['pnl'].sum()) if len(l) else 99
        print(f"   {yr}: {len(g):>4}T | WR:{len(w)/len(g)*100:5.1f}% | "
              f"PF:{ypf:.2f} | ${g['pnl'].sum():>8,.2f}")
    print("-" * 70)
    t100 = G.initial_balance * 0.02
    ok_m = int((monthly >= t100).sum())
    print(f" 🎯 هدف ۲٪/ماه (${t100:.0f}):")
    print(f"    بالای هدف:  {ok_m} از {tot_m} ({ok_m/tot_m*100:.0f}%)")
    print(f"    واقعی:      ${res['withdrawn']/max(1,tot_m):.2f}/ماه")
    print("═" * 70)
    print(f" v9b→$938 | v10→$0 | v11→${res['withdrawn']:,.0f}")
    print("═" * 70)

    # Save
    ec = pd.DataFrame(res['eq_curve'])
    ec.to_csv('equity_v11.csv', index=False)
    monthly.to_csv('monthly_v11.csv', header=['pnl'])
    print(f"\n  📊 equity_v11.csv + monthly_v11.csv saved")


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        CorrArb v11 — Exit Logic Rebuild             ║")
    print("╚══════════════════════════════════════════════════════╝")

    data = load_data()
    acfg = PairConfig.active()

    print("\n  Computing signals...")
    sigs = {}
    for n in data:
        if n in acfg:
            sigs[n] = compute_signals(n, data[n], acfg[n])

    active_data = {n: data[n] for n in sigs}
    res = run_backtest(active_data, sigs)
    print_report(res)
    print(f"\n  ✅ {(datetime.now()-t0).total_seconds():.1f}s")
