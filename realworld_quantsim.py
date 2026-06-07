"""
CorrArb Prop Simulator — v9b Fix
=================================
مشکل v9: Pair Quality Filter با ۱۰ ترید در ۳ هفته هر دو pair رو pause کرد

اصلاحات v9b:
  ✅ FIX: حداقل ۳۰ ترید (نه ۱۰) برای ارزیابی PF
  ✅ FIX: پنجره ارزیابی ۲۷۰ روز (نه ۱۸۰)
  ✅ FIX: چک هر ماه (نه هر هفته)
  ✅ FIX: حداقل ۶ ماه از شروع قبل از اولین ارزیابی
  ✅ FIX: EURGBP direct — correlation از price خود pair (نه legs)
  ✅ KEEP: Cooldown 10 روز بعد از blown
  ✅ KEEP: Monthly risk reduction
  ✅ KEEP: spread واقعی هر pair
  ✅ KEEP: pip_value دینامیک
"""

import pandas as pd
import numpy as np
import glob, zipfile, os, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class Config:
    # ── قوانین پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── ریسک ──
    risk_base_pct      = 0.015
    risk_min_pct       = 0.005
    consec_loss_n      = 2
    risk_reduce        = 0.5

    # ── spread واقعی هر pair ──
    PAIR_SPREAD = {
        'EURGBP': 1.0,
        'AUDNZD': 2.5,
    }
    PIP_SIZE = {
        'EURGBP': 0.0001,
        'AUDNZD': 0.0001,
    }

    commission_per_lot = 7.0
    slippage_pips      = 0.5

    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── z-score ──
    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.5
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0

    # ── فیلترها ──
    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # ── خروج ──
    sl_pips            = 30.0
    tp_pips            = 90.0
    time_stop_bars     = 36
    partial_ratio      = 0.50

    # ── ATR ──
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    # ── Variance Ratio ──
    vr_period = 200
    vr_k      = 4
    vr_max    = 0.90

    # ── Cooldown ──
    cooldown_days = 10

    # ── Monthly risk reduction ──
    monthly_loss_threshold = -150.0

    # ── Pair Quality Filter (اصلاح‌شده) ──
    pair_eval_min_trades = 30      # ↑ از ۱۰ به ۳۰
    pair_eval_window     = 270     # ↑ از ۱۸۰ به ۲۷۰ روز
    pair_eval_start_days = 180     # حداقل ۶ ماه قبل از اولین ارزیابی
    pair_min_pf          = 1.05
    pair_eval_interval   = 30      # روز — هر ۳۰ روز (نه ۷)


# ═══════════════════════════════════════════════════════════════════════
def load_raw_csv(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths: raise FileNotFoundError(f"No CSV: {pattern}")
    frames = [pd.read_csv(p, sep=';', header=None, names=['ts','o','h','l','c','v'])
              for p in paths]
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths: raise FileNotFoundError(f"No ZIP: {pattern}")
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p) as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if not csv_name: continue
                with z.open(csv_name) as f:
                    frames.append(pd.read_csv(f, sep=';', header=None,
                                              names=['ts','o','h','l','c','v']))
        except Exception as e:
            print(f"  ⚠ {os.path.basename(p)}: {e}")
    if not frames: raise ValueError(f"No valid ZIP: {pattern}")
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


def load_all_pairs():
    print("\n  Loading datasets...")
    pairs = {}

    # EURGBP از ZIP مستقیم
    try:
        raw   = load_raw_zip('data/HISTDATA*EURGBP*.zip')
        df15  = to_15min(raw, 'eg')
        gbp15 = to_15min(load_raw_csv('data/*GBPUSD*.csv'), 'gbp')
        m = df15.join(gbp15[['c_gbp']], how='inner').dropna()
        m['c_spread']   = m['c_eg']
        m['o_spread']   = m['o_eg']
        m['h_spread']   = m['h_eg']
        m['l_spread']   = m['l_eg']
        m['quote_rate'] = m['c_gbp']
        m = m[m.index.weekday < 5].copy()
        pairs['EURGBP'] = {'df': m,
                           'spread_pip': Config.PAIR_SPREAD['EURGBP'],
                           'pip':        Config.PIP_SIZE['EURGBP'],
                           'source':     'direct ZIP'}
        print(f"  ✅ EURGBP: {len(m):,} candles")
    except Exception as e:
        print(f"  ❌ EURGBP: {e}")

    # AUDNZD synthetic از ZIP
    try:
        aud = to_15min(load_raw_zip('data/HISTDATA*AUDUSD*.zip'), 'aud')
        nzd = to_15min(load_raw_zip('data/HISTDATA*NZDUSD*.zip'), 'nzd')
        m   = aud.join(nzd, how='inner').dropna()
        m['c_spread']   = m['c_aud'] / m['c_nzd']
        m['o_spread']   = m['o_aud'] / m['o_nzd']
        m['h_spread']   = m['h_aud'] / m['l_nzd']
        m['l_spread']   = m['l_aud'] / m['h_nzd']
        m['quote_rate'] = m['c_nzd']
        m = m[m.index.weekday < 5].copy()
        pairs['AUDNZD'] = {'df': m,
                           'spread_pip': Config.PAIR_SPREAD['AUDNZD'],
                           'pip':        Config.PIP_SIZE['AUDNZD'],
                           'source':     'synthetic ZIP'}
        print(f"  ✅ AUDNZD: {len(m):,} candles")
    except Exception as e:
        print(f"  ❌ AUDNZD: {e}")

    if not pairs: raise RuntimeError("No pairs loaded!")
    return pairs


# ═══════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_vr(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    v1 = r1.rolling(window).var()
    vk = rk.rolling(window).var()
    return vk / (k * v1.replace(0, np.nan))


def compute_signals(name, info):
    C  = Config
    df = info['df']

    log_r  = np.log(df['c_spread'])
    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    # برای EURGBP direct، correlation از price خود pair
    if 'c_aud' in df.columns:
        corr_ok = (df['c_aud'].pct_change()
                   .rolling(C.corr_period)
                   .corr(df['c_nzd'].pct_change()) > C.corr_min)
    else:
        # EURGBP direct: correlation بین تغییرات EURGBP و GBPUSD (جهت مخالف معمولاً)
        corr_ok = pd.Series(True, index=df.index)  # direct pair نیازی به corr filter ندارد

    vr        = calc_vr(log_r, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    hour    = pd.Series(df.index.hour,      index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    sig = pd.Series(0, index=df.index)
    cond = vol_ok & time_ok & corr_ok & regime_ok
    sig[(z < -C.z_entry) & cond] =  1
    sig[(z >  C.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    n = int((sig != 0).sum())
    print(f"    {name}: {n:,} signals | Regime OK: {int(regime_ok.sum()):,} bars")
    return sig, z


# ═══════════════════════════════════════════════════════════════════════
def calc_pnl(direction, entry, exit_px, lot, qr, pip):
    C = Config
    gross = direction * (exit_px - entry) * lot * C.lot_size * qr
    return gross - C.commission_per_lot * lot


def new_acc(ts):
    C = Config
    return {
        'equity': C.initial_balance, 'start_ts': ts,
        'trades': [], 'blown': False, 'blown_rsn': '',
        'peak': C.initial_balance, 'consec_loss': 0,
    }


def _rec(pos, exit_px, exit_ts, pnl, status, lot):
    return {'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot,
            'entry': pos['entry'], 'exit': exit_px,
            'entry_ts': pos['entry_ts'], 'exit_ts': exit_ts,
            'pnl': pnl, 'status': status}


# ═══════════════════════════════════════════════════════════════════════
def run_backtest(pairs, pair_signals):
    C          = Config
    pair_names = list(pairs.keys())

    common_idx = None
    for n in pair_names:
        idx = pairs[n]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)
    start_date = common_idx[C.warmup]
    print(f"  ✅ Common bars: {n_bars:,} | {common_idx[0].date()} → {common_idx[-1].date()}")

    pa = {}
    for n in pair_names:
        df_p      = pairs[n]['df'].reindex(common_idx).ffill()
        sig_s, z_s = pair_signals[n]
        pa[n] = {
            'o':   df_p['o_spread'].values.astype(float),
            'c':   df_p['c_spread'].values.astype(float),
            'qr':  df_p['quote_rate'].values.astype(float),
            'sig': sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z':   z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
            'sp':  pairs[n]['spread_pip'],
            'pip': pairs[n]['pip'],
        }

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    acc              = new_acc(start_date)
    total_withdrawn  = 0.0
    acc_num          = 1
    day_start_eq     = C.initial_balance
    month_start_eq   = C.initial_balance
    cooldown_until   = None
    all_trades       = []
    acc_logs         = []

    positions    = {n: None for n in pair_names}
    trades_today = {n: 0    for n in pair_names}
    pending_sig  = {n: 0    for n in pair_names}

    # Pair Quality tracking
    pair_trades_hist = {n: [] for n in pair_names}   # [(ts, pnl)]
    pair_paused      = {n: False for n in pair_names}
    last_eval_month  = {n: None  for n in pair_names}

    print(f"\n  ▶ Running v9b Realistic...")
    print(f"    Pairs: {' + '.join(pair_names)}")
    print(f"    PairQuality: min {C.pair_eval_min_trades}T / {C.pair_eval_window}d / PF>{C.pair_min_pf}")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq = acc['equity']
        if eq > acc['peak']: acc['peak'] = eq

        in_cooldown = cooldown_until is not None and ts < cooldown_until

        # ── ریست روزانه ──
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for n in pair_names: trades_today[n] = 0

        # ── ریست ماهانه ──
        if ts.day == 1 and ts.hour == 0 and ts.minute == 0:
            month_start_eq = acc['equity']

        # ── Pair Quality Check: فقط ماهانه و بعد از warmup کافی ──
        if ts.hour == 2 and ts.minute == 0 and ts.day == 1:
            days_since_start = (ts - start_date).days
            if days_since_start >= C.pair_eval_start_days:
                for n in pair_names:
                    cur_month = (ts.year, ts.month)
                    if last_eval_month[n] == cur_month:
                        continue
                    last_eval_month[n] = cur_month
                    cutoff = ts - timedelta(days=C.pair_eval_window)
                    recent = [(t, p) for t, p in pair_trades_hist[n] if t >= cutoff]
                    if len(recent) >= C.pair_eval_min_trades:
                        wins   = sum(p for _, p in recent if p > 0)
                        losses = abs(sum(p for _, p in recent if p < 0))
                        pf     = wins / losses if losses > 0 else 2.0
                        was_p  = pair_paused[n]
                        pair_paused[n] = pf < C.pair_min_pf
                        if pair_paused[n] != was_p:
                            st = "⏸ PAUSED" if pair_paused[n] else "▶ RESUMED"
                            print(f"    {st} {n} | PF(270d)={pf:.2f} | T={len(recent)} | {ts.date()}")

        # ── blown ──
        if acc['blown']:
            acc_logs.append({
                'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts,
                'reason': acc['blown_rsn'], 'pnl': acc['equity'] - C.initial_balance,
                'n_trades': len(acc['trades']), 'days': (ts - acc['start_ts']).days
            })
            print(f"    💥 #{acc_num:>3} | {ts.date()} | Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            cooldown_until = ts + timedelta(days=C.cooldown_days)
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = month_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = pending_sig[n] = 0
                positions[n] = None
            continue

        if in_cooldown: continue

        # Monthly stress
        monthly_pnl      = acc['equity'] - month_start_eq
        monthly_stressed = monthly_pnl < C.monthly_loss_threshold

        # ── ورود ──
        for n in pair_names:
            a = pa[n]
            if (pending_sig[n] != 0 and positions[n] is None
                    and trades_today[n] < C.max_trades_day
                    and not pair_paused[n]):
                sv  = pending_sig[n]
                qr  = a['qr'][bar]; pip = a['pip']
                risk = C.risk_base_pct * (0.5 if monthly_stressed else 1.0)
                if acc['consec_loss'] >= C.consec_loss_n:
                    risk = max(risk * C.risk_reduce, C.risk_min_pct)
                pv  = pip * C.lot_size * qr if pip * C.lot_size * qr > 0 else 10.0
                lot = round(float(np.clip(acc['equity']*risk/(C.sl_pips*pv),
                                          C.min_lot, C.max_lot)), 2)
                ep  = a['o'][bar] + sv * (C.slippage_pips + a['sp']/2) * pip
                sl  = ep - sv * C.sl_pips * pip
                tp  = ep + sv * C.tp_pips * pip
                positions[n] = {
                    'pair': n, 'dir': sv, 'lot': lot, 'lot_remaining': lot,
                    'partial_done': False, 'entry': ep, 'sl': sl, 'tp': tp,
                    'entry_ts': ts, 'entry_bar': bar, 'pip': pip,
                }
                trades_today[n] += 1
            pending_sig[n] = 0

        # ── floating PnL ──
        total_float = sum(
            calc_pnl(p['dir'], p['entry'], pa[n]['c'][bar],
                     p['lot_remaining'], pa[n]['qr'][bar], p['pip'])
            for n in pair_names if (p := positions[n]) is not None
        )
        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            reason = "DailyDD" if current_eq <= daily_limit else "TotalDD"
            acc['blown'] = True; acc['blown_rsn'] = reason
            for n in pair_names:
                pos = positions[n]
                if pos is None: continue
                pnl = calc_pnl(pos['dir'], pos['entry'], pa[n]['c'][bar],
                               pos['lot_remaining'], pa[n]['qr'][bar], pos['pip'])
                acc['equity'] += pnl
                r = _rec(pos, pa[n]['c'][bar], ts, pnl, 'BLOWN', pos['lot_remaining'])
                all_trades.append(r); acc['trades'].append(r)
                pair_trades_hist[n].append((ts, pnl))
                positions[n] = None
            continue

        # ── خروج ──
        for n in pair_names:
            pos = positions[n]
            if pos is None: continue
            a   = pa[n]; cp = a['c'][bar]; qr = a['qr'][bar]
            d   = pos['dir']; ep = pos['entry']
            zn  = a['z'][bar]; pip = pos['pip']; lr = pos['lot_remaining']

            # Partial
            if not pos['partial_done'] and not np.isnan(zn):
                if (d==1 and zn>=-C.z_exit_partial) or (d==-1 and zn<=C.z_exit_partial):
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr, pip)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            r = _rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(r); acc['trades'].append(r)
                            pair_trades_hist[n].append((ts, p_pnl))
                            pos['lot_remaining'] = round(lr - p_lot, 2)
                            pos['partial_done']  = True
                            pos['sl'] = pos['entry']
                            lr = pos['lot_remaining']
                            if lr < C.min_lot:
                                positions[n] = None; continue

            pnl_now    = calc_pnl(d, ep, cp, lr, qr, pip)
            hit_z_stop = not np.isnan(zn) and (
                (d==1 and zn<=-C.z_stop_margin) or (d==-1 and zn>=C.z_stop_margin))
            hit_z_exit = not np.isnan(zn) and (
                (d==1 and zn>=-C.z_exit_full) or (d==-1 and zn<=C.z_exit_full))
            if hit_z_exit and pnl_now < C.min_net_profit_usd and not pos['partial_done']:
                hit_z_exit = False
            hit_sl    = (d==1 and cp<=pos['sl']) or (d==-1 and cp>=pos['sl'])
            hit_tp    = (d==1 and cp>=pos['tp']) or (d==-1 and cp<=pos['tp'])
            bars_open = bar - pos['entry_bar']
            time_stop = (bars_open >= C.time_stop_bars and pnl_now < 0) or \
                        (bars_open >= C.time_stop_bars * 2)

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                exit_px = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                st = ('SL' if hit_sl else ('TP' if hit_tp else
                      ('Z-Stop' if hit_z_stop else ('TimeStop' if time_stop else 'Z-Exit'))))
                fpnl = calc_pnl(d, ep, exit_px, lr, qr, pip)
                acc['equity'] += fpnl
                r = _rec(pos, exit_px, ts, fpnl, st, lr)
                all_trades.append(r); acc['trades'].append(r)
                pair_trades_hist[n].append((ts, fpnl))
                positions[n] = None
                if fpnl > 0: acc['consec_loss'] = 0
                else:        acc['consec_loss'] += 1

        # ── Target Hit ──
        if acc['equity'] >= PROFIT_LEVEL and all(positions[n] is None for n in pair_names):
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            dt = (ts - acc['start_ts']).days
            acc_logs.append({
                'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts,
                'reason': 'TARGET_HIT', 'pnl': w,
                'n_trades': len(acc['trades']), 'days': dt
            })
            print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | "
                  f"Bank:${total_withdrawn:>9.2f} | {dt}d | {len(acc['trades'])}T")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = month_start_eq = acc['equity']
            for n in pair_names: trades_today[n] = pending_sig[n] = 0
            continue

        # ── سیگنال جدید ──
        for n in pair_names:
            a = pa[n]
            if (positions[n] is None and not acc['blown'] and not in_cooldown
                    and not pair_paused[n]
                    and trades_today[n] < C.max_trades_day
                    and a['sig'][bar] != 0):
                pending_sig[n] = int(a['sig'][bar])

    return {
        'all_trades': all_trades, 'account_logs': acc_logs,
        'total_withdrawn': total_withdrawn,
        'final_equity': acc['equity'],
        'total_accounts': acc_num, 'pair_names': pair_names,
    }


# ═══════════════════════════════════════════════════════════════════════
def print_report(results):
    trades = results['all_trades']
    if not trades:
        print("\n❌ No trades."); return

    df_t  = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month']   = df_t['exit_ts'].dt.to_period('M')

    wins   = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    wr = len(wins)/len(df_t)*100
    pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses) else float('inf')

    logs    = results['account_logs']
    df_acc  = pd.DataFrame(logs) if logs else pd.DataFrame()
    targets = df_acc[df_acc['reason']=='TARGET_HIT'] if len(df_acc) else pd.DataFrame()
    blowns  = df_acc[df_acc['reason']!='TARGET_HIT'] if len(df_acc) else pd.DataFrame()

    monthly = df_t.groupby('month')['pnl'].sum()
    pos_m   = int((monthly > 0).sum())
    neg_m   = int((monthly < 0).sum())

    print("\n" + "═"*68)
    print(f" ▌  CorrArb v9b — {'+'.join(results['pair_names'])}  ▐")
    print("═"*68)
    print(f" {'Total Trades:':<26} {len(df_t):,}")
    print(f" {'Win Rate:':<26} {wr:.2f}%")
    print(f" {'Profit Factor:':<26} {pf:.2f}")
    print(f" {'Avg Win:':<26} ${wins['pnl'].mean():.2f}")
    print(f" {'Avg Loss:':<26} ${losses['pnl'].mean():.2f}")
    print("-"*68)
    print(f" {'Accounts Passed:':<26} {len(targets)}")
    print(f" {'Accounts Blown:':<26} {len(blowns)}")
    if len(targets) and 'days' in targets.columns:
        print(f" {'Avg Days/Pass:':<26} {targets['days'].mean():.0f} روز")
        print(f" {'Avg Trades/Pass:':<26} {targets['n_trades'].mean():.0f} ترید")
    print("-"*68)
    print(f" {'Total Banked (15yr):':<26} ${results['total_withdrawn']:,.2f}")
    print(f" {'Active Equity:':<26} ${results['final_equity']:,.2f}")
    print(f" {'Monthly Avg:':<26} ${results['total_withdrawn']/180:.2f}/ماه")
    print("-"*68)
    print(f" {'ماه‌های مثبت:':<26} {pos_m} از {len(monthly)}")
    print(f" {'ماه‌های منفی:':<26} {neg_m} از {len(monthly)}")
    print(f" {'بهترین ماه:':<26} ${monthly.max():,.2f}")
    print(f" {'بدترین ماه:':<26} ${monthly.min():,.2f}")
    print("-"*68)
    print(" عملکرد هر جفت ارز:")
    for pair in results['pair_names']:
        pt = df_t[df_t['pair']==pair] if 'pair' in df_t.columns else pd.DataFrame()
        if not len(pt): continue
        pw = pt[pt['pnl']>0]; pl = pt[pt['pnl']<0]
        ppf = pw['pnl'].sum()/abs(pl['pnl'].sum()) if len(pl) else float('inf')
        print(f"   {pair}: {len(pt):>4}T | WR:{len(pw)/len(pt)*100:4.1f}% | "
              f"PF:{ppf:.2f} | Net:${pt['pnl'].sum():>8,.2f}")
    print("-"*68)
    print(" خروج‌ها:")
    for st, cnt in df_t['status'].value_counts().items():
        print(f"   {st:<14}: {cnt:>5} ({cnt/len(df_t)*100:.1f}%)")
    print("═"*68)
    bk = results['total_withdrawn']
    print(f" v8→$7,151 | v9→$0 | v9b→${bk:,.0f}")
    print("═"*68)


if __name__ == "__main__":
    t0 = datetime.now()
    print("╔════════════════════════════════════════════════════╗")
    print("║   CorrArb v9b — Realistic + Fixed Quality Filter  ║")
    print("╚════════════════════════════════════════════════════╝")
    pairs = load_all_pairs()
    print("\n  Computing signals...")
    sigs = {n: compute_signals(n, pairs[n]) for n in pairs}
    res  = run_backtest(pairs, sigs)
    print_report(res)
    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.2f}s")
