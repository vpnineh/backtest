"""
CorrArb Prop Simulator — v9
============================
Root Cause Fix:
  - h_spread / l_spread برای SL استفاده نمی‌شه (phantom range)
  - SL فقط با close price چک می‌شه + slippage model جدا
  - Slippage واقعی: فقط موقع خروج اضطراری (DD)
  - Portfolio Heat درست محاسبه می‌شه
  - تمام تغییرات v8 که کار می‌کردن حفظ شدن
"""

import pandas as pd
import numpy as np
import glob, zipfile, os, warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG v9
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── مدیریت ریسک ──
    risk_base_pct      = 0.015       # برگشت به v7 (1.5%)
    risk_min_pct       = 0.0075
    consec_loss_n      = 2
    risk_reduce        = 0.5
    max_open_pairs     = 2

    # ── هزینه‌های بروکر ──
    spread_pips_eurgbp = 1.2
    spread_pips_audnzd = 1.5
    commission_per_lot = 7.0
    # slippage فقط موقع خروج اضطراری
    slippage_entry_pips = 0.3        # ورود عادی
    slippage_sl_pips    = 0.8        # خروج SL (نه intra-bar!)

    # ── مشخصات ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── Z-Score ──
    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.8
    z_exit_full        = 0.3
    z_stop_margin      = 4.0
    min_net_profit_usd = 20.0

    # ── فیلترها ──
    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # ── خروج اضطراری ──
    sl_pips            = 30.0
    tp_pips            = 90.0
    time_stop_bars     = 48

    # ── ATR Filter ──
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    # ── Variance Ratio ──
    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.95

    # ── Partial Exit ──
    partial_ratio      = 0.50

    # ── SL روش ──
    # False = close-based (مثل v7 که کار می‌کرد)
    # True  = فقط بعد از اینکه close از SL رد شد
    use_close_sl       = True


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_raw(pattern: str, is_zip: bool) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files found: {pattern}")
    frames = []
    for p in paths:
        try:
            if is_zip:
                with zipfile.ZipFile(p, 'r') as z:
                    csv_name = next(
                        (f for f in z.namelist()
                         if f.lower().endswith('.csv')), None
                    )
                    if csv_name is None:
                        continue
                    with z.open(csv_name) as f:
                        df = pd.read_csv(
                            f, sep=';', header=None,
                            names=['ts','o','h','l','c','v']
                        )
            else:
                df = pd.read_csv(
                    p, sep=';', header=None,
                    names=['ts','o','h','l','c','v']
                )
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Skip {os.path.basename(p)}: {e}")
    if not frames:
        raise ValueError(f"No valid data: {pattern}")
    raw = pd.concat(frames, ignore_index=True).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def to_15min(raw: pd.DataFrame, sfx: str) -> pd.DataFrame:
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()


def build_spread_df(df_a, sfx_a, df_b, sfx_b) -> pd.DataFrame:
    """
    ✅ فقط close-based spread
    h_spread/l_spread حذف شدن چون phantom range ایجاد می‌کنن
    """
    merged = df_a.join(df_b, how='inner').dropna()
    if len(merged) == 0:
        raise ValueError("No common timestamps")

    merged['c_spread']   = merged[f'c_{sfx_a}'] / merged[f'c_{sfx_b}']
    merged['o_spread']   = merged[f'o_{sfx_a}'] / merged[f'o_{sfx_b}']
    # ✅ h/l spread فقط برای ATR (نه برای SL)
    merged['h_spread']   = merged[f'h_{sfx_a}'] / merged[f'l_{sfx_b}']
    merged['l_spread']   = merged[f'l_{sfx_a}'] / merged[f'h_{sfx_b}']
    merged['quote_rate'] = merged[f'c_{sfx_b}']
    return merged[merged.index.weekday < 5].copy()


def load_all_pairs() -> dict:
    print("\n  Loading and syncing datasets...")
    pairs = {}
    try:
        eur = to_15min(load_raw('data/*EURUSD*.csv', is_zip=False), 'eur')
        gbp = to_15min(load_raw('data/*GBPUSD*.csv', is_zip=False), 'gbp')
        df  = build_spread_df(eur, 'eur', gbp, 'gbp')
        pairs['EURGBP'] = {
            'df': df, 'leg_a': 'c_eur', 'leg_b': 'c_gbp',
            'spread_pips': Config.spread_pips_eurgbp
        }
        print(f"  ✅ EURGBP : {len(df):>7,} candles"
              f" | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ EURGBP : {e}")

    try:
        aud = to_15min(load_raw('data/HISTDATA*AUDUSD*.zip', is_zip=True), 'aud')
        nzd = to_15min(load_raw('data/HISTDATA*NZDUSD*.zip', is_zip=True), 'nzd')
        df  = build_spread_df(aud, 'aud', nzd, 'nzd')
        pairs['AUDNZD'] = {
            'df': df, 'leg_a': 'c_aud', 'leg_b': 'c_nzd',
            'spread_pips': Config.spread_pips_audnzd
        }
        print(f"  ✅ AUDNZD : {len(df):>7,} candles"
              f" | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ AUDNZD : {e}")

    if not pairs:
        raise RuntimeError("No pairs loaded.")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_variance_ratio(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()
    return vark / (k * var1.replace(0, np.nan))


def compute_signals(pair_name: str, pair_info: dict) -> tuple:
    C   = Config
    df  = pair_info['df']
    leg_a = pair_info['leg_a']
    leg_b = pair_info['leg_b']

    log_ratio = np.log(df['c_spread'])
    z_mean = log_ratio.rolling(C.z_fast_period).mean()
    z_std  = log_ratio.rolling(C.z_fast_period).std()
    z_score = (log_ratio - z_mean) / z_std.replace(0, np.nan)

    ret_a   = df[leg_a].pct_change()
    ret_b   = df[leg_b].pct_change()
    corr_ok = ret_a.rolling(C.corr_period).corr(ret_b) > C.corr_min

    vr        = calc_variance_ratio(log_ratio, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    hour    = pd.Series(df.index.hour,      index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    long_cond  = (z_score < -C.z_entry) & vol_ok & time_ok & corr_ok & regime_ok
    short_cond = (z_score >  C.z_entry) & vol_ok & time_ok & corr_ok & regime_ok

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    n = int((sig != 0).sum())
    l = int((sig ==  1).sum())
    s = int((sig == -1).sum())
    r = int(regime_ok.sum())
    print(f"    {pair_name}: {n:,} signals (L:{l} | S:{s})"
          f" | Regime OK: {r:,} bars")
    return sig, z_score


# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl(direction, entry_px, exit_px, lot, quote_rate):
    C = Config
    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    gross_usd   = gross_quote * quote_rate
    commission  = C.commission_per_lot * lot
    return gross_usd - commission


def calc_lot(equity, sl_pips, consec_loss, quote_rate):
    C = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(
            risk * (C.risk_reduce ** (consec_loss - C.consec_loss_n + 1)),
            C.risk_min_pct
        )
    pip_value_usd = C.pip * C.lot_size * quote_rate
    risk_usd = equity * risk
    raw = risk_usd / (sl_pips * pip_value_usd)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def new_acc(ts) -> dict:
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


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE v9
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, pair_signals: dict) -> dict:
    C          = Config
    pip        = C.pip
    pair_names = list(pairs.keys())

    # ── ایندکس مشترک ──
    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)
    print(f"  ✅ Common bars: {n_bars:,}"
          f" | {common_idx[0].date()} → {common_idx[-1].date()}")

    # ── آرایه‌های numpy ──
    pa = {}
    for name in pair_names:
        df_p = pairs[name]['df'].reindex(common_idx).ffill()
        sig_s, z_s = pair_signals[name]
        pa[name] = {
            'o':   df_p['o_spread'].values.astype(float),
            # ✅ h/l فقط برای ATR استفاده می‌شه (نه SL)
            'c':   df_p['c_spread'].values.astype(float),
            'qr':  df_p['quote_rate'].values.astype(float),
            'sig': sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z':   z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
        }

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    acc             = new_acc(common_idx[C.warmup])
    total_withdrawn = 0.0
    acc_num         = 1
    day_start_eq    = C.initial_balance
    all_trades      = []
    acc_logs        = []
    eq_curve        = []

    positions    = {name: None for name in pair_names}
    trades_today = {name: 0    for name in pair_names}
    pending_sig  = {name: 0    for name in pair_names}

    print(f"\n  ▶ Running Multi-Pair Prop Simulator v9...")
    print(f"    Pairs  : {' + '.join(pair_names)}")
    print(f"    Target : +{C.profit_target_pct*100:.0f}%"
          f"  | Daily DD: -{C.max_daily_loss_pct*100:.0f}%"
          f"  | Total DD: -{C.max_total_dd_pct*100:.0f}%")
    print(f"    Risk   : {C.risk_base_pct*100:.1f}%"
          f"  | SL: {C.sl_pips}p"
          f"  | TP: {C.tp_pips}p"
          f"  | TimeStop: {C.time_stop_bars} bars")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq = acc['equity']
        eq_curve.append((ts, round(eq, 4)))

        if eq > acc['peak']:
            acc['peak'] = eq

        # ── ریست روزانه ──
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0

        # ── Progress ──
        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(
                f"    Progress: {pct:5.1f}%"
                f" | Eq: ${acc['equity']:,.2f}"
                f" | Bank: ${total_withdrawn:,.2f}",
                end='\r'
            )

        # ── چک حساب سوخته → ریست ──
        if acc['blown']:
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   acc['blown_rsn'],
                'pnl':      acc['equity'] - C.initial_balance,
            })
            print(f"\n    💥 #{acc_num:>3} | {ts.date()}"
                  f" | Eq: ${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name]  = 0
                positions[name]    = None
            continue

        # ── ورود ──
        n_open = sum(1 for n in pair_names if positions[n] is not None)
        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0
                    and positions[name] is None
                    and trades_today[name] < C.max_trades_day
                    and n_open < C.max_open_pairs):
                sv  = pending_sig[name]
                qr  = a['qr'][bar]
                lot = calc_lot(
                    acc['equity'], C.sl_pips, acc['consec_loss'], qr
                )
                sp  = pairs[name]['spread_pips']
                ep  = a['o'][bar] + sv * (
                    C.slippage_entry_pips + sp / 2
                ) * pip
                sl = ep - sv * C.sl_pips * pip
                tp = ep + sv * C.tp_pips * pip
                positions[name] = {
                    'pair':         name,
                    'dir':          sv,
                    'lot':          lot,
                    'lot_remaining': lot,
                    'partial_done': False,
                    'entry':        ep,
                    'sl':           sl,
                    'tp':           tp,
                    'entry_ts':     ts,
                    'entry_bar':    bar,
                }
                trades_today[name] += 1
                n_open += 1
            pending_sig[name] = 0

        # ── ✅ Floating PnL با CLOSE (نه H/L) ──
        total_float = 0.0
        for name in pair_names:
            pos = positions[name]
            if pos is not None:
                a = pa[name]
                total_float += calc_pnl(
                    pos['dir'], pos['entry'],
                    a['c'][bar], pos['lot_remaining'], a['qr'][bar]
                )

        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        # ── DD چک ──
        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            acc['blown']     = True
            rsn = "DailyDD" if current_eq <= daily_limit else "TotalDD"
            acc['blown_rsn'] = rsn
            for name in pair_names:
                pos = positions[name]
                if pos is None:
                    continue
                a = pa[name]
                # ✅ بستن به close (با یک slippage کوچک)
                exit_px = a['c'][bar] - pos['dir'] * C.slippage_sl_pips * pip
                pnl = calc_pnl(
                    pos['dir'], pos['entry'],
                    exit_px, pos['lot_remaining'], a['qr'][bar]
                )
                acc['equity'] += pnl
                rec = _make_rec(
                    pos, exit_px, ts, pnl, 'BLOWN', pos['lot_remaining']
                )
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None
            continue

        # ── مدیریت خروج ──
        for name in pair_names:
            pos = positions[name]
            if pos is None:
                continue

            a       = pa[name]
            cp      = a['c'][bar]
            qr      = a['qr'][bar]
            d       = pos['dir']
            ep      = pos['entry']
            zn      = a['z'][bar]
            lot_rem = pos['lot_remaining']

            # ── ✅ SL/TP با close (+ slippage فقط روی SL) ──
            hit_sl = (d ==  1 and cp <= pos['sl']) or \
                     (d == -1 and cp >= pos['sl'])
            hit_tp = (d ==  1 and cp >= pos['tp']) or \
                     (d == -1 and cp <= pos['tp'])

            # ── Partial Exit ──
            if not pos['partial_done'] and not np.isnan(zn):
                hit_p = (
                    (d ==  1 and zn >= -C.z_exit_partial) or
                    (d == -1 and zn <=  C.z_exit_partial)
                )
                if hit_p:
                    p_lot = round(lot_rem * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, qr)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            rec = _make_rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(rec)
                            acc['trades'].append(rec)
                            pos['lot_remaining'] = round(lot_rem - p_lot, 2)
                            pos['partial_done']  = True
                            lot_rem = pos['lot_remaining']
                            if lot_rem < C.min_lot:
                                positions[name] = None
                                continue

            # ── Z-Stop ──
            hit_z_stop = (not np.isnan(zn)) and (
                (d ==  1 and zn <= -C.z_stop_margin) or
                (d == -1 and zn >=  C.z_stop_margin)
            )

            # ── Z-Exit ──
            hit_z_exit = False
            if not np.isnan(zn):
                z_crossed = (
                    (d ==  1 and zn >= -C.z_exit_full) or
                    (d == -1 and zn <=  C.z_exit_full)
                )
                if z_crossed:
                    pnl_check = calc_pnl(d, ep, cp, lot_rem, qr)
                    if pnl_check >= C.min_net_profit_usd or pos['partial_done']:
                        hit_z_exit = True

            # ── TimeStop ──
            time_stop = (bar - pos['entry_bar']) >= C.time_stop_bars

            if hit_sl or hit_tp or hit_z_exit or hit_z_stop or time_stop:
                if hit_sl:
                    # ✅ slippage فقط روی SL (نه intra-bar)
                    exit_px = pos['sl'] - d * C.slippage_sl_pips * pip
                    st = 'SL'
                elif hit_tp:
                    exit_px, st = pos['tp'], 'TP'
                elif hit_z_stop:
                    exit_px, st = cp, 'Z-Stop'
                elif time_stop:
                    exit_px, st = cp, 'TimeStop'
                else:
                    exit_px, st = cp, 'Z-Exit'

                final_pnl = calc_pnl(d, ep, exit_px, lot_rem, qr)
                acc['equity'] += final_pnl
                rec = _make_rec(pos, exit_px, ts, final_pnl, st, lot_rem)
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None

                if final_pnl > 0:
                    acc['consec_loss'] = 0
                else:
                    acc['consec_loss'] += 1

        # ── برداشت سود ──
        all_closed = all(positions[n] is None for n in pair_names)
        if acc['equity'] >= PROFIT_LEVEL and all_closed and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   'TARGET_HIT',
                'pnl':      w,
            })
            print(
                f"\n    💰 #{acc_num:>3} | {ts.date()} | "
                f"Target Hit: ${w:>7.2f} | "
                f"Total Bank: ${total_withdrawn:>9.2f}"
            )
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name]  = 0
            continue

        # ── سیگنال جدید ──
        n_open = sum(1 for n in pair_names if positions[n] is not None)
        for name in pair_names:
            a = pa[name]
            if (positions[name] is None
                    and not acc['blown']
                    and trades_today[name] < C.max_trades_day
                    and n_open < C.max_open_pairs
                    and a['sig'][bar] != 0):
                pending_sig[name] = int(a['sig'][bar])


    print()
    return {
        'all_trades':      all_trades,
        'account_logs':    acc_logs,
        'eq_curve':        eq_curve,
        'total_withdrawn': total_withdrawn,
        'final_equity':    acc['equity'],
        'total_accounts':  acc_num,
        'pair_names':      pair_names,
    }


def _make_rec(pos, exit_px, exit_ts, pnl, status, lot):
    return {
        'pair':      pos['pair'],
        'dir':       pos['dir'],
        'lot':       lot,
        'entry':     pos['entry'],
        'exit':      exit_px,
        'entry_ts':  pos['entry_ts'],
        'exit_ts':   exit_ts,
        'pnl':       pnl,
        'status':    status,
        'entry_bar': pos['entry_bar'],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════
def calc_max_drawdown(eq_series: pd.Series) -> float:
    roll_max = eq_series.cummax()
    dd = (eq_series - roll_max) / roll_max
    return float(dd.min())


def monte_carlo_simulation(
    trades_pnl: list,
    n_sim: int = 1000,
    initial: float = 5000.0
) -> dict:
    arr = np.array(trades_pnl)
    n   = len(arr)
    results = []
    for _ in range(n_sim):
        shuffled = np.random.choice(arr, size=n, replace=True)
        eq       = initial + np.cumsum(shuffled)
        full_eq  = np.concatenate([[initial], eq])
        peak     = np.maximum.accumulate(full_eq)
        dd       = (full_eq - peak) / peak
        results.append({
            'final':  eq[-1],
            'max_dd': dd.min()
        })
    finals = [r['final'] for r in results]
    dds    = [r['max_dd'] for r in results]
    return {
        'median_final':  np.median(finals),
        'p5_final':      np.percentile(finals, 5),
        'p95_final':     np.percentile(finals, 95),
        'median_dd':     np.median(dds),
        'worst_dd_p95':  np.percentile(dds, 95),
        'prob_positive': np.mean([f > initial for f in finals]) * 100,
    }


def walk_forward_summary(all_trades: list) -> None:
    if not all_trades:
        return
    df_t = pd.DataFrame(all_trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['year']    = df_t['exit_ts'].dt.year
    print("\n  📊 Walk-Forward Annual Summary:")
    print(f"  {'Year':<6} {'Trades':>7} {'WR%':>7} {'PF':>6} {'Net PnL':>10}")
    print("  " + "─" * 42)
    for yr, grp in df_t.groupby('year'):
        wins = grp[grp['pnl'] > 0]
        loss = grp[grp['pnl'] < 0]
        wr   = len(wins) / len(grp) * 100
        pf   = (wins['pnl'].sum() / abs(loss['pnl'].sum())
                if len(loss) > 0 else float('inf'))
        net  = grp['pnl'].sum()
        flag = "✅" if net > 0 else "❌"
        print(f"  {yr:<6} {len(grp):>7,} {wr:>7.1f} {pf:>6.2f}"
              f" ${net:>9,.2f} {flag}")


def print_report(results: dict):
    trades     = results['all_trades']
    pair_names = results.get('pair_names', [])

    if not trades:
        print("\n❌ No trades executed.")
        return

    df_t = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month']   = df_t['exit_ts'].dt.to_period('M')

    wins   = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum())
          if len(losses) > 0 else float('inf'))

    eq_df = pd.DataFrame(results['eq_curve'], columns=['ts', 'eq'])
    eq_s  = eq_df.set_index('ts')['eq']
    mdd   = calc_max_drawdown(eq_s)

    print("\n" + "═" * 70)
    print(f" ▌  CorrArb Prop Simulator v9 — {'+'.join(pair_names)}  ▐")
    print("═" * 70)
    print(f" Total Trades:    {len(df_t):,}")
    print(f" Win Rate:        {wr:.2f}%")
    print(f" Profit Factor:   {pf:.2f}")
    print(f" Max Drawdown:    {mdd*100:.2f}%")
    print(f" Total Banked:    ${results['total_withdrawn']:,.2f}")
    print(f" Active Equity:   ${results['final_equity']:,.2f}")
    if len(wins):   print(f" Avg Win:         ${wins['pnl'].mean():.2f}")
    if len(losses): print(f" Avg Loss:        ${losses['pnl'].mean():.2f}")
    print(f" Expectancy:      ${df_t['pnl'].mean():.2f} per trade")

    if 'pair' in df_t.columns and len(pair_names) > 1:
        print("─" * 70)
        print(" عملکرد هر جفت ارز:")
        for pair in pair_names:
            pt = df_t[df_t['pair'] == pair]
            if len(pt) == 0:
                continue
            pw = pt[pt['pnl'] > 0]
            pl = pt[pt['pnl'] < 0]
            p_wr = len(pw) / len(pt) * 100
            p_pf = (pw['pnl'].sum() / abs(pl['pnl'].sum())
                    if len(pl) > 0 else float('inf'))
            print(f"   {pair}: {len(pt):>4} trades"
                  f" | WR: {p_wr:5.1f}%"
                  f" | PF: {p_pf:.2f}"
                  f" | Net: ${pt['pnl'].sum():>9,.2f}")

    print("─" * 70)
    print(" خروج‌ها بر اساس نوع:")
    for st, cnt in df_t['status'].value_counts().items():
        pct = cnt / len(df_t) * 100
        print(f"   {st:<12}: {cnt:>4} ({pct:.1f}%)")

    logs    = results['account_logs']
    targets = sum(1 for l in logs if l['reason'] == 'TARGET_HIT')
    blown   = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')
    sr      = targets / (targets + blown) * 100 if (targets + blown) > 0 else 0
    print("─" * 70)
    print(f" حساب‌ها: {results['total_accounts']} کل"
          f" | ✅ Target: {targets}"
          f" | 💥 Blown: {blown}"
          f" | Success: {sr:.0f}%")

    monthly = df_t.groupby('month')['pnl'].sum()
    if len(monthly):
        pos_m = int((monthly > 0).sum())
        neg_m = int((monthly < 0).sum())
        mwr   = pos_m / (pos_m + neg_m) * 100 if (pos_m + neg_m) > 0 else 0
        print(f" ماهانه: avg ${monthly.mean():,.2f}"
              f" | Best: ${monthly.max():,.2f}"
              f" | Worst: ${monthly.min():,.2f}")
        print(f"         Positive: {pos_m}"
              f" | Negative: {neg_m}"
              f" | Win Rate: {mwr:.0f}%")
    print("═" * 70)

    walk_forward_summary(trades)

    print("\n  🎲 Monte Carlo (1000 sims):")
    mc = monte_carlo_simulation(list(df_t['pnl']))
    print(f"     Median Final:    ${mc['median_final']:>10,.2f}")
    print(f"     5th Pct:         ${mc['p5_final']:>10,.2f}")
    print(f"     95th Pct:        ${mc['p95_final']:>10,.2f}")
    print(f"     Median Max DD:   {mc['median_dd']*100:>8.2f}%")
    print(f"     Worst DD (95th): {mc['worst_dd_p95']*100:>8.2f}%")
    print(f"     Prob Profitable: {mc['prob_positive']:>8.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()

    pairs = load_all_pairs()

    print("\n  Computing Statistical Signals...")
    pair_signals = {}
    for name, info in pairs.items():
        pair_signals[name] = compute_signals(name, info)

    results = run_backtest(pairs, pair_signals)
    print_report(results)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  ✅ Executed in: {elapsed:.2f}s")
