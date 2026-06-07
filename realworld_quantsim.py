"""
CorrArb Prop Simulator — v8.1 REALISTIC BASELINE
==============================================================================
استراتژی: دقیقاً همان منطق v8 کاربر (ثابت بودن Z-Score، اسمارت استاپ، فیلتر رژیم)
اصلاحات ریاضی (فقط رفع خوش‌بینی کاذب):
  ۱. کمیسیون و اسپرد دو برابر شد (چون ترید سینتتیک روی 2 جفت ارز باز می‌شود).
  ۲. محاسبه Drawdown بر اساس بدترین قیمت کندل (High/Low) برای پوشش سایه‌ها.
"""

import pandas as pd
import numpy as np
import glob
import zipfile
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05        
    max_daily_loss_pct = 0.05        
    max_total_dd_pct   = 0.10        

    # ── مدیریت ریسک ──
    risk_base_pct      = 0.015       
    risk_min_pct       = 0.0075      
    consec_loss_n      = 2
    risk_reduce        = 0.5

    # 🔴 اصلاح واقع‌گرایانه: هزینه‌های بروکر ضرب در ۲ (برای دو لگ معامله)
    spread_pips        = 2.4         # قبلاً 1.2 بود
    commission_per_lot = 14.0        # قبلاً 7.0 بود
    slippage_pips      = 0.5         # کمی افزایش برای خروج‌های مارکت

    # ── مشخصات ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── پارامترهای z-score (دقیقاً استراتژی خودتان) ──
    z_fast_period      = 96          
    z_entry            = 2.1
    z_exit_partial     = 0.5         
    z_exit_full        = 0.0         
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
    time_stop_bars     = 36          

    # ── فیلتر ATR ──
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    # ── فیلتر رژیم: Variance Ratio ──
    vr_period          = 200         
    vr_k               = 4           
    vr_max             = 0.90        

    # ── Partial Exit ──
    partial_ratio      = 0.50        

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING 
# ═══════════════════════════════════════════════════════════════════════════
def load_raw(pattern: str, is_zip: bool) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths: raise FileNotFoundError(f"No files found: {pattern}")
    frames = []
    for p in paths:
        try:
            if is_zip:
                with zipfile.ZipFile(p, 'r') as z:
                    csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                    if csv_name is None: continue
                    with z.open(csv_name) as f:
                        df = pd.read_csv(f, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            else:
                df = pd.read_csv(p, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Skip {os.path.basename(p)}: {e}")

    raw = pd.concat(frames, ignore_index=True).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw

def to_15min(raw: pd.DataFrame, sfx: str) -> pd.DataFrame:
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()

def build_spread_df(df_a: pd.DataFrame, sfx_a: str, df_b: pd.DataFrame, sfx_b: str) -> pd.DataFrame:
    merged = df_a.join(df_b, how='inner').dropna()
    merged['c_spread'] = merged[f'c_{sfx_a}'] / merged[f'c_{sfx_b}']
    merged['o_spread'] = merged[f'o_{sfx_a}'] / merged[f'o_{sfx_b}']
    merged['h_spread'] = merged[f'h_{sfx_a}'] / merged[f'l_{sfx_b}']
    merged['l_spread'] = merged[f'l_{sfx_a}'] / merged[f'h_{sfx_b}']
    merged['quote_rate'] = merged[f'c_{sfx_b}']
    return merged[merged.index.weekday < 5].copy()

def load_all_pairs() -> dict:
    print("\n  Loading and syncing datasets...")
    pairs = {}
    try:
        eur = to_15min(load_raw('data/*EURUSD*.csv', is_zip=False), 'eur')
        gbp = to_15min(load_raw('data/*GBPUSD*.csv', is_zip=False), 'gbp')
        df = build_spread_df(eur, 'eur', gbp, 'gbp')
        pairs['EURGBP'] = {'df': df, 'leg_a': 'c_eur', 'leg_b': 'c_gbp'}
        print(f"  ✅ EURGBP : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e: print(f"  ❌ EURGBP : {e}")

    try:
        aud = to_15min(load_raw('data/HISTDATA*AUDUSD*.zip', is_zip=True), 'aud')
        nzd = to_15min(load_raw('data/HISTDATA*NZDUSD*.zip', is_zip=True), 'nzd')
        df = build_spread_df(aud, 'aud', nzd, 'nzd')
        pairs['AUDNZD'] = {'df': df, 'leg_a': 'c_aud', 'leg_b': 'c_nzd'}
        print(f"  ✅ AUDNZD : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e: print(f"  ❌ AUDNZD : {e}")
    return pairs

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION (دقیقاً نسخه اصلی شما)
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_variance_ratio(series: pd.Series, k: int, window: int) -> pd.Series:
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
    z_mean    = log_ratio.rolling(C.z_fast_period).mean()
    z_std     = log_ratio.rolling(C.z_fast_period).std()
    z_score   = (log_ratio - z_mean) / z_std.replace(0, np.nan)

    ret_a   = df[leg_a].pct_change()
    ret_b   = df[leg_b].pct_change()
    corr_ok = ret_a.rolling(C.corr_period).corr(ret_b) > C.corr_min

    vr        = calc_variance_ratio(log_ratio, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    hour    = pd.Series(df.index.hour,       index=df.index)
    dow     = pd.Series(df.index.dayofweek,  index=df.index)
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
    print(f"    {pair_name}: {n:,} signals (L:{l} | S:{s}) | Regime OK: {r:,} bars")
    return sig, z_score

# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl(direction: int, entry_px: float, exit_px: float, lot: float, quote_rate: float) -> float:
    C = Config
    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    gross_usd   = gross_quote * quote_rate
    # کمیسیون در Config دو برابر شده است (۱۴ دلار)، پس نیازی به ضرب مجدد در اینجا نیست
    commission  = C.commission_per_lot * lot
    return gross_usd - commission

def calc_lot(equity: float, sl_pips: float, consec_loss: int, quote_rate: float) -> float:
    C = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
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
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, pair_signals: dict) -> dict:
    C          = Config
    pip        = C.pip
    pair_names = list(pairs.keys())

    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)

    # 🔴 اضافه شدن h و l برای بررسی سایه کندل‌ها
    pa = {}
    for name in pair_names:
        df_p  = pairs[name]['df'].reindex(common_idx).ffill()
        sig_s, z_s = pair_signals[name]
        pa[name] = {
            'o':   df_p['o_spread'].values.astype(float),
            'c':   df_p['c_spread'].values.astype(float),
            'h':   df_p['h_spread'].values.astype(float),
            'l':   df_p['l_spread'].values.astype(float),
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

    print(f"\n  ▶ Running Baseline Math-Corrected Simulator v8.1...")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq = acc['equity']
        eq_curve.append((ts, round(eq, 4)))

        if eq > acc['peak']: acc['peak'] = eq

        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0

        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(f"    Progress: {pct:5.1f}% | Eq: ${acc['equity']:,.2f} | Bank: ${total_withdrawn:,.2f}", end='\r')

        if acc['blown']:
            acc_logs.append({
                'account':  acc_num,
                'start_ts': acc['start_ts'],
                'end_ts':   ts,
                'reason':   acc['blown_rsn'],
                'pnl':      acc['equity'] - C.initial_balance,
            })
            print(f"\n    💥 #{acc_num:>3} | {ts.date()} | Eq: ${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name]  = 0
                positions[name]    = None
            continue

        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0 and positions[name] is None and trades_today[name] < C.max_trades_day):
                sv  = pending_sig[name]
                qr  = a['qr'][bar]
                lot = calc_lot(acc['equity'], C.sl_pips, acc['consec_loss'], qr)
                ep  = a['o'][bar] + sv * (C.slippage_pips + C.spread_pips / 2) * pip
                sl  = ep - sv * C.sl_pips * pip
                tp  = ep + sv * C.tp_pips * pip
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
            pending_sig[name] = 0

        # ── 🔴 محاسبه منطقی‌تر Drawdown با استفاده از بدترین قیمت ──
        total_float = 0.0
        for name in pair_names:
            pos = positions[name]
            if pos is not None:
                a = pa[name]
                # درگیر شدن با سایه کندل برای Drawdown واقعی‌تر
                worst_px = a['l'][bar] if pos['dir'] == 1 else a['h'][bar]
                total_float += calc_pnl(pos['dir'], pos['entry'], worst_px, pos['lot_remaining'], a['qr'][bar])

        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            acc['blown']     = True
            acc['blown_rsn'] = "Wick/DailyDD" if current_eq <= daily_limit else "TotalDD"
            for name in pair_names:
                pos = positions[name]
                if pos is None: continue
                a   = pa[name]
                pnl = calc_pnl(pos['dir'], pos['entry'], a['c'][bar], pos['lot_remaining'], a['qr'][bar])
                acc['equity'] += pnl
                rec = _make_rec(pos, a['c'][bar], ts, pnl, 'BLOWN', pos['lot_remaining'])
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None
            continue

        for name in pair_names:
            pos = positions[name]
            if pos is None: continue

            a       = pa[name]
            cp      = a['c'][bar]
            qr      = a['qr'][bar]
            d       = pos['dir']
            ep      = pos['entry']
            zn      = a['z'][bar]
            lot_rem = pos['lot_remaining']

            if not pos['partial_done'] and not np.isnan(zn):
                hit_p = ((d ==  1 and zn >= -C.z_exit_partial) or 
                         (d == -1 and zn <=  C.z_exit_partial))
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
                            pos['sl'] = pos['entry']
                            lot_rem = pos['lot_remaining']
                            if lot_rem < C.min_lot:
                                positions[name] = None
                                continue

            hit_z_stop = (not np.isnan(zn)) and ((d == 1 and zn <= -C.z_stop_margin) or (d == -1 and zn >= C.z_stop_margin))

            hit_z_exit = False
            if not np.isnan(zn):
                z_crossed = ((d == 1 and zn >= -C.z_exit_full) or (d == -1 and zn <= C.z_exit_full))
                if z_crossed:
                    pnl_check = calc_pnl(d, ep, cp, lot_rem, qr)
                    if pnl_check >= C.min_net_profit_usd or pos['partial_done']:
                        hit_z_exit = True

            hit_sl = (d == 1 and a['l'][bar] <= pos['sl']) or (d == -1 and a['h'][bar] >= pos['sl'])
            hit_tp = (d == 1 and a['h'][bar] >= pos['tp']) or (d == -1 and a['l'][bar] <= pos['tp'])

            bars_open = bar - pos['entry_bar']
            current_pos_pnl = calc_pnl(d, ep, cp, lot_rem, qr)
            time_stop = ((bars_open >= C.time_stop_bars and current_pos_pnl < 0) or (bars_open >= C.time_stop_bars * 2))

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                if hit_sl:       exit_px, st = pos['sl'], 'SL'
                elif hit_tp:     exit_px, st = pos['tp'], 'TP'
                elif hit_z_stop: exit_px, st = cp, 'Z-Stop'
                elif time_stop:  exit_px, st = cp, 'TimeStop'
                else:            exit_px, st = cp, 'Z-Exit'

                final_pnl = calc_pnl(d, ep, exit_px, lot_rem, qr)
                acc['equity'] += final_pnl

                rec = _make_rec(pos, exit_px, ts, final_pnl, st, lot_rem)
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None

                if final_pnl > 0: acc['consec_loss'] = 0
                else:             acc['consec_loss'] += 1

        all_closed = all(positions[name] is None for name in pair_names)
        if acc['equity'] >= PROFIT_LEVEL and all_closed and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': 'TARGET_HIT', 'pnl': w})
            print(f"\n    💰 #{acc_num:>3} | {ts.date()} | Target Hit: ${w:>7.2f} | Total Bank: ${total_withdrawn:>9.2f}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name]  = 0
            continue

        for name in pair_names:
            a = pa[name]
            if (positions[name] is None and not acc['blown'] and trades_today[name] < C.max_trades_day and a['sig'][bar] != 0):
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

def _make_rec(pos: dict, exit_px: float, exit_ts, pnl: float, status: str, lot: float) -> dict:
    return {'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot, 'entry': pos['entry'], 'exit': exit_px, 'entry_ts': pos['entry_ts'], 'exit_ts': exit_ts, 'pnl': pnl, 'status': status, 'entry_bar': pos['entry_bar']}

# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING (بدون تغییر نسبت به نسخه خودتان)
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results: dict):
    trades = results['all_trades']
    pair_names = results.get('pair_names', [])
    if not trades: return print("\n❌ No trades executed.")
    df_t = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month']   = df_t['exit_ts'].dt.to_period('M')

    wins, losses = df_t[df_t['pnl'] > 0], df_t[df_t['pnl'] < 0]
    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'))

    print("\n" + "═" * 70)
    print(f" ▌  CorrArb Prop Simulator v8.1 (Math Fixed) — {'+'.join(pair_names)}  ▐")
    print("═" * 70)
    print(f" Total Trades:    {len(df_t):,}\n Win Rate:        {wr:.2f}%\n Profit Factor:   {pf:.2f}")
    print(f" Total Banked:    ${results['total_withdrawn']:,.2f}\n Active Equity:   ${results['final_equity']:,.2f}")
    
    if 'pair' in df_t.columns and len(pair_names) > 1:
        print("-" * 70 + "\n عملکرد هر جفت ارز:")
        for pair in pair_names:
            pt = df_t[df_t['pair'] == pair]
            if len(pt) == 0: continue
            pw, pl = pt[pt['pnl'] > 0], pt[pt['pnl'] < 0]
            p_wr = len(pw) / len(pt) * 100 if len(pt) else 0
            p_pf = (pw['pnl'].sum() / abs(pl['pnl'].sum()) if len(pl) > 0 else float('inf'))
            print(f"   {pair}: {len(pt):>4} trades | WR: {p_wr:5.1f}% | PF: {p_pf:.2f} | Net PnL: ${pt['pnl'].sum():>8,.2f}")

    print("-" * 70 + "\n خروج‌ها بر اساس نوع:")
    for st, cnt in df_t['status'].value_counts().items(): print(f"   {st:<12}: {cnt:>4} معامله")
    
    logs = results['account_logs']
    targets = sum(1 for l in logs if l['reason'] == 'TARGET_HIT')
    blown = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')
    print("-" * 70 + f"\n حساب‌ها: {results['total_accounts']} کل | ✅ Target Hit: {targets} | 💥 Blown: {blown}")
    print("═" * 70)

if __name__ == "__main__":
    t0 = datetime.now()
    pairs = load_all_pairs()
    print("\n  Computing Statistical Signals...")
    pair_signals = {name: compute_signals(name, info) for name, info in pairs.items()}
    results = run_backtest(pairs, pair_signals)
    print_report(results)
    print(f"  ✅ Executed in: {(datetime.now() - t0).total_seconds():.2f}s")
