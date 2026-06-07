"""
CorrArb Prop Simulator — v10 THE ULTIMATE EDGE (Direct Data + 3 Optimizations)
==============================================================================
تغییرات این نسخه:
 🔴 ۱. ATR Dynamic SL/TP: استاپ و تارگت بر اساس نوسانات لحظه‌ای (حجم هم داینامیک می‌شود).
 🔴 ۲. Full Exit (No Partial): حذف خروج‌های ۵۰ درصدی برای رشد Profit Factor.
 🔴 ۳. Trend Filter: مسدود کردن سیگنال‌ها در روندهای شدید یک‌طرفه.
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
#  CONFIG (تنظیمات سوئیچ‌ها)
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05        
    max_daily_loss_pct = 0.05        
    max_total_dd_pct   = 0.10        

    risk_base_pct      = 0.015       
    risk_min_pct       = 0.0075      
    consec_loss_n      = 2
    risk_reduce        = 0.5

    spread_pips        = 2.0         
    commission_per_lot = 7.0         
    slippage_pips      = 0.5         

    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── پارامترهای پایه ──
    z_fast_period      = 96          
    z_entry            = 2.1
    z_exit_full        = 0.0         
    z_stop_margin      = 4.0         
    min_net_profit_usd = 20.0        

    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2           
    time_stop_bars     = 36          

    # ── فیلتر رژیم ──
    vr_period          = 200         
    vr_k               = 4           
    vr_max             = 0.90        

    # =======================================================
    # 🔴 سه گزینه بهینه‌سازی (The 3 Optimizations)
    # =======================================================
    
    # [Option 1] استاپ و تارگت داینامیک (ATR)
    use_dynamic_stops  = True
    atr_sl_mult        = 1.5         # SL = 1.5 * ATR
    atr_tp_mult        = 2.0         # TP = 2.0 * ATR
    # در صورت خاموش بودن بالا، از این مقادیر ثابت استفاده می‌شود:
    sl_pips_static     = 30.0
    tp_pips_static     = 90.0
    atr_period         = 14

    # [Option 2] خروج پله‌ای (Partial Exit)
    # شما خواستید حذف شود، پس False است. اگر خواستید تست کنید True کنید.
    enable_partial_exit = False
    partial_ratio       = 0.50
    z_exit_partial      = 0.5

    # [Option 3] فیلتر روند (Trend Filter)
    use_trend_filter   = True
    trend_ma_period    = 200
    trend_max_slope    = 0.0003      # حداکثر شیب مجاز (3 پیپ در هر کندل 15m)

    # =======================================================
    approx_quote_rate = {'EURGBP': 1.25, 'AUDNZD': 0.60}

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING 
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_zip(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths: raise FileNotFoundError(f"No files found: {pattern}")
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p, 'r') as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if csv_name is None: continue
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            frames.append(df)
        except Exception as e: pass
    raw = pd.concat(frames, ignore_index=True).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw

def to_15min(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        'o': raw['o'].resample('15min').first(),
        'h': raw['h'].resample('15min').max(),
        'l': raw['l'].resample('15min').min(),
        'c': raw['c'].resample('15min').last(),
    }).dropna()
    return df[df.index.weekday < 5].copy()

def load_all_direct_pairs() -> dict:
    print("\n  Loading Direct Cross Pair Datasets...")
    pairs = {}
    try:
        df_eg = to_15min(load_raw_zip('data/*EURGBP*.zip'))
        pairs['EURGBP'] = {'df': df_eg}
        print(f"  ✅ EURGBP : {len(df_eg):>7,} candles")
    except Exception as e: print(f"  ❌ EURGBP : {e}")

    try:
        df_an = to_15min(load_raw_zip('data/*AUDNZD*.zip'))
        pairs['AUDNZD'] = {'df': df_an}
        print(f"  ✅ AUDNZD : {len(df_an):>7,} candles")
    except Exception as e: print(f"  ❌ AUDNZD : {e}")
    return pairs

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION (با فیلتر روند)
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_signals(pair_name: str, df: pd.DataFrame) -> dict:
    C = Config
    log_price = np.log(df['c'])
    z_mean    = log_price.rolling(C.z_fast_period).mean()
    z_std     = log_price.rolling(C.z_fast_period).std()
    z_score   = (log_price - z_mean) / z_std.replace(0, np.nan)

    # 🔴 فیلتر روند (Option 3)
    trend_ma = df['c'].rolling(C.trend_ma_period).mean()
    # محاسبه شیب مطلق (قدر مطلق تغییرات MA در 5 کندل گذشته تقسیم بر 5)
    trend_slope = (trend_ma.diff(5).abs() / 5).fillna(0)
    
    vr = (log_price.diff(C.vr_k).rolling(C.vr_period).var() / 
         (C.vr_k * log_price.diff(1).rolling(C.vr_period).var().replace(0, np.nan)))
    regime_ok = vr < C.vr_max

    atr = calc_atr(df['h'], df['l'], df['c'], C.atr_period)

    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # اعمال فیلتر روند اگر روشن باشد
    trend_ok = (trend_slope < C.trend_max_slope) if C.use_trend_filter else True

    long_cond  = (z_score < -C.z_entry) & time_ok & regime_ok & trend_ok
    short_cond = (z_score >  C.z_entry) & time_ok & regime_ok & trend_ok

    # ردگیری سیگنال‌هایی که فقط بخاطر Trend مسدود شدند (برای لاگ)
    long_raw  = (z_score < -C.z_entry) & time_ok & regime_ok
    short_raw = (z_score >  C.z_entry) & time_ok & regime_ok
    blocked_by_trend = ((long_raw | short_raw) & ~trend_ok).sum() if C.use_trend_filter else 0

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0) 

    print(f"    {pair_name}: Computed. Trend Filter blocked {blocked_by_trend:,} signals.")
    return {'sig': sig, 'z': z_score, 'atr': atr, 'blocked': blocked_by_trend}

# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl(direction: int, entry_px: float, exit_px: float, lot: float, pair_name: str, apply_slippage: bool = False) -> float:
    C = Config
    quote_rate = C.approx_quote_rate.get(pair_name, 1.0)
    if apply_slippage: exit_px = exit_px - (direction * C.slippage_pips * C.pip)
    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    return (gross_quote * quote_rate) - (C.commission_per_lot * lot)

def calc_lot(equity: float, dynamic_sl_dist: float, consec_loss: int, pair_name: str) -> float:
    C = Config
    quote_rate = C.approx_quote_rate.get(pair_name, 1.0)
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n: risk = max(risk * C.risk_reduce, C.risk_min_pct)
    pip_value_usd = C.pip * C.lot_size * quote_rate
    sl_in_pips = dynamic_sl_dist / C.pip if dynamic_sl_dist > 0 else 30.0
    risk_usd = equity * risk
    raw = risk_usd / (sl_in_pips * pip_value_usd)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)

def new_acc(ts) -> dict:
    return {'equity': Config.initial_balance, 'start_ts': ts, 'trades': [], 'blown': False, 'blown_rsn': '', 'peak': Config.initial_balance, 'consec_loss': 0}

# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, signals_data: dict) -> dict:
    C = Config
    pair_names = list(pairs.keys())
    
    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)

    pa = {}
    total_blocked = 0
    for name in pair_names:
        df_p  = pairs[name]['df'].reindex(common_idx).ffill()
        sdata = signals_data[name]
        total_blocked += sdata['blocked']
        pa[name] = {
            'o': df_p['o'].values.astype(float), 'c': df_p['c'].values.astype(float),
            'h': df_p['h'].values.astype(float), 'l': df_p['l'].values.astype(float),
            'sig': sdata['sig'].reindex(common_idx).fillna(0).values.astype(int),
            'z':   sdata['z'].reindex(common_idx).fillna(np.nan).values.astype(float),
            'atr': sdata['atr'].reindex(common_idx).fillna(0.0010).values.astype(float),
        }

    acc             = new_acc(common_idx[C.warmup])
    total_withdrawn = 0.0
    acc_num         = 1
    day_start_eq    = C.initial_balance
    all_trades      = []
    acc_logs        = []

    positions    = {name: None for name in pair_names}
    trades_today = {name: 0 for name in pair_names}
    pending_sig  = {name: 0 for name in pair_names}

    # برای لاگ گرفتن سایز استاپ‌ها
    tracker_sl_pips = []
    tracker_tp_pips = []

    print(f"\n  ▶ Running v10 Simulator (Dynamic SL/TP: {C.use_dynamic_stops} | Partial: {C.enable_partial_exit} | TrendFilter: {C.use_trend_filter})...")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        if acc['equity'] > acc['peak']: acc['peak'] = acc['equity']

        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0

        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(f"    Progress: {pct:5.1f}% | Eq: ${acc['equity']:,.2f} | Bank: ${total_withdrawn:,.2f}", end='\r')

        if acc['blown']:
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': acc['blown_rsn'], 'pnl': acc['equity'] - C.initial_balance})
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0; pending_sig[name] = 0; positions[name] = None
            continue

        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0 and positions[name] is None and trades_today[name] < C.max_trades_day):
                sv  = pending_sig[name]
                
                # 🔴 محاسبه استاپ‌لاس و تارگت داینامیک
                if C.use_dynamic_stops:
                    current_atr = a['atr'][bar]
                    sl_dist = current_atr * C.atr_sl_mult
                    tp_dist = current_atr * C.atr_tp_mult
                else:
                    sl_dist = C.sl_pips_static * C.pip
                    tp_dist = C.tp_pips_static * C.pip

                lot = calc_lot(acc['equity'], sl_dist, acc['consec_loss'], name)
                ep  = a['o'][bar] + sv * (C.slippage_pips + C.spread_pips / 2) * C.pip
                sl  = ep - sv * sl_dist
                tp  = ep + sv * tp_dist
                
                tracker_sl_pips.append(sl_dist / C.pip)
                tracker_tp_pips.append(tp_dist / C.pip)

                positions[name] = {
                    'pair': name, 'dir': sv, 'lot': lot, 'lot_remaining': lot, 
                    'partial_done': False, 'entry': ep, 'sl': sl, 'tp': tp, 
                    'entry_ts': ts, 'entry_bar': bar
                }
                trades_today[name] += 1
            pending_sig[name] = 0

        total_float = 0.0
        for name in pair_names:
            pos = positions[name]
            if pos is not None:
                worst_px = pa[name]['l'][bar] if pos['dir'] == 1 else pa[name]['h'][bar]
                total_float += calc_pnl(pos['dir'], pos['entry'], worst_px, pos['lot_remaining'], name)

        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= C.initial_balance * (1 - C.max_total_dd_pct):
            acc['blown'] = True
            acc['blown_rsn'] = "Wick/DailyDD" if current_eq <= daily_limit else "TotalDD"
            for name in pair_names:
                if positions[name] is not None:
                    pnl = calc_pnl(positions[name]['dir'], positions[name]['entry'], pa[name]['c'][bar], positions[name]['lot_remaining'], name)
                    acc['equity'] += pnl
                    all_trades.append(_make_rec(positions[name], pa[name]['c'][bar], ts, pnl, 'BLOWN', positions[name]['lot_remaining']))
                    positions[name] = None
            continue

        for name in pair_names:
            pos = positions[name]
            if pos is None: continue

            a, cp, d, ep, zn, lot_rem = pa[name], pa[name]['c'][bar], pos['dir'], pos['entry'], pa[name]['z'][bar], pos['lot_remaining']

            # 🔴 Option 2: Partial Exit Check
            if C.enable_partial_exit and not pos['partial_done'] and not np.isnan(zn):
                if (d == 1 and zn >= -C.z_exit_partial) or (d == -1 and zn <= C.z_exit_partial):
                    p_lot = round(lot_rem * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, name, apply_slippage=True)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            all_trades.append(_make_rec(pos, cp, ts, p_pnl, 'Partial', p_lot))
                            pos['lot_remaining'] = round(lot_rem - p_lot, 2)
                            pos['partial_done'] = True
                            pos['sl'] = pos['entry']
                            if pos['lot_remaining'] < C.min_lot: positions[name] = None; continue

            hit_z_stop = (not np.isnan(zn)) and ((d == 1 and zn <= -C.z_stop_margin) or (d == -1 and zn >= C.z_stop_margin))
            hit_z_exit = False
            if not np.isnan(zn) and ((d == 1 and zn >= -C.z_exit_full) or (d == -1 and zn <= C.z_exit_full)):
                if calc_pnl(d, ep, cp, lot_rem, name) >= C.min_net_profit_usd or pos['partial_done']: hit_z_exit = True

            hit_sl = (d == 1 and a['l'][bar] <= pos['sl']) or (d == -1 and a['h'][bar] >= pos['sl'])
            hit_tp = (d == 1 and a['h'][bar] >= pos['tp']) or (d == -1 and a['l'][bar] <= pos['tp'])

            bars_open = bar - pos['entry_bar']
            time_stop = ((bars_open >= C.time_stop_bars and calc_pnl(d, ep, cp, lot_rem, name) < 0) or (bars_open >= C.time_stop_bars * 2))

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                has_slippage = True if (hit_sl or time_stop) else False
                st = 'SL' if hit_sl else 'TP' if hit_tp else 'Z-Stop' if hit_z_stop else 'TimeStop' if time_stop else 'Z-Exit'
                exit_px = pos['sl'] if hit_sl else pos['tp'] if hit_tp else cp
                final_pnl = calc_pnl(d, ep, exit_px, lot_rem, name, apply_slippage=has_slippage)
                
                acc['equity'] += final_pnl
                all_trades.append(_make_rec(pos, exit_px, ts, final_pnl, st, lot_rem))
                positions[name] = None

                if final_pnl > 0: acc['consec_loss'] = 0
                else:             acc['consec_loss'] += 1

        if acc['equity'] >= C.initial_balance * (1 + C.profit_target_pct) and all(positions[n] is None for n in pair_names) and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': 'TARGET_HIT', 'pnl': w})
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0; pending_sig[name] = 0
            continue

        for name in pair_names:
            if positions[name] is None and not acc['blown'] and trades_today[name] < C.max_trades_day and pa[name]['sig'][bar] != 0:
                pending_sig[name] = int(pa[name]['sig'][bar])

    print() 
    avg_sl = sum(tracker_sl_pips)/len(tracker_sl_pips) if tracker_sl_pips else 0
    avg_tp = sum(tracker_tp_pips)/len(tracker_tp_pips) if tracker_tp_pips else 0

    return {
        'all_trades': all_trades, 'account_logs': acc_logs, 'total_withdrawn': total_withdrawn,
        'final_equity': acc['equity'], 'total_accounts': acc_num, 'pair_names': pair_names,
        'diag_blocked': total_blocked, 'diag_avg_sl': avg_sl, 'diag_avg_tp': avg_tp
    }

def _make_rec(pos, exit_px, exit_ts, pnl, status, lot):
    return {'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot, 'entry': pos['entry'], 'exit': exit_px, 'entry_ts': pos['entry_ts'], 'exit_ts': exit_ts, 'pnl': pnl, 'status': status, 'entry_bar': pos['entry_bar']}

# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results: dict):
    trades = results['all_trades']
    if not trades: return print("\n❌ No trades executed.")
    df_t = pd.DataFrame(trades)
    
    wins, losses = df_t[df_t['pnl'] > 0], df_t[df_t['pnl'] < 0]
    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'))

    print("\n" + "═" * 70)
    print(f" ▌  v10 ULTIMATE EDGE (Dynamic SL/TP + Trend Filter + No Partial)  ▐")
    print("═" * 70)
    print(f" Total Trades:    {len(df_t):,}\n Win Rate:        {wr:.2f}%\n Profit Factor:   {pf:.2f}")
    print(f" Total Banked:    ${results['total_withdrawn']:,.2f}\n Active Equity:   ${results['final_equity']:,.2f}")
    print("-" * 70 + "\n خروج‌ها بر اساس نوع:")
    for st, cnt in df_t['status'].value_counts().items(): print(f"   {st:<12}: {cnt:>4} معامله")
    
    logs = results['account_logs']
    targets = sum(1 for l in logs if l['reason'] == 'TARGET_HIT')
    blown = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')
    print("-" * 70 + f"\n حساب‌ها: {results['total_accounts']} کل | ✅ Target Hit: {targets} | 💥 Blown: {blown}")
    
    # 🔴 گزارش عارضه‌یابی فیلترها (Filter Diagnostics)
    print("═" * 70)
    print(" 🛠 گزارش تاثیر فیلترها (Diagnostic Report):")
    print(f"   ➤ فیلتر روند (Trend Filter): {results['diag_blocked']:,} سیگنالِ خطرناک را مسدود کرد.")
    print(f"   ➤ استاپ داینامیک (ATR SL): میانگین {results['diag_avg_sl']:.1f} پیپ (به جای 30 پیپ ثابت).")
    print(f"   ➤ تارگت داینامیک (ATR TP): میانگین {results['diag_avg_tp']:.1f} پیپ (به جای 90 پیپ ثابت).")
    print(f"   ➤ خروج پله‌ای (Partial): غیرفعال شد تا سودهای کامل (Full TP/Z-Exit) رشد کنند.")
    print("═" * 70)

if __name__ == "__main__":
    t0 = datetime.now()
    pairs = load_all_direct_pairs()
    signals_data = {name: compute_signals(name, info['df']) for name, info in pairs.items()}
    results = run_backtest(pairs, signals_data)
    print_report(results)
    print(f"  ✅ Executed in: {(datetime.now() - t0).total_seconds():.2f}s")
