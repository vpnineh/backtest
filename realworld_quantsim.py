"""
CorrArb Prop Simulator — v11 THE A/B TEST ENGINE
==============================================================================
مقایسه دو استراتژی خروج (Exit Logic Battle):
  [A] Pure R:R (بدون هیچ خروج زودهنگام، فقط SL یا TP)
  [B] Break-Even (وقتی Z=0 شد، استاپ‌لاس به نقطه ورود منتقل می‌شود)
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

    z_fast_period      = 96          
    z_entry            = 2.1
    z_exit_full        = 0.0         
    z_stop_margin      = 4.0         
    min_net_profit_usd = 20.0        

    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2           

    vr_period          = 200         
    vr_k               = 4           
    vr_max             = 0.90        

    # 🔴 پارامترهای بهینه‌شده
    use_dynamic_stops  = True
    atr_sl_mult        = 3.0         
    atr_tp_mult        = 6.0         
    atr_period         = 14

    enable_partial_exit = False
    
    use_trend_filter   = True
    trend_ma_period    = 200
    trend_max_slope    = 0.00003     

    # این متغیرها توسط موتور A/B در زمان اجرا تغییر می‌کنند:
    z_exit_mode        = 'OFF'       # 'OFF' | 'BE' | 'CLOSE'
    use_time_stop      = False       

    approx_quote_rate = {'EURGBP': 1.25, 'AUDNZD': 0.60}

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & SIGNALS
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_zip(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths: return None
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p, 'r') as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if csv_name is None: continue
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            frames.append(df)
        except: pass
    if not frames: return None
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
    df_eg = load_raw_zip('data/*EURGBP*.zip')
    if df_eg is not None:
        pairs['EURGBP'] = {'df': to_15min(df_eg)}
        print(f"  ✅ EURGBP Loaded")
        
    df_an = load_raw_zip('data/*AUDNZD*.zip')
    if df_an is not None:
        pairs['AUDNZD'] = {'df': to_15min(df_an)}
        print(f"  ✅ AUDNZD Loaded")
    return pairs

def calc_atr(h, l, c, period):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_signals(pair_name: str, df: pd.DataFrame) -> dict:
    C = Config
    log_price = np.log(df['c'])
    z_mean    = log_price.rolling(C.z_fast_period).mean()
    z_std     = log_price.rolling(C.z_fast_period).std()
    z_score   = (log_price - z_mean) / z_std.replace(0, np.nan)

    trend_ma = df['c'].rolling(C.trend_ma_period).mean()
    trend_slope = (trend_ma.diff(5).abs() / 5).fillna(0)
    
    vr = (log_price.diff(C.vr_k).rolling(C.vr_period).var() / (C.vr_k * log_price.diff(1).rolling(C.vr_period).var().replace(0, np.nan)))
    regime_ok = vr < C.vr_max
    atr = calc_atr(df['h'], df['l'], df['c'], C.atr_period)

    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)
    trend_ok = (trend_slope < C.trend_max_slope) if C.use_trend_filter else True

    long_cond  = (z_score < -C.z_entry) & time_ok & regime_ok & trend_ok
    short_cond = (z_score >  C.z_entry) & time_ok & regime_ok & trend_ok

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0) 
    return {'sig': sig, 'z': z_score, 'atr': atr}

# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl(direction, entry_px, exit_px, lot, pair_name, apply_slippage=False):
    C = Config
    if apply_slippage: exit_px = exit_px - (direction * C.slippage_pips * C.pip)
    gross_usd = direction * (exit_px - entry_px) * lot * C.lot_size * C.approx_quote_rate.get(pair_name, 1.0)
    return gross_usd - (C.commission_per_lot * lot)

def calc_lot(equity, dynamic_sl_dist, consec_loss, pair_name):
    C = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n: risk = max(risk * C.risk_reduce, C.risk_min_pct)
    pip_value_usd = C.pip * C.lot_size * C.approx_quote_rate.get(pair_name, 1.0)
    sl_in_pips = dynamic_sl_dist / C.pip if dynamic_sl_dist > 0 else 30.0
    return round(float(np.clip((equity * risk) / (sl_in_pips * pip_value_usd), C.min_lot, C.max_lot)), 2)

def new_acc(ts): return {'equity': Config.initial_balance, 'start_ts': ts, 'trades': [], 'blown': False, 'blown_rsn': '', 'peak': Config.initial_balance, 'consec_loss': 0}

# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, signals_data: dict, scenario_name: str) -> dict:
    C = Config
    pair_names = list(pairs.keys())
    
    idx_list = [pairs[n]['df'].index for n in pair_names]
    common_idx = idx_list[0]
    for i in idx_list[1:]: common_idx = common_idx.intersection(i)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)

    pa = {}
    for name in pair_names:
        df_p, sdata = pairs[name]['df'].reindex(common_idx).ffill(), signals_data[name]
        pa[name] = {'o': df_p['o'].values.astype(float), 'c': df_p['c'].values.astype(float), 'h': df_p['h'].values.astype(float), 'l': df_p['l'].values.astype(float), 'sig': sdata['sig'].reindex(common_idx).fillna(0).values.astype(int), 'z': sdata['z'].reindex(common_idx).fillna(np.nan).values.astype(float), 'atr': sdata['atr'].reindex(common_idx).fillna(0.0010).values.astype(float)}

    acc = new_acc(common_idx[C.warmup])
    total_withdrawn, acc_num, all_trades, acc_logs = 0.0, 1, [], []
    positions = {name: None for name in pair_names}
    trades_today = {name: 0 for name in pair_names}
    pending_sig = {name: 0 for name in pair_names}
    day_start_eq = C.initial_balance

    print(f"\n  ▶ Running: {scenario_name}")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        if acc['equity'] > acc['peak']: acc['peak'] = acc['equity']

        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0

        if (bar - C.warmup) % 150_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(f"    Progress: {pct:5.1f}% | Accounts Blown: {acc_num-1} | Target Hits: {len([l for l in acc_logs if l['reason']=='TARGET_HIT'])}", end='\r')

        if acc['blown']:
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': acc['blown_rsn'], 'pnl': acc['equity'] - C.initial_balance})
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0; pending_sig[name] = 0; positions[name] = None
            continue

        for name in pair_names:
            if pending_sig[name] != 0 and positions[name] is None and trades_today[name] < C.max_trades_day:
                sv = pending_sig[name]
                sl_dist = pa[name]['atr'][bar] * C.atr_sl_mult
                tp_dist = pa[name]['atr'][bar] * C.atr_tp_mult
                lot = calc_lot(acc['equity'], sl_dist, acc['consec_loss'], name)
                ep  = pa[name]['o'][bar] + sv * (C.slippage_pips + C.spread_pips / 2) * C.pip
                
                positions[name] = {'pair': name, 'dir': sv, 'lot': lot, 'lot_remaining': lot, 'entry': ep, 'sl': ep - sv * sl_dist, 'tp': ep + sv * tp_dist, 'entry_ts': ts, 'entry_bar': bar, 'be_activated': False}
                trades_today[name] += 1
            pending_sig[name] = 0

        total_float = sum([calc_pnl(p['dir'], p['entry'], pa[n]['l'][bar] if p['dir']==1 else pa[n]['h'][bar], p['lot_remaining'], n) for n, p in positions.items() if p])
        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= C.initial_balance * (1 - C.max_total_dd_pct):
            acc['blown'] = True
            acc['blown_rsn'] = "Wick/DailyDD" if current_eq <= daily_limit else "TotalDD"
            for n in pair_names:
                if positions[n]:
                    pnl = calc_pnl(positions[n]['dir'], positions[n]['entry'], pa[n]['c'][bar], positions[n]['lot_remaining'], n)
                    acc['equity'] += pnl
                    all_trades.append({'pair': n, 'pnl': pnl, 'status': 'BLOWN'})
                    positions[n] = None
            continue

        for name in pair_names:
            pos = positions[name]
            if pos is None: continue

            a, cp, d, ep, zn = pa[name], pa[name]['c'][bar], pos['dir'], pos['entry'], pa[name]['z'][bar]
            hit_z_exit = False
            
            # 🔴 Logic Exit BATTLE: (BE vs OFF)
            if not np.isnan(zn) and ((d == 1 and zn >= -C.z_exit_full) or (d == -1 and zn <= C.z_exit_full)):
                if C.z_exit_mode == 'BE' and not pos['be_activated']:
                    pos['sl'] = pos['entry'] # Break Even!
                    pos['be_activated'] = True
                elif C.z_exit_mode == 'CLOSE':
                    hit_z_exit = True

            hit_z_stop = (not np.isnan(zn)) and ((d == 1 and zn <= -C.z_stop_margin) or (d == -1 and zn >= C.z_stop_margin))
            hit_sl = (d == 1 and a['l'][bar] <= pos['sl']) or (d == -1 and a['h'][bar] >= pos['sl'])
            hit_tp = (d == 1 and a['h'][bar] >= pos['tp']) or (d == -1 and a['l'][bar] <= pos['tp'])

            time_stop = False
            if C.use_time_stop:
                bars_open = bar - pos['entry_bar']
                time_stop = ((bars_open >= 36 and calc_pnl(d, ep, cp, pos['lot_remaining'], name) < 0) or (bars_open >= 72))

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                has_slippage = True if (hit_sl or time_stop) else False
                
                if hit_sl:
                    st = 'BE-Stop' if pos['be_activated'] and pos['sl'] == pos['entry'] else 'SL'
                    exit_px = pos['sl']
                elif hit_tp:     exit_px, st = pos['tp'], 'TP'
                elif hit_z_stop: exit_px, st = cp, 'Z-Stop'
                elif time_stop:  exit_px, st = cp, 'TimeStop'
                else:            exit_px, st = cp, 'Z-Exit'

                final_pnl = calc_pnl(d, ep, exit_px, pos['lot_remaining'], name, apply_slippage=has_slippage)
                acc['equity'] += final_pnl
                all_trades.append({'pair': name, 'pnl': final_pnl, 'status': st})
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

    print(" " * 80, end='\r') # clear line
    return {'trades': all_trades, 'logs': acc_logs, 'bank': total_withdrawn, 'eq': acc['equity'], 'name': scenario_name}

# ═══════════════════════════════════════════════════════════════════════════
#  A/B TEST RUNNER & REPORTER
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    pairs = load_all_direct_pairs()
    print("  Computing Statistical Signals...")
    signals_data = {name: compute_signals(name, info['df']) for name, info in pairs.items()}

    # 🔴 تعریف سناریوهای تست
    scenarios = [
        {'name': '[Strategy A] PURE RISK/REWARD (Z-Exit OFF, TimeStop OFF)', 'z_mode': 'OFF', 'ts': False},
        {'name': '[Strategy B] SMART BREAK-EVEN (Z-Exit -> BE, TimeStop OFF)', 'z_mode': 'BE', 'ts': False}
    ]

    results = []
    for s in scenarios:
        Config.z_exit_mode = s['z_mode']
        Config.use_time_stop = s['ts']
        res = run_backtest(pairs, signals_data, s['name'])
        results.append(res)

    print("\n" + "═" * 75)
    print(" 🏆 BATTLE REPORT: Strategy A vs Strategy B")
    print("═" * 75)
    
    for r in results:
        df_t = pd.DataFrame(r['trades'])
        if len(df_t) == 0: continue
        wins, losses = df_t[df_t['pnl'] > 0], df_t[df_t['pnl'] < 0]
        wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
        pf = (wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'))
        targets = sum(1 for l in r['logs'] if l['reason'] == 'TARGET_HIT')
        blown = sum(1 for l in r['logs'] if l['reason'] != 'TARGET_HIT')

        print(f" {r['name']}")
        print(f"   Target Hits : {targets} ✅")
        print(f"   Blown Accs  : {blown} 💥")
        print(f"   Total Banked: ${r['bank']:,.2f}")
        print(f"   Win Rate    : {wr:.2f}%")
        print(f"   Profit Fact : {pf:.2f}")
        
        # Breakdown of Exits
        exit_counts = df_t['status'].value_counts()
        tp_cnt = exit_counts.get('TP', 0)
        sl_cnt = exit_counts.get('SL', 0)
        be_cnt = exit_counts.get('BE-Stop', 0)
        print(f"   Exits       : TP: {tp_cnt} | SL: {sl_cnt} | BE-Stop: {be_cnt}")
        print("-" * 75)
        
    print(f"  ✅ Experiment completed in {(datetime.now() - t0).total_seconds():.2f}s")
