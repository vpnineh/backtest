"""
CorrArb Prop Simulator — v12 THE HOLY GRAIL RUN
==============================================================================
هسته استراتژی: [Strategy A] برنده بلامنازع تست A/B (تارگت کامل در برابر استاپ داینامیک)
فیلترهای نهایی اضافه شده:
  ۱. Mid-Week Trading: فقط روزهای سه‌شنبه، چهارشنبه و پنج‌شنبه (حذف گپ دوشنبه و نویز جمعه)
  ۲. Deep Extreme Entry: ورود سخت‌گیرانه Z=2.5 (فقط موقعیت‌های قطعی و کشیدگی‌های شدید)
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

    # 🔴 فیلتر کشیدگی عمیق (ورود بسیار سخت‌گیرانه‌تر)
    z_entry            = 2.5         # قبلا 2.1 بود
    z_fast_period      = 96          

    # 🔴 فیلتر روزهای هفته (دوشنبه=0، جمعه=4 مسدود شدند)
    hour_start         = 2
    hour_end           = 19
    trade_days         = [1, 2, 3]   # فقط سه‌شنبه، چهارشنبه، پنج‌شنبه

    vr_period          = 200         
    vr_k               = 4           
    vr_max             = 0.90        

    # ── فیلتر روند بیدار شده ──
    use_trend_filter   = True
    trend_ma_period    = 200
    trend_max_slope    = 0.00003     

    # ── استاپ و تارگت داینامیک (نسبت 1 به 2 خالص) ──
    atr_sl_mult        = 3.0         
    atr_tp_mult        = 6.0         
    atr_period         = 14

    approx_quote_rate = {'EURGBP': 1.25, 'AUDNZD': 0.60}

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING 
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

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════
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
    trend_ok = (trend_slope < C.trend_max_slope)

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
#  BACKTEST ENGINE (PURE R:R LOGIC)
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, signals_data: dict) -> dict:
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
        pa[name] = {'o': df_p['o'].values.astype(float), 'c': df_p['c'].values.astype(float), 'h': df_p['h'].values.astype(float), 'l': df_p['l'].values.astype(float), 'sig': sdata['sig'].reindex(common_idx).fillna(0).values.astype(int), 'atr': sdata['atr'].reindex(common_idx).fillna(0.0010).values.astype(float)}

    acc = new_acc(common_idx[C.warmup])
    total_withdrawn, acc_num, all_trades, acc_logs = 0.0, 1, [], []
    positions = {name: None for name in pair_names}
    trades_today = {name: 0 for name in pair_names}
    pending_sig = {name: 0 for name in pair_names}
    day_start_eq = C.initial_balance

    print(f"\n  ▶ Running v12 Holy Grail Simulator...")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        if acc['equity'] > acc['peak']: acc['peak'] = acc['equity']

        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0

        if (bar - C.warmup) % 150_000 == 0 and bar > C.warmup:
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
            if pending_sig[name] != 0 and positions[name] is None and trades_today[name] < C.max_trades_day:
                sv = pending_sig[name]
                sl_dist = pa[name]['atr'][bar] * C.atr_sl_mult
                tp_dist = pa[name]['atr'][bar] * C.atr_tp_mult
                lot = calc_lot(acc['equity'], sl_dist, acc['consec_loss'], name)
                ep  = pa[name]['o'][bar] + sv * (C.slippage_pips + C.spread_pips / 2) * C.pip
                
                positions[name] = {'pair': name, 'dir': sv, 'lot': lot, 'entry': ep, 'sl': ep - sv * sl_dist, 'tp': ep + sv * tp_dist}
                trades_today[name] += 1
            pending_sig[name] = 0

        total_float = sum([calc_pnl(p['dir'], p['entry'], pa[n]['l'][bar] if p['dir']==1 else pa[n]['h'][bar], p['lot'], n) for n, p in positions.items() if p])
        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= C.initial_balance * (1 - C.max_total_dd_pct):
            acc['blown'] = True
            acc['blown_rsn'] = "Wick/DailyDD" if current_eq <= daily_limit else "TotalDD"
            for n in pair_names:
                if positions[n]:
                    pnl = calc_pnl(positions[n]['dir'], positions[n]['entry'], pa[n]['c'][bar], positions[n]['lot'], n)
                    acc['equity'] += pnl
                    all_trades.append({'pair': n, 'pnl': pnl, 'status': 'BLOWN'})
                    positions[n] = None
            continue

        for name in pair_names:
            pos = positions[name]
            if pos is None: continue

            a, d, ep = pa[name], pos['dir'], pos['entry']

            # Pure R:R -> Only SL or TP exits
            hit_sl = (d == 1 and a['l'][bar] <= pos['sl']) or (d == -1 and a['h'][bar] >= pos['sl'])
            hit_tp = (d == 1 and a['h'][bar] >= pos['tp']) or (d == -1 and a['l'][bar] <= pos['tp'])

            if hit_sl or hit_tp:
                st = 'SL' if hit_sl else 'TP'
                exit_px = pos['sl'] if hit_sl else pos['tp']
                final_pnl = calc_pnl(d, ep, exit_px, pos['lot'], name, apply_slippage=hit_sl) # Slippage only on SL
                
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

    print(" " * 80, end='\r') 
    return {'trades': all_trades, 'logs': acc_logs, 'bank': total_withdrawn, 'eq': acc['equity']}

# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    pairs = load_all_direct_pairs()
    print("  Computing Statistical Signals...")
    signals_data = {name: compute_signals(name, info['df']) for name, info in pairs.items()}

    res = run_backtest(pairs, signals_data)
    
    df_t = pd.DataFrame(res['trades'])
    if len(df_t) > 0:
        wins, losses = df_t[df_t['pnl'] > 0], df_t[df_t['pnl'] < 0]
        wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
        pf = (wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'))
        targets = sum(1 for l in res['logs'] if l['reason'] == 'TARGET_HIT')
        blown = sum(1 for l in res['logs'] if l['reason'] != 'TARGET_HIT')

        print("\n" + "═" * 75)
        print(" 🏆 THE HOLY GRAIL REPORT (v12) 🏆")
        print("═" * 75)
        print(f"   Target Hits : {targets} ✅")
        print(f"   Blown Accs  : {blown} 💥")
        print(f"   Total Banked: ${res['bank']:,.2f}")
        print(f"   Win Rate    : {wr:.2f}%")
        print(f"   Profit Fact : {pf:.2f}")
        print("-" * 75)
        print("   Exits Breakdown:")
        for st, cnt in df_t['status'].value_counts().items(): print(f"     {st:<8}: {cnt:>4}")
        print("═" * 75)
    else:
        print("\n❌ No trades executed.")
        
    print(f"  ✅ Executed in {(datetime.now() - t0).total_seconds():.2f}s")
