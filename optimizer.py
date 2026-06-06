"""
CorrArb Prop Optimizer — Grid Search
هدف: یافتن بهترین پارامترها برای عبور از پراپ فرم (2010-2025)
"""

import pandas as pd
import numpy as np
import glob
import warnings
import itertools
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  Grid Search Parameters (پارامترهایی که ترکیب می‌شوند)
# ═══════════════════════════════════════════════════════════════════════════
PARAM_GRID = {
    'z_entry': [1.90, 2.00, 2.10],        # مرز ورود
    'z_stop_margin': [3.2, 3.6, 4.0],     # کات لاس داینامیک
    'corr_min': [0.70, 0.75, 0.80],       # فیلتر همبستگی
    'time_stop_bars': [48, 72, 96]        # حداکثر زمان پوزیشن (۱۲، ۱۸ و ۲۴ ساعت)
}

# ═══════════════════════════════════════════════════════════════════════════
#  کلاس Config پایه (ثابت‌ها)
# ═══════════════════════════════════════════════════════════════════════════
class BaseConfig:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10
    risk_base_pct      = 0.01
    risk_min_pct       = 0.005
    consec_loss_n      = 2
    risk_reduce        = 0.5
    spread_pips        = 1.2
    commission_per_lot = 7.0
    slippage_pips      = 0.3
    pip                = 0.0001
    lot_size           = 100_000
    max_lot            = 5.0
    min_lot            = 0.01
    warmup             = 500
    z_fast_period      = 96
    z_exit             = 0.1
    min_net_profit_usd = 10.0
    corr_period        = 96
    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 3
    sl_pips            = 40.0
    tp_pips            = 80.0
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

# ── توابع پایه (دقیقاً مشابه نسخه ۶) ──
def load_data() -> pd.DataFrame:
    print("  Loading 15 Years of Data... Please wait.")
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    
    def read_pair(paths, suffix):
        frames = [pd.read_csv(p, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v']) for p in paths]
        df = pd.concat(frames).sort_values('ts')
        df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
        df = df.set_index('ts')[~df['ts'].duplicated(keep='last')]
        df.columns = [f'{col}_{suffix}' for col in df.columns]
        return df

    raw = read_pair(files_eur, 'eur').join(read_pair(files_gbp, 'gbp'), how='inner').dropna()
    df = pd.DataFrame({
        'o_eur': raw['o_eur'].resample('15min').first(),
        'c_eur': raw['c_eur'].resample('15min').last(),
        'h_eur': raw['h_eur'].resample('15min').max(),
        'l_eur': raw['l_eur'].resample('15min').min(),
        'o_gbp': raw['o_gbp'].resample('15min').first(),
        'c_gbp': raw['c_gbp'].resample('15min').last(),
        'h_gbp': raw['h_gbp'].resample('15min').max(),
        'l_gbp': raw['l_gbp'].resample('15min').min(),
    }).dropna()
    df['c_eg'] = df['c_eur'] / df['c_gbp']
    df['o_eg'] = df['o_eur'] / df['o_gbp']
    df = df[df.index.weekday < 5]
    print(f"✅ Data Loaded: {len(df):,} Candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df

def calc_atr(h, l, c, period=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_dynamic_pnl(dir_trade, ep, cp, lot, cg, C):
    return (dir_trade * (cp - ep) * lot * C.lot_size * cg) - (C.commission_per_lot * lot)

def calc_lot(equity, sl_pips, consec_loss, C):
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n: risk = max(risk * C.risk_reduce, C.risk_min_pct)
    return round(float(np.clip(equity * risk / (sl_pips * 12.5), C.min_lot, C.max_lot)), 2)

# ═══════════════════════════════════════════════════════════════════════════
#  هسته بک‌تست (فشرده‌شده برای سرعت در Loop)
# ═══════════════════════════════════════════════════════════════════════════
def run_fast_backtest(df, params):
    C = BaseConfig()
    # Apply dynamic params
    C.z_entry = params['z_entry']
    C.z_stop_margin = params['z_stop_margin']
    C.corr_min = params['corr_min']
    C.time_stop_bars = params['time_stop_bars']

    # 1. Signals Pre-calculation
    log_ratio = np.log(df['c_eg'])
    z_score = (log_ratio - log_ratio.rolling(C.z_fast_period).mean()) / log_ratio.rolling(C.z_fast_period).std().replace(0, np.nan)
    corr_ok = df['c_eur'].pct_change().rolling(C.corr_period).corr(df['c_gbp'].pct_change()) > C.corr_min
    atr_eg = calc_atr(df['h_eur']/df['l_gbp'], df['l_eur']/df['h_gbp'], df['c_eg'], C.atr_period)
    atr_ma = atr_eg.rolling(C.atr_ma_period).mean()
    vol_ok = (atr_eg > atr_ma * C.atr_min_mult) & (atr_eg < atr_ma * C.atr_max_mult)
    time_ok = pd.Series(df.index.hour, index=df.index).between(C.hour_start, C.hour_end) & pd.Series(df.index.dayofweek, index=df.index).isin(C.trade_days)

    sig = pd.Series(0, index=df.index)
    sig[(z_score < -C.z_entry) & vol_ok & time_ok & corr_ok] = 1
    sig[(z_score > C.z_entry) & vol_ok & time_ok & corr_ok] = -1
    sig = sig.where(sig != sig.shift(), 0)

    # 2. Engine Vectors
    open_eg, close_eg, close_g = df['o_eg'].values, df['c_eg'].values, df['c_gbp'].values
    sig_a, z_a = sig.values, z_score.values
    ts_a = df.index

    # 3. Fast Engine
    PROP_FLOOR, PROFIT_LEVEL = C.initial_balance * (1 - C.max_total_dd_pct), C.initial_balance * (1 + C.profit_target_pct)
    total_withdrawn = 0.0
    acc_num = 1
    trades_list = []
    
    eq = C.initial_balance
    day_start_eq = eq
    consec_loss = 0
    pos = None
    trades_today = 0
    pending_sig = 0
    blown = False

    for bar in range(C.warmup, len(ts_a)):
        ts = ts_a[bar]
        if ts.hour == 0 and ts.minute == 0: day_start_eq, trades_today = eq, 0

        if blown:
            eq, day_start_eq, consec_loss, pos, trades_today, pending_sig, blown = C.initial_balance, C.initial_balance, 0, None, 0, 0, False
            acc_num += 1
            continue

        if pending_sig != 0 and pos is None and trades_today < C.max_trades_day:
            sv = pending_sig
            lot = calc_lot(eq, C.sl_pips, consec_loss, C)
            ep = open_eg[bar] + (sv * (C.slippage_pips + C.spread_pips/2) * C.pip)
            pos = {'dir': sv, 'lot': lot, 'entry': ep, 'sl': ep - sv * C.sl_pips * C.pip, 'tp': ep + sv * C.tp_pips * C.pip, 'entry_bar': bar}
            trades_today += 1
        pending_sig = 0

        if pos is not None:
            cp, cg, d, ep, zn = close_eg[bar], close_g[bar], pos['dir'], pos['entry'], z_a[bar]
            current_pnl = calc_dynamic_pnl(d, ep, cp, pos['lot'], cg, C)
            current_eq = eq + current_pnl

            if current_eq <= day_start_eq * (1 - C.max_daily_loss_pct) or current_eq <= PROP_FLOOR:
                trades_list.append({'pnl': current_pnl, 'status': 'BLOWN'})
                blown = True; pos = None; continue

            hit_z_stop = not np.isnan(zn) and ((d == 1 and zn <= -C.z_stop_margin) or (d == -1 and zn >= C.z_stop_margin))
            hit_z_exit = not np.isnan(zn) and ((d == 1 and zn >= -C.z_exit) or (d == -1 and zn <= C.z_exit))
            if hit_z_exit and current_pnl < C.min_net_profit_usd: hit_z_exit = False
            
            hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
            hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])
            time_stop = (bar - pos['entry_bar']) >= C.time_stop_bars

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                st = 'SL' if hit_sl else ('TP' if hit_tp else ('Z-Stop' if hit_z_stop else ('TimeStop' if time_stop else 'Z-Exit')))
                final_pnl = calc_dynamic_pnl(d, ep, pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp), pos['lot'], cg, C)
                eq += final_pnl
                trades_list.append({'pnl': final_pnl, 'status': st})
                if final_pnl > 0: consec_loss = 0
                else: consec_loss += 1
                pos = None

        if eq >= PROFIT_LEVEL and pos is None and not blown:
            total_withdrawn += (eq - C.initial_balance)
            acc_num += 1
            eq, day_start_eq, consec_loss, trades_today, pending_sig = C.initial_balance, C.initial_balance, 0, 0, 0

        if pos is None and not blown and trades_today < C.max_trades_day and sig_a[bar] != 0:
            pending_sig = sig_a[bar]

    # Metrics Calc
    if not trades_list: return {'pf': 0, 'wr': 0, 'trades': 0, 'banked': 0, 'blown': acc_num-1}
    df_t = pd.DataFrame(trades_list)
    wins, losses = df_t[df_t['pnl'] > 0], df_t[df_t['pnl'] < 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    blown_count = len(df_t[df_t['status'] == 'BLOWN'])
    target_hits = acc_num - 1 - blown_count

    return {
        'pf': round(pf, 2),
        'wr': round(len(wins) / len(df_t) * 100, 1),
        'trades': len(df_t),
        'banked': round(total_withdrawn, 0),
        'blown': blown_count,
        'passed': target_hits
    }

# ═══════════════════════════════════════════════════════════════════════════
#  اجرای Optimizer
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n🚀 Starting Grid Search: {len(combinations)} combinations to test.")
    print("-" * 80)
    print(f"{'Z_Ent':<6} | {'Z_Stp':<5} | {'Corr':<4} | {'Time':<4} || {'PF':<4} | {'WR%':<4} | {'Trades':<6} | {'Passed':<6} | {'Blown':<5} | {'Banked $':<8}")
    print("-" * 80)

    results_list = []
    t0 = time.time()

    for idx, params in enumerate(combinations):
        res = run_fast_backtest(df, params)
        res.update(params)
        results_list.append(res)
        
        # چاپ زنده نتایجی که ارزش دیدن دارند (PF بالای 1.15)
        if res['pf'] > 1.15:
            print(f"{params['z_entry']:<6.2f} | {params['z_stop_margin']:<5.1f} | {params['corr_min']:<4.2f} | {params['time_stop_bars']:<4} || "
                  f"{res['pf']:<4.2f} | {res['wr']:<4.1f} | {res['trades']:<6} | {res['passed']:<6} | {res['blown']:<5} | ${res['banked']:<8.0f}")

    print("-" * 80)
    print(f"⏱ Completed in {round(time.time() - t0, 1)} seconds.")
    
    # پیدا کردن بهترین نتیجه بر اساس Profit Factor
    best_pf = max(results_list, key=lambda x: x['pf'])
    best_bank = max(results_list, key=lambda x: x['banked'])

    print("\n🏆 BEST BY PROFIT FACTOR:")
    print(best_pf)
    print("\n💰 BEST BY TOTAL MONEY BANKED:")
    print(best_bank)
