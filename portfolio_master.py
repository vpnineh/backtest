"""
CorrArb Portfolio ML Master — v10 (Auto-Calibrator & OOS Forward Test)
هدف: کشف خودکار پارامترها در 10 سال گذشته (Train) و تست پورتفولیو در 5 سال آینده (OOS)
"""

import os
import glob
import zipfile
import itertools
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG 
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    initial_balance    = 5000.0
    prop_cost          = 300.0      
    profit_target_pct  = 0.05       
    max_daily_loss_pct = 0.05       
    max_total_dd_pct   = 0.10       
    
    risk_per_trade     = 0.005      
    assumed_sl_pct     = 0.004      
    max_concurrent     = 3          
    
    # ── تقسیم دیتای 15 ساله ──
    train_end_date     = '2020-12-31' # 10 سال آموزش و کالیبراسیون
    test_start_date    = '2021-01-01' # 5 سال تست واقعی و نابینا (OOS)
    
    lot_size           = 100_000
    commission         = 7.0
    spread_pips        = 1.2

PORTFOLIO_PAIRS = {
    'EURGBP': ('EURUSD', 'GBPUSD'),
    'AUDNZD': ('AUDUSD', 'NZDUSD'),
    'XAUXAG': ('XAUUSD', 'XAGUSD')
}

# ── شبکه جستجو (Grid Search) برای کشف داینامیک ──
GRID_PARAMS = {
    'z_entry': [1.8, 2.2, 2.5],
    'trend_atr_thresh': [0.0010, 0.0015],
    'ml_prob_thresh': [0.52, 0.55, 0.60]
}

# ═══════════════════════════════════════════════════════════════════════════
#  توابع پایه‌ای و پاک‌سازی
# ═══════════════════════════════════════════════════════════════════════════
def clean_and_extract_data(data_path='data'):
    zips = glob.glob(os.path.join(data_path, '*.zip'))
    if zips:
        print(f"📦 Found {len(zips)} ZIP files. Extracting and cleaning up...")
        for z in zips:
            try:
                with zipfile.ZipFile(z, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
                os.remove(z) 
            except Exception: pass
        txt_files = glob.glob(os.path.join(data_path, '*.txt'))
        for txt in txt_files:
            try: os.remove(txt)
            except Exception: pass
        print("✅ Cleanup complete. Only CSVs remain.")
    else:
        print("✅ Directory is clean. Reading directly from CSVs.")

def load_symbol(symbol, all_csvs):
    files = [f for f in all_csvs if symbol in f.upper()]
    if not files: return None
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=r'[;,]', engine='python', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S', errors='coerce')
            df = df.dropna(subset=['ts']).drop_duplicates('ts').set_index('ts').sort_index()
            resampled = pd.DataFrame({
                'o': df['o'].resample('15min').first(),
                'h': df['h'].resample('15min').max(),
                'l': df['l'].resample('15min').min(),
                'c': df['c'].resample('15min').last(),
            }).dropna()
            frames.append(resampled)
        except Exception: pass
    if not frames: return None
    final_df = pd.concat(frames).sort_index()
    return final_df[~final_df.index.duplicated(keep='last')]

def calc_atr(h, l, c, period=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ═══════════════════════════════════════════════════════════════════════════
#  موتور کالیبراسیون و آموزش (Auto-Calibrator)
# ═══════════════════════════════════════════════════════════════════════════
def process_and_calibrate():
    clean_and_extract_data('data')
    print("\n⚙️ STARTING AUTO-CALIBRATION (Train Data: 2010 - 2020) ...")
    all_csvs = glob.glob('data/*.csv')
    ml_data = {}
    
    keys, values = zip(*GRID_PARAMS.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for pair_name, (base, quote) in PORTFOLIO_PAIRS.items():
        print(f"\n  🔍 Optimizing [{pair_name}]...")
        df_base = load_symbol(base, all_csvs)
        df_quote = load_symbol(quote, all_csvs)
        if df_base is None or df_quote is None:
            print(f"     ⚠️ Missing data, skipping...")
            continue
            
        raw = df_base.join(df_quote, lsuffix='_b', rsuffix='_q').dropna()
        raw['c_cross'] = raw['c_b'] / raw['c_q']
        raw['atr'] = calc_atr(raw['h_b']/raw['l_q'], raw['l_b']/raw['h_q'], raw['c_cross'], 14)
        raw['atr_norm'] = raw['atr'] / raw['c_cross']
        raw['log_ratio'] = np.log(raw['c_cross'])
        raw['z_score_raw'] = (raw['log_ratio'] - raw['log_ratio'].rolling(96).mean()) / raw['log_ratio'].rolling(96).std()
        raw['mom_96'] = raw['c_cross'] / raw['c_cross'].shift(96) - 1
        raw['future_ret'] = raw['c_cross'].shift(-12) / raw['c_cross'] - 1
        raw = raw.dropna()
        
        train_full = raw[raw.index <= Config.train_end_date].copy()
        test_full  = raw[raw.index >= Config.test_start_date].copy()
        
        best_pf = -1.0
        best_params = None
        best_model = None
        
        # گرید سرچ روی دیتای Train
        for params in grid_combinations:
            tdf = train_full.copy()
            tdf['regime'] = np.where(tdf['atr_norm'] > params['trend_atr_thresh'], 'TREND', 'RANGE')
            
            range_l = (tdf['regime'] == 'RANGE') & (tdf['z_score_raw'] < -params['z_entry'])
            range_s = (tdf['regime'] == 'RANGE') & (tdf['z_score_raw'] > params['z_entry'])
            trend_l = (tdf['regime'] == 'TREND') & (tdf['mom_96'] > 0.005)
            trend_s = (tdf['regime'] == 'TREND') & (tdf['mom_96'] < -0.005)
            
            tdf['raw_sig'] = 0
            tdf.loc[range_l | trend_l, 'raw_sig'] = 1
            tdf.loc[range_s | trend_s, 'raw_sig'] = -1
            tdf['raw_sig'] = tdf['raw_sig'].where(tdf['raw_sig'] != tdf['raw_sig'].shift(), 0)
            
            tdf['label'] = 0
            tdf.loc[(tdf['raw_sig'] == 1) & (tdf['future_ret'] > 0), 'label'] = 1
            tdf.loc[(tdf['raw_sig'] == -1) & (tdf['future_ret'] < 0), 'label'] = 1
            
            train_sigs = tdf[tdf['raw_sig'] != 0]
            if len(train_sigs) < 20: continue
                
            features = ['z_score_raw', 'mom_96', 'atr_norm']
            rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
            rf.fit(train_sigs[features], train_sigs['label'])
            
            train_sigs['ml_prob'] = rf.predict_proba(train_sigs[features])[:, 1]
            approved = train_sigs[train_sigs['ml_prob'] >= params['ml_prob_thresh']]
            
            if len(approved) == 0: continue
            
            wins = approved[approved['label'] == 1]['future_ret'].sum()
            losses = abs(approved[approved['label'] == 0]['future_ret'].sum())
            pf = wins / losses if losses > 0 else wins
            
            if pf > best_pf:
                best_pf = pf
                best_params = params
                best_model = rf
                
        if best_params is None:
            print(f"     ⚠️ Could not find profitable parameters for {pair_name}")
            continue
            
        print(f"     ✅ Best Train Params: Z_Entry={best_params['z_entry']} | ATR={best_params['trend_atr_thresh']} | ML_Thresh={best_params['ml_prob_thresh']}")
        print(f"     🏆 Train PF: {best_pf:.2f}")
        
        # اعمال بهترین پارامترها روی دیتای Test (OOS)
        test_full['regime'] = np.where(test_full['atr_norm'] > best_params['trend_atr_thresh'], 'TREND', 'RANGE')
        range_l = (test_full['regime'] == 'RANGE') & (test_full['z_score_raw'] < -best_params['z_entry'])
        range_s = (test_full['regime'] == 'RANGE') & (test_full['z_score_raw'] > best_params['z_entry'])
        trend_l = (test_full['regime'] == 'TREND') & (test_full['mom_96'] > 0.005)
        trend_s = (test_full['regime'] == 'TREND') & (test_full['mom_96'] < -0.005)
        
        test_full['raw_sig'] = 0
        test_full.loc[range_l | trend_l, 'raw_sig'] = 1
        test_full.loc[range_s | trend_s, 'raw_sig'] = -1
        test_full['raw_sig'] = test_full['raw_sig'].where(test_full['raw_sig'] != test_full['raw_sig'].shift(), 0)
        
        test_full['ml_prob'] = 0.0
        sig_idx = test_full[test_full['raw_sig'] != 0].index
        if len(sig_idx) > 0:
            test_full.loc[sig_idx, 'ml_prob'] = best_model.predict_proba(test_full.loc[sig_idx, features])[:, 1]
            test_full['ml_approved'] = np.where((test_full['raw_sig'] != 0) & (test_full['ml_prob'] >= best_params['ml_prob_thresh']), test_full['raw_sig'], 0)
        else:
            test_full['ml_approved'] = 0
            
        ml_data[pair_name] = test_full
        
    return ml_data

# ═══════════════════════════════════════════════════════════════════════════
#  موتور اجرای فوروارد تست پورتفولیو (Out of Sample)
# ═══════════════════════════════════════════════════════════════════════════
def run_forward_test(ml_data):
    print("\n🚀 RUNNING FORWARD PORTFOLIO BACKTEST (OOS Data: 2021 - 2025)...")
    
    all_timestamps = []
    for df in ml_data.values(): all_timestamps.extend(df.index.tolist())
    if not all_timestamps:
        print("❌ No data for backtest.")
        return
        
    master_index = pd.DatetimeIndex(sorted(list(set(all_timestamps))))
    
    eq = day_start_eq = Config.initial_balance
    total_banked = 0.0
    blown_count = passed_count = 0
    trades_log = []
    
    for ts in master_index:
        if ts.hour == 0 and ts.minute == 0: day_start_eq = eq
            
        if eq <= day_start_eq * (1 - Config.max_daily_loss_pct) or eq <= Config.initial_balance * (1 - Config.max_total_dd_pct):
            blown_count += 1
            eq = day_start_eq = Config.initial_balance 
            continue
            
        step_pnl = 0
        active_trades_this_step = 0
        
        for sym, df in ml_data.items():
            if ts in df.index:
                row = df.loc[ts]
                sig = row['ml_approved']
                
                if sig != 0 and active_trades_this_step < Config.max_concurrent:
                    trade_ret = row['future_ret'] if sig == 1 else -row['future_ret']
                    position_size_usd = (eq * Config.risk_per_trade) / Config.assumed_sl_pct
                    lots = position_size_usd / Config.lot_size
                    transaction_costs = lots * (Config.commission + (Config.spread_pips * 10))
                    
                    trade_pnl = (trade_ret * position_size_usd) - transaction_costs
                    step_pnl += trade_pnl
                    active_trades_this_step += 1
                    trades_log.append({'sym': sym, 'pnl': trade_pnl, 'regime': row['regime']})
                    
        eq += step_pnl

        if eq >= Config.initial_balance * (1 + Config.profit_target_pct):
            passed_count += 1
            total_banked += ((eq - Config.initial_balance) - Config.prop_cost)
            eq = day_start_eq = Config.initial_balance

    df_res = pd.DataFrame(trades_log)
    if df_res.empty:
        print("❌ No valid trades found in Forward Test.")
        return
        
    wins = df_res[df_res['pnl'] > 0]
    losses = df_res[df_res['pnl'] < 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else float('inf')
    
    print("\n" + "═" * 65)
    print(" ▌  Final Portfolio Reality Check (2021-2025 OOS)  ▐")
    print("═" * 65)
    print(f" Total Trades:          {len(df_res):,}")
    print(f" Win Rate:              {(len(wins)/len(df_res))*100:.1f}%")
    print(f" Profit Factor:         {pf:.2f}")
    print(f" Accounts Passed:       {passed_count}")
    print(f" Accounts Blown:        {blown_count}")
    print(f" Total Net Banked:      ${total_banked:,.2f} (Clean Profit)")
    print("═" * 65)
    print("\nTrades Executed per Asset & Regime:")
    print(pd.crosstab(df_res['sym'], df_res['regime']))

if __name__ == "__main__":
    ml_data = process_and_calibrate()
    if ml_data: run_forward_test(ml_data)
