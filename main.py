import os
import glob
import zipfile
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ==========================================
# 1. FAST DATA INGESTION
# ==========================================
def load_historical_data(pair, data_dir='data'):
    print(f"[*] Scanning '{data_dir}' for {pair} data...")
    zip_files = glob.glob(os.path.join(data_dir, f'*{pair}*.zip'))
    all_dfs = []
    
    for zf in zip_files:
        with zipfile.ZipFile(zf, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.csv'):
                    with z.open(file_info) as f:
                        df = pd.read_csv(f, sep=';', header=None,
                                         names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                                         engine='c', on_bad_lines='skip')
                        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"[!] No data found for {pair} in '{data_dir}'.")

    print(f"[*] Merging and sorting data...")
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df['datetime'] = pd.to_datetime(master_df['datetime'], format='%Y%m%d %H%M%S')
    master_df.set_index('datetime', inplace=True)
    master_df.sort_index(inplace=True)
    
    return master_df[['open', 'high', 'low', 'close']]

# ==========================================
# 2. VECTORIZED INDICATORS (ZERO LOOK-AHEAD)
# ==========================================
def prepare_multi_timeframe_data(df):
    print("[*] Calculating Multi-Timeframe Indicators (Vectorized)...")
    
    # --- H4 Calculations ---
    df_h4 = df['close'].resample('4h', label='left', closed='left').last().to_frame(name='close')
    df_h4['ema_200'] = EMAIndicator(close=df_h4['close'], window=200).ema_indicator()
    df_h4.index = df_h4.index + pd.Timedelta(hours=4) 

    # --- M15 Calculations ---
    df_m15 = df.resample('15min', label='left', closed='left').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'})
    
    df_m15['rsi'] = RSIIndicator(close=df_m15['close'], window=14).rsi()
    
    bb = BollingerBands(close=df_m15['close'], window=20, window_dev=2.0)
    df_m15['bb_bot'] = bb.bollinger_lband()
    df_m15['bb_top'] = bb.bollinger_hband()
        
    atr100 = AverageTrueRange(high=df_m15['high'], low=df_m15['low'], close=df_m15['close'], window=100)
    df_m15['atr_100'] = atr100.average_true_range()
    
    adx = ADXIndicator(high=df_m15['high'], low=df_m15['low'], close=df_m15['close'], window=14)
    df_m15['adx'] = adx.adx()
    
    # Simple Engulfing Pattern Logic
    df_m15['prev_open'] = df_m15['open'].shift(1)
    df_m15['prev_close'] = df_m15['close'].shift(1)
    df_m15['bull_engulf'] = (df_m15['prev_close'] < df_m15['prev_open']) & (df_m15['close'] > df_m15['open']) & (df_m15['close'] > df_m15['prev_open']) & (df_m15['open'] < df_m15['prev_close'])
    df_m15['bear_engulf'] = (df_m15['prev_close'] > df_m15['prev_open']) & (df_m15['close'] < df_m15['open']) & (df_m15['close'] < df_m15['prev_open']) & (df_m15['open'] > df_m15['prev_close'])
    
    df_m15.index = df_m15.index + pd.Timedelta(minutes=15)

    # --- M5 Calculations ---
    df_m5 = df.resample('5min', label='left', closed='left').agg({'high':'max', 'low':'min', 'close':'last'})
    atr14 = AverageTrueRange(high=df_m5['high'], low=df_m5['low'], close=df_m5['close'], window=14)
    df_m5['atr_14_m5'] = atr14.average_true_range()
    df_m5.index = df_m5.index + pd.Timedelta(minutes=5)

    # --- Merge All to M1 ---
    print("[*] Merging Timeframes without bias...")
    merged = df.copy()
    
    merged = merged.join(df_h4[['ema_200']], how='left')
    merged = merged.join(df_m15[['rsi', 'bb_bot', 'bb_top', 'atr_100', 'adx', 'bull_engulf', 'bear_engulf']], how='left')
    merged = merged.join(df_m5[['atr_14_m5']], how='left')
    
    merged.ffill(inplace=True)
    merged.dropna(inplace=True) 
    
    return merged

# ==========================================
# 3. HIGH-SPEED ENGINE (NUMPY ITERATION)
# ==========================================
def run_fast_backtest(df, pair):
    print(f"[*] Starting High-Speed Backtest Engine over {len(df)} 1-Minute candles...")
    
    closes = df['close'].values
    ema200 = df['ema_200'].values
    rsi = df['rsi'].values
    bb_bot = df['bb_bot'].values
    bb_top = df['bb_top'].values
    atr_100 = df['atr_100'].values
    adx = df['adx'].values
    bull_eng = df['bull_engulf'].values
    bear_eng = df['bear_engulf'].values
    atr_14_m5 = df['atr_14_m5'].values
    
    hours = df.index.hour.values
    days = df.index.dayofweek.values
    
    pip_val = 0.0001 if 'JPY' not in pair else 0.01
    grid_multipliers = [1.00, 1.35, 1.80, 2.40, 3.20, 4.30, 5.0]
    max_levels = 7
    target_profit_pct = 0.01
    
    equity = 10000.0
    
    basket_dir = 0
    positions = [] 
    basket_start_equity = 0.0
    
    total_trades = 0
    winning_baskets = 0
    losing_baskets = 0
    max_dd = 0.0
    peak_equity = equity
    
    for i in range(100, len(closes)):
        price = closes[i]
        
        if equity > peak_equity: peak_equity = equity
        current_dd = (peak_equity - equity) / peak_equity
        if current_dd > max_dd: max_dd = current_dd
            
        if basket_dir != 0:
            unrealized = 0.0
            for p, s in positions:
                unrealized += (price - p) * s * basket_dir * (1/pip_val) 
                
            target_profit = basket_start_equity * target_profit_pct
            
            if unrealized >= target_profit or adx[i] > 35:
                equity += unrealized
                if unrealized > 0: winning_baskets += 1
                else: losing_baskets += 1
                
                basket_dir = 0
                positions.clear()
                continue
                
            levels_open = len(positions)
            if levels_open < max_levels:
                last_price = positions[-1][0]
                
                dist_pips = (0.8 * atr_14_m5[i]) / pip_val
                dist_pips = max(25, min(dist_pips, 45))
                grid_dist = dist_pips * pip_val
                
                if basket_dir == 1 and price <= (last_price - grid_dist):
                    new_size = 10 * grid_multipliers[levels_open] 
                    positions.append((price, new_size))
                    total_trades += 1
                    
                elif basket_dir == -1 and price >= (last_price + grid_dist):
                    new_size = 10 * grid_multipliers[levels_open]
                    positions.append((price, new_size))
                    total_trades += 1
            
            continue 

        if not (8 <= hours[i] <= 17): continue
        if days[i] == 4 and hours[i] >= 12: continue
        
        if adx[i] < 25 and atr_100[i] < (1.3 * atr_100[i-1]):
            if price > ema200[i] and rsi[i] < 28 and price < bb_bot[i] and bull_eng[i]:
                basket_dir = 1
                basket_start_equity = equity
                positions.append((price, 10)) 
                total_trades += 1
                
            elif price < ema200[i] and rsi[i] > 72 and price > bb_top[i] and bear_eng[i]:
                basket_dir = -1
                basket_start_equity = equity
                positions.append((price, 10))
                total_trades += 1

    return equity, max_dd, total_trades, winning_baskets, losing_baskets

if __name__ == '__main__':
    pair_to_test = 'AUDNZD' 
    print("="*50)
    print("ULTRA-FAST PROFESSIONAL BACKTEST PIPELINE")
    print("="*50)
    
    try:
        raw_df = load_historical_data(pair=pair_to_test, data_dir='data')
        ready_df = prepare_multi_timeframe_data(raw_df)
        
        final_equity, mdd, t_trades, w_baskets, l_baskets = run_fast_backtest(ready_df, pair_to_test)
        
        print("\n" + "="*50)
        print(f"BACKTEST RESULTS: {pair_to_test}")
        print("="*50)
        print(f"Final Balance:    ${final_equity:.2f}")
        print(f"Max Drawdown:     {mdd*100:.2f}%")
        print(f"Total Grids/Trades: {w_baskets + l_baskets} / {t_trades}")
        
        total_baskets = w_baskets + l_baskets
        if total_baskets > 0:
            win_rate = (w_baskets / total_baskets) * 100
            print(f"Basket Win Rate:  {win_rate:.2f}%")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during execution: {e}")
