import os
import glob
import zipfile
import pandas as pd
import numpy as np
import matplotlib
# تنظیم بک‌اند غیرگرافیکی برای محیط‌های CI/CD مانند GitHub Actions
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback

# ==========================================
# 1. توابع خواندن دیتا و اندیکاتورها
# ==========================================

def load_hist_data(symbol, start_year, end_year, data_folder="data"):
    """خواندن خودکار فایل‌های CSV یا ZIP از پوشه دیتا"""
    print(f"[1/4] Loading data for {symbol} ({start_year}-{end_year})...")
    all_data = []
    for year in range(start_year, end_year + 1):
        patterns = [
            os.path.join(data_folder, f"*{symbol}*M1*{year}*.csv"),
            os.path.join(data_folder, f"*{symbol}*M1*{year}*.zip"),
            os.path.join(data_folder, f"*{symbol}*M1*{year}*.CSV"),
            os.path.join(data_folder, f"*{symbol}*M1*{year}*.ZIP")
        ]
        
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        
        for file in files:
            try:
                if file.lower().endswith('.zip'):
                    with zipfile.ZipFile(file, 'r') as z:
                        csv_name = [f for f in z.namelist() if f.lower().endswith('.csv')][0]
                        with z.open(csv_name) as f:
                            df = pd.read_csv(f, sep=';', header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                else:
                    df = pd.read_csv(file, sep=';', header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Error reading {file}: {e}")
                
    if not all_data:
        raise FileNotFoundError(f"No data found for {symbol} between {start_year} and {end_year}")
        
    df = pd.concat(all_data, ignore_index=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
    df.set_index('DateTime', inplace=True)
    df.sort_index(inplace=True)
    print(f"[1/4] Data loaded successfully. Total M1 rows: {len(df)}")
    return df

def resample_and_indicators(df_m1):
    """تبدیل به مولتی تایم فریم و محاسبه اندیکاتورها بدون Look-ahead"""
    print("[2/4] Resampling M1 to M15 and H4...")
    df_m15 = df_m1.resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    del df_m1
    import gc
    gc.collect()
    
    print("[2/4] Calculating H4 Trend (EMA 100)...")
    df_h4 = df_m15.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    
    df_h4['EMA_100'] = df_h4['Close'].ewm(span=100, adjust=False).mean()
    df_h4 = df_h4.shift(1) # جلوگیری از Look-ahead
    
    df_m15 = df_m15.join(df_h4[['EMA_100', 'Close']].rename(columns={'EMA_100': 'EMA_100_H4', 'Close': 'Close_H4'})).ffill()
    
    print("[3/4] Calculating M15 Indicators (ATR, RSI)...")
    high_low = df_m15['High'] - df_m15['Low']
    high_close = np.abs(df_m15['High'] - df_m15['Close'].shift())
    low_close = np.abs(df_m15['Low'] - df_m15['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df_m15['ATR'] = true_range.rolling(14).mean()
    
    delta = df_m15['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df_m15['RSI'] = 100 - (100 / (1 + rs))
    
    df_m15.dropna(inplace=True)
    print(f"[3/4] Indicators calculated. Total valid M15 candles: {len(df_m15)}")
    return df_m15

# ==========================================
# 2. موتور بک‌تست بهینه (itertuples)
# ==========================================

def run_backtest(df, initial_balance=10000, risk_pct=0.01, use_martingale=False):
    print(f"[4/4] Starting backtest engine... (Martingale Enabled: {use_martingale})")
    equity = initial_balance
    trades = []
    current_trade = None
    
    for row in df.itertuples(index=True):
        if current_trade:
            hit_sl = False
            hit_tp = False
            
            if current_trade['type'] == 'long':
                if row.Low <= current_trade['sl']:
                    hit_sl = True
                elif row.High >= current_trade['tp']:
                    hit_tp = True
            else: 
                if row.High >= current_trade['sl']:
                    hit_sl = True
                elif row.Low <= current_trade['tp']:
                    hit_tp = True
            
            if hit_sl:
                loss = current_trade['size'] * abs(current_trade['entry'] - current_trade['sl'])
                equity -= loss
                trades.append({**current_trade, 'exit': current_trade['sl'], 'profit': -loss, 'equity': equity})
                current_trade = None
            elif hit_tp:
                profit = current_trade['size'] * abs(current_trade['tp'] - current_trade['entry'])
                equity += profit
                trades.append({**current_trade, 'exit': current_trade['tp'], 'profit': profit, 'equity': equity})
                current_trade = None
            else:
                price = row.Close
                initial_entry = current_trade['initial_entry']
                initial_atr = current_trade['initial_atr']
                
                # 1. مارتینگل (در صورت روشن بودن)
                if use_martingale and current_trade['martingale_step'] < 1:
                    trigger_price = initial_entry - initial_atr if current_trade['type'] == 'long' else initial_entry + initial_atr
                    if (current_trade['type'] == 'long' and price <= trigger_price) or \
                       (current_trade['type'] == 'short' and price >= trigger_price):
                        new_size = (equity * risk_pct / initial_atr) * 1.5
                        current_trade['size'] += new_size
                        current_trade['entry'] = (current_trade['entry'] + price) / 2 
                        current_trade['martingale_step'] = 1
                        if current_trade['type'] == 'long':
                            current_trade['sl'] = initial_entry - (3 * initial_atr)
                        else:
                            current_trade['sl'] = initial_entry + (3 * initial_atr)
                
                # 2. پیرامید کردن (افزایش حجم در سود)
                if current_trade['pyramid_step'] < 2:
                    trigger_price = initial_entry + (1.5 * initial_atr) if current_trade['type'] == 'long' else initial_entry - (1.5 * initial_atr)
                    if (current_trade['type'] == 'long' and price >= trigger_price) or \
                       (current_trade['type'] == 'short' and price <= trigger_price):
                        new_size = (equity * risk_pct / initial_atr) * 0.5
                        current_trade['size'] += new_size
                        current_trade['pyramid_step'] += 1
                        if current_trade['type'] == 'long':
                            current_trade['sl'] = initial_entry + (0.5 * initial_atr)
                        else:
                            current_trade['sl'] = initial_entry - (0.5 * initial_atr)

        # --- منطق ورود جدید با تاییدیه پرایس اکشن ---
        if not current_trade:
            if pd.isna(row.ATR) or pd.isna(row.RSI) or pd.isna(row.Close_H4) or pd.isna(row.EMA_100_H4):
                continue
                
            trend_bullish = row.Close_H4 > row.EMA_100_H4
            trend_bearish = row.Close_H4 < row.EMA_100_H4
            
            # تاییدیه کندل: کندل فعلی باید در جهت معامله بسته شود (سبز برای خرید، قرمز برای فروش)
            bullish_candle = row.Close > row.Open
            bearish_candle = row.Close < row.Open
            
            # ورود Long: روند صعودی + RSI زیر 40 (اصلاح) + کندل صعودی (تاییدیه بازگشت)
            if trend_bullish and row.RSI < 40 and bullish_candle:
                entry = row.Open 
                atr = row.ATR
                size = (equity * risk_pct) / atr
                sl = entry - (1.5 * atr)   # ریسک 1.5 ATR
                tp = entry + (3.0 * atr)   # سود 3 ATR (ریسک به ریوارد 1 به 2)
                current_trade = {
                    'type': 'long', 'entry': entry, 'initial_entry': entry, 'sl': sl, 'tp': tp,
                    'size': size, 'initial_atr': atr, 'martingale_step': 0, 'pyramid_step': 0,
                    'entry_time': row.Index
                }
                
            # ورود Short: روند نزولی + RSI بالای 60 (اصلاح) + کندل نزولی (تاییدیه بازگشت)
            elif trend_bearish and row.RSI > 60 and bearish_candle:
                entry = row.Open
                atr = row.ATR
                size = (equity * risk_pct) / atr
                sl = entry + (1.5 * atr)
                tp = entry - (3.0 * atr)
                current_trade = {
                    'type': 'short', 'entry': entry, 'initial_entry': entry, 'sl': sl, 'tp': tp,
                    'size': size, 'initial_atr': atr, 'martingale_step': 0, 'pyramid_step': 0,
                    'entry_time': row.Index
                }

    print(f"[4/4] Backtest finished. Total trades executed: {len(trades)}")
    return trades

# ==========================================
# 3. اجرای اسکریپت و تحلیل نتایج
# ==========================================

if __name__ == "__main__":
    SYMBOL = "EURUSD"
    START_YEAR = 2015
    END_YEAR = 2023
    
    print(f"\n=== Backtest Started for {SYMBOL} ({START_YEAR}-{END_YEAR}) ===\n")
    
    try:
        df_m1 = load_hist_data(SYMBOL, START_YEAR, END_YEAR, data_folder="data")
        df_m15 = resample_and_indicators(df_m1)
        
        # مارتینگل در این مرحله خاموش است تا پایه استراتژی سنجیده شود
        trades = run_backtest(df_m15, initial_balance=10000, risk_pct=0.01, use_martingale=False)
        
        if not trades:
            print("\nResult: No trades were executed.")
        else:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv("trades_log.csv", index=False)
            
            total_trades = len(trades_df)
            win_rate = (trades_df['profit'] > 0).mean() * 100
            total_profit = trades_df['equity'].iloc[-1] - 10000
            
            # محاسبه دراوداون
            rolling_max = trades_df['equity'].cummax()
            drawdown = (trades_df['equity'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            gross_profit = trades_df.loc[trades_df['profit'] > 0, 'profit'].sum()
            gross_loss = abs(trades_df.loc[trades_df['profit'] < 0, 'profit'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            print("\n--- Backtest Results ---")
            print(f"Symbol: {SYMBOL} | Period: {START_YEAR}-{END_YEAR}")
            print(f"Initial Balance: $10,000")
            print(f"Final Balance: ${trades_df['equity'].iloc[-1]:.2f}")
            print(f"Total Net Profit: ${total_profit:.2f}")
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2f}%")
            
            plt.figure(figsize=(12, 6))
            plt.plot(trades_df['equity'])
            plt.title(f"Equity Curve - {SYMBOL} (Baseline Test)")
            plt.xlabel("Trade Number")
            plt.ylabel("Balance ($)")
            plt.grid(True)
            plt.savefig("equity_curve.png")
            print("\nEquity curve saved as 'equity_curve.png'")
            print("Trades log saved as 'trades_log.csv'")
            
    except Exception as e:
        print("\n!!! FATAL ERROR OCCURRED !!!")
        traceback.print_exc()
