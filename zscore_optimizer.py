import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import logging
import requests
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PairsTradingOptimizerMultiTF:
    def __init__(self, asset1: str, asset2: str):
        self.asset1 = asset1
        self.asset2 = asset2
        # هزینه واقعی حساب ECN (شامل کمیسیون + اسپرد) = 2 پیپ
        self.transaction_cost = 0.0002 

    def fetch_data(self, period: str, interval: str) -> pd.DataFrame:
        logging.info(f"دانلود دیتا: تایم‌فریم {interval} برای {period}...")
        try:
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            df = yf.download(
                [self.asset1, self.asset2], 
                period=period, 
                interval=interval, 
                session=session
            )['Close']
            return df.dropna()
        except Exception as e:
            logging.error(f"خطا در دانلود {interval}: {e}")
            return pd.DataFrame()

    def backtest(self, data: pd.DataFrame, window: int, z_entry: float, z_exit: float) -> dict:
        df = data.copy()
        
        df['Spread'] = np.log(df[self.asset1]) - np.log(df[self.asset2])
        df['Spread_Mean'] = df['Spread'].rolling(window=window).mean()
        df['Spread_Std'] = df['Spread'].rolling(window=window).std()
        df['Z_Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']
        
        df['Signal'] = 0
        df.loc[df['Z_Score'] < -z_entry, 'Signal'] = 1
        df.loc[df['Z_Score'] > z_entry, 'Signal'] = -1
        df.loc[abs(df['Z_Score']) <= z_exit, 'Signal'] = 0
        
        df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').shift(1)
        df.dropna(inplace=True)
        
        if df.empty:
            return None

        ret1 = np.log(df[self.asset1] / df[self.asset1].shift(1))
        ret2 = np.log(df[self.asset2] / df[self.asset2].shift(1))
        
        position_changes = df['Position'].diff().abs()
        costs = position_changes * self.transaction_cost
        
        df['Strategy_Return'] = (df['Position'] * (ret1 - ret2)) - costs
        
        total_trades = position_changes.sum() / 2 
        cumulative_return = np.exp(df['Strategy_Return'].cumsum().iloc[-1]) - 1
        
        cum_ret_series = np.exp(df['Strategy_Return'].cumsum())
        rolling_max = cum_ret_series.cummax()
        drawdown = (cum_ret_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'Window': window,
            'Z_Entry': z_entry,
            'Z_Exit': z_exit,
            'Total_Return_%': round(cumulative_return * 100, 2),
            'Max_Drawdown_%': round(max_drawdown * 100, 2),
            'Total_Trades': int(total_trades)
        }

    def run_optimization(self):
        # تعریف تایم‌فریم‌ها با حداکثر دیتای مجازشان در یاهو
        configs = [
            {'interval': '1d', 'period': '5y'},    # 5 سال گذشته روزانه
            {'interval': '1h', 'period': '730d'},  # 2 سال گذشته 1 ساعته
            {'interval': '15m', 'period': '60d'},  # 60 روز گذشته 15 دقیقه
        ]
        
        windows = [50, 100, 150, 200]
        z_entries = [2.0, 2.5, 3.0]
        z_exits = [0.0, 0.5]
        combinations = list(itertools.product(windows, z_entries, z_exits))
        
        all_results = pd.DataFrame()
        
        for cfg in configs:
            interval = cfg['interval']
            period = cfg['period']
            
            data = self.fetch_data(period, interval)
            if data.empty:
                continue
                
            results = []
            for window, z_entry, z_exit in combinations:
                res = self.backtest(data, window, z_entry, z_exit)
                if res and res['Total_Trades'] > 2:  
                    res['Timeframe'] = interval
                    res['Period'] = period
                    results.append(res)
                    
            if results:
                df_res = pd.DataFrame(results)
                # مرتب‌سازی برای هر تایم‌فریم
                df_res = df_res.sort_values(by='Total_Return_%', ascending=False)
                # اضافه کردن به دیتافریم کلی
                all_results = pd.concat([all_results, df_res], ignore_index=True)
                
        return all_results

if __name__ == "__main__":
    optimizer = PairsTradingOptimizerMultiTF(asset1="EURUSD=X", asset2="GBPUSD=X")
    best_results = optimizer.run_optimization()
    
    # مرتب‌سازی نهایی
    best_results = best_results[['Timeframe', 'Period', 'Window', 'Z_Entry', 'Z_Exit', 'Total_Return_%', 'Max_Drawdown_%', 'Total_Trades']]
    
    output_file = "multi_timeframe_optimization.csv"
    best_results.to_csv(output_file, index=False)
    
    print(f"\n--- نتایج بک‌تست چندگانه در {output_file} ذخیره شد ---")
