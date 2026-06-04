import yfinance as yf
import pandas as pd
import numpy as np
import itertools
import logging
import requests
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PairsTradingOptimizer:
    def __init__(self, asset1: str, asset2: str, period: str = "730d", interval: str = "1h"):
        self.asset1 = asset1
        self.asset2 = asset2
        self.period = period
        self.interval = interval
        self.data = pd.DataFrame()
        
        # هزینه واقعی و دقیق حساب ECN (شامل کمیسیون + اسپرد)
        # مجموعاً 2 پیپ هزینه برای باز و بسته کردن هر دو جفت ارز در یک سیگنال
        self.transaction_cost = 0.0002 

    def fetch_intraday_data(self):
        logging.info(f"در حال دانلود دیتای {self.interval} برای {self.period} اخیر...")
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        
        df = yf.download(
            [self.asset1, self.asset2], 
            period=self.period, 
            interval=self.interval, 
            session=session
        )['Close']
        self.data = df.dropna()
        logging.info(f"تعداد کندل‌های دریافت شده: {len(self.data)}")

    def backtest(self, window: int, z_entry: float, z_exit: float) -> dict:
        df = pd.DataFrame()
        df[self.asset1] = self.data[self.asset1]
        df[self.asset2] = self.data[self.asset2]
        
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
        
        # کسر دقیق هزینه اسپرد و کمیسیون در هر بار تغییر پوزیشن
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

    def optimize(self):
        # بازه‌های جدید برای شکار روندهای بزرگتر و غلبه بر کمیسیون
        windows = [50, 100, 150, 200, 250]
        z_entries = [2.0, 2.25, 2.5, 2.75, 3.0, 3.5]
        z_exits = [0.0, 0.25, 0.5]
        
        combinations = list(itertools.product(windows, z_entries, z_exits))
        logging.info(f"شروع Grid Search روی {len(combinations)} ترکیب مختلف با احتساب اسپرد واقعی ECN...")
        
        results = []
        for window, z_entry, z_exit in combinations:
            res = self.backtest(window, z_entry, z_exit)
            if res and res['Total_Trades'] > 5:  
                results.append(res)
                
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Total_Return_%', ascending=False).reset_index(drop=True)
        
        return results_df

if __name__ == "__main__":
    optimizer = PairsTradingOptimizer(asset1="EURUSD=X", asset2="GBPUSD=X")
    optimizer.fetch_intraday_data()
    
    best_results = optimizer.optimize()
    
    output_file = "optimized_parameters.csv"
    best_results.head(30).to_csv(output_file, index=False)
    
    print("\n--- بهترین تنظیمات با احتساب اسپرد و کمیسیون واقعی ECN ---")
    print(best_results.head(5))
