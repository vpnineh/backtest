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
        # هزینه تراکنش (اسپرد + کمیسیون) معادل 2 پیپ برای هر جفت ارز در هر معامله
        self.transaction_cost = 0.0004 

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
        """اجرای یک بک‌تست سریع و وکتورایز شده برای یک ترکیب از پارامترها"""
        df = pd.DataFrame()
        df[self.asset1] = self.data[self.asset1]
        df[self.asset2] = self.data[self.asset2]
        
        # محاسبه لگاریتم اسپرد
        df['Spread'] = np.log(df[self.asset1]) - np.log(df[self.asset2])
        
        # محاسبه Z-Score
        df['Spread_Mean'] = df['Spread'].rolling(window=window).mean()
        df['Spread_Std'] = df['Spread'].rolling(window=window).std()
        df['Z_Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']
        
        # سیگنال‌ها
        df['Signal'] = 0
        df.loc[df['Z_Score'] < -z_entry, 'Signal'] = 1
        df.loc[df['Z_Score'] > z_entry, 'Signal'] = -1
        df.loc[abs(df['Z_Score']) <= z_exit, 'Signal'] = 0
        
        df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').shift(1)
        df.dropna(inplace=True)
        
        if df.empty:
            return None

        # محاسبه سود و زیان (بازدهی)
        ret1 = np.log(df[self.asset1] / df[self.asset1].shift(1))
        ret2 = np.log(df[self.asset2] / df[self.asset2].shift(1))
        
        # اضافه کردن کسر هزینه اسپرد هنگام تغییر پوزیشن
        position_changes = df['Position'].diff().abs()
        costs = position_changes * self.transaction_cost
        
        df['Strategy_Return'] = (df['Position'] * (ret1 - ret2)) - costs
        
        # محاسبه فاکتورهای کلیدی
        total_trades = position_changes.sum() / 2  # تقسیم بر ۲ چون هر ورود یک خروج دارد
        cumulative_return = np.exp(df['Strategy_Return'].cumsum().iloc[-1]) - 1
        
        # محاسبه Max Drawdown
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
        # تعریف محدوده‌هایی که می‌خواهیم تست کنیم
        windows = [20, 30, 40, 50, 80, 100]
        z_entries = [1.5, 1.75, 2.0, 2.25, 2.5]
        z_exits = [0.0, 0.25, 0.5]
        
        combinations = list(itertools.product(windows, z_entries, z_exits))
        logging.info(f"شروع Grid Search روی {len(combinations)} ترکیب مختلف...")
        
        results = []
        for window, z_entry, z_exit in combinations:
            res = self.backtest(window, z_entry, z_exit)
            if res and res['Total_Trades'] > 10:  # فیلتر کردن حالت‌هایی که ترید خیلی کمی داشتند
                results.append(res)
                
        # مرتب‌سازی نتایج بر اساس بیشترین سود
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Total_Return_%', ascending=False).reset_index(drop=True)
        
        return results_df

if __name__ == "__main__":
    # تمرکز روی جفت ارز قدرتمندی که پیدا کرده بودیم
    optimizer = PairsTradingOptimizer(asset1="EURUSD=X", asset2="GBPUSD=X")
    optimizer.fetch_intraday_data()
    
    best_results = optimizer.optimize()
    
    # ذخیره 20 تنظیمات برتر برای بررسی
    output_file = "optimized_parameters.csv"
    best_results.head(20).to_csv(output_file, index=False)
    
    print("\n--- 5 تنظیمات برتر با احتساب اسپرد ---")
    print(best_results.head(5))
    logging.info(f"نتایج کامل در {output_file} ذخیره شد.")
