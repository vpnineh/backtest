import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PairsTradingBacktester:
    def __init__(self, asset1: str, asset2: str, start: str, end: str, z_entry: float = 2.0, z_exit: float = 0.0, window: int = 50):
        self.asset1 = asset1
        self.asset2 = asset2
        self.start = start
        self.end = end
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.window = window
        self.data = pd.DataFrame()

    def fetch_data(self):
        logging.info(f"در حال دانلود دیتای {self.asset1} و {self.asset2}...")
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        
        df = yf.download([self.asset1, self.asset2], start=self.start, end=self.end, session=session)['Close']
        self.data = df.dropna()
        
    def calculate_signals(self):
        logging.info("در حال محاسبه اسپرد و Z-Score...")
        df = self.data.copy()
        
        # محاسبه ضریب پوشش ریسک (Hedge Ratio) با استفاده از رگرسیون خطی
        df['Spread'] = np.log(df[self.asset1]) - np.log(df[self.asset2])
        
        # محاسبه Z-Score به صورت متحرک (Rolling)
        df['Spread_Mean'] = df['Spread'].rolling(window=self.window).mean()
        df['Spread_Std'] = df['Spread'].rolling(window=self.window).std()
        df['Z_Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']
        
        # تولید سیگنال‌های معاملاتی (1 برای خرید اسپرد، -1 برای فروش اسپرد)
        df['Signal'] = 0
        df.loc[df['Z_Score'] < -self.z_entry, 'Signal'] = 1      # خرید
        df.loc[df['Z_Score'] > self.z_entry, 'Signal'] = -1      # فروش
        df.loc[abs(df['Z_Score']) <= self.z_exit, 'Signal'] = 0  # خروج در نقطه تعادل
        
        # پر کردن روزهای خالی با وضعیت قبلی (حفظ پوزیشن)
        df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
        df['Position'] = df['Position'].shift(1) # شیفت برای جلوگیری از خطای دید در آینده
        
        self.data = df.dropna()

    def calculate_pnl(self):
        logging.info("در حال محاسبه سود و زیان (PnL)...")
        # محاسبه بازدهی روزانه هر جفت ارز
        ret1 = np.log(self.data[self.asset1] / self.data[self.asset1].shift(1))
        ret2 = np.log(self.data[self.asset2] / self.data[self.asset2].shift(1))
        
        # بازدهی کل استراتژی
        self.data['Strategy_Return'] = self.data['Position'] * (ret1 - ret2)
        self.data['Cumulative_Return'] = self.data['Strategy_Return'].cumsum()
        
        total_return = np.exp(self.data['Cumulative_Return'].iloc[-1]) - 1
        logging.info(f"*** بازدهی کل استراتژی: {total_return * 100:.2f}% ***")
        
    def plot_results(self, filename="backtest_result.png"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # نمودار Z-Score
        ax1.plot(self.data.index, self.data['Z_Score'], label='Z-Score', color='blue')
        ax1.axhline(self.z_entry, color='red', linestyle='--', label='Sell Threshold (+2)')
        ax1.axhline(-self.z_entry, color='green', linestyle='--', label='Buy Threshold (-2)')
        ax1.axhline(self.z_exit, color='black', linestyle='-', label='Mean (0)')
        ax1.set_title(f"Z-Score for {self.asset1} and {self.asset2}")
        ax1.legend()
        
        # نمودار بازدهی سرمایه (Equity Curve)
        cumulative_pct = (np.exp(self.data['Cumulative_Return']) - 1) * 100
        ax2.plot(self.data.index, cumulative_pct, label='Strategy Equity Curve (%)', color='purple')
        ax2.set_title("Cumulative Returns (%)")
        ax2.set_ylabel("Profit %")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
        logging.info(f"نمودار نتایج در {filename} ذخیره شد.")

if __name__ == "__main__":
    # بک‌تست روی جفت‌ارزی که در مرحله قبل پیدا کردیم
    tester = PairsTradingBacktester(
        asset1="EURUSD=X", 
        asset2="GBPUSD=X", 
        start="2021-01-01", 
        end="2024-01-01",
        window=50  # میانگین متحرک 50 روزه
    )
    
    tester.fetch_data()
    tester.calculate_signals()
    tester.calculate_pnl()
    tester.plot_results("eur_gbp_backtest.png")
