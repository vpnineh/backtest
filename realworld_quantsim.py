import pandas as pd
import numpy as np
import logging
import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')

class RealWorldSimulator:
    def __init__(self):
        self.initial_balance = 5000.0  
        self.risk_per_trade = 0.01     
        self.tc = 0.0001               
        
    def load_data(self):
        # خواندن تمامی فایل‌های CSV در پوشه data
        all_files = glob.glob('data/*.csv')
        eur_files = [f for f in all_files if 'eurusd' in f.lower()]
        gbp_files = [f for f in all_files if 'gbpusd' in f.lower()]

        def process_files(files):
            dfs = []
            for f in files:
                df = pd.read_csv(f, sep=';', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S').dt.tz_localize(None) # حذف تایم‌زون
                dfs.append(df[['timestamp', 'close']])
            return pd.concat(dfs).sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')

        df_eur = process_files(eur_files).rename(columns={'close': 'EURUSD'})
        df_gbp = process_files(gbp_files).rename(columns={'close': 'GBPUSD'})
        
        self.base_data = df_eur.join(df_gbp, how='inner').dropna()
        print(f"✅ دیتای 5 ساله محلی لود شد: {len(self.base_data):,} کندل")

    def run_simulation(self):
        df = self.base_data.resample('15min').last().dropna()
        
        # استراتژی آربیتراژ آماری (تست شده)
        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        roll = df['Spread'].rolling(150)
        df['Z'] = (df['Spread'] - roll.mean()) / (roll.std() + 1e-8)
        
        # منطق معاملاتی (ورود Z > 2.5 و خروج Z = 0)
        df['Pos'] = 0
        df.loc[df['Z'] < -2.5, 'Pos'] = 1
        df.loc[df['Z'] > 2.5, 'Pos'] = -1
        df.loc[abs(df['Z']) <= 0.5, 'Pos'] = 0
        df['Pos'] = df['Pos'].replace(0, np.nan).ffill().fillna(0).shift(1)
        
        # محاسبه سود و زیان دلاری
        r_eur = df['EURUSD'].pct_change()
        r_gbp = df['GBPUSD'].pct_change()
        returns = (df['Pos'] * (r_eur - r_gbp) - self.tc).fillna(0)
        
        # سود مرکب با لوریج 4
        df['Equity'] = self.initial_balance * (1 + (returns * 4)).cumprod()
        df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax() * 100
        
        # خروجی نهایی
        df.to_csv("Equity_Curve_Report.csv")
        print(f"💰 موجودی نهایی: ${df['Equity'].iloc[-1]:,.2f}")
        print(f"📉 بیشترین افت حساب: {df['Drawdown'].min():.2f}%")

if __name__ == "__main__":
    sim = RealWorldSimulator()
    sim.load_data()
    sim.run_simulation()
