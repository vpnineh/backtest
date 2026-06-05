import pandas as pd
import numpy as np
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

class PropFirmReadyEngine:
    def __init__(self):
        self.initial_balance = 5000.0  # سرمایه اولیه
        self.tc = 0.0001               # هزینه تراکنش ECN (1 پیپ)
        self.leverage = 2.0            # اهرم کاهش‌یافته برای امنیت پراپ
        
    def load_data(self):
        # لود کردن فایل‌های EURUSD و GBPUSD از پوشه data
        all_files = glob.glob('data/*.csv')
        eur_files = [f for f in all_files if 'eurusd' in f.lower()]
        gbp_files = [f for f in all_files if 'gbpusd' in f.lower()]

        def process(files):
            dfs = [pd.read_csv(f, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v']) for f in files]
            df = pd.concat(dfs).sort_values('ts').drop_duplicates('ts')
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            return df.set_index('ts')[['c']]

        self.base_data = process(eur_files).join(process(gbp_files), lsuffix='_eur', rsuffix='_gbp').dropna()
        print(f"✅ دیتا با موفقیت لود شد: {len(self.base_data):,} کندل")

    def run_simulation(self):
        df = self.base_data.resample('15min').last().dropna()
        
        # فرمول آربیتراژ کلاسیک
        df['Spread'] = np.log(df['c_eur']) - np.log(df['c_gbp'])
        df['Mean'] = df['Spread'].rolling(150).mean()
        df['Std'] = df['Spread'].rolling(150).std()
        df['Z'] = (df['Spread'] - df['Mean']) / (df['Std'] + 1e-8)
        
        # استراتژی ورود و خروج بهینه (بدون فیلترهای اضافه)
        df['Pos'] = 0
        df.loc[df['Z'] < -2.5, 'Pos'] = 1  # Buy
        df.loc[df['Z'] > 2.5, 'Pos'] = -1  # Sell
        df.loc[abs(df['Z']) <= 0.2, 'Pos'] = 0 # خروج زودهنگام برای سود کوچک اما ایمن
        df['Pos'] = df['Pos'].replace(0, np.nan).ffill().fillna(0).shift(1)
        
        # محاسبات بازدهی با اهرم مدیریت‌شده
        r_eur = df['c_eur'].pct_change()
        r_gbp = df['c_gbp'].pct_change()
        strat_ret = (df['Pos'] * (r_eur - r_gbp) - self.tc) * self.leverage
        
        # شبیه‌ساز اکوییتی با سود مرکب
        df['Equity'] = self.initial_balance * (1 + strat_ret).cumprod()
        df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax() * 100
        
        # خروجی نهایی
        df.to_csv("Equity_Curve_Report.csv")
        print(f"💰 نتیجه نهایی: ${df['Equity'].iloc[-1]:,.2f} | 📉 دراودان: {df['Drawdown'].min():.2f}%")

if __name__ == "__main__":
    engine = PropFirmReadyEngine()
    engine.load_data()
    engine.run_simulation()
