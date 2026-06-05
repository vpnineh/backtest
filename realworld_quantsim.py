import pandas as pd
import numpy as np
import logging
import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')

class RealWorldSimulator:
    def __init__(self):
        self.initial_balance = 5000.0  # سرمایه اولیه 5000 دلار
        self.risk_per_trade = 0.01     # 1 درصد ریسک در هر معامله
        self.sl_pips_estimate = 0.0025 # تخمین 25 پیپ حد ضرر برای محاسبه حجم
        self.tc = 0.0001               # 1 پیپ کمیسیون و اسپرد در حساب ECN
        
        # محاسبه ضریب لوریج (اهرم) بر اساس ریسک
        self.leverage = self.risk_per_trade / self.sl_pips_estimate
        
        self.base_data = pd.DataFrame()

    def load_data(self):
        print(f"\n{'='*60}")
        print(f"🏦 شبیه‌ساز حساب واقعی (صندوق سرمایه‌گذاری)")
        print(f"💰 سرمایه اولیه: ${self.initial_balance:,}")
        print(f"⚠️ ریسک در هر معامله: {self.risk_per_trade*100}%")
        print(f"⚙️ لوریج اتوماتیک: {self.leverage:.2f}x")
        print(f"{'='*60}\n")
        
        all_files = glob.glob('data/*.csv')
        eur_files = [f for f in all_files if 'eurusd' in f.lower()]
        gbp_files = [f for f in all_files if 'gbpusd' in f.lower()]

        dfs_eur, dfs_gbp = [], []
        for f in eur_files:
            df = pd.read_csv(f, sep=';', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
            dfs_eur.append(df[['timestamp', 'close']])
            
        for f in gbp_files:
            df = pd.read_csv(f, sep=';', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
            dfs_gbp.append(df[['timestamp', 'close']])

        df_eur = pd.concat(dfs_eur).sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp').rename(columns={'close': 'EURUSD'})
        df_gbp = pd.concat(dfs_gbp).sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp').rename(columns={'close': 'GBPUSD'})
        
        self.base_data = df_eur.join(df_gbp, how='inner').dropna()

    def run_simulation(self):
        # استفاده از تنظیمات طلایی به دست آمده از تست قبلی (M15, Window=150, Z=2.5)
        df = self.base_data.resample('15min').last().dropna()
        
        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        roll = df['Spread'].rolling(150)
        df['Z'] = (df['Spread'] - roll.mean()) / (roll.std() + 1e-8)
        
        # تولید سیگنال
        df['Pos'] = 0
        df.loc[df['Z'] < -2.5, 'Pos'] = 1
        df.loc[df['Z'] > 2.5, 'Pos'] = -1
        df.loc[abs(df['Z']) <= 0.5, 'Pos'] = 0
        df['Pos'] = df['Pos'].replace(0, np.nan).ffill().fillna(0).shift(1)
        
        # محاسبه بازدهی درصدی پایه
        r_eur = (df['EURUSD'] - df['EURUSD'].shift(1)) / df['EURUSD'].shift(1)
        r_gbp = (df['GBPUSD'] - df['GBPUSD'].shift(1)) / df['GBPUSD'].shift(1)
        
        pos_changes = df['Pos'].diff().abs()
        costs = pos_changes * self.tc
        
        # سود خالص خام (بدون لوریج)
        raw_strat_return = (df['Pos'] * (r_eur - r_gbp) - costs).fillna(0)
        
        # === اعمال سود مرکب و لوریج واقعی ===
        # بازدهی هر کندل ضربدر ضریب اهرم می‌شود
        leveraged_return = raw_strat_return * self.leverage
        
        # محاسبه رشد حساب با فرمول بهره مرکب (Compound Interest)
        df['Equity'] = self.initial_balance * (1 + leveraged_return).cumprod()
        df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax() * 100
        
        # محاسبه آمار نهایی
        final_balance = df['Equity'].iloc[-1]
        max_dd = df['Drawdown'].min()
        total_trades = int(pos_changes.sum() / 2)
        net_profit = final_balance - self.initial_balance
        roi_percent = (net_profit / self.initial_balance) * 100
        
        # خلاصه سالانه
        df['Year'] = df.index.year
        yearly_perf = df.groupby('Year').apply(lambda x: (x['Equity'].iloc[-1] / x['Equity'].iloc[0] - 1) * 100)
        
        print(f"📊 گزارش نهایی شبیه‌ساز (5 ساله):")
        print(f"----------------------------------------")
        print(f"تعداد کل معاملات : {total_trades} معامله")
        print(f"موجودی نهایی حساب: ${final_balance:,.2f}")
        print(f"سود خالص دلاری   : ${net_profit:,.2f} (+{roi_percent:.1f}%)")
        print(f"بیشترین افت حساب : {max_dd:.2f}% (Max Drawdown)")
        print(f"----------------------------------------")
        print("📅 عملکرد تفکیک‌شده هر سال (سود مرکب):")
        for year, ret in yearly_perf.items():
            print(f" سال {year} : {ret:+.2f}%")
        print(f"{'='*60}\n")
        
        # ذخیره فایل گزارش برای نمودار
        export_df = df[['EURUSD', 'GBPUSD', 'Pos', 'Equity', 'Drawdown']].copy()
        export_df.to_csv("Equity_Curve_Report.csv")

if __name__ == "__main__":
    sim = RealWorldSimulator()
    sim.load_data()
    sim.run_simulation()
