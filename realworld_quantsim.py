import pandas as pd
import numpy as np
import logging
import glob
import warnings
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')

class RealWorldSimulator:
    def __init__(self):
        self.initial_balance = 5000.0  
        self.risk_per_trade = 0.01     
        self.sl_pips_estimate = 0.0025 
        self.tc = 0.0001               
        
        self.leverage = self.risk_per_trade / self.sl_pips_estimate
        self.base_data = pd.DataFrame()

    def load_data(self):
        print(f"\n{'='*60}")
        print(f"🏦 شبیه‌ساز کوانت پیشرفته (مجهز به فیلتر ADF)")
        print(f"💰 سرمایه اولیه: ${self.initial_balance:,}")
        print(f"⚠️ ریسک در معامله: {self.risk_per_trade*100}% | لوریج: {self.leverage:.2f}x")
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
        df = self.base_data.resample('15min').last().dropna()
        
        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        roll = df['Spread'].rolling(150)
        df['Z'] = (df['Spread'] - roll.mean()) / (roll.std() + 1e-8)
        
        # استخراج آرایه‌ها برای حلقه پردازشی سریع
        z_vals = df['Z'].values
        spread_vals = df['Spread'].values
        pos = np.zeros(len(df))
        current_pos = 0
        
        print("⏳ در حال اجرای آزمون هم‌جمعی (ADF) روی سیگنال‌ها. لطفاً منتظر بمانید...")
        
        # حلقه هوشمند با فیلتر ADF و Stop Loss
        for i in range(150, len(df)):
            z = z_vals[i]
            
            if current_pos == 0:
                if z < -2.5 or z > 2.5:
                    # اجرای تست دیکی-فولر فقط در لحظه برخورد به حد ورود
                    subset = spread_vals[i-150:i]
                    if np.var(subset) > 1e-10: 
                        try:
                            # اگر p-value کمتر از 0.05 باشد یعنی جفت‌ارزها همگرا هستند
                            pval = adfuller(subset, maxlag=1)[1]
                            if pval < 0.05:
                                current_pos = 1 if z < -2.5 else -1
                        except:
                            pass
                            
            elif current_pos == 1:
                # خروج با سود (Z >= -0.5) یا خروج اضطراری با ضرر (Z <= -4.0)
                if z >= -0.5 or z <= -4.0:
                    current_pos = 0
                    
            elif current_pos == -1:
                # خروج با سود (Z <= 0.5) یا خروج اضطراری با ضرر (Z >= 4.0)
                if z <= 0.5 or z >= 4.0:
                    current_pos = 0
                    
            pos[i] = current_pos
            
        # اعمال پوزیشن‌ها با یک کندل تاخیر (برای شبیه‌سازی ورود واقعی)
        df['Pos'] = pos
        df['Pos'] = df['Pos'].shift(1).fillna(0)
        
        # محاسبه بازدهی درصدی
        r_eur = (df['EURUSD'] - df['EURUSD'].shift(1)) / df['EURUSD'].shift(1)
        r_gbp = (df['GBPUSD'] - df['GBPUSD'].shift(1)) / df['GBPUSD'].shift(1)
        
        pos_changes = df['Pos'].diff().abs()
        costs = pos_changes * self.tc
        
        raw_strat_return = (df['Pos'] * (r_eur - r_gbp) - costs).fillna(0)
        leveraged_return = raw_strat_return * self.leverage
        
        df['Equity'] = self.initial_balance * (1 + leveraged_return).cumprod()
        df['Drawdown'] = (df['Equity'] - df['Equity'].cummax()) / df['Equity'].cummax() * 100
        
        final_balance = df['Equity'].iloc[-1]
        max_dd = df['Drawdown'].min()
        total_trades = int(pos_changes.sum() / 2)
        net_profit = final_balance - self.initial_balance
        roi_percent = (net_profit / self.initial_balance) * 100
        
        df['Year'] = df.index.year
        yearly_perf = df.groupby('Year').apply(lambda x: (x['Equity'].iloc[-1] / x['Equity'].iloc[0] - 1) * 100)
        
        print(f"\n📊 گزارش نهایی شبیه‌ساز (ADF Filtered):")
        print(f"----------------------------------------")
        print(f"تعداد کل معاملات : {total_trades} معامله (معاملات سمی فیلتر شدند)")
        print(f"موجودی نهایی حساب: ${final_balance:,.2f}")
        print(f"سود خالص دلاری   : ${net_profit:,.2f} (+{roi_percent:.1f}%)")
        print(f"بیشترین افت حساب : {max_dd:.2f}% (کنترلِ ریسکِ عالی)")
        print(f"----------------------------------------")
        print("📅 عملکرد تفکیک‌شده هر سال:")
        for year, ret in yearly_perf.items():
            print(f" سال {year} : {ret:+.2f}%")
        print(f"{'='*60}\n")
        
        export_df = df[['EURUSD', 'GBPUSD', 'Pos', 'Equity', 'Drawdown']].copy()
        export_df.to_csv("Equity_Curve_Report.csv")

if __name__ == "__main__":
    sim = RealWorldSimulator()
    sim.load_data()
    sim.run_simulation()
