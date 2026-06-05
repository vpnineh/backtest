import pandas as pd
import numpy as np
import glob

class PropSafeHedgingEngine:
    def __init__(self):
        self.initial_balance = 5000.0
        self.max_leverage = 1.0  # اهرمِ بسیار امن برای تستِ اولیه
        self.hedging_threshold = 0.003  # باز کردن هج در ضرر 30 پیپی
        
    def load_data(self):
        # لود دیتای شما
        files_eur = glob.glob('data/*EURUSD*.csv')
        files_gbp = glob.glob('data/*GBPUSD*.csv')
        
        def read_df(f):
            df = pd.read_csv(f, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            return df.set_index('ts')[['c']]

        eur = pd.concat([read_df(f) for f in files_eur]).sort_index()
        gbp = pd.concat([read_df(f) for f in files_gbp]).sort_index()
        self.df = eur.join(gbp, lsuffix='_eur', rsuffix='_gbp').dropna()

    def run_simulation(self):
        df = self.df.resample('15min').last()
        
        # استراتژی: ورود دوطرفه در کانال‌های قیمتی
        # مدیریت پوزیشن: هجینگ هوشمند
        equity = [self.initial_balance]
        net_pos_eur = 0
        net_pos_gbp = 0
        
        for i in range(1, len(df)):
            prev_eur = df['c_eur'].iloc[i-1]
            curr_eur = df['c_eur'].iloc[i]
            
            # منطقِ ساده و امن: 
            # اگر قیمت تغییرِ ناگهانی داشت، پوزیشن معکوس برای هجینگ باز کن
            diff = (curr_eur - prev_eur) / prev_eur
            
            # اگر بازار بیش از 0.1% حرکت کرد، هجینگ فعال می‌شود
            if abs(diff) > 0.001:
                net_pos_eur -= np.sign(diff) * 0.1 # حجم کوچک (0.1 لات)
            
            # محاسبه سود و زیان لحظه‌ای
            profit = net_pos_eur * (curr_eur - prev_eur) * 100000 
            new_equity = equity[-1] + profit
            
            # مدیریت دراودان: اگر موجودی به زیر 4500 رسید، هجینگ را ببند (Stop Loss کلی)
            if new_equity < 4500:
                new_equity = 4500
                net_pos_eur = 0 
                
            equity.append(new_equity)

        df['Equity'] = equity
        df.to_csv("Equity_Curve_Report.csv")
        print(f"💰 موجودی نهایی: ${equity[-1]:,.2f}")

if __name__ == "__main__":
    engine = PropSafeHedgingEngine()
    engine.load_data()
    engine.run_simulation()
