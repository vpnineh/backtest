import pandas as pd
import numpy as np
import yfinance as yf
import logging
from statsmodels.tsa.stattools import adfuller

logging.basicConfig(level=logging.INFO, format='%(message)s')

class OnlineSimulator:
    def __init__(self):
        self.initial_balance = 5000.0
        self.tc = 0.0001 # 1 pip cost
        self.leverage = 4.0 # اهرم ۴ (معادل ریسک ۱٪ در استاپ ۲۵ پیپی)

    def load_data(self, timeframe):
        # دانلود دیتا از یاهو فایننس
        # 15m برای 60 روز (محدودیت یاهو)، 1h و 4h برای دیتای طولانی‌تر
        period = "2y" if timeframe in ["1h", "4h"] else "60d"
        eur = yf.download("EURUSD=X", period=period, interval=timeframe)
        gbp = yf.download("GBPUSD=X", period=period, interval=timeframe)
        
        df = pd.DataFrame({'EURUSD': eur['Close'].squeeze(), 'GBPUSD': gbp['Close'].squeeze()})
        return df.dropna()

    def run_simulation(self):
        timeframes = {'15m': '15min', '1h': '1hour', '4h': '4hour'}
        
        for tf_code, tf_name in timeframes.items():
            print(f"\n🚀 شروع تحلیل میدان نبرد روی تایم‌فریم: {tf_name}")
            df = self.load_data(tf_code)
            
            df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
            roll = df['Spread'].rolling(100)
            df['Z'] = (df['Spread'] - roll.mean()) / (roll.std() + 1e-8)
            
            # منطق معاملاتی با فیلتر ADF
            pos = np.zeros(len(df))
            current_pos = 0
            for i in range(100, len(df)):
                z = df['Z'].iloc[i]
                if current_pos == 0:
                    if abs(z) > 2.5:
                        subset = df['Spread'].iloc[i-100:i].values
                        if adfuller(subset)[1] < 0.05:
                            current_pos = 1 if z < -2.5 else -1
                elif current_pos == 1 and z >= -0.5: current_pos = 0
                elif current_pos == -1 and z <= 0.5: current_pos = 0
                pos[i] = current_pos
            
            df['Pos'] = pd.Series(pos, index=df.index).shift(1).fillna(0)
            
            # محاسبات سود مرکب
            r_eur = df['EURUSD'].pct_change()
            r_gbp = df['GBPUSD'].pct_change()
            returns = (df['Pos'] * (r_eur - r_gbp) - self.tc).fillna(0) * self.leverage
            
            equity = self.initial_balance * (1 + returns).cumprod()
            roi = (equity.iloc[-1] / self.initial_balance - 1) * 100
            
            print(f"📊 نتیجه تایم‌فریم {tf_name}: رشد {roi:+.2f}%")
            
        print("\n✅ تست آنلاین تمام تایم‌فریم‌ها با موفقیت انجام شد.")

if __name__ == "__main__":
    sim = OnlineSimulator()
    sim.run_simulation()
