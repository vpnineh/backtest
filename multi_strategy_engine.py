import pandas as pd
import numpy as np
import logging
import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MultiStrategyArena:
    def __init__(self):
        self.tc = 0.0001  # هزینه تراکنش (ECN) = 1 پیپ
        self.base_data = pd.DataFrame()

    def _load_histdata(self, file_list):
        dfs = []
        for f in file_list:
            if not f.endswith('.csv'): continue
            df = pd.read_csv(f, sep=';', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
            dfs.append(df[['timestamp', 'close']])
        return pd.concat(dfs, ignore_index=True).sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp') if dfs else pd.DataFrame()

    def load_data(self):
        all_files = glob.glob('data/*.csv')
        eur_files = [f for f in all_files if 'eurusd' in f.lower()]
        gbp_files = [f for f in all_files if 'gbpusd' in f.lower()]

        logging.info("در حال ساخت دیتابیس 5 ساله برای میدان نبرد...")
        df_eur = self._load_histdata(eur_files).rename(columns={'close': 'EURUSD'})
        df_gbp = self._load_histdata(gbp_files).rename(columns={'close': 'GBPUSD'})
        
        self.base_data = df_eur.join(df_gbp, how='inner').dropna()
        logging.info(f"✅ دیتای یکپارچه آماده شد. ({len(self.base_data):,} کندل)")

    def calculate_metrics(self, df, pos_col, name, tf):
        r_eur = (df['EURUSD'] - df['EURUSD'].shift(1)) / df['EURUSD'].shift(1)
        
        # اگر استراتژی StatArb است، بازدهی اسپرد محاسبه شود، در غیر این صورت فقط بازدهی EURUSD
        if name == 'Statistical_Arbitrage':
            r_gbp = (df['GBPUSD'] - df['GBPUSD'].shift(1)) / df['GBPUSD'].shift(1)
            strategy_returns = df[pos_col] * (r_eur - r_gbp)
        else:
            strategy_returns = df[pos_col] * r_eur
            
        pos_changes = df[pos_col].diff().abs()
        costs = pos_changes * self.tc
        
        net_returns = (strategy_returns - costs).dropna()
        trades = int(pos_changes.sum() / 2)
        
        if trades < 20: return None
        
        total_ret = net_returns.sum() * 100
        cum_ret = net_returns.cumsum() * 100
        max_dd = (cum_ret - cum_ret.cummax()).min()
        
        std = net_returns.std() * np.sqrt(252 * 1440)
        sharpe = (net_returns.mean() * 252 * 1440 / std) if std > 0 else 0
        
        return {
            'Strategy': name,
            'Timeframe': tf,
            'Total_Return_%': round(total_ret, 2),
            'Max_Drawdown_%': round(max_dd, 2),
            'Sharpe_Ratio': round(sharpe, 2),
            'Total_Trades': trades
        }

    def evaluate_strategies(self):
        timeframes = {'15min': 'M15', '5min': 'M5'}
        results = []

        for tf, tf_name in timeframes.items():
            logging.info(f"شبیه‌سازی استراتژی‌ها روی تایم‌فریم {tf_name}...")
            df = self.base_data.resample(tf).last().dropna()
            
            # ==========================================
            # 1. Statistical Arbitrage (Pairs Trading)
            # ==========================================
            df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
            roll = df['Spread'].rolling(150)
            df['Z'] = (df['Spread'] - roll.mean()) / (roll.std() + 1e-8)
            
            df['Pos_Arb'] = 0
            df.loc[df['Z'] < -2.5, 'Pos_Arb'] = 1
            df.loc[df['Z'] > 2.5, 'Pos_Arb'] = -1
            df.loc[abs(df['Z']) <= 0.5, 'Pos_Arb'] = 0
            df['Pos_Arb'] = df['Pos_Arb'].replace(0, np.nan).ffill().fillna(0).shift(1)
            
            res_arb = self.calculate_metrics(df, 'Pos_Arb', 'Statistical_Arbitrage', tf_name)
            if res_arb: results.append(res_arb)

            # ==========================================
            # 2. Bollinger Bands Mean Reversion
            # ==========================================
            roll_bb = df['EURUSD'].rolling(50)
            df['BB_Mid'] = roll_bb.mean()
            df['BB_Std'] = roll_bb.std()
            df['BB_Up'] = df['BB_Mid'] + (2.0 * df['BB_Std'])
            df['BB_Low'] = df['BB_Mid'] - (2.0 * df['BB_Std'])
            
            df['Pos_BB'] = 0
            df.loc[df['EURUSD'] < df['BB_Low'], 'Pos_BB'] = 1  # خرید در کف
            df.loc[df['EURUSD'] > df['BB_Up'], 'Pos_BB'] = -1  # فروش در سقف
            df.loc[abs(df['EURUSD'] - df['BB_Mid']) < 0.0005, 'Pos_BB'] = 0 # خروج در خط میانی
            df['Pos_BB'] = df['Pos_BB'].replace(0, np.nan).ffill().fillna(0).shift(1)
            
            res_bb = self.calculate_metrics(df, 'Pos_BB', 'Bollinger_MeanRev', tf_name)
            if res_bb: results.append(res_bb)

            # ==========================================
            # 3. EMA Golden/Death Cross (Trend Following)
            # ==========================================
            df['EMA_20'] = df['EURUSD'].ewm(span=20, adjust=False).mean()
            df['EMA_100'] = df['EURUSD'].ewm(span=100, adjust=False).mean()
            
            df['Pos_EMA'] = 0
            df.loc[df['EMA_20'] > df['EMA_100'], 'Pos_EMA'] = 1  # روند صعودی
            df.loc[df['EMA_20'] < df['EMA_100'], 'Pos_EMA'] = -1 # روند نزولی
            df['Pos_EMA'] = df['Pos_EMA'].shift(1)
            
            res_ema = self.calculate_metrics(df, 'Pos_EMA', 'EMA_Trend_Cross', tf_name)
            if res_ema: results.append(res_ema)

            # ==========================================
            # 4. RSI Extremes (Momentum Reversal)
            # ==========================================
            delta = df['EURUSD'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            df['Pos_RSI'] = 0
            df.loc[df['RSI'] < 25, 'Pos_RSI'] = 1   # اشباع فروش
            df.loc[df['RSI'] > 75, 'Pos_RSI'] = -1  # اشباع خرید
            df.loc[(df['RSI'] > 40) & (df['RSI'] < 60), 'Pos_RSI'] = 0 # خروج در حالت خنثی
            df['Pos_RSI'] = df['Pos_RSI'].replace(0, np.nan).ffill().fillna(0).shift(1)
            
            res_rsi = self.calculate_metrics(df, 'Pos_RSI', 'RSI_Extremes', tf_name)
            if res_rsi: results.append(res_rsi)

            # ==========================================
            # 5. Donchian Volatility Breakout
            # ==========================================
            df['High_20'] = df['EURUSD'].rolling(40).max()
            df['Low_20'] = df['EURUSD'].rolling(40).min()
            
            df['Pos_Breakout'] = 0
            df.loc[df['EURUSD'] >= df['High_20'].shift(1), 'Pos_Breakout'] = 1   # شکست سقف
            df.loc[df['EURUSD'] <= df['Low_20'].shift(1), 'Pos_Breakout'] = -1  # شکست کف
            df['Pos_Breakout'] = df['Pos_Breakout'].replace(0, np.nan).ffill().fillna(0).shift(1)
            
            res_brk = self.calculate_metrics(df, 'Pos_Breakout', 'Volatility_Breakout', tf_name)
            if res_brk: results.append(res_brk)

        return pd.DataFrame(results)

if __name__ == "__main__":
    arena = MultiStrategyArena()
    arena.load_data()
    report = arena.evaluate_strategies()
    
    # مرتب‌سازی بر اساس بیشترین سود برای تعیین پادشاه استراتژی‌ها
    report = report.sort_values(by='Total_Return_%', ascending=False)
    
    output_file = "Ultimate_5_Strategies_Arena.csv"
    report.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("🏆 نتایج نبرد 5 استراتژی در میدان 5 ساله (رتبه‌بندی بر اساس سود):")
    print(f"{'='*60}")
    print(report.to_string(index=False))
