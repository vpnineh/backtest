import pandas as pd
import numpy as np
import itertools
import logging
import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class HistDataMultiTimeframeOptimizer:
    def __init__(self):
        self.transaction_cost = 0.0002 # هزینه تراکنش (حدود 2 پیپ)
        self.base_data = pd.DataFrame()

    def _load_histdata_files(self, file_list):
        dfs = []
        for f in file_list:
            if not f.endswith('.csv'): continue
            logging.info(f"در حال خواندن فایل: {f}")
            
            # خواندن فرمت مخصوص HistData
            df = pd.read_csv(
                f, 
                sep=';', 
                header=None, 
                names=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            # تبدیل تاریخ به فرمت استاندارد پایتون
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d %H%M%S')
            dfs.append(df[['timestamp', 'close']])
            
        if not dfs:
            return pd.DataFrame()
            
        # چسباندن همه سال‌ها به هم و مرتب‌سازی بر اساس تاریخ
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').drop_duplicates('timestamp')
        return combined.set_index('timestamp')

    def load_base_m1_data(self):
        all_files = glob.glob('data/*.csv')
        if not all_files:
            raise FileNotFoundError("❌ هیچ فایل CSV در پوشه data یافت نشد!")

        # پیدا کردن اتوماتیک فایل‌های هر جفت ارز (مهم نیست چندتا باشند)
        eurusd_files = [f for f in all_files if 'eurusd' in f.lower()]
        gbpusd_files = [f for f in all_files if 'gbpusd' in f.lower()]

        if not eurusd_files or not gbpusd_files:
            raise ValueError(f"❌ فایل‌های هر دو جفت ارز کامل نیست. فایل‌های موجود: {all_files}")

        logging.info("در حال پردازش و تجمیع فایل‌های HistData...")
        
        df_eur = self._load_histdata_files(eurusd_files).rename(columns={'close': 'EURUSD'})
        df_gbp = self._load_histdata_files(gbpusd_files).rename(columns={'close': 'GBPUSD'})

        self.base_data = df_eur.join(df_gbp, how='inner').dropna()
        logging.info(f"✅ دیتای یکپارچه ساخته شد. تعداد کل کندل‌های مشترک: {len(self.base_data):,}")

    def resample_to_timeframe(self, tf_code: str) -> pd.DataFrame:
        if tf_code == '1min':
            return self.base_data.copy()
        return self.base_data.resample(tf_code).last().dropna()

    def backtest_single(self, data: pd.DataFrame, window: int, z_entry: float, z_exit: float):
        df = data.copy()
        if len(df) < window + 100:
            return None

        # اسپرد و Z-Score
        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        roll = df['Spread'].rolling(window=window)
        df['Mean'] = roll.mean()
        df['Std'] = roll.std()
        
        # جلوگیری از خطای تقسیم بر صفر در زمان‌های راکد بازار
        df.loc[df['Std'] == 0, 'Std'] = 1e-8 
        df['Z'] = (df['Spread'] - df['Mean']) / df['Std']
        df.dropna(inplace=True)

        # سیگنال‌دهی
        z = df['Z'].values
        signal = np.zeros(len(df))
        signal[z < -z_entry] =  1
        signal[z >  z_entry] = -1
        signal[np.abs(z) <= z_exit] = 0

        pos = pd.Series(signal, index=df.index).replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)

        # محاسبه سود خطی (درصدی)
        r_eur = (df['EURUSD'] - df['EURUSD'].shift(1)) / df['EURUSD'].shift(1)
        r_gbp = (df['GBPUSD'] - df['GBPUSD'].shift(1)) / df['GBPUSD'].shift(1)

        pos_change = pos.diff().abs()
        cost = pos_change * self.transaction_cost
        
        strat_ret = (pos * (r_eur - r_gbp) - cost).dropna()

        if strat_ret.empty:
            return None

        total_trades = int(pos_change.sum() / 2)
        if total_trades < 10:
            return None

        # محاسبات نهایی واقعی
        total_ret = strat_ret.sum() * 100
        cum_ret = strat_ret.cumsum() * 100
        rolling_max = cum_ret.cummax()
        drawdown = cum_ret - rolling_max
        max_dd = drawdown.min()

        std = strat_ret.std() * np.sqrt(252 * 1440)
        mean_ret = strat_ret.mean() * 252 * 1440
        sharpe = (mean_ret / std) if std > 0 else 0

        return {
            'Window': window,
            'Z_Entry': z_entry,
            'Z_Exit': z_exit,
            'Total_Return_%': round(total_ret, 2),
            'Max_Drawdown_%': round(max_dd, 2),
            'Sharpe': round(sharpe, 2),
            'Total_Trades': total_trades,
        }

    def run_optimization(self):
        timeframes = {'15min': '15min', '5min': '5min', '2min': '2min', '1min': '1min'}
        windows = [50, 100, 150]
        z_entries = [2.0, 2.5, 3.0]
        z_exits = [0.0, 0.5]
        combos = list(itertools.product(windows, z_entries, z_exits))

        all_results = []
        for tf_code, tf_name in timeframes.items():
            tf_data = self.resample_to_timeframe(tf_code)
            logging.info(f"شروع بک‌تست تایم‌فریم: {tf_name} ({len(tf_data):,} کندل)")

            tf_results = []
            for w, ze, zx in combos:
                res = self.backtest_single(tf_data, w, ze, zx)
                if res:
                    res['Timeframe'] = tf_name
                    tf_results.append(res)

            if tf_results:
                df_tf = pd.DataFrame(tf_results).sort_values('Total_Return_%', ascending=False)
                all_results.append(df_tf.head(3))

        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

if __name__ == "__main__":
    optimizer = HistDataMultiTimeframeOptimizer()
    optimizer.load_base_m1_data()
    report = optimizer.run_optimization()
    
    output_file = "ultimate_histdata_mtf_report.csv"
    report.to_csv(output_file, index=False)
    print(f"\n✅ بک‌تست کامل شد. فایل نهایی: {output_file}")
    print(report.to_string(index=False))
