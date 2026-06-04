import pandas as pd
import numpy as np
import itertools
import logging
import glob
import os
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class DukascopyMultiTimeframeOptimizer:
    def __init__(self):
        self.transaction_cost = 0.0002
        self.base_data = pd.DataFrame()

    def load_base_m1_data(self):
        logging.info("در حال بارگذاری دیتای خام 1 دقیقه‌ای از فایل‌های دانلود شده...")
        try:
            all_files = glob.glob('data/*.csv')
            logging.info(f"فایل‌های یافت شده: {all_files}")

            if not all_files:
                raise FileNotFoundError("هیچ فایل CSV در پوشه data یافت نشد!")

            eurusd_file = None
            gbpusd_file = None

            for f in all_files:
                f_lower = f.lower()
                if 'eurusd' in f_lower:
                    eurusd_file = f
                elif 'gbpusd' in f_lower:
                    gbpusd_file = f

            if not eurusd_file:
                raise FileNotFoundError(f"فایل EURUSD یافت نشد. فایل‌های موجود: {all_files}")
            if not gbpusd_file:
                raise FileNotFoundError(f"فایل GBPUSD یافت نشد. فایل‌های موجود: {all_files}")

            logging.info(f"EURUSD: {eurusd_file}")
            logging.info(f"GBPUSD: {gbpusd_file}")

            df_eur = pd.read_csv(eurusd_file)
            df_gbp = pd.read_csv(gbpusd_file)

            logging.info(f"ستون‌های EURUSD: {df_eur.columns.tolist()}")
            logging.info(f"نمونه داده:\n{df_eur.head(3)}")

            df_eur['timestamp'] = pd.to_datetime(df_eur['timestamp'])
            df_gbp['timestamp'] = pd.to_datetime(df_gbp['timestamp'])

            df_eur.set_index('timestamp', inplace=True)
            df_gbp.set_index('timestamp', inplace=True)

            df_eur = df_eur[['close']].rename(columns={'close': 'EURUSD'})
            df_gbp = df_gbp[['close']].rename(columns={'close': 'GBPUSD'})

            self.base_data = df_eur.join(df_gbp, how='inner').dropna()
            logging.info(f"✅ دیتا لود شد | تعداد کندل: {len(self.base_data):,}")

        except Exception as e:
            logging.error(f"خطا در پردازش دیتای پایه: {e}")
            raise

    def resample_data(self, timeframe: str) -> pd.DataFrame:
        if timeframe == '1min':
            return self.base_data.copy()
        logging.info(f"ساخت چارت {timeframe} از روی دیتای 1 دقیقه...")
        return self.base_data.resample(timeframe).last().dropna()

    def backtest(self, data: pd.DataFrame, window: int, z_entry: float, z_exit: float) -> dict:
        df = data.copy()

        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        df['Spread_Mean'] = df['Spread'].rolling(window=window).mean()
        df['Spread_Std'] = df['Spread'].rolling(window=window).std()
        df['Z_Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']

        df['Signal'] = 0
        df.loc[df['Z_Score'] < -z_entry, 'Signal'] = 1
        df.loc[df['Z_Score'] > z_entry, 'Signal'] = -1
        df.loc[abs(df['Z_Score']) <= z_exit, 'Signal'] = 0

        df['Position'] = df['Signal'].replace(to_replace=0, method='ffill').shift(1)
        df.dropna(inplace=True)

        if df.empty:
            return None

        ret1 = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
        ret2 = np.log(df['GBPUSD'] / df['GBPUSD'].shift(1))

        position_changes = df['Position'].diff().abs()
        costs = position_changes * self.transaction_cost

        df['Strategy_Return'] = (df['Position'] * (ret1 - ret2)) - costs

        total_trades = position_changes.sum() / 2
        cumulative_return = np.exp(df['Strategy_Return'].cumsum().iloc[-1]) - 1

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

    def run_all_timeframes(self):
        timeframes = ['15min', '5min', '2min', '1min']
        windows = [50, 100, 150, 200]
        z_entries = [2.0, 2.5, 3.0]
        z_exits = [0.0, 0.5]
        combinations = list(itertools.product(windows, z_entries, z_exits))

        all_results = pd.DataFrame()

        for tf in timeframes:
            tf_data = self.resample_data(tf)
            logging.info(f"شروع Grid Search روی تایم‌فریم {tf} ({len(tf_data)} کندل)...")

            results = []
            for window, z_entry, z_exit in combinations:
                res = self.backtest(tf_data, window, z_entry, z_exit)
                if res and res['Total_Trades'] > 20:
                    res['Timeframe'] = tf
                    results.append(res)

            if results:
                df_res = pd.DataFrame(results)
                df_res = df_res.sort_values(by='Total_Return_%', ascending=False)
                all_results = pd.concat([all_results, df_res.head(5)], ignore_index=True)

        return all_results


if __name__ == "__main__":
    optimizer = DukascopyMultiTimeframeOptimizer()
    optimizer.load_base_m1_data()

    final_report = optimizer.run_all_timeframes()

    final_report = final_report[[
        'Timeframe', 'Window', 'Z_Entry', 'Z_Exit',
        'Total_Return_%', 'Max_Drawdown_%', 'Total_Trades'
    ]]

    output_file = "ultimate_5years_mtf_report.csv"
    final_report.to_csv(output_file, index=False)

    print(f"\n{'='*50}")
    print(f"✅ بک‌تست پایان یافت")
    print(f"📄 نتایج ذخیره شد: {output_file}")
    print(f"{'='*50}")
    print(final_report.to_string(index=False))
