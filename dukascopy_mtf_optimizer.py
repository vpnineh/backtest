import pandas as pd
import numpy as np
import itertools
import logging
import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class DukascopyMultiTimeframeOptimizer:
    def __init__(self):
        self.transaction_cost = 0.0002
        self.base_data = pd.DataFrame()

    def load_base_m1_data(self):
        logging.info("بارگذاری دیتا...")
        try:
            all_files = glob.glob('data/*.csv')
            logging.info(f"فایل‌های یافت شده: {all_files}")

            eurusd_file = None
            gbpusd_file = None
            for f in all_files:
                fl = f.lower()
                if 'eurusd' in fl:
                    eurusd_file = f
                elif 'gbpusd' in fl:
                    gbpusd_file = f

            if not eurusd_file or not gbpusd_file:
                raise FileNotFoundError(f"فایل‌ها یافت نشدند: {all_files}")

            logging.info(f"EURUSD: {eurusd_file}")
            logging.info(f"GBPUSD: {gbpusd_file}")

            df_eur = pd.read_csv(eurusd_file)
            df_gbp = pd.read_csv(gbpusd_file)

            # ======= تبدیل timestamp =======
            df_eur['timestamp'] = pd.to_datetime(df_eur['timestamp'])
            df_gbp['timestamp'] = pd.to_datetime(df_gbp['timestamp'])

            # ======= حذف duplicate ها =======
            before = len(df_eur)
            df_eur = df_eur.drop_duplicates(subset='timestamp', keep='last')
            df_gbp = df_gbp.drop_duplicates(subset='timestamp', keep='last')
            after = len(df_eur)
            logging.info(f"حذف duplicate: {before:,} → {after:,} کندل")

            # ======= حذف کندل‌های بی‌معنی (volume=0 و OHLC یکسان) =======
            mask_eur = ~(
                (df_eur['volume'] == 0) &
                (df_eur['open'] == df_eur['close']) &
                (df_eur['high'] == df_eur['low'])
            )
            mask_gbp = ~(
                (df_gbp['volume'] == 0) &
                (df_gbp['open'] == df_gbp['close']) &
                (df_gbp['high'] == df_gbp['low'])
            )
            df_eur = df_eur[mask_eur]
            df_gbp = df_gbp[mask_gbp]
            logging.info(f"بعد از حذف کندل‌های بی‌معنی - EURUSD: {len(df_eur):,} | GBPUSD: {len(df_gbp):,}")

            df_eur.set_index('timestamp', inplace=True)
            df_gbp.set_index('timestamp', inplace=True)

            df_eur = df_eur[['close']].rename(columns={'close': 'EURUSD'})
            df_gbp = df_gbp[['close']].rename(columns={'close': 'GBPUSD'})

            df_eur = df_eur.sort_index()
            df_gbp = df_gbp.sort_index()

            self.base_data = df_eur.join(df_gbp, how='inner').dropna()

            # ======= نمونه داده برای بررسی =======
            logging.info(f"بازه زمانی: {self.base_data.index[0]} تا {self.base_data.index[-1]}")
            logging.info(f"✅ تعداد کندل نهایی M1: {len(self.base_data):,}")
            logging.info(f"نمونه:\n{self.base_data.head(5)}")

            # ======= چک منطقی =======
            expected_min = 200_000
            expected_max = 800_000
            if len(self.base_data) < expected_min:
                logging.warning(f"⚠️ تعداد کندل خیلی کم است: {len(self.base_data):,}")
            elif len(self.base_data) > expected_max:
                logging.warning(f"⚠️ تعداد کندل خیلی زیاد است: {len(self.base_data):,} - احتمال duplicate")

        except Exception as e:
            logging.error(f"خطا: {e}")
            raise

    def resample_data(self, timeframe: str) -> pd.DataFrame:
        """تبدیل M1 به تایم‌فریم بالاتر"""
        if timeframe == '1min':
            return self.base_data.copy()

        logging.info(f"ساخت چارت {timeframe}...")

        resampled = self.base_data.resample(timeframe).agg({
            'EURUSD': 'last',
            'GBPUSD': 'last',
        }).dropna()

        logging.info(f"تعداد کندل {timeframe}: {len(resampled):,}")
        return resampled

    def backtest(self, data: pd.DataFrame, window: int, z_entry: float, z_exit: float) -> dict | None:
        df = data.copy()

        if len(df) < window + 10:
            return None

        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        df['Spread_Mean'] = df['Spread'].rolling(window=window).mean()
        df['Spread_Std'] = df['Spread'].rolling(window=window).std()

        # جلوگیری از تقسیم بر صفر
        df = df[df['Spread_Std'] > 1e-10]
        df['Z_Score'] = (df['Spread'] - df['Spread_Mean']) / df['Spread_Std']

        df['Signal'] = 0
        df.loc[df['Z_Score'] < -z_entry, 'Signal'] = 1
        df.loc[df['Z_Score'] > z_entry, 'Signal'] = -1
        df.loc[abs(df['Z_Score']) <= z_exit, 'Signal'] = 0

        # جایگزین روش deprecated
        df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).shift(1)
        df.dropna(inplace=True)

        if df.empty or len(df) < 10:
            return None

        ret1 = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
        ret2 = np.log(df['GBPUSD'] / df['GBPUSD'].shift(1))

        position_changes = df['Position'].diff().abs()
        costs = position_changes * self.transaction_cost

        strategy_returns = df['Position'] * (ret1 - ret2) - costs
        strategy_returns = strategy_returns.dropna()

        if strategy_returns.empty:
            return None

        # ======= چک overflow =======
        cumsum = strategy_returns.cumsum()
        if cumsum.abs().max() > 500:
            logging.warning(f"⚠️ overflow در window={window}, z_entry={z_entry}")
            return None

        cumulative_return = np.exp(cumsum.iloc[-1]) - 1
        total_trades = int(position_changes.sum() / 2)

        cum_ret_series = np.exp(cumsum)
        rolling_max = cum_ret_series.cummax()
        drawdown = (cum_ret_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Sharpe Ratio
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252 * 1440)
        else:
            sharpe = 0.0

        return {
            'Window': window,
            'Z_Entry': z_entry,
            'Z_Exit': z_exit,
            'Total_Return_%': round(cumulative_return * 100, 4),
            'Max_Drawdown_%': round(max_drawdown * 100, 4),
            'Total_Trades': total_trades,
            'Sharpe': round(sharpe, 3),
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
            logging.info(f"Grid Search روی {tf} ({len(tf_data):,} کندل) - {len(combinations)} ترکیب...")

            results = []
            for window, z_entry, z_exit in combinations:
                res = self.backtest(tf_data, window, z_entry, z_exit)
                if res and res['Total_Trades'] > 20:
                    res['Timeframe'] = tf
                    results.append(res)

            if results:
                df_res = pd.DataFrame(results)
                df_res = df_res.sort_values(by='Total_Return_%', ascending=False)
                logging.info(f"✅ {tf}: {len(df_res)} نتیجه معتبر یافت شد")
                all_results = pd.concat([all_results, df_res.head(5)], ignore_index=True)
            else:
                logging.warning(f"⚠️ {tf}: هیچ نتیجه معتبری یافت نشد")

        return all_results


if __name__ == "__main__":
    optimizer = DukascopyMultiTimeframeOptimizer()
    optimizer.load_base_m1_data()

    final_report = optimizer.run_all_timeframes()

    if final_report.empty:
        logging.error("❌ هیچ نتیجه‌ای تولید نشد!")
    else:
        cols = ['Timeframe', 'Window', 'Z_Entry', 'Z_Exit',
                'Total_Return_%', 'Max_Drawdown_%', 'Sharpe', 'Total_Trades']
        final_report = final_report[cols]
        final_report = final_report.sort_values('Total_Return_%', ascending=False)

        output_file = "ultimate_5years_mtf_report.csv"
        final_report.to_csv(output_file, index=False)

        print(f"\n{'='*60}")
        print(f"✅ بک‌تست پایان یافت | نتایج: {output_file}")
        print(f"{'='*60}")
        print(final_report.to_string(index=False))
