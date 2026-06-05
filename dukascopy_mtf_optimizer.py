import pandas as pd
import numpy as np
import itertools
import logging
import glob
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


class DukascopyMultiTimeframeOptimizer:
    def __init__(self):
        self.transaction_cost = 0.0002  # 2 پیپ
        self.base_data = pd.DataFrame()

    # ──────────────────────────────────────────
    def load_base_m1_data(self):
        logging.info("بارگذاری دیتای M1...")

        all_files = glob.glob('data/*.csv')
        logging.info(f"فایل‌های یافت شده: {all_files}")

        if not all_files:
            raise FileNotFoundError("هیچ فایل CSV در پوشه data یافت نشد!")

        eurusd_file = None
        gbpusd_file = None
        for f in all_files:
            fl = f.lower()
            if 'eurusd' in fl:
                eurusd_file = f
            elif 'gbpusd' in fl:
                gbpusd_file = f

        if not eurusd_file:
            raise FileNotFoundError(f"فایل EURUSD یافت نشد | موجود: {all_files}")
        if not gbpusd_file:
            raise FileNotFoundError(f"فایل GBPUSD یافت نشد | موجود: {all_files}")

        logging.info(f"EURUSD ← {eurusd_file}")
        logging.info(f"GBPUSD ← {gbpusd_file}")

        df_eur = pd.read_csv(eurusd_file)
        df_gbp = pd.read_csv(gbpusd_file)

        logging.info(f"ستون‌ها: {df_eur.columns.tolist()}")
        logging.info(f"نمونه خام EURUSD:\n{df_eur.head(5).to_string()}")
        logging.info(f"تعداد خام: EURUSD={len(df_eur):,} | GBPUSD={len(df_gbp):,}")

        # پارس timestamp
        df_eur['timestamp'] = pd.to_datetime(df_eur['timestamp'])
        df_gbp['timestamp'] = pd.to_datetime(df_gbp['timestamp'])

        # مرتب‌سازی
        df_eur = df_eur.sort_values('timestamp')
        df_gbp = df_gbp.sort_values('timestamp')

        # حذف duplicate
        before_eur = len(df_eur)
        df_eur = df_eur.drop_duplicates(subset='timestamp', keep='last')
        df_gbp = df_gbp.drop_duplicates(subset='timestamp', keep='last')
        logging.info(f"حذف duplicate EURUSD: {before_eur:,} → {len(df_eur):,}")

        # حذف کندل‌های بی‌معنی (volume=0 و قیمت ثابت)
        def remove_bad_candles(df, name):
            bad = (df['volume'] == 0) & (df['open'] == df['close']) & (df['high'] == df['low'])
            cleaned = df[~bad]
            logging.info(f"حذف کندل بد {name}: {bad.sum():,} عدد | باقی: {len(cleaned):,}")
            return cleaned

        df_eur = remove_bad_candles(df_eur, 'EURUSD')
        df_gbp = remove_bad_candles(df_gbp, 'GBPUSD')

        # ست کردن index
        df_eur = df_eur.set_index('timestamp')[['close']].rename(columns={'close': 'EURUSD'})
        df_gbp = df_gbp.set_index('timestamp')[['close']].rename(columns={'close': 'GBPUSD'})

        # join
        self.base_data = df_eur.join(df_gbp, how='inner').dropna()

        # اطلاعات نهایی
        n = len(self.base_data)
        t_start = self.base_data.index[0]
        t_end = self.base_data.index[-1]
        days = (t_end - t_start).days

        logging.info(f"{'='*50}")
        logging.info(f"✅ دیتای M1 آماده:")
        logging.info(f"   کندل‌ها: {n:,}")
        logging.info(f"   از: {t_start}")
        logging.info(f"   تا: {t_end}")
        logging.info(f"   روزها: {days}")
        logging.info(f"   میانگین کندل/روز: {n/days:.0f}")
        logging.info(f"{'='*50}")

        # چک منطقی
        if n < 100_000:
            logging.warning(f"⚠️ کندل خیلی کم: {n:,} (انتظار ~500K برای ۲ سال)")
        elif n > 1_500_000:
            logging.warning(f"⚠️ کندل خیلی زیاد: {n:,} - احتمال duplicate باقیمانده")

    # ──────────────────────────────────────────
    def resample_to_timeframe(self, timeframe: str) -> pd.DataFrame:
        if timeframe == '1T':
            return self.base_data.copy()

        resampled = self.base_data.resample(timeframe).agg(
            EURUSD=('EURUSD', 'last'),
            GBPUSD=('GBPUSD', 'last'),
        ).dropna()

        logging.info(f"Resample {timeframe}: {len(resampled):,} کندل")
        return resampled

    # ──────────────────────────────────────────
    def backtest_single(
        self,
        data: pd.DataFrame,
        window: int,
        z_entry: float,
        z_exit: float,
    ) -> dict | None:

        df = data.copy()
        if len(df) < window + 50:
            return None

        # محاسبه spread و z-score
        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        roll = df['Spread'].rolling(window=window)
        df['Mean'] = roll.mean()
        df['Std']  = roll.std()
        df = df[df['Std'] > 1e-8].copy()
        df['Z'] = (df['Spread'] - df['Mean']) / df['Std']
        df.dropna(inplace=True)

        if len(df) < 50:
            return None

        # سیگنال
        signal = np.zeros(len(df))
        signal[df['Z'].values < -z_entry] =  1   # خرید spread
        signal[df['Z'].values >  z_entry] = -1   # فروش spread
        signal[np.abs(df['Z'].values) <= z_exit] = 0

        # تبدیل سیگنال به پوزیشن (hold تا خروج)
        position = pd.Series(signal, index=df.index)
        position = position.replace(0, np.nan).ffill().fillna(0)
        position = position.shift(1).fillna(0)

        # بازده
        r_eur = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
        r_gbp = np.log(df['GBPUSD'] / df['GBPUSD'].shift(1))

        pos_change = position.diff().abs()
        cost = pos_change * self.transaction_cost
        strat_ret = position * (r_eur - r_gbp) - cost
        strat_ret = strat_ret.dropna()

        if strat_ret.empty:
            return None

        total_trades = int(pos_change.sum() / 2)
        if total_trades < 20:
            return None

        # چک overflow
        cumsum = strat_ret.cumsum()
        if cumsum.abs().max() > 300:
            return None

        cum_series = np.exp(cumsum)
        total_return = cum_series.iloc[-1] - 1

        # Max Drawdown
        rolling_max = cum_series.cummax()
        drawdown = (cum_series - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Sharpe (annualized)
        periods_per_year = {
            '1T':   252 * 1440,
            '2T':   252 * 720,
            '5T':   252 * 288,
            '15T':  252 * 96,
        }
        # پیدا کردن timeframe از فاصله زمانی
        if len(df) > 1:
            avg_minutes = (df.index[-1] - df.index[0]).total_seconds() / 60 / len(df)
            ann_factor = 252 * 1440 / max(avg_minutes, 1)
        else:
            ann_factor = 252 * 1440

        std = strat_ret.std()
        sharpe = (strat_ret.mean() / std * np.sqrt(ann_factor)) if std > 0 else 0.0

        # Win Rate
        wins = (strat_ret > 0).sum()
        win_rate = wins / len(strat_ret) * 100

        return {
            'Window':          window,
            'Z_Entry':         z_entry,
            'Z_Exit':          z_exit,
            'Total_Return_%':  round(total_return * 100, 4),
            'Max_Drawdown_%':  round(max_dd * 100, 4),
            'Sharpe':          round(sharpe, 3),
            'Win_Rate_%':      round(win_rate, 2),
            'Total_Trades':    total_trades,
        }

    # ──────────────────────────────────────────
    def run_optimization(self):
        # تایم‌فریم‌ها با فرمت pandas
        timeframes = {
            '15T': '15 دقیقه',
            '5T':  '5 دقیقه',
            '2T':  '2 دقیقه',
            '1T':  '1 دقیقه',
        }

        windows   = [50, 100, 150, 200]
        z_entries = [1.5, 2.0, 2.5, 3.0]
        z_exits   = [0.0, 0.5, 1.0]

        combinations = list(itertools.product(windows, z_entries, z_exits))
        logging.info(f"تعداد ترکیب برای هر تایم‌فریم: {len(combinations)}")

        all_results = []

        for tf_code, tf_name in timeframes.items():
            tf_data = self.resample_to_timeframe(tf_code)
            n_candles = len(tf_data)

            logging.info(f"\n{'─'*50}")
            logging.info(f"تایم‌فریم {tf_name} ({tf_code}) | {n_candles:,} کندل")

            if n_candles < 500:
                logging.warning(f"⚠️ کندل خیلی کم برای {tf_code}، رد شد")
                continue

            tf_results = []
            for window, z_entry, z_exit in combinations:
                res = self.backtest_single(tf_data, window, z_entry, z_exit)
                if res:
                    res['Timeframe'] = tf_name
                    res['TF_Code']   = tf_code
                    tf_results.append(res)

            if tf_results:
                df_tf = pd.DataFrame(tf_results)
                df_tf = df_tf.sort_values('Total_Return_%', ascending=False)
                top5 = df_tf.head(5)
                all_results.append(top5)
                logging.info(f"✅ {tf_name}: {len(tf_results)} نتیجه معتبر | بهترین بازده: {top5.iloc[0]['Total_Return_%']:.2f}%")
            else:
                logging.warning(f"⚠️ {tf_name}: هیچ نتیجه معتبری یافت نشد")

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)


# ══════════════════════════════════════════════════
if __name__ == "__main__":
    optimizer = DukascopyMultiTimeframeOptimizer()

    # بارگذاری دیتا
    optimizer.load_base_m1_data()

    # اجرای بهینه‌سازی
    logging.info("\nشروع Grid Search روی همه تایم‌فریم‌ها...")
    report = optimizer.run_optimization()

    if report.empty:
        logging.error("❌ هیچ نتیجه‌ای تولید نشد!")
        exit(1)

    # مرتب‌سازی نهایی
    cols = [
        'Timeframe', 'Window', 'Z_Entry', 'Z_Exit',
        'Total_Return_%', 'Max_Drawdown_%',
        'Sharpe', 'Win_Rate_%', 'Total_Trades'
    ]
    report = report[cols].sort_values('Total_Return_%', ascending=False)

    # ذخیره
    output_file = "ultimate_5years_mtf_report.csv"
    report.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"✅ بک‌تست کامل شد")
    print(f"📄 فایل: {output_file}")
    print(f"{'='*60}")
    print(report.to_string(index=False))
