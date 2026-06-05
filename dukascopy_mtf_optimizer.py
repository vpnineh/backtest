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
        self.transaction_cost = 0.0002
        self.base_data = pd.DataFrame()

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

        # خواندن فایل‌ها
        df_eur = pd.read_csv(eurusd_file, parse_dates=['timestamp'])
        df_gbp = pd.read_csv(gbpusd_file, parse_dates=['timestamp'])

        logging.info(f"خام | EURUSD: {len(df_eur):,} | GBPUSD: {len(df_gbp):,}")
        logging.info(f"نمونه:\n{df_eur.head(3).to_string()}")

        # چک timestamp
        hours_eur = df_eur['timestamp'].dt.hour.nunique()
        logging.info(f"ساعت‌های یکتا EURUSD: {hours_eur}")
        if hours_eur < 5:
            raise ValueError(
                f"❌ timestamp ها اشتباه هستند! فقط {hours_eur} ساعت یکتا.\n"
                f"کش قدیمی را پاک کن و downloader را دوباره اجرا کن."
            )

        # مرتب‌سازی و حذف duplicate
        df_eur = df_eur.sort_values('timestamp').drop_duplicates('timestamp', keep='last')
        df_gbp = df_gbp.sort_values('timestamp').drop_duplicates('timestamp', keep='last')

        # فیلتر کندل‌های بی‌معنی
        def clean(df, name):
            bad = (df['volume'] == 0) & (df['open'] == df['close']) & (df['high'] == df['low'])
            result = df[~bad]
            logging.info(f"فیلتر {name}: {len(df):,} → {len(result):,} (حذف {bad.sum():,})")
            return result

        df_eur = clean(df_eur, 'EURUSD')
        df_gbp = clean(df_gbp, 'GBPUSD')

        # ست کردن index
        df_eur = df_eur.set_index('timestamp')[['close']].rename(columns={'close': 'EURUSD'})
        df_gbp = df_gbp.set_index('timestamp')[['close']].rename(columns={'close': 'GBPUSD'})

        self.base_data = df_eur.join(df_gbp, how='inner').dropna()

        n         = len(self.base_data)
        t_start   = self.base_data.index[0]
        t_end     = self.base_data.index[-1]
        days      = (t_end - t_start).days
        avg_day   = n / max(days, 1)

        logging.info(f"{'='*50}")
        logging.info(f"✅ دیتای M1 آماده:")
        logging.info(f"   کندل‌ها    : {n:,}")
        logging.info(f"   از         : {t_start}")
        logging.info(f"   تا         : {t_end}")
        logging.info(f"   روزها      : {days}")
        logging.info(f"   کندل/روز   : {avg_day:.0f}")
        logging.info(f"{'='*50}")

        # اعتبارسنجی
        if avg_day < 100:
            raise ValueError(
                f"❌ میانگین {avg_day:.0f} کندل/روز خیلی کمه!\n"
                f"باید ~400-500 باشه. کش را پاک کن و downloader را دوباره اجرا کن."
            )

        if avg_day > 1500:
            raise ValueError(
                f"❌ میانگین {avg_day:.0f} کندل/روز خیلی زیاده! احتمال duplicate."
            )

        logging.info(f"✅ اعتبارسنجی موفق - داده سالم است")

    def resample_to_timeframe(self, tf_code: str) -> pd.DataFrame:
        if tf_code == '1min':
            return self.base_data.copy()

        resampled = self.base_data.resample(tf_code).agg(
            EURUSD=('EURUSD', 'last'),
            GBPUSD=('GBPUSD', 'last'),
        ).dropna()

        logging.info(f"Resample {tf_code}: {len(resampled):,} کندل")
        return resampled

    def backtest_single(
        self,
        data: pd.DataFrame,
        window: int,
        z_entry: float,
        z_exit: float,
    ) -> dict | None:

        df = data.copy()
        if len(df) < window + 100:
            return None

        # spread و z-score
        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        roll         = df['Spread'].rolling(window=window)
        df['Mean']   = roll.mean()
        df['Std']    = roll.std()
        df           = df[df['Std'] > 1e-8].copy()
        df['Z']      = (df['Spread'] - df['Mean']) / df['Std']
        df.dropna(inplace=True)

        if len(df) < 100:
            return None

        # سیگنال
        z = df['Z'].values
        signal = np.zeros(len(df))
        signal[z < -z_entry] =  1
        signal[z >  z_entry] = -1
        signal[np.abs(z) <= z_exit] = 0

        # پوزیشن (hold تا خروج)
        pos = pd.Series(signal, index=df.index)
        pos = pos.replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)

        # بازده
        r_eur = np.log(df['EURUSD'] / df['EURUSD'].shift(1))
        r_gbp = np.log(df['GBPUSD'] / df['GBPUSD'].shift(1))

        pos_change = pos.diff().abs()
        cost       = pos_change * self.transaction_cost
        strat_ret  = (pos * (r_eur - r_gbp) - cost).dropna()

        if strat_ret.empty:
            return None

        total_trades = int(pos_change.sum() / 2)
        if total_trades < 20:
            return None

        # چک overflow
        cumsum = strat_ret.cumsum()
        if cumsum.abs().max() > 300:
            return None

        cum_series  = np.exp(cumsum)
        total_ret   = cum_series.iloc[-1] - 1
        rolling_max = cum_series.cummax()
        drawdown    = (cum_series - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        # Sharpe annualized
        std = strat_ret.std()
        if std > 0:
            # محاسبه ann_factor از فاصله واقعی بین کندل‌ها
            total_minutes = (df.index[-1] - df.index[0]).total_seconds() / 60
            minutes_per_candle = total_minutes / max(len(df) - 1, 1)
            ann_factor = 252 * 1440 / max(minutes_per_candle, 1)
            sharpe = strat_ret.mean() / std * np.sqrt(ann_factor)
        else:
            sharpe = 0.0

        win_rate = (strat_ret > 0).sum() / len(strat_ret) * 100

        return {
            'Window':         window,
            'Z_Entry':        z_entry,
            'Z_Exit':         z_exit,
            'Total_Return_%': round(total_ret * 100, 4),
            'Max_Drawdown_%': round(max_dd * 100, 4),
            'Sharpe':         round(sharpe, 3),
            'Win_Rate_%':     round(win_rate, 2),
            'Total_Trades':   total_trades,
        }

    def run_optimization(self) -> pd.DataFrame:
        timeframes = {
            '15min': '15 دقیقه',
            '5min':  '5 دقیقه',
            '2min':  '2 دقیقه',
            '1min':  '1 دقیقه',
        }

        windows   = [30, 50, 100, 150, 200]
        z_entries = [1.5, 2.0, 2.5, 3.0]
        z_exits   = [0.0, 0.5, 1.0]
        combos    = list(itertools.product(windows, z_entries, z_exits))

        logging.info(f"ترکیب‌ها برای هر تایم‌فریم: {len(combos)}")

        all_results = []

        for tf_code, tf_name in timeframes.items():
            tf_data   = self.resample_to_timeframe(tf_code)
            n_candles = len(tf_data)

            logging.info(f"\n{'─'*50}")
            logging.info(f"تایم‌فریم: {tf_name} | کندل: {n_candles:,}")

            if n_candles < 1000:
                logging.warning(f"⚠️ {tf_name}: کندل کم ({n_candles})، رد شد")
                continue

            tf_results = []
            for window, z_entry, z_exit in combos:
                res = self.backtest_single(tf_data, window, z_entry, z_exit)
                if res:
                    res['Timeframe'] = tf_name
                    tf_results.append(res)

            if tf_results:
                df_tf = (
                    pd.DataFrame(tf_results)
                    .sort_values('Total_Return_%', ascending=False)
                )
                best = df_tf.iloc[0]
                logging.info(
                    f"✅ {tf_name}: {len(tf_results)} نتیجه | "
                    f"بهترین: {best['Total_Return_%']:.2f}% | "
                    f"Sharpe: {best['Sharpe']:.2f}"
                )
                all_results.append(df_tf.head(5))
            else:
                logging.warning(f"⚠️ {tf_name}: نتیجه معتبر یافت نشد")

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    optimizer = DukascopyMultiTimeframeOptimizer()
    optimizer.load_base_m1_data()

    logging.info("\nشروع بهینه‌سازی...")
    report = optimizer.run_optimization()

    if report.empty:
        logging.error("❌ هیچ نتیجه‌ای تولید نشد!")
        exit(1)

    cols = [
        'Timeframe', 'Window', 'Z_Entry', 'Z_Exit',
        'Total_Return_%', 'Max_Drawdown_%',
        'Sharpe', 'Win_Rate_%', 'Total_Trades',
    ]
    report = report[cols].sort_values('Total_Return_%', ascending=False)

    output_file = "ultimate_5years_mtf_report.csv"
    report.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"✅ بک‌تست ۵ ساله کامل شد")
    print(f"📄 {output_file}")
    print(f"{'='*60}")
    print(report.to_string(index=False))
