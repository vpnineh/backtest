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
        all_files = glob.glob('data/*.csv')
        eurusd_file = next((f for f in all_files if 'eurusd' in f.lower()), None)
        gbpusd_file = next((f for f in all_files if 'gbpusd' in f.lower()), None)

        df_eur = pd.read_csv(eurusd_file, parse_dates=['timestamp']).drop_duplicates('timestamp')
        df_gbp = pd.read_csv(gbpusd_file, parse_dates=['timestamp']).drop_duplicates('timestamp')

        df_eur = df_eur.set_index('timestamp')[['close']].rename(columns={'close': 'EURUSD'})
        df_gbp = df_gbp.set_index('timestamp')[['close']].rename(columns={'close': 'GBPUSD'})

        self.base_data = df_eur.join(df_gbp, how='inner').dropna()
        logging.info(f"✅ دیتای M1 با موفقیت لود شد. تعداد کندل مشترک: {len(self.base_data):,}")

    def resample_to_timeframe(self, tf_code: str) -> pd.DataFrame:
        if tf_code == '1min':
            return self.base_data.copy()
        return self.base_data.resample(tf_code).last().dropna()

    def backtest_single(self, data: pd.DataFrame, window: int, z_entry: float, z_exit: float):
        df = data.copy()
        if len(df) < window + 100:
            return None

        df['Spread'] = np.log(df['EURUSD']) - np.log(df['GBPUSD'])
        roll = df['Spread'].rolling(window=window)
        df['Mean'] = roll.mean()
        df['Std'] = roll.std()
        df['Z'] = (df['Spread'] - df['Mean']) / df['Std']
        df.dropna(inplace=True)

        z = df['Z'].values
        signal = np.zeros(len(df))
        signal[z < -z_entry] =  1
        signal[z >  z_entry] = -1
        signal[np.abs(z) <= z_exit] = 0

        pos = pd.Series(signal, index=df.index).replace(0, np.nan).ffill().fillna(0).shift(1).fillna(0)

        # فرمول اصلاح شده: درصد سود واقعی به جای لگاریتم
        pct_ret_eur = (df['EURUSD'] - df['EURUSD'].shift(1)) / df['EURUSD'].shift(1)
        pct_ret_gbp = (df['GBPUSD'] - df['GBPUSD'].shift(1)) / df['GBPUSD'].shift(1)

        pos_change = pos.diff().abs()
        cost = pos_change * self.transaction_cost
        
        # محاسبه سود هر معامله
        strat_ret = (pos * (pct_ret_eur - pct_ret_gbp) - cost).dropna()

        if strat_ret.empty:
            return None

        total_trades = int(pos_change.sum() / 2)
        if total_trades < 10:
            return None

        # محاسبه سود تجمعی واقعی (Compound)
        cum_series = (1 + strat_ret).cumprod()
        total_ret = cum_series.iloc[-1] - 1
        
        rolling_max = cum_series.cummax()
        drawdown = (cum_series - rolling_max) / rolling_max
        max_dd = drawdown.min()

        std = strat_ret.std()
        sharpe = (strat_ret.mean() / std * np.sqrt(252 * 1440)) if std > 0 else 0

        return {
            'Window': window,
            'Z_Entry': z_entry,
            'Z_Exit': z_exit,
            'Total_Return_%': round(total_ret * 100, 2),
            'Max_Drawdown_%': round(max_dd * 100, 2),
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
            logging.info(f"پردازش تایم‌فریم: {tf_name}")

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
    optimizer = DukascopyMultiTimeframeOptimizer()
    optimizer.load_base_m1_data()
    report = optimizer.run_optimization()
    
    output_file = "ultimate_1year_mtf_report.csv"
    report.to_csv(output_file, index=False)
    print(f"\n✅ خروجی ۱ ساله ذخیره شد: {output_file}")
    print(report.to_string(index=False))
