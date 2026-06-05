import pandas as pd
import numpy as np
import glob

class PropSafeHedgingEngine:
    def __init__(self):
        self.initial_balance = 5000.0
        self.max_lot           = 0.1          # حجم هر معامله (لات)
        self.max_net_lot       = 1.0          # سقف خالص پوزیشن
        self.move_threshold    = 0.001        # 0.1٪ برای فعال‌سازی هج
        self.dd_floor          = 4500.0       # کف دراودان → بستن همه پوزیشن‌ها
        self.pip_value         = 100_000      # ارزش هر لات در واحد قیمتی

    # ------------------------------------------------------------------ #
    def load_data(self):
        files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
        files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))

        if not files_eur:
            raise FileNotFoundError("فایل EURUSD پیدا نشد در پوشه data/")
        if not files_gbp:
            raise FileNotFoundError("فایل GBPUSD پیدا نشد در پوشه data/")

        def read_df(path):
            df = pd.read_csv(
                path, sep=';', header=None,
                names=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            df = df.set_index('ts')[['c']]
            df = df[~df.index.duplicated(keep='last')]   # حذف تکراری‌ها
            return df

        eur = pd.concat([read_df(f) for f in files_eur]).sort_index()
        gbp = pd.concat([read_df(f) for f in files_gbp]).sort_index()

        combined = eur.join(gbp, lsuffix='_eur', rsuffix='_gbp')
        combined = combined.dropna()

        if combined.empty:
            raise ValueError("بعد از join هیچ داده مشترکی باقی نماند!")

        self.df = combined
        print(f"✅ دیتا لود شد | ردیف‌ها: {len(self.df):,} | "
              f"از {self.df.index[0]} تا {self.df.index[-1]}")

    # ------------------------------------------------------------------ #
    def run_simulation(self):
        # ریسمپل + حذف NaN احتمالی بعد از ریسمپل
        df = (
            self.df
            .resample('15min')
            .last()
            .dropna()          # ← مهم: کندل‌های خالی را حذف می‌کند
            .reset_index()
        )

        if len(df) < 2:
            raise ValueError("بعد از ریسمپل، داده کافی برای شبیه‌سازی وجود ندارد.")

        # ---------------------------------------------------------------- #
        # متغیرهای وضعیت
        equity      = self.initial_balance
        net_pos_eur = 0.0          # خالص پوزیشن فعلی (لات)
        all_equity  = [equity]
        trades      = 0
        forced_cls  = 0

        prices_eur = df['c_eur'].values    # numpy → سریع‌تر
        prices_gbp = df['c_gbp'].values

        for i in range(1, len(df)):
            prev_eur = prices_eur[i - 1]
            curr_eur = prices_eur[i]

            # ---- ۱. بررسی صحت قیمت‌ها ----
            if prev_eur <= 0 or curr_eur <= 0 or np.isnan(prev_eur) or np.isnan(curr_eur):
                all_equity.append(equity)
                continue

            # ---- ۲. محاسبه حرکت بازار ----
            diff = (curr_eur - prev_eur) / prev_eur   # بازده نسبی

            # ---- ۳. منطق هجینگ ----
            if abs(diff) > self.move_threshold:
                signal = -np.sign(diff) * self.max_lot

                # سقف پوزیشن خالص رعایت شود
                new_net = net_pos_eur + signal
                new_net = np.clip(new_net, -self.max_net_lot, self.max_net_lot)

                if new_net != net_pos_eur:
                    net_pos_eur = new_net
                    trades += 1

            # ---- ۴. محاسبه P&L لحظه‌ای ----
            price_change = curr_eur - prev_eur          # تغییر قیمت (دلار)
            profit       = net_pos_eur * price_change * self.pip_value

            # ---- ۵. به‌روزرسانی موجودی ----
            equity += profit

            # ---- ۶. مدیریت دراودان (Stop Loss کلی) ----
            if equity <= self.dd_floor:
                equity      = self.dd_floor
                net_pos_eur = 0.0           # بستن همه پوزیشن‌ها
                forced_cls += 1

            # ---- ۷. حفاظت در برابر NaN/Inf ----
            if not np.isfinite(equity):
                print(f"⚠️  کندل {i}: equity نامعتبر شد → ریست به آخرین مقدار معتبر")
                equity      = all_equity[-1]
                net_pos_eur = 0.0

            all_equity.append(equity)

        # ---------------------------------------------------------------- #
        # ساخت گزارش
        df = df.iloc[:len(all_equity)].copy()
        df['Equity'] = all_equity

        max_equity  = max(all_equity)
        min_equity  = min(all_equity)
        final_eq    = all_equity[-1]
        profit_pct  = (final_eq - self.initial_balance) / self.initial_balance * 100

        # محاسبه Max Drawdown واقعی
        running_max = pd.Series(all_equity).cummax()
        drawdown    = (pd.Series(all_equity) - running_max) / running_max * 100
        max_dd      = drawdown.min()

        print("\n" + "=" * 45)
        print("       📊 گزارش شبیه‌سازی PropSafe")
        print("=" * 45)
        print(f"  موجودی اولیه   : ${self.initial_balance:>10,.2f}")
        print(f"  موجودی نهایی   : ${final_eq:>10,.2f}")
        print(f"  بازده کل       : {profit_pct:>+10.2f}%")
        print(f"  بیشترین موجودی : ${max_equity:>10,.2f}")
        print(f"  کمترین موجودی  : ${min_equity:>10,.2f}")
        print(f"  Max Drawdown   : {max_dd:>10.2f}%")
        print(f"  تعداد معاملات  : {trades:>10,}")
        print(f"  بسته‌شدن اضطراری: {forced_cls:>10,} بار")
        print("=" * 45)

        df.to_csv("Equity_Curve_Report.csv", index=False)
        print("✅ فایل Equity_Curve_Report.csv ذخیره شد.")
        return df


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    engine = PropSafeHedgingEngine()
    engine.load_data()
    engine.run_simulation()
