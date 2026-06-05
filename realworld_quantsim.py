import pandas as pd
import numpy as np
import glob
from dataclasses import dataclass, field
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# ================================================================== #
#                        CONFIG مرکزی                                #
# ================================================================== #
@dataclass
class PropConfig:
    initial_balance: float = 5000.0
    
    # محدودیت‌های پراپ (استاندارد FTMO/MyForexFunds)
    max_daily_loss_pct: float = 0.05      # 5% ضرر روزانه
    max_total_dd_pct:   float = 0.10      # 10% دراودان کل
    profit_target_pct:  float = 0.10      # 10% هدف سود
    
    # مدیریت پوزیشن
    risk_per_trade_pct: float = 0.01      # 1% ریسک هر معامله
    max_lot_per_trade:  float = 0.5
    spread_eur:         float = 0.00012   # 1.2 پیپ
    spread_gbp:         float = 0.00015   # 1.5 پیپ
    commission_per_lot: float = 7.0       # دلار per lot (round trip)


# ================================================================== #
#                      کلاس Trade                                    #
# ================================================================== #
@dataclass
class Trade:
    strategy:   str
    symbol:     str
    direction:  int        # +1 خرید / -1 فروش
    lot:        float
    entry_price: float
    entry_time:  pd.Timestamp
    sl:         float
    tp:         float
    exit_price: float  = 0.0
    exit_time:  pd.Timestamp = None
    pnl:        float  = 0.0
    status:     str    = 'open'   # open / closed


# ================================================================== #
#              کلاس مدیریت ریسک مرکزی                               #
# ================================================================== #
class RiskManager:
    def __init__(self, config: PropConfig):
        self.cfg          = config
        self.equity       = config.initial_balance
        self.peak_equity  = config.initial_balance
        self.daily_start_equity = config.initial_balance
        self.current_date = None
        self.trading_halted = False
        self.halt_reason    = ""
        self.equity_curve   = [config.initial_balance]
        self.daily_pnl_log  = {}

    def update_daily(self, date: pd.Timestamp):
        """در شروع روز جدید ریست می‌شود"""
        day = date.date()
        if self.current_date != day:
            self.current_date       = day
            self.daily_start_equity = self.equity
            self.trading_halted     = False   # روز جدید → ریست

    def calculate_lot(self, sl_pips: float, pip_value: float = 10.0) -> float:
        """محاسبه حجم بر اساس ریسک ثابت"""
        if sl_pips <= 0:
            return 0.01
        risk_amount = self.equity * self.cfg.risk_per_trade_pct
        lot = risk_amount / (sl_pips * pip_value)
        lot = round(min(lot, self.cfg.max_lot_per_trade), 2)
        return max(lot, 0.01)

    def register_pnl(self, pnl: float, date: pd.Timestamp) -> bool:
        """
        ثبت P&L و بررسی محدودیت‌های پراپ
        Returns: False اگر معامله باید متوقف شود
        """
        self.equity += pnl
        self.equity_curve.append(self.equity)
        self.peak_equity = max(self.peak_equity, self.equity)

        day = date.date()
        self.daily_pnl_log[day] = self.daily_pnl_log.get(day, 0) + pnl

        # چک ۱: ضرر روزانه
        daily_dd = (self.equity - self.daily_start_equity) / self.daily_start_equity
        if daily_dd <= -self.cfg.max_daily_loss_pct:
            self.trading_halted = True
            self.halt_reason    = f"Daily Loss {daily_dd*100:.1f}%"
            return False

        # چک ۲: دراودان کل
        total_dd = (self.equity - self.peak_equity) / self.peak_equity
        if total_dd <= -self.cfg.max_total_dd_pct:
            self.trading_halted = True
            self.halt_reason    = f"Max DD {total_dd*100:.1f}%"
            return False

        # چک ۳: هدف سود رسیده؟
        total_profit = (self.equity - self.cfg.initial_balance) / self.cfg.initial_balance
        if total_profit >= self.cfg.profit_target_pct:
            self.trading_halted = True
            self.halt_reason    = f"✅ Profit Target {total_profit*100:.1f}%"
            return False

        return True

    @property
    def max_drawdown(self) -> float:
        curve = pd.Series(self.equity_curve)
        roll_max = curve.cummax()
        dd = (curve - roll_max) / roll_max
        return dd.min() * 100

    @property
    def sharpe_ratio(self) -> float:
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252 * 96)  # 96 کندل ۱۵ دقیقه‌ای در روز


# ================================================================== #
#            Strategy 1: Correlation Arbitrage                       #
# ================================================================== #
class CorrelationArbitrageStrategy:
    """
    منطق: EURUSD و GBPUSD معمولاً correlation بالایی دارند (>0.85)
    وقتی correlation موقتاً کاهش می‌یابد → divergence → mean reversion
    
    سیگنال:
    - Spread = EUR_normalized - GBP_normalized
    - وقتی spread > 2σ → EUR گران است → EUR بفروش، GBP بخر
    - وقتی spread < -2σ → GBP گران است → GBP بفروش، EUR بخر
    """
    def __init__(self, config: PropConfig):
        self.cfg        = config
        self.name       = "Correlation_Arb"
        self.lookback   = 96        # ۲۴ ساعت (۹۶ کندل ۱۵ دقیقه‌ای)
        self.z_entry    = 2.0       # ورود در ۲ انحراف معیار
        self.z_exit     = 0.5       # خروج در بازگشت به میانه
        self.min_corr   = 0.70      # حداقل correlation برای فعال بودن
        self.trades: List[Trade] = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # نرمال‌سازی قیمت‌ها (Z-score rolling)
        eur_mean = df['c_eur'].rolling(self.lookback).mean()
        eur_std  = df['c_eur'].rolling(self.lookback).std()
        gbp_mean = df['c_gbp'].rolling(self.lookback).mean()
        gbp_std  = df['c_gbp'].rolling(self.lookback).std()

        df['eur_z'] = (df['c_eur'] - eur_mean) / eur_std
        df['gbp_z'] = (df['c_gbp'] - gbp_mean) / gbp_std

        # Spread بین دو جفت‌ارز نرمال‌شده
        df['spread_z'] = df['eur_z'] - df['gbp_z']

        # میانگین و انحراف معیار spread
        df['spread_mean'] = df['spread_z'].rolling(self.lookback).mean()
        df['spread_std']  = df['spread_z'].rolling(self.lookback).std()
        df['z_score']     = (df['spread_z'] - df['spread_mean']) / df['spread_std'].replace(0, np.nan)

        # Correlation rolling
        df['correlation'] = df['c_eur'].rolling(self.lookback).corr(df['c_gbp'])

        # سیگنال‌ها
        df['arb_signal'] = 0
        
        valid = (
            df['correlation'].abs() >= self.min_corr
        ) & df['z_score'].notna()

        # EUR گران → فروش EUR، خرید GBP
        df.loc[valid & (df['z_score'] >  self.z_entry), 'arb_signal'] = -1
        # GBP گران → فروش GBP، خرید EUR
        df.loc[valid & (df['z_score'] < -self.z_entry), 'arb_signal'] =  1

        return df


# ================================================================== #
#         Strategy 2: London/NY Session Breakout                     #
# ================================================================== #
class SessionBreakoutStrategy:
    """
    منطق: در اول Session لندن (08:00 GMT) و نیویورک (13:00 GMT)
    بیشترین نوسان اتفاق می‌افتد.
    
    سیگنال:
    - Range ساعت ۰۶:۰۰-۰۸:۰۰ GMT را محاسبه کن (Pre-London Range)
    - شکست بالای range → Buy
    - شکست پایین range → Sell
    - ATR Filter برای جلوگیری از سیگنال کاذب در بازار آرام
    """
    def __init__(self, config: PropConfig):
        self.cfg      = config
        self.name     = "Session_Breakout"
        self.atr_mult = 1.5         # ضریب ATR برای SL
        self.trades: List[Trade] = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df.index.hour
        df['date'] = df.index.date

        # ATR
        df['tr']  = np.maximum(
            df['h_eur'] - df['l_eur'],
            np.maximum(
                abs(df['h_eur'] - df['c_eur'].shift(1)),
                abs(df['l_eur'] - df['c_eur'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # Pre-London Range: کندل‌های ۰۶:۰۰ تا ۰۷:۴۵ GMT
        pre_london = df[df['hour'].isin([6, 7])].copy()
        daily_range = pre_london.groupby('date').agg(
            range_high=('h_eur', 'max'),
            range_low=('l_eur', 'min')
        )

        df = df.join(daily_range, on='date')

        # سیگنال فقط در ساعات ۰۸:۰۰ تا ۱۶:۰۰
        df['breakout_signal'] = 0
        trading_hours = df['hour'].between(8, 16)

        # شکست صعودی
        df.loc[
            trading_hours &
            (df['c_eur'] > df['range_high']) &
            (df['atr'] > df['atr'].rolling(20).mean() * 0.8),   # بازار فعال
            'breakout_signal'
        ] = 1

        # شکست نزولی
        df.loc[
            trading_hours &
            (df['c_eur'] < df['range_low']) &
            (df['atr'] > df['atr'].rolling(20).mean() * 0.8),
            'breakout_signal'
        ] = -1

        return df


# ================================================================== #
#         Strategy 3: ATR-Based Mean Reversion                       #
# ================================================================== #
class ATRMeanReversionStrategy:
    """
    منطق: بازار بعد از حرکات شارپ (بیشتر از N×ATR) 
    تمایل به بازگشت به میانگین دارد.
    
    سیگنال:
    - اگر قیمت بیش از ۲×ATR از EMA(50) فاصله گرفت → Mean Reversion
    - فقط در زمان‌هایی که RSI اشباع خرید/فروش نشان می‌دهد
    - SL پشت آخرین swing high/low
    """
    def __init__(self, config: PropConfig):
        self.cfg        = config
        self.name       = "ATR_Mean_Rev"
        self.ema_period = 50
        self.atr_mult   = 2.0       # فاصله از EMA برای ورود
        self.rsi_period = 14
        self.rsi_ob     = 65        # اشباع خرید (محافظه‌کارانه)
        self.rsi_os     = 35        # اشباع فروش
        self.trades: List[Trade] = []

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta  = series.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMA
        df['ema50'] = df['c_eur'].ewm(span=self.ema_period, adjust=False).mean()

        # ATR
        df['tr']  = np.maximum(
            df['h_eur'] - df['l_eur'],
            np.maximum(
                abs(df['h_eur'] - df['c_eur'].shift(1)),
                abs(df['l_eur'] - df['c_eur'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # RSI
        df['rsi'] = self._rsi(df['c_eur'], self.rsi_period)

        # فاصله از EMA نسبت به ATR
        df['dist_from_ema'] = (df['c_eur'] - df['ema50']) / df['atr'].replace(0, np.nan)

        df['mr_signal'] = 0

        # قیمت خیلی بالای EMA + RSI اشباع خرید → فروش (انتظار بازگشت)
        df.loc[
            (df['dist_from_ema'] >  self.atr_mult) &
            (df['rsi'] > self.rsi_ob),
            'mr_signal'
        ] = -1

        # قیمت خیلی زیر EMA + RSI اشباع فروش → خرید
        df.loc[
            (df['dist_from_ema'] < -self.atr_mult) &
            (df['rsi'] < self.rsi_os),
            'mr_signal'
        ] =  1

        return df


# ================================================================== #
#                      موتور اصلی Backtest                          #
# ================================================================== #
class PropBacktestEngine:
    def __init__(self):
        self.config = PropConfig()
        self.risk   = RiskManager(self.config)
        self.strategies = {
            'corr_arb':    CorrelationArbitrageStrategy(self.config),
            'breakout':    SessionBreakoutStrategy(self.config),
            'mean_rev':    ATRMeanReversionStrategy(self.config),
        }
        self.all_trades: List[Trade] = []

    # ---------------------------------------------------------------- #
    def load_data(self):
        files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
        files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))

        if not files_eur or not files_gbp:
            raise FileNotFoundError("فایل‌های CSV پیدا نشدند در پوشه data/")

        def read_df(path, suffix):
            df = pd.read_csv(
                path, sep=';', header=None,
                names=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            df = df.set_index('ts')
            df = df[~df.index.duplicated(keep='last')]
            df.columns = [f'{col}_{suffix}' for col in df.columns]
            return df

        eur = pd.concat([read_df(f, 'eur') for f in files_eur]).sort_index()
        gbp = pd.concat([read_df(f, 'gbp') for f in files_gbp]).sort_index()

        df = eur.join(gbp, how='inner').dropna()

        # ریسمپل به ۱۵ دقیقه
        df_15 = pd.DataFrame({
            'o_eur': df['o_eur'].resample('15min').first(),
            'h_eur': df['h_eur'].resample('15min').max(),
            'l_eur': df['l_eur'].resample('15min').min(),
            'c_eur': df['c_eur'].resample('15min').last(),
            'o_gbp': df['o_gbp'].resample('15min').first(),
            'h_gbp': df['h_gbp'].resample('15min').max(),
            'l_gbp': df['l_gbp'].resample('15min').min(),
            'c_gbp': df['c_gbp'].resample('15min').last(),
        }).dropna()

        self.df = df_15
        print(f"✅ دیتا لود شد | {len(self.df):,} کندل ۱۵ دقیقه‌ای")
        print(f"   از: {self.df.index[0]}  تا: {self.df.index[-1]}")

    # ---------------------------------------------------------------- #
    def _apply_spread(self, price: float, direction: int, spread: float) -> float:
        """اعمال اسپرد واقعی روی قیمت ورود"""
        return price + (spread / 2 * direction)

    def _calculate_pnl(self, trade: Trade, exit_price: float) -> float:
        """محاسبه P&L با کمیسیون"""
        spread = (self.config.spread_eur 
                  if trade.symbol == 'EURUSD' 
                  else self.config.spread_gbp)
        
        price_diff   = (exit_price - trade.entry_price) * trade.direction
        gross_pnl    = price_diff * trade.lot * 100_000
        commission   = self.config.commission_per_lot * trade.lot
        spread_cost  = spread * trade.lot * 100_000
        return gross_pnl - commission - spread_cost

    # ---------------------------------------------------------------- #
    def run(self):
        print("\n⚙️  در حال محاسبه سیگنال‌ها...")

        # تولید سیگنال توسط هر استراتژی
        df_arb = self.strategies['corr_arb'].generate_signals(self.df)
        df_brk = self.strategies['breakout'].generate_signals(self.df)
        df_mr  = self.strategies['mean_rev'].generate_signals(self.df)

        # ادغام سیگنال‌ها
        df = self.df.copy()
        df['sig_arb'] = df_arb['arb_signal'].fillna(0).astype(int)
        df['sig_brk'] = df_brk['breakout_signal'].fillna(0).astype(int)
        df['sig_mr']  = df_mr['mr_signal'].fillna(0).astype(int)
        df['atr']     = df_mr['atr']
        df['z_score'] = df_arb.get('z_score', pd.Series(0, index=df.index))

        # پوزیشن‌های باز هر استراتژی
        open_trades: Dict[str, Trade] = {}

        print("⚙️  در حال اجرای شبیه‌سازی...\n")

        for i in range(100, len(df)):    # از ۱۰۰ام به بعد (warmup اندیکاتورها)
            row      = df.iloc[i]
            curr_ts  = df.index[i]
            curr_eur = row['c_eur']
            curr_gbp = row['c_gbp']
            atr      = row['atr'] if pd.notna(row['atr']) else 0.0010

            # ---- به‌روزرسانی روزانه ----
            self.risk.update_daily(curr_ts)

            if self.risk.trading_halted:
                # بستن همه پوزیشن‌های باز
                for key, tr in list(open_trades.items()):
                    price = curr_eur if tr.symbol == 'EURUSD' else curr_gbp
                    tr.pnl    = self._calculate_pnl(tr, price)
                    tr.status = 'closed_halt'
                    self.all_trades.append(tr)
                    self.risk.register_pnl(tr.pnl, curr_ts)
                open_trades.clear()
                continue

            # ================================================================
            # Strategy 1: Correlation Arbitrage
            # ================================================================
            key_arb_eur = 'arb_eur'
            key_arb_gbp = 'arb_gbp'

            if key_arb_eur not in open_trades and key_arb_gbp not in open_trades:
                sig = int(row['sig_arb'])
                if sig != 0:
                    sl_pips = atr * 10_000 * 1.5
                    lot     = self.risk.calculate_lot(sl_pips)

                    # EUR position
                    tr_eur = Trade(
                        strategy    = self.strategies['corr_arb'].name,
                        symbol      = 'EURUSD',
                        direction   = sig,
                        lot         = lot,
                        entry_price = self._apply_spread(curr_eur, sig, self.config.spread_eur),
                        entry_time  = curr_ts,
                        sl          = curr_eur - sig * atr * 1.5,
                        tp          = curr_eur + sig * atr * 2.5,
                    )
                    # GBP position (معکوس)
                    tr_gbp = Trade(
                        strategy    = self.strategies['corr_arb'].name,
                        symbol      = 'GBPUSD',
                        direction   = -sig,
                        lot         = lot,
                        entry_price = self._apply_spread(curr_gbp, -sig, self.config.spread_gbp),
                        entry_time  = curr_ts,
                        sl          = curr_gbp + sig * atr * 1.5,
                        tp          = curr_gbp - sig * atr * 2.5,
                    )
                    open_trades[key_arb_eur] = tr_eur
                    open_trades[key_arb_gbp] = tr_gbp

            else:
                # بررسی SL/TP/خروج برای Arb
                for key in [key_arb_eur, key_arb_gbp]:
                    if key not in open_trades:
                        continue
                    tr    = open_trades[key]
                    price = curr_eur if tr.symbol == 'EURUSD' else curr_gbp
                    z     = row.get('z_score', 0)

                    hit_sl = (tr.direction ==  1 and price <= tr.sl) or \
                             (tr.direction == -1 and price >= tr.sl)
                    hit_tp = (tr.direction ==  1 and price >= tr.tp) or \
                             (tr.direction == -1 and price <= tr.tp)
                    # خروج وقتی spread به میانه برگشت
                    z_exit = abs(z) < self.strategies['corr_arb'].z_exit \
                             if pd.notna(z) else False

                    if hit_sl or hit_tp or z_exit:
                        tr.exit_price = price
                        tr.exit_time  = curr_ts
                        tr.pnl        = self._calculate_pnl(tr, price)
                        tr.status     = 'sl' if hit_sl else ('tp' if hit_tp else 'z_exit')
                        self.all_trades.append(tr)
                        self.risk.register_pnl(tr.pnl, curr_ts)
                        del open_trades[key]

            # ================================================================
            # Strategy 2: Session Breakout
            # ================================================================
            key_brk = 'breakout'
            if key_brk not in open_trades:
                sig = int(row['sig_brk'])
                if sig != 0:
                    sl_pips = atr * 10_000 * self.strategies['breakout'].atr_mult
                    lot     = self.risk.calculate_lot(sl_pips)
                    tr = Trade(
                        strategy    = self.strategies['breakout'].name,
                        symbol      = 'EURUSD',
                        direction   = sig,
                        lot         = lot,
                        entry_price = self._apply_spread(curr_eur, sig, self.config.spread_eur),
                        entry_time  = curr_ts,
                        sl          = curr_eur - sig * atr * self.strategies['breakout'].atr_mult,
                        tp          = curr_eur + sig * atr * 3.0,   # RR = 2:1
                    )
                    open_trades[key_brk] = tr
            else:
                tr    = open_trades[key_brk]
                price = curr_eur
                hit_sl = (tr.direction ==  1 and price <= tr.sl) or \
                         (tr.direction == -1 and price >= tr.sl)
                hit_tp = (tr.direction ==  1 and price >= tr.tp) or \
                         (tr.direction == -1 and price <= tr.tp)

                # Trailing Stop: اگر سود > 1ATR، SL را به BE ببر
                if tr.direction == 1 and price > tr.entry_price + atr:
                    tr.sl = max(tr.sl, tr.entry_price)
                elif tr.direction == -1 and price < tr.entry_price - atr:
                    tr.sl = min(tr.sl, tr.entry_price)

                if hit_sl or hit_tp:
                    tr.exit_price = price
                    tr.exit_time  = curr_ts
                    tr.pnl        = self._calculate_pnl(tr, price)
                    tr.status     = 'sl' if hit_sl else 'tp'
                    self.all_trades.append(tr)
                    self.risk.register_pnl(tr.pnl, curr_ts)
                    del open_trades[key_brk]

            # ================================================================
            # Strategy 3: ATR Mean Reversion
            # ================================================================
            key_mr = 'mean_rev'
            if key_mr not in open_trades:
                sig = int(row['sig_mr'])
                if sig != 0:
                    sl_pips = atr * 10_000 * 2.0
                    lot     = self.risk.calculate_lot(sl_pips)
                    tr = Trade(
                        strategy    = self.strategies['mean_rev'].name,
                        symbol      = 'EURUSD',
                        direction   = sig,
                        lot         = lot,
                        entry_price = self._apply_spread(curr_eur, sig, self.config.spread_eur),
                        entry_time  = curr_ts,
                        sl          = curr_eur - sig * atr * 2.0,
                        tp          = curr_eur + sig * atr * 1.5,   # هدف: برگشت به EMA
                    )
                    open_trades[key_mr] = tr
            else:
                tr    = open_trades[key_mr]
                price = curr_eur
                hit_sl = (tr.direction ==  1 and price <= tr.sl) or \
                         (tr.direction == -1 and price >= tr.sl)
                hit_tp = (tr.direction ==  1 and price >= tr.tp) or \
                         (tr.direction == -1 and price <= tr.tp)

                if hit_sl or hit_tp:
                    tr.exit_price = price
                    tr.exit_time  = curr_ts
                    tr.pnl        = self._calculate_pnl(tr, price)
                    tr.status     = 'sl' if hit_sl else 'tp'
                    self.all_trades.append(tr)
                    self.risk.register_pnl(tr.pnl, curr_ts)
                    del open_trades[key_mr]

        # بستن پوزیشن‌های باز در پایان
        last_ts  = df.index[-1]
        last_eur = df['c_eur'].iloc[-1]
        last_gbp = df['c_gbp'].iloc[-1]
        for key, tr in open_trades.items():
            price     = last_eur if tr.symbol == 'EURUSD' else last_gbp
            tr.pnl    = self._calculate_pnl(tr, price)
            tr.status = 'closed_eod'
            self.all_trades.append(tr)
            self.risk.register_pnl(tr.pnl, last_ts)

    # ---------------------------------------------------------------- #
    def report(self):
        if not self.all_trades:
            print("❌ هیچ معامله‌ای انجام نشد!")
            return

        trades_df = pd.DataFrame([{
            'strategy':    t.strategy,
            'symbol':      t.symbol,
            'direction':   t.direction,
            'lot':         t.lot,
            'entry_time':  t.entry_time,
            'exit_time':   t.exit_time,
            'entry_price': t.entry_price,
            'exit_price':  t.exit_price,
            'pnl':         t.pnl,
            'status':      t.status,
        } for t in self.all_trades])

        equity_curve = pd.Series(self.risk.equity_curve)
        final_equity = self.risk.equity
        total_return = (final_equity - self.config.initial_balance) / self.config.initial_balance * 100
        win_rate     = (trades_df['pnl'] > 0).mean() * 100
        avg_win      = trades_df.loc[trades_df['pnl'] > 0, 'pnl'].mean()
        avg_loss     = trades_df.loc[trades_df['pnl'] < 0, 'pnl'].mean()
        profit_factor = (
            trades_df.loc[trades_df['pnl'] > 0, 'pnl'].sum() /
            abs(trades_df.loc[trades_df['pnl'] < 0, 'pnl'].sum())
            if trades_df['pnl'].lt(0).any() else float('inf')
        )

        print("\n" + "=" * 55)
        print("         📊 گزارش کامل Backtest (2020-2025)")
        print("=" * 55)
        print(f"  موجودی اولیه     : ${self.config.initial_balance:>10,.2f}")
        print(f"  موجودی نهایی     : ${final_equity:>10,.2f}")
        print(f"  بازده کل         : {total_return:>+10.2f}%")
        print(f"  Max Drawdown     : {self.risk.max_drawdown:>10.2f}%")
        print(f"  Sharpe Ratio     : {self.risk.sharpe_ratio:>10.2f}")
        print(f"  Profit Factor    : {profit_factor:>10.2f}")
        print(f"  Win Rate         : {win_rate:>10.1f}%")
        print(f"  Avg Win          : ${avg_win:>10.2f}")
        print(f"  Avg Loss         : ${avg_loss:>10.2f}")
        print(f"  تعداد کل معاملات : {len(trades_df):>10,}")
        print(f"  توقف به دلیل     : {self.risk.halt_reason}")
        print("-" * 55)

        # گزارش هر استراتژی
        print("\n  📈 عملکرد هر استراتژی:")
        print(f"  {'استراتژی':<25} {'معاملات':>8} {'Win%':>7} {'PnL':>10}")
        print("  " + "-" * 52)
        for strat_name in trades_df['strategy'].unique():
            sub   = trades_df[trades_df['strategy'] == strat_name]
            wr    = (sub['pnl'] > 0).mean() * 100
            total = sub['pnl'].sum()
            print(f"  {strat_name:<25} {len(sub):>8,} {wr:>6.1f}% ${total:>10,.2f}")
        print("=" * 55)

        # ذخیره فایل‌ها
        trades_df.to_csv("trades_report.csv", index=False)
        equity_curve.to_csv("equity_curve.csv", index=False)
        print("\n✅ فایل‌های خروجی:")
        print("   → trades_report.csv")
        print("   → equity_curve.csv")


# ================================================================== #
if __name__ == "__main__":
    engine = PropBacktestEngine()
    engine.load_data()
    engine.run()
    engine.report()
