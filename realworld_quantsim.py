import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')


# ================================================================== #
#                         CONFIG                                     #
# ================================================================== #
class Config:
    initial_balance      = 5_000.0
    risk_per_trade_pct   = 0.005      # 0.5% per trade
    max_daily_loss_pct   = 0.045      # 4.5% daily limit (پراپ)
    max_total_dd_pct     = 0.09       # 9% max DD (زیر 10% پراپ)
    profit_target_pct    = 10.0       # بدون سقف عملی
    spread_eur_pips      = 1.0
    spread_gbp_pips      = 1.2
    commission_per_lot   = 6.0
    pip                  = 0.0001
    lot_size             = 100_000
    max_lot              = 2.0
    atr_period           = 14
    min_rr               = 1.8       # حداقل RR قابل قبول


# ================================================================== #
#                        DATA LOADER                                 #
# ================================================================== #
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur or not files_gbp:
        raise FileNotFoundError("CSV files not found in data/")

    def read(paths, suffix):
        frames = []
        for p in paths:
            df = pd.read_csv(p, sep=';', header=None,
                             names=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            df = df.set_index('ts')
            df = df[~df.index.duplicated(keep='last')]
            df.columns = [f'{c}_{suffix}' for c in df.columns]
            frames.append(df)
        return pd.concat(frames).sort_index()

    eur = read(files_eur, 'eur')
    gbp = read(files_gbp, 'gbp')
    raw = eur.join(gbp, how='inner').dropna()

    df = pd.DataFrame({
        'o_eur': raw['o_eur'].resample('15min').first(),
        'h_eur': raw['h_eur'].resample('15min').max(),
        'l_eur': raw['l_eur'].resample('15min').min(),
        'c_eur': raw['c_eur'].resample('15min').last(),
        'v_eur': raw['v_eur'].resample('15min').sum(),
        'o_gbp': raw['o_gbp'].resample('15min').first(),
        'h_gbp': raw['h_gbp'].resample('15min').max(),
        'l_gbp': raw['l_gbp'].resample('15min').min(),
        'c_gbp': raw['c_gbp'].resample('15min').last(),
        'v_gbp': raw['v_gbp'].resample('15min').sum(),
    }).dropna()

    df = df[df.index.weekday < 5]
    print(f"✅ {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ================================================================== #
#                      INDICATORS                                    #
# ================================================================== #
def calc_atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(close, period=14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_adx(high, low, close, period=14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_n = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_s = tr.rolling(period).sum()
    di_p = 100 * dm_p.rolling(period).sum() / atr_s.replace(0, np.nan)
    di_n = 100 * dm_n.rolling(period).sum() / atr_s.replace(0, np.nan)
    dx = (abs(di_p - di_n) / (di_p + di_n).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig


def calc_bbands(close, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + mult * std, mid, mid - mult * std


def calc_stochastic(high, low, close, k_period=14, d_period=3):
    lowest = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


# ================================================================== #
#                     UTILITY FUNCTIONS                              #
# ================================================================== #
def trade_cost(lot: float, symbol: str) -> float:
    sp = Config.spread_eur_pips if symbol == 'EUR' else Config.spread_gbp_pips
    return sp * Config.pip * lot * Config.lot_size + Config.commission_per_lot * lot


def calc_pnl(direction, lot, entry, exit_p, symbol) -> float:
    raw = direction * (exit_p - entry) * lot * Config.lot_size
    return raw - trade_cost(lot, symbol)


def lot_size_calc(equity, sl_pips) -> float:
    if sl_pips <= 0:
        return 0.01
    risk_usd = equity * Config.risk_per_trade_pct
    lot = risk_usd / (sl_pips * Config.pip * Config.lot_size)
    return round(np.clip(lot, 0.01, Config.max_lot), 2)


# ================================================================== #
#                      RISK MANAGER                                  #
# ================================================================== #
class RiskManager:
    def __init__(self, name="Portfolio"):
        self.name = name
        self.equity = Config.initial_balance
        self.peak = Config.initial_balance
        self.day_start_eq = Config.initial_balance
        self.cur_day = None
        self.halted = False
        self.halt_reason = "Running"
        self.curve = [Config.initial_balance]
        self.curve_ts = [None]
        self.daily_pnl = {}

    def new_bar(self, ts: pd.Timestamp):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day = day
            self.day_start_eq = self.equity
            # روزانه ریست نمی‌شود - فقط DD کل چک می‌شود

    def check_daily_loss(self) -> bool:
        daily_dd = (self.equity - self.day_start_eq) / self.day_start_eq
        if daily_dd <= -Config.max_daily_loss_pct:
            return False
        return True

    def add_pnl(self, amount: float, ts: pd.Timestamp) -> bool:
        self.equity += amount
        self.peak = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))
        self.curve_ts.append(ts)

        d_key = str(ts.date())
        self.daily_pnl[d_key] = self.daily_pnl.get(d_key, 0) + amount

        # Daily loss check
        if not self.check_daily_loss():
            self.halted = True
            daily_dd = (self.equity - self.day_start_eq) / self.day_start_eq
            self.halt_reason = f"Daily Loss {daily_dd * 100:.1f}%"
            return False

        # Total DD check
        total_dd = (self.equity - self.peak) / self.peak
        if total_dd <= -Config.max_total_dd_pct:
            self.halted = True
            self.halt_reason = f"Max DD {total_dd * 100:.1f}%"
            return False

        return True

    def can_trade(self) -> bool:
        """بررسی آیا مجاز به ترید هستیم"""
        if self.halted:
            return False
        if not self.check_daily_loss():
            return False
        return True

    @property
    def max_dd(self):
        s = pd.Series(self.curve)
        return ((s - s.cummax()) / s.cummax() * 100).min()

    @property
    def max_dd_abs(self):
        s = pd.Series(self.curve)
        return (s - s.cummax()).min()

    @property
    def sharpe(self):
        r = pd.Series(self.curve).pct_change().dropna()
        if r.std() == 0:
            return 0
        return r.mean() / r.std() * np.sqrt(252 * 96)

    @property
    def sortino(self):
        r = pd.Series(self.curve).pct_change().dropna()
        neg = r[r < 0]
        ds = neg.std() if len(neg) > 0 else 1e-10
        if ds == 0:
            return 0
        return r.mean() / ds * np.sqrt(252 * 96)

    @property
    def calmar(self):
        ret = (self.equity / Config.initial_balance - 1)
        dd = abs(self.max_dd / 100)
        return ret / dd if dd > 0 else 0


# ================================================================== #
#        STRATEGY 1: Correlation Arbitrage (اصلاح شده)              #
#                                                                    #
#  تنها استراتژی سودده از نسخه قبل (WR=75%, PF=4.48)              #
#  بهبود: فیلترهای بیشتر + مدیریت پوزیشن بهتر                     #
# ================================================================== #
def strategy_corr_arb(df: pd.DataFrame) -> tuple:
    """
    Mean-reversion روی نسبت EUR/GBP
    وقتی Z-score از 2 رد شد → برگشت به میانگین
    """
    eurgbp = df['c_eur'] / df['c_gbp']
    period = 96  # 1 روز کاری (96 کندل 15 دقیقه‌ای)

    mean = eurgbp.rolling(period).mean()
    std = eurgbp.rolling(period).std()
    z = (eurgbp - mean) / std.replace(0, np.nan)

    # فیلتر نوسان کافی
    std_ok = std > std.rolling(period * 5).mean() * 0.35

    # ADX هر دو جفت‌ارز - رنج باشد نه ترند
    adx_eur = calc_adx(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    adx_gbp = calc_adx(df['h_gbp'], df['l_gbp'], df['c_gbp'], 14)
    adx_ok = (adx_eur < 30) & (adx_gbp < 30)

    # RSI نه اشباع خرید/فروش شدید (اگر RSI=90 شده احتمالا ترند قوی)
    rsi_eur = calc_rsi(df['c_eur'], 14)
    rsi_ok = rsi_eur.between(25, 75)

    # ساعات فعال: لندن + NY overlap
    hour = pd.Series(df.index.hour, index=df.index)
    time_ok = hour.between(8, 18)

    # Volume filter: نه وقتی بازار مرده
    vol_ma = df['v_eur'].rolling(96).mean()
    vol_ok = df['v_eur'] > vol_ma * 0.3

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['sl_pips'] = 0.0
    signals['tp_pips'] = 0.0

    # Z > 2.2 → Short (EUR overvalued vs GBP)
    short_cond = (z > 2.2) & std_ok & adx_ok & rsi_ok & time_ok & vol_ok
    # Z < -2.2 → Long
    long_cond = (z < -2.2) & std_ok & adx_ok & rsi_ok & time_ok & vol_ok

    signals.loc[short_cond, 'signal'] = -1
    signals.loc[long_cond, 'signal'] = 1
    signals.loc[short_cond, 'sl_pips'] = 22.0
    signals.loc[long_cond, 'sl_pips'] = 22.0
    signals.loc[short_cond, 'tp_pips'] = 40.0
    signals.loc[long_cond, 'tp_pips'] = 40.0

    # حذف سیگنال‌های تکراری پشت سر هم
    mask = signals['signal'] != signals['signal'].shift()
    signals.loc[~mask, 'signal'] = 0

    return signals, z


# ================================================================== #
#        STRATEGY 2: RSI Divergence + Structure                     #
#                                                                    #
#  واگرایی RSI + شکست ساختار = یکی از قابل‌اعتمادترین ستاپ‌ها      #
#  WR بالا چون فقط وقتی واگرایی تایید شده وارد می‌شویم             #
# ================================================================== #
def strategy_rsi_divergence(df: pd.DataFrame) -> tuple:
    """
    RSI Bullish/Bearish Divergence with structure confirmation
    - قیمت lower low ولی RSI higher low → Bullish
    - قیمت higher high ولی RSI lower high → Bearish
    + تایید با کندل و EMA
    """
    c = df['c_eur']
    h = df['h_eur']
    l = df['l_eur']
    o = df['o_eur']

    rsi = calc_rsi(c, 14)
    atr = calc_atr(h, l, c, 14)
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    hour = pd.Series(df.index.hour, index=df.index)
    active = hour.between(7, 18)

    # پیدا کردن swing points (lookback=10 کندل = 2.5 ساعت)
    lb = 10
    swing_low = l.rolling(lb * 2 + 1, center=True).min()
    swing_high = h.rolling(lb * 2 + 1, center=True).max()

    is_swing_low = (l == swing_low) & (l < l.shift(1)) & (l < l.shift(-1))
    is_swing_high = (h == swing_high) & (h > h.shift(1)) & (h > h.shift(-1))

    # ذخیره آخرین swing
    last_swing_low_price = l.where(is_swing_low).ffill()
    last_swing_low_rsi = rsi.where(is_swing_low).ffill()
    prev_swing_low_price = last_swing_low_price.shift(lb)
    prev_swing_low_rsi = last_swing_low_rsi.shift(lb)

    last_swing_high_price = h.where(is_swing_high).ffill()
    last_swing_high_rsi = rsi.where(is_swing_high).ffill()
    prev_swing_high_price = last_swing_high_price.shift(lb)
    prev_swing_high_rsi = last_swing_high_rsi.shift(lb)

    # Bullish Divergence: price lower low + RSI higher low
    bull_div = (
            (last_swing_low_price < prev_swing_low_price) &
            (last_swing_low_rsi > prev_swing_low_rsi + 3) &  # RSI حداقل 3 واحد بالاتر
            (rsi < 40) &  # RSI در ناحیه oversold
            (rsi > rsi.shift(1)) &  # RSI شروع به بالا رفتن کرده
            (c > o) &  # کندل صعودی (تایید)
            active
    )

    # Bearish Divergence: price higher high + RSI lower high
    bear_div = (
            (last_swing_high_price > prev_swing_high_price) &
            (last_swing_high_rsi < prev_swing_high_rsi - 3) &
            (rsi > 60) &
            (rsi < rsi.shift(1)) &
            (c < o) &
            active
    )

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['sl_pips'] = 0.0
    signals['tp_pips'] = 0.0

    for idx in df.index:
        i = df.index.get_loc(idx)
        if i < 250:
            continue

        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue

        atr_pips = atr_v / Config.pip

        if bull_div.iloc[i]:
            sl_p = max(15, min(atr_pips * 1.5, 30))
            tp_p = sl_p * 2.2
            signals.iloc[i, signals.columns.get_loc('signal')] = 1
            signals.iloc[i, signals.columns.get_loc('sl_pips')] = sl_p
            signals.iloc[i, signals.columns.get_loc('tp_pips')] = tp_p

        elif bear_div.iloc[i]:
            sl_p = max(15, min(atr_pips * 1.5, 30))
            tp_p = sl_p * 2.2
            signals.iloc[i, signals.columns.get_loc('signal')] = -1
            signals.iloc[i, signals.columns.get_loc('sl_pips')] = sl_p
            signals.iloc[i, signals.columns.get_loc('tp_pips')] = tp_p

    # حذف تکراری
    mask = signals['signal'] != signals['signal'].shift()
    signals.loc[~mask, 'signal'] = 0

    # حداکثر 1 سیگنال هر 4 ساعت
    nz = signals[signals['signal'] != 0].copy()
    if len(nz) > 0:
        keep = [nz.index[0]]
        for idx in nz.index[1:]:
            if (idx - keep[-1]).total_seconds() >= 4 * 3600:
                keep.append(idx)
        drop = [idx for idx in nz.index if idx not in keep]
        signals.loc[drop, 'signal'] = 0

    return signals, rsi


# ================================================================== #
#        STRATEGY 3: Session Momentum Continuation                  #
#                                                                    #
#  مومنتوم لندن: اگر ۲ ساعت اول لندن جهت مشخص داشت               #
#  + تایید حجم + EMA → ادامه حرکت تا پایان سشن                    #
# ================================================================== #
def strategy_session_momentum(df: pd.DataFrame) -> tuple:
    """
    اگر بازار لندن (7-9 GMT) یک حرکت قوی داشته:
    - جهت: بالای 60% رنج روز قبل
    - تایید: EMA21 > EMA50 (یا برعکس)
    - RSI هم‌جهت
    → ورود در pullback ساعت 9-10 برای ادامه حرکت تا 14-15 GMT
    """
    c = df['c_eur']
    h = df['h_eur']
    l = df['l_eur']
    o = df['o_eur']

    ema21 = c.ewm(span=21, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    rsi = calc_rsi(c, 14)
    atr = calc_atr(h, l, c, 14)
    adx = calc_adx(h, l, c, 14)
    _, _, macd_hist = calc_macd(c)
    stoch_k, stoch_d = calc_stochastic(h, l, c)

    hour = pd.Series(df.index.hour, index=df.index)
    date_s = pd.Series(df.index.date, index=df.index)

    # محاسبه رنج اول لندن (7:00-8:45)
    early_london = df[(hour >= 7) & (hour <= 8)]
    london_range = early_london.groupby(date_s[early_london.index]).agg(
        london_h=('h_eur', 'max'),
        london_l=('l_eur', 'min'),
        london_c=('c_eur', 'last'),
        london_o=('o_eur', 'first')
    )
    london_range['london_dir'] = np.sign(
        london_range['london_c'] - london_range['london_o']
    )
    london_range['london_rng'] = london_range['london_h'] - london_range['london_l']

    # رنج روز قبل
    daily = df.groupby(date_s).agg(
        day_h=('h_eur', 'max'),
        day_l=('l_eur', 'min')
    )
    daily['prev_rng'] = (daily['day_h'] - daily['day_l']).shift(1)

    # Join
    d = df.copy()
    d['date'] = d.index.date
    d = d.join(london_range, on='date')
    d = d.join(daily[['prev_rng']], on='date')

    signals = pd.DataFrame(index=df.index)
    signals['signal'] = 0
    signals['sl_pips'] = 0.0
    signals['tp_pips'] = 0.0

    # ورود: ساعت 9-11 GMT (بعد از تشکیل momentum اولیه)
    entry_time = hour.between(9, 11)
    weekday = pd.Series(df.index.weekday, index=df.index)
    weekday_ok = weekday.between(0, 3)

    for idx in df.index:
        i = df.index.get_loc(idx)
        if i < 300:
            continue
        if not entry_time.iloc[i] or not weekday_ok.iloc[i]:
            continue

        try:
            l_dir = d['london_dir'].iloc[i]
            l_rng = d['london_rng'].iloc[i]
            p_rng = d['prev_rng'].iloc[i]
        except (KeyError, IndexError):
            continue

        if pd.isna(l_dir) or pd.isna(l_rng) or pd.isna(p_rng) or l_dir == 0:
            continue

        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue

        atr_pips = atr_v / Config.pip

        # فیلتر: حرکت لندن باید قابل توجه باشد
        # حداقل 40% رنج روز قبل
        if p_rng <= 0 or l_rng < p_rng * 0.35:
            continue

        # ADX > 20: بازار ترند دارد
        if pd.isna(adx.iloc[i]) or adx.iloc[i] < 20:
            continue

        # EMA alignment
        ema21_v = ema21.iloc[i]
        ema50_v = ema50.iloc[i]
        rsi_v = rsi.iloc[i]
        close_v = c.iloc[i]
        macd_v = macd_hist.iloc[i]

        if l_dir > 0:  # London bullish
            if not (ema21_v > ema50_v):
                continue
            if not (rsi_v > 45 and rsi_v < 70):
                continue
            if pd.notna(macd_v) and macd_v < 0:
                continue
            # Pullback: قیمت بین EMA21 و london_high
            london_h = d['london_h'].iloc[i]
            if pd.notna(london_h) and close_v > london_h:
                continue  # قبلا رد شده، pullback نیست

            sl_p = max(12, min(atr_pips * 1.3, 25))
            tp_p = sl_p * 2.5
            signals.iloc[i, signals.columns.get_loc('signal')] = 1
            signals.iloc[i, signals.columns.get_loc('sl_pips')] = sl_p
            signals.iloc[i, signals.columns.get_loc('tp_pips')] = tp_p

        elif l_dir < 0:  # London bearish
            if not (ema21_v < ema50_v):
                continue
            if not (rsi_v < 55 and rsi_v > 30):
                continue
            if pd.notna(macd_v) and macd_v > 0:
                continue
            london_l = d['london_l'].iloc[i]
            if pd.notna(london_l) and close_v < london_l:
                continue

            sl_p = max(12, min(atr_pips * 1.3, 25))
            tp_p = sl_p * 2.5
            signals.iloc[i, signals.columns.get_loc('signal')] = -1
            signals.iloc[i, signals.columns.get_loc('sl_pips')] = sl_p
            signals.iloc[i, signals.columns.get_loc('tp_pips')] = tp_p

    # فقط 1 سیگنال در روز
    nz = signals[signals['signal'] != 0].copy()
    if len(nz) > 0:
        first_per_day = nz.groupby(nz.index.date).head(1).index
        drop_idx = [idx for idx in nz.index if idx not in first_per_day]
        signals.loc[drop_idx, 'signal'] = 0

    return signals, atr


# ================================================================== #
#                    BACKTEST ENGINE (INDEPENDENT)                   #
# ================================================================== #
def run_single_strategy(df: pd.DataFrame, strategy_name: str,
                        signals: pd.DataFrame, symbol: str = 'EUR',
                        use_trailing: bool = True,
                        time_stop_hours: int = 72) -> tuple:
    """
    بک‌تست مستقل برای یک استراتژی
    """
    risk = RiskManager(name=strategy_name)
    trades = []
    position = None
    warmup = 300

    atr = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)

    for i in range(warmup, len(df)):
        ts = df.index[i]
        row = df.iloc[i]

        if symbol == 'EUR':
            hi, lo, cp = row['h_eur'], row['l_eur'], row['c_eur']
        else:
            hi, lo, cp = row['h_gbp'], row['l_gbp'], row['c_gbp']

        risk.new_bar(ts)

        if risk.halted:
            if position is not None:
                p_pnl = calc_pnl(position['dir'], position['lot'],
                                 position['entry'], cp, symbol)
                trades.append({
                    **position, 'exit': cp, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': 'halt_close'
                })
                risk.add_pnl(p_pnl, ts)
                position = None
            continue

        # ── CHECK EXIT ──
        if position is not None:
            d_dir = position['dir']
            entry = position['entry']
            sl = position['sl']
            tp = position['tp']

            hit_sl = (d_dir == 1 and lo <= sl) or (d_dir == -1 and hi >= sl)
            hit_tp = (d_dir == 1 and hi >= tp) or (d_dir == -1 and lo <= tp)

            # Trailing Stop
            if use_trailing:
                atr_v = atr.iloc[i]
                if pd.notna(atr_v) and atr_v > 0:
                    move = d_dir * (cp - entry)
                    # Break-even after 1 ATR profit
                    if move > atr_v * 1.0:
                        be = entry + d_dir * atr_v * 0.2
                        if d_dir == 1:
                            position['sl'] = max(position['sl'], be)
                        else:
                            position['sl'] = min(position['sl'], be)
                    # Lock 50% after 1.5 ATR
                    if move > atr_v * 1.5:
                        lock = entry + d_dir * atr_v * 0.7
                        if d_dir == 1:
                            position['sl'] = max(position['sl'], lock)
                        else:
                            position['sl'] = min(position['sl'], lock)

                    sl = position['sl']
                    hit_sl = ((d_dir == 1 and lo <= sl) or
                              (d_dir == -1 and hi >= sl))

            # Time stop
            elapsed = (ts - position['entry_ts']).total_seconds() / 3600
            if elapsed >= time_stop_hours and not hit_tp:
                p_pnl = calc_pnl(d_dir, position['lot'],
                                 entry, cp, symbol)
                trades.append({
                    **position, 'exit': cp, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': 'TimeStop'
                })
                risk.add_pnl(p_pnl, ts)
                position = None
                continue

            # End of week close (جمعه ساعت 20)
            if ts.weekday() == 4 and ts.hour >= 20:
                p_pnl = calc_pnl(d_dir, position['lot'],
                                 entry, cp, symbol)
                trades.append({
                    **position, 'exit': cp, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': 'WeekEnd'
                })
                risk.add_pnl(p_pnl, ts)
                position = None
                continue

            exit_r = None
            exit_p = None
            if hit_sl and hit_tp:
                # اگر هر دو خورده، فرض SL اول (محافظه‌کارانه)
                exit_r = 'SL'
                exit_p = sl
            elif hit_sl:
                exit_r = 'SL'
                exit_p = sl
            elif hit_tp:
                exit_r = 'TP'
                exit_p = tp

            if exit_r:
                p_pnl = calc_pnl(d_dir, position['lot'],
                                 entry, exit_p, symbol)
                trades.append({
                    **position, 'exit': exit_p, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': exit_r
                })
                risk.add_pnl(p_pnl, ts)
                position = None

        # ── CHECK ENTRY ──
        if position is None and risk.can_trade():
            sig_val = signals['signal'].iloc[i]
            if sig_val != 0:
                sv = int(sig_val)
                sl_pips = signals['sl_pips'].iloc[i]
                tp_pips = signals['tp_pips'].iloc[i]

                if sl_pips <= 0 or tp_pips <= 0:
                    continue

                # RR check
                if tp_pips / sl_pips < Config.min_rr:
                    continue

                lot = lot_size_calc(risk.equity, sl_pips)
                spread = (Config.spread_eur_pips if symbol == 'EUR'
                          else Config.spread_gbp_pips)
                half_sp = spread * Config.pip / 2
                ep = cp + sv * half_sp

                position = dict(
                    strategy=strategy_name, symbol=symbol,
                    dir=sv, lot=lot, entry=ep,
                    sl=ep - sv * sl_pips * Config.pip,
                    tp=ep + sv * tp_pips * Config.pip,
                    entry_ts=ts,
                )

    # Close remaining position
    if position is not None:
        last_ts = df.index[-1]
        last_p = df['c_eur'].iloc[-1] if symbol == 'EUR' else df['c_gbp'].iloc[-1]
        p_pnl = calc_pnl(position['dir'], position['lot'],
                         position['entry'], last_p, symbol)
        trades.append({
            **position, 'exit': last_p, 'exit_ts': last_ts,
            'pnl': p_pnl, 'status': 'eod_close'
        })
        risk.add_pnl(p_pnl, last_ts)

    return trades, risk


# ================================================================== #
#                    COMBINED BACKTEST                                #
# ================================================================== #
def run_combined_backtest(df: pd.DataFrame,
                          strategies: dict) -> tuple:
    """
    اجرای ترکیبی استراتژی‌ها با یک حساب مشترک
    """
    risk = RiskManager(name="Combined")
    all_trades = []
    open_positions = {}
    warmup = 300

    atr = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)

    for i in range(warmup, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        c_eur = row['c_eur']
        h_eur = row['h_eur']
        l_eur = row['l_eur']

        risk.new_bar(ts)

        if risk.halted:
            for key in list(open_positions.keys()):
                p = open_positions.pop(key)
                p_pnl = calc_pnl(p['dir'], p['lot'],
                                 p['entry'], c_eur, 'EUR')
                all_trades.append({
                    **p, 'exit': c_eur, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': 'halt_close'
                })
                risk.add_pnl(p_pnl, ts)
            continue

        # ── EXIT CHECK ──
        for key in list(open_positions.keys()):
            p = open_positions[key]
            d_dir = p['dir']
            entry = p['entry']

            hit_sl = ((d_dir == 1 and l_eur <= p['sl']) or
                      (d_dir == -1 and h_eur >= p['sl']))
            hit_tp = ((d_dir == 1 and h_eur >= p['tp']) or
                      (d_dir == -1 and l_eur <= p['tp']))

            # Trailing
            atr_v = atr.iloc[i]
            if pd.notna(atr_v) and atr_v > 0:
                move = d_dir * (c_eur - entry)
                if move > atr_v * 1.0:
                    be = entry + d_dir * atr_v * 0.2
                    if d_dir == 1:
                        p['sl'] = max(p['sl'], be)
                    else:
                        p['sl'] = min(p['sl'], be)
                if move > atr_v * 1.5:
                    lock = entry + d_dir * atr_v * 0.7
                    if d_dir == 1:
                        p['sl'] = max(p['sl'], lock)
                    else:
                        p['sl'] = min(p['sl'], lock)

                hit_sl = ((d_dir == 1 and l_eur <= p['sl']) or
                          (d_dir == -1 and h_eur >= p['sl']))

            # Time stop
            elapsed = (ts - p['entry_ts']).total_seconds() / 3600
            max_hours = {'CorrArb': 72, 'RSI_Div': 48,
                         'SessionMom': 36}
            max_h = max_hours.get(p['strategy'], 48)
            if elapsed >= max_h:
                p_pnl = calc_pnl(d_dir, p['lot'], entry, c_eur, 'EUR')
                all_trades.append({
                    **p, 'exit': c_eur, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': 'TimeStop'
                })
                risk.add_pnl(p_pnl, ts)
                del open_positions[key]
                continue

            # Weekend close
            if ts.weekday() == 4 and ts.hour >= 20:
                p_pnl = calc_pnl(d_dir, p['lot'], entry, c_eur, 'EUR')
                all_trades.append({
                    **p, 'exit': c_eur, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': 'WeekEnd'
                })
                risk.add_pnl(p_pnl, ts)
                del open_positions[key]
                continue

            exit_r = None
            exit_p = None
            if hit_sl:
                exit_r = 'SL'
                exit_p = p['sl']
            elif hit_tp:
                exit_r = 'TP'
                exit_p = p['tp']

            if exit_r:
                p_pnl = calc_pnl(d_dir, p['lot'], entry, exit_p, 'EUR')
                all_trades.append({
                    **p, 'exit': exit_p, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': exit_r
                })
                risk.add_pnl(p_pnl, ts)
                del open_positions[key]

        # ── ENTRY CHECK ──
        if not risk.can_trade():
            continue

        max_positions = 3  # حداکثر 3 پوزیشن همزمان
        if len(open_positions) >= max_positions:
            continue

        for strat_name, sigs in strategies.items():
            if strat_name in open_positions:
                continue
            if len(open_positions) >= max_positions:
                break

            sig_val = sigs['signal'].iloc[i]
            if sig_val == 0:
                continue

            sv = int(sig_val)
            sl_pips = sigs['sl_pips'].iloc[i]
            tp_pips = sigs['tp_pips'].iloc[i]

            if sl_pips <= 0 or tp_pips <= 0:
                continue
            if tp_pips / sl_pips < Config.min_rr:
                continue

            lot = lot_size_calc(risk.equity, sl_pips)
            half_sp = Config.spread_eur_pips * Config.pip / 2
            ep = c_eur + sv * half_sp

            open_positions[strat_name] = dict(
                strategy=strat_name, symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

    # Close remaining
    last_ts = df.index[-1]
    last_p = df['c_eur'].iloc[-1]
    for key, p in open_positions.items():
        p_pnl = calc_pnl(p['dir'], p['lot'], p['entry'], last_p, 'EUR')
        all_trades.append({
            **p, 'exit': last_p, 'exit_ts': last_ts,
            'pnl': p_pnl, 'status': 'eod_close'
        })
        risk.add_pnl(p_pnl, last_ts)

    return all_trades, risk


# ================================================================== #
#                    REPORT GENERATOR                                #
# ================================================================== #
def generate_report(trades: list, risk: RiskManager, title: str = ""):
    if not trades:
        print(f"\n❌ [{title}] No trades!")
        return None

    t = pd.DataFrame(trades)
    t['pnl'] = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts'] = pd.to_datetime(t['entry_ts'])
    t['exit_ts'] = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    start_d = t['entry_ts'].min()
    end_d = t['exit_ts'].max()
    total_days = max((end_d - start_d).days, 1)
    total_months = total_days / 30.44
    total_years = total_days / 365.25

    final_eq = risk.equity
    total_pnl = final_eq - Config.initial_balance
    total_ret = total_pnl / Config.initial_balance * 100
    ann_ret = (((final_eq / Config.initial_balance) **
                (365.25 / total_days) - 1) * 100) if total_days > 1 else 0

    win_t = t[t['pnl'] > 0]
    loss_t = t[t['pnl'] < 0]
    win_r = len(win_t) / len(t) * 100 if len(t) > 0 else 0
    avg_w = win_t['pnl'].mean() if len(win_t) > 0 else 0
    avg_l = loss_t['pnl'].mean() if len(loss_t) > 0 else 0
    gw = win_t['pnl'].sum()
    gl = abs(loss_t['pnl'].sum())
    pf = gw / gl if gl > 0 else float('inf')
    exp = t['pnl'].mean()
    rr = abs(avg_w / avg_l) if avg_l != 0 else 0

    # Monthly return
    monthly_ret = total_ret / total_months if total_months > 0 else 0

    # Consecutive
    sign = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cw, cl, mcw, mcl = 0, 0, 0, 0
    for s in sign:
        if s > 0:
            cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0:
            cl += 1; cw = 0; mcl = max(mcl, cl)
        else:
            cw = cl = 0

    W = 72
    SEP = "═" * W

    def rw(label, value):
        lbl = f"  {label}"
        val = str(value)
        dots = "·" * max(2, W - len(lbl) - len(val) - 2)
        return f"{lbl} {dots} {val}"

    def box_top(t_):
        inner = f"─ {t_} "
        return "┌" + inner + "─" * (W - len(inner) - 1) + "┐"

    def box_bot():
        return "└" + "─" * (W - 1) + "┘"

    # Prop Fitness Score
    prop_score = 0
    if win_r >= 40:
        prop_score += 20
    if pf >= 1.3:
        prop_score += 20
    if abs(risk.max_dd) <= 5:
        prop_score += 25
    elif abs(risk.max_dd) <= 8:
        prop_score += 15
    if monthly_ret >= 10:
        prop_score += 25
    elif monthly_ret >= 5:
        prop_score += 15
    if rr >= 1.5:
        prop_score += 10

    prop_grade = "F"
    if prop_score >= 90:
        prop_grade = "A+"
    elif prop_score >= 80:
        prop_grade = "A"
    elif prop_score >= 70:
        prop_grade = "B+"
    elif prop_score >= 60:
        prop_grade = "B"
    elif prop_score >= 50:
        prop_grade = "C"
    elif prop_score >= 40:
        prop_grade = "D"

    lines = [
        "", SEP,
        f"  {'▌':>6}  {title}  ▐",
        SEP, "",
        box_top("General"),
        rw("Period", f"{start_d.date()} → {end_d.date()}"),
        rw("Total Days", f"{total_days:,}"),
        rw("Total Trades", f"{len(t):,}"),
        rw("Trades/Week", f"{len(t) / (total_days / 7):.1f}"),
        box_bot(), "",

        box_top("Financial Results"),
        rw("Initial Balance", f"${Config.initial_balance:,.2f}"),
        rw("Final Balance", f"${final_eq:,.2f}"),
        rw("Total PnL", f"${total_pnl:+,.2f}"),
        rw("Total Return", f"{total_ret:+.2f}%"),
        rw("Monthly Return (avg)", f"{monthly_ret:+.2f}%"),
        rw("Annualized Return", f"{ann_ret:+.2f}%"),
        rw("Best Trade", f"${t['pnl'].max():+.2f}"),
        rw("Worst Trade", f"${t['pnl'].min():+.2f}"),
        box_bot(), "",

        box_top("Risk Metrics"),
        rw("Max Drawdown %", f"{risk.max_dd:.2f}%"),
        rw("Max Drawdown $", f"${risk.max_dd_abs:+.2f}"),
        rw("Sharpe Ratio", f"{risk.sharpe:.2f}"),
        rw("Sortino Ratio", f"{risk.sortino:.2f}"),
        rw("Calmar Ratio", f"{risk.calmar:.2f}"),
        rw("Profit Factor", f"{pf:.2f}"),
        rw("Status", risk.halt_reason),
        box_bot(), "",

        box_top("Trade Statistics"),
        rw("Win Rate", f"{win_r:.1f}%"),
        rw("Winning Trades", f"{len(win_t):,}"),
        rw("Losing Trades", f"{len(loss_t):,}"),
        rw("Avg Win", f"${avg_w:+.2f}"),
        rw("Avg Loss", f"${avg_l:+.2f}"),
        rw("Reward:Risk", f"{rr:.2f}"),
        rw("Expectancy", f"${exp:+.2f}"),
        rw("Max Consec Wins", f"{mcw}"),
        rw("Max Consec Losses", f"{mcl}"),
        rw("Avg Duration", f"{t['duration_min'].mean():.0f} min"),
        box_bot(), "",

        box_top("Prop Fitness"),
        rw("Prop Score", f"{prop_score}/100"),
        rw("Grade", prop_grade),
        rw("DD < 5%?", "✅" if abs(risk.max_dd) <= 5 else "❌"),
        rw("DD < 10%?", "✅" if abs(risk.max_dd) <= 10 else "❌"),
        rw("Monthly > 10%?", "✅" if monthly_ret >= 10 else "❌"),
        rw("PF > 1.3?", "✅" if pf >= 1.3 else "❌"),
        rw("WR > 40%?", "✅" if win_r >= 40 else "❌"),
        box_bot(), "",
    ]

    # Exit distribution
    lines.append(box_top("Exit Distribution"))
    for status, cnt in t['status'].value_counts().items():
        pct = cnt / len(t) * 100
        avg_p = t.loc[t['status'] == status, 'pnl'].mean()
        bar = "█" * max(1, int(pct / 2.5))
        lines.append(
            f"  {status:<13} {cnt:>5} ({pct:>5.1f}%)  "
            f"{bar:<28}  avg=${avg_p:>+.2f}"
        )
    lines.append(box_bot())

    # Monthly breakdown
    t['ym'] = t['entry_ts'].dt.to_period('M')
    monthly = (t.groupby('ym')
               .agg(n=('pnl', 'count'), pnl=('pnl', 'sum'),
                    wins=('pnl', lambda x: (x > 0).sum()),
                    best=('pnl', 'max'), worst=('pnl', 'min'))
               .reset_index())
    monthly['wr'] = monthly['wins'] / monthly['n'] * 100
    monthly['ret'] = monthly['pnl'] / Config.initial_balance * 100
    monthly['cum_pnl'] = monthly['pnl'].cumsum()
    monthly['cum_ret'] = monthly['cum_pnl'] / Config.initial_balance * 100

    lines += ["", box_top("Monthly Report")]
    lines.append(
        f"  {'Month':>7}  {'#':>4}  {'WR%':>5}  {'PnL':>10}  "
        f"{'Ret%':>6}  {'CumPnL':>10}  {'CumRet':>7}"
    )
    lines.append("  " + "─" * (W - 3))

    positive_months = 0
    negative_months = 0
    for _, mr in monthly.iterrows():
        arrow = "▲" if mr['pnl'] >= 0 else "▼"
        if mr['pnl'] >= 0:
            positive_months += 1
        else:
            negative_months += 1
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>4}  "
            f"{mr['wr']:>4.0f}%  ${mr['pnl']:>9.2f}  "
            f"{mr['ret']:>+5.1f}%  ${mr['cum_pnl']:>9.2f}  "
            f"{mr['cum_ret']:>+6.1f}% {arrow}"
        )

    lines.append("  " + "─" * (W - 3))
    lines.append(
        f"  Profitable Months: {positive_months}/{len(monthly)} "
        f"({positive_months / len(monthly) * 100:.0f}%)"
    )
    lines += [box_bot(), ""]

    # Yearly breakdown
    t['yr'] = t['entry_ts'].dt.year
    yearly = (t.groupby('yr')
              .agg(n=('pnl', 'count'), pnl=('pnl', 'sum'),
                   wins=('pnl', lambda x: (x > 0).sum()))
              .reset_index())
    yearly['wr'] = yearly['wins'] / yearly['n'] * 100
    yearly['ret'] = yearly['pnl'] / Config.initial_balance * 100

    lines.append(box_top("Yearly Report"))
    lines.append(
        f"  {'Year':>5}  {'#':>5}  {'WR%':>5}  {'PnL':>10}  {'Ret%':>7}"
    )
    lines.append("  " + "─" * (W - 3))
    for _, yr in yearly.iterrows():
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>4.0f}%  ${yr['pnl']:>9.2f}  "
            f"{yr['ret']:>+6.1f}%"
        )
    lines += [box_bot(), "", SEP]

    output = "\n".join(lines)
    print(output)

    # Return stats dict
    return {
        'name': title,
        'trades': len(t),
        'total_pnl': total_pnl,
        'total_ret': total_ret,
        'monthly_ret': monthly_ret,
        'win_rate': win_r,
        'pf': pf,
        'rr': rr,
        'max_dd': risk.max_dd,
        'sharpe': risk.sharpe,
        'sortino': risk.sortino,
        'calmar': risk.calmar,
        'exp': exp,
        'prop_score': prop_score,
        'prop_grade': prop_grade,
        'positive_months': positive_months,
        'total_months': len(monthly),
        'output': output,
        'risk': risk,
        'trades_df': t,
        'monthly_df': monthly,
    }


# ================================================================== #
#                    SAVE RESULTS                                    #
# ================================================================== #
def save_results(results: list, combined_result=None):
    """ذخیره تمام نتایج در فایل"""

    # ── Full Report TXT ──
    with open("Backtest_Report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  PROFESSIONAL BACKTEST REPORT\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 72 + "\n\n")

        f.write("=" * 72 + "\n")
        f.write("  INDIVIDUAL STRATEGY RESULTS\n")
        f.write("=" * 72 + "\n")

        for r in results:
            if r:
                f.write(r['output'])
                f.write("\n\n")

        if combined_result:
            f.write("\n" + "=" * 72 + "\n")
            f.write("  COMBINED PORTFOLIO RESULTS\n")
            f.write("=" * 72 + "\n")
            f.write(combined_result['output'])

        # Summary comparison
        f.write("\n\n" + "=" * 72 + "\n")
        f.write("  STRATEGY COMPARISON SUMMARY\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"  {'Strategy':<20} {'Trades':>6} {'WR%':>6} "
                f"{'PF':>6} {'RR':>5} {'DD%':>7} "
                f"{'MonRet%':>8} {'PnL':>10} {'Score':>6}\n")
        f.write("  " + "─" * 68 + "\n")
        for r in results:
            if r:
                pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "  ∞"
                f.write(
                    f"  {r['name']:<20} {r['trades']:>6} "
                    f"{r['win_rate']:>5.1f}% {pf_s:>6} "
                    f"{r['rr']:>5.2f} {r['max_dd']:>6.2f}% "
                    f"{r['monthly_ret']:>+7.2f}% "
                    f"${r['total_pnl']:>9.2f} "
                    f"{r['prop_score']:>3}/100\n"
                )
        if combined_result:
            r = combined_result
            pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "  ∞"
            f.write("  " + "─" * 68 + "\n")
            f.write(
                f"  {'COMBINED':<20} {r['trades']:>6} "
                f"{r['win_rate']:>5.1f}% {pf_s:>6} "
                f"{r['rr']:>5.2f} {r['max_dd']:>6.2f}% "
                f"{r['monthly_ret']:>+7.2f}% "
                f"${r['total_pnl']:>9.2f} "
                f"{r['prop_score']:>3}/100\n"
            )

    # ── CSV Summary ──
    rows = [
        ["BACKTEST SUMMARY", "", ""],
        ["Generated", datetime.now().strftime('%Y-%m-%d %H:%M'), ""],
        ["", "", ""],
        ["STRATEGY COMPARISON", "", "", "", "", "", "", "", "", ""],
        ["Strategy", "Trades", "WinRate%", "PF", "RR", "MaxDD%",
         "MonthlyRet%", "TotalPnL", "PropScore", "Grade"],
    ]
    for r in results:
        if r:
            rows.append([
                r['name'], r['trades'], round(r['win_rate'], 1),
                round(r['pf'], 2), round(r['rr'], 2),
                round(r['max_dd'], 2), round(r['monthly_ret'], 2),
                round(r['total_pnl'], 2), r['prop_score'], r['prop_grade']
            ])
    if combined_result:
        r = combined_result
        rows.append([
            "COMBINED", r['trades'], round(r['win_rate'], 1),
            round(r['pf'], 2), round(r['rr'], 2),
            round(r['max_dd'], 2), round(r['monthly_ret'], 2),
            round(r['total_pnl'], 2), r['prop_score'], r['prop_grade']
        ])

    pd.DataFrame(rows).to_csv(
        "Backtest_Summary.csv", index=False, header=False,
        encoding="utf-8-sig"
    )

    # ── Equity Curves ──
    for r in results:
        if r and r['risk']:
            eq_df = pd.DataFrame({
                'ts': r['risk'].curve_ts,
                'equity': r['risk'].curve
            })
            eq_df['dd_pct'] = (
                    (eq_df['equity'] - eq_df['equity'].cummax())
                    / eq_df['equity'].cummax() * 100
            ).round(4)
            safe_name = r['name'].replace(' ', '_').replace(':', '')
            eq_df.to_csv(f"equity_{safe_name}.csv",
                         index=False, encoding="utf-8-sig")

    if combined_result and combined_result['risk']:
        eq_df = pd.DataFrame({
            'ts': combined_result['risk'].curve_ts,
            'equity': combined_result['risk'].curve
        })
        eq_df['dd_pct'] = (
                (eq_df['equity'] - eq_df['equity'].cummax())
                / eq_df['equity'].cummax() * 100
        ).round(4)
        eq_df.to_csv("equity_combined.csv",
                     index=False, encoding="utf-8-sig")

    print(f"\n✅ Files saved:")
    print(f"   → Backtest_Report.txt")
    print(f"   → Backtest_Summary.csv")
    for r in results:
        if r:
            safe_name = r['name'].replace(' ', '_').replace(':', '')
            print(f"   → equity_{safe_name}.csv")
    if combined_result:
        print(f"   → equity_combined.csv")


# ================================================================== #
#                         MAIN                                       #
# ================================================================== #
if __name__ == "__main__":
    print("=" * 72)
    print("  PROFESSIONAL PROP TRADING BACKTEST SYSTEM")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # Load data
    df = load_data()

    # ═══════════════════════════════════════════════════
    #  PHASE 1: Build signals for each strategy
    # ═══════════════════════════════════════════════════
    print("\n⚙️  Building Strategy 1: Correlation Arbitrage...")
    sig_corr, z_score = strategy_corr_arb(df)
    n_sig_corr = (sig_corr['signal'] != 0).sum()
    print(f"   → {n_sig_corr} raw signals")

    print("⚙️  Building Strategy 2: RSI Divergence...")
    sig_rsi, rsi_vals = strategy_rsi_divergence(df)
    n_sig_rsi = (sig_rsi['signal'] != 0).sum()
    print(f"   → {n_sig_rsi} raw signals")

    print("⚙️  Building Strategy 3: Session Momentum...")
    sig_mom, atr_mom = strategy_session_momentum(df)
    n_sig_mom = (sig_mom['signal'] != 0).sum()
    print(f"   → {n_sig_mom} raw signals")

    # ═══════════════════════════════════════════════════
    #  PHASE 2: Independent backtests
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PHASE 2: INDEPENDENT STRATEGY BACKTESTS")
    print("=" * 72)

    results = []

    # Strategy 1: CorrArb
    print("\n🔄 Running: Correlation Arbitrage...")
    trades_corr, risk_corr = run_single_strategy(
        df, 'CorrArb', sig_corr,
        symbol='EUR', use_trailing=True, time_stop_hours=72
    )
    r1 = generate_report(trades_corr, risk_corr,
                         "Strategy 1: Correlation Arbitrage")
    results.append(r1)

    # Strategy 2: RSI Divergence
    print("\n🔄 Running: RSI Divergence...")
    trades_rsi, risk_rsi = run_single_strategy(
        df, 'RSI_Div', sig_rsi,
        symbol='EUR', use_trailing=True, time_stop_hours=48
    )
    r2 = generate_report(trades_rsi, risk_rsi,
                         "Strategy 2: RSI Divergence")
    results.append(r2)

    # Strategy 3: Session Momentum
    print("\n🔄 Running: Session Momentum...")
    trades_mom, risk_mom = run_single_strategy(
        df, 'SessionMom', sig_mom,
        symbol='EUR', use_trailing=True, time_stop_hours=36
    )
    r3 = generate_report(trades_mom, risk_mom,
                         "Strategy 3: Session Momentum")
    results.append(r3)

    # ═══════════════════════════════════════════════════
    #  PHASE 3: Filter - only profitable strategies
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PHASE 3: STRATEGY SELECTION")
    print("=" * 72)

    profitable_strategies = {}
    profitable_results = []

    strategy_signals = {
        'CorrArb': sig_corr,
        'RSI_Div': sig_rsi,
        'SessionMom': sig_mom,
    }
    strategy_results = {
        'CorrArb': r1,
        'RSI_Div': r2,
        'SessionMom': r3,
    }

    for name, r in strategy_results.items():
        if r is None:
            print(f"  ❌ {name}: No trades - REMOVED")
            continue

        passed = True
        reasons = []

        if r['total_pnl'] <= 0:
            passed = False
            reasons.append(f"Losing (PnL=${r['total_pnl']:.2f})")

        if r['win_rate'] < 35:
            passed = False
            reasons.append(f"Low WR ({r['win_rate']:.1f}%)")

        if r['pf'] < 1.0:
            passed = False
            reasons.append(f"PF<1 ({r['pf']:.2f})")

        if abs(r['max_dd']) > 9:
            passed = False
            reasons.append(f"High DD ({r['max_dd']:.1f}%)")

        if passed:
            print(f"  ✅ {name}: PASSED "
                  f"(PnL=${r['total_pnl']:+.2f}, "
                  f"WR={r['win_rate']:.1f}%, "
                  f"PF={r['pf']:.2f}, "
                  f"DD={r['max_dd']:.1f}%)")
            profitable_strategies[name] = strategy_signals[name]
            profitable_results.append(r)
        else:
            print(f"  ❌ {name}: REMOVED - {', '.join(reasons)}")

    # ═══════════════════════════════════════════════════
    #  PHASE 4: Combined Portfolio (only profitable)
    # ═══════════════════════════════════════════════════
    combined_result = None
    if len(profitable_strategies) > 0:
        print(f"\n{'=' * 72}")
        print(f"  PHASE 4: COMBINED PORTFOLIO "
              f"({len(profitable_strategies)} strategies)")
        print("=" * 72)

        print(f"\n🔄 Running combined backtest...")
        trades_comb, risk_comb = run_combined_backtest(
            df, profitable_strategies
        )
        combined_result = generate_report(
            trades_comb, risk_comb,
            f"COMBINED PORTFOLIO ({len(profitable_strategies)} strategies)"
        )
    else:
        print("\n⚠️  No profitable strategies found for combination!")

    # ═══════════════════════════════════════════════════
    #  PHASE 5: Final Summary & Save
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)

    print(f"\n  {'Strategy':<25} {'Trades':>6} {'WR%':>6} "
          f"{'PF':>6} {'DD%':>7} {'MonRet%':>8} "
          f"{'PnL':>10} {'Grade':>6}")
    print("  " + "─" * 68)

    for r in results:
        if r:
            status = "✅" if r['total_pnl'] > 0 else "❌"
            pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "  ∞"
            print(
                f"  {r['name']:<25} {r['trades']:>6} "
                f"{r['win_rate']:>5.1f}% {pf_s:>6} "
                f"{r['max_dd']:>6.2f}% "
                f"{r['monthly_ret']:>+7.2f}% "
                f"${r['total_pnl']:>9.2f} "
                f"{r['prop_grade']:>4} {status}"
            )

    if combined_result:
        r = combined_result
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "  ∞"
        print("  " + "─" * 68)
        print(
            f"  {'COMBINED':<25} {r['trades']:>6} "
            f"{r['win_rate']:>5.1f}% {pf_s:>6} "
            f"{r['max_dd']:>6.2f}% "
            f"{r['monthly_ret']:>+7.2f}% "
            f"${r['total_pnl']:>9.2f} "
            f"{r['prop_grade']:>4} "
            f"{'✅' if r['total_pnl'] > 0 else '❌'}"
        )

    # Save everything
    save_results(results, combined_result)

    print("\n" + "=" * 72)
    print("  ✅ BACKTEST COMPLETE")
    print("=" * 72)
