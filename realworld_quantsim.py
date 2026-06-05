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
    risk_per_trade_pct   = 0.012      # ↑ از 0.5% به 1.2%
    max_daily_loss_pct   = 0.05       # ↑ کمی بیشتر
    max_total_dd_pct     = 0.10       # ↑ از 8% به 10%
    profit_target_pct    = 0.40       # ↑ از 10% به 40% (برای پراپ فاز ۲)
    spread_eur_pips      = 1.0        # ↓ واقعی‌تر
    spread_gbp_pips      = 1.2
    commission_per_lot   = 6.0        # ↓ کمی کمتر
    pip                  = 0.0001
    lot_size             = 100_000
    max_open_positions   = 3          # همزمان حداکثر ۳ پوزیشن
    atr_period           = 14
    max_lot              = 2.0        # ↑ از 1.0


# ================================================================== #
#                        ابزارهای کمکی                              #
# ================================================================== #
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur or not files_gbp:
        raise FileNotFoundError("فایل CSV پیدا نشد در data/")

    def read(paths, suffix):
        frames = []
        for p in paths:
            df = pd.read_csv(p, sep=';', header=None,
                             names=['ts','o','h','l','c','v'])
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

    # حذف کندل‌های آخر هفته
    df = df[df.index.weekday < 5]

    print(f"✅ {len(df):,} کندل | {df.index[0].date()} → {df.index[-1].date()}")
    return df


def calc_atr(high, low, close, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(close, period=14) -> pd.Series:
    d    = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_adx(high, low, close, period=14) -> pd.Series:
    """ADX برای فیلتر روند قوی"""
    tr    = calc_atr(high, low, close, 1)
    dm_p  = (high.diff()).clip(lower=0)
    dm_n  = (-low.diff()).clip(lower=0)
    dm_p  = dm_p.where(dm_p > dm_n, 0)
    dm_n  = dm_n.where(dm_n > dm_p, 0)
    atr14 = tr.rolling(period).sum()
    di_p  = 100 * dm_p.rolling(period).sum() / atr14.replace(0, np.nan)
    di_n  = 100 * dm_n.rolling(period).sum() / atr14.replace(0, np.nan)
    dx    = (abs(di_p - di_n) / (di_p + di_n).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f  = close.ewm(span=fast,   adjust=False).mean()
    ema_s  = close.ewm(span=slow,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal,  adjust=False).mean()
    hist   = macd - sig
    return macd, sig, hist


def calc_bollinger(close, period=20, std_mult=2.0):
    mid  = close.rolling(period).mean()
    std  = close.rolling(period).std()
    return mid + std_mult * std, mid, mid - std_mult * std


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
#                        Risk Manager                                #
# ================================================================== #
class RiskManager:
    def __init__(self):
        self.equity       = Config.initial_balance
        self.peak         = Config.initial_balance
        self.day_start_eq = Config.initial_balance
        self.cur_day      = None
        self.halted       = False
        self.halt_reason  = "در حال اجرا"
        self.curve        = [Config.initial_balance]
        self.curve_ts     = [None]
        self.daily_pnl    = {}   # date → pnl
        self.monthly_pnl  = {}   # YYYY-MM → pnl

    def new_bar(self, ts: pd.Timestamp):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day      = day
            self.day_start_eq = self.equity
            # ریست daily halt
            if self.halted and "Daily" in self.halt_reason:
                self.halted      = False
                self.halt_reason = "در حال اجرا"

    def add_pnl(self, amount: float, ts: pd.Timestamp) -> bool:
        self.equity += amount
        self.peak    = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))
        self.curve_ts.append(ts)

        # ثبت روزانه/ماهانه
        d_key = str(ts.date())
        m_key = ts.strftime('%Y-%m')
        self.daily_pnl[d_key]   = self.daily_pnl.get(d_key, 0) + amount
        self.monthly_pnl[m_key] = self.monthly_pnl.get(m_key, 0) + amount

        daily_dd = (self.equity - self.day_start_eq) / self.day_start_eq
        if daily_dd <= -Config.max_daily_loss_pct:
            self.halted      = True
            self.halt_reason = f"Daily Loss {daily_dd*100:.1f}%"
            return False

        total_dd = (self.equity - self.peak) / self.peak
        if total_dd <= -Config.max_total_dd_pct:
            self.halted      = True
            self.halt_reason = f"Max DD {total_dd*100:.1f}%"
            return False

        profit_pct = (self.equity - Config.initial_balance) / Config.initial_balance
        if profit_pct >= Config.profit_target_pct:
            self.halted      = True
            self.halt_reason = f"Profit Target {profit_pct*100:.1f}%"
            return False

        return True

    @property
    def max_dd(self):
        s = pd.Series(self.curve)
        roll_max = s.cummax()
        dd = (s - roll_max) / roll_max * 100
        return dd.min()

    @property
    def max_dd_abs(self):
        s = pd.Series(self.curve)
        return (s - s.cummax()).min()

    @property
    def sharpe(self):
        r = pd.Series(self.curve).pct_change().dropna()
        return (r.mean() / r.std() * np.sqrt(252 * 96)) if r.std() > 0 else 0

    @property
    def sortino(self):
        r = pd.Series(self.curve).pct_change().dropna()
        neg = r[r < 0]
        down_std = neg.std() if len(neg) > 0 else 1e-10
        return (r.mean() / down_std * np.sqrt(252 * 96)) if down_std > 0 else 0

    @property
    def calmar(self):
        ann_ret = (self.equity / Config.initial_balance - 1)
        dd = abs(self.max_dd / 100)
        return ann_ret / dd if dd > 0 else 0


# ================================================================== #
#   Strategy 1: Correlation Pair Trading (بهینه‌شده)                #
# ================================================================== #
def build_corr_arb_signals(df):
    """
    منطق بهبودیافته:
    - Z-score با باند ±2.2 (سخت‌گیرانه‌تر)
    - فیلتر volatility: std باید بالاتر از میانگین بلند باشد
    - فیلتر روند: ADX < 25 (بازار رنجینگ بهتر است)
    - خروج با Z < 0.3
    """
    eurgbp   = df['c_eur'] / df['c_gbp']
    period   = 96
    period_l = period * 5

    mean   = eurgbp.rolling(period).mean()
    std    = eurgbp.rolling(period).std()
    z      = (eurgbp - mean) / std.replace(0, np.nan)

    # فیلتر: بازار نباید خیلی ترند باشد
    adx_eur = calc_adx(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    std_ok  = std > std.rolling(period_l).mean() * 0.4
    adx_ok  = adx_eur < 30   # رنج‌بازار بهتر عمل می‌کند

    sig = pd.Series(0, index=df.index)
    sig[(z >  2.2) & std_ok & adx_ok] = -1   # EURGBP بالاست → EUR کوتاه
    sig[(z < -2.2) & std_ok & adx_ok] =  1   # EURGBP پایینه → EUR بلند
    sig = sig.where(sig != sig.shift(), 0)

    return sig, z


# ================================================================== #
#   Strategy 2: Asian Range Breakout (بهینه‌شده)                    #
# ================================================================== #
def build_asian_breakout_signals(df):
    """
    بهبودها:
    - Range باید بین 12 تا 50 پیپ (بهینه‌تر)
    - تایید: Volume spike در لندن
    - فیلتر ADX: بالای 20 (روند داریم)
    - SL داخل رنج (نه بیرون) → RR بهتر
    - TP = 2.5x SL
    """
    d = df.copy()
    d['hour']    = d.index.hour
    d['weekday'] = d.index.weekday
    d['date']    = d.index.date

    atr     = calc_atr(d['h_eur'], d['l_eur'], d['c_eur'], 14)
    adx_eur = calc_adx(d['h_eur'], d['l_eur'], d['c_eur'], 14)

    # محاسبه رنج آسیا
    asian = d[d['hour'].between(1, 6)]
    rng   = asian.groupby('date').agg(
        ah=('h_eur', 'max'),
        al=('l_eur', 'min')
    )
    rng['rng_pips'] = (rng['ah'] - rng['al']) / Config.pip

    # volume میانگین آسیا برای مقایسه
    vol_asia = asian.groupby('date')['v_eur'].mean().rename('vol_asia_avg')
    rng      = rng.join(vol_asia)

    d = d.join(rng, on='date')

    # فیلترها
    valid = (
        d['rng_pips'].between(12, 50) &
        d['weekday'].between(0, 3)          # دوشنبه-پنجشنبه
    )

    # لندن ساعت ۷ تا ۱۲
    london    = d['hour'].between(7, 11)
    # حجم لندن vs میانگین آسیا
    vol_spike = d['v_eur'] > d['vol_asia_avg'] * 1.3

    sig = pd.Series(0, index=d.index)
    # Long breakout
    long_cond = (
        london & valid & vol_spike &
        (d['c_eur'] > d['ah']) &
        (d['o_eur'] <= d['ah']) &
        (adx_eur > 18)
    )
    # Short breakout
    short_cond = (
        london & valid & vol_spike &
        (d['c_eur'] < d['al']) &
        (d['o_eur'] >= d['al']) &
        (adx_eur > 18)
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1

    # فقط اولین سیگنال هر روز
    nonzero   = sig[sig != 0]
    first_idx = nonzero.groupby(nonzero.index.date).head(1).index
    final     = pd.Series(0, index=d.index)
    final[first_idx] = sig[first_idx]

    return final, d['ah'], d['al'], atr


# ================================================================== #
#   Strategy 3: Multi-EMA Trend Pullback (بهینه‌شده کامل)           #
# ================================================================== #
def build_trend_pullback_signals(df):
    """
    بهبودها:
    - اضافه کردن EMA200 برای تایید روند بزرگ
    - MACD برای تایید momentum
    - RSI pullback zone بهینه‌شده: 38-62
    - فیلتر ADX > 22 (روند کافی)
    - استفاده از Bollinger برای تشخیص pullback
    """
    c      = df['c_eur']
    h      = df['h_eur']
    l      = df['l_eur']

    ema21  = c.ewm(span=21,  adjust=False).mean()
    ema55  = c.ewm(span=55,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rsi    = calc_rsi(c, 14)
    atr    = calc_atr(h, l, c, 14)
    adx    = calc_adx(h, l, c, 14)
    _, _, macd_hist = calc_macd(c)
    _, bb_mid, _    = calc_bollinger(c, 20, 2.0)

    # فاصله از EMA21 به واحد ATR
    dist_ema21 = (c - ema21) / atr.replace(0, np.nan)

    sig = pd.Series(0, index=df.index)

    # Long: روند صعودی، pullback به EMA21، momentum برگشته
    long_cond = (
        (ema21  > ema55)   &   # روند کوتاه‌مدت صعودی
        (ema55  > ema200)  &   # روند بلندمدت صعودی
        dist_ema21.between(-1.2, 0.3) &  # pullback به EMA
        rsi.between(38, 58)   &   # RSI در ناحیه pullback
        (macd_hist > macd_hist.shift(1)) &  # MACD در حال بهبود
        (adx > 22)            &   # روند قوی کافی
        (df.index.hour.isin(range(7, 18)))  # ساعات فعال
    )

    # Short: روند نزولی، pullback به EMA21، momentum برگشته
    short_cond = (
        (ema21  < ema55)   &
        (ema55  < ema200)  &
        dist_ema21.between(-0.3, 1.2) &
        rsi.between(42, 62)   &
        (macd_hist < macd_hist.shift(1)) &
        (adx > 22)            &
        (df.index.hour.isin(range(7, 18)))
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    return sig, ema21, ema55, atr, adx


# ================================================================== #
#   Strategy 4: London Open Momentum (جایگزین OverlapMom)           #
#                                                                    #
#  منطق:                                                             #
#  ساعت ۰۷:۰۰-۰۹:۰۰ GMT → اوج نقدینگی لندن                        #
#  شرط: حرکت قوی در ۴ کندل اول + تایید MACD                        #
#  SL: ATR-based، TP: 3x SL (RR عالی)                              #
# ================================================================== #
def build_london_momentum_signals(df):
    """
    استراتژی لحظه‌ای باز شدن لندن:
    - قوی‌ترین ساعات بازار فارکس
    - حرکت جهت‌دار قوی اول روز
    - فیلتر: بازار نباید شب قبل رنج بزرگی زده باشد
    """
    c   = df['c_eur']
    h   = df['h_eur']
    l   = df['l_eur']

    ema9  = c.ewm(span=9,  adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    rsi   = calc_rsi(c, 14)
    atr   = calc_atr(h, l, c, 14)
    adx   = calc_adx(h, l, c, 14)
    _, sig_line, macd_hist = calc_macd(c)

    hour    = pd.Series(df.index.hour,    index=df.index)
    weekday = pd.Series(df.index.weekday, index=df.index)

    # قدرت کندل (بدنه نسبت به ATR)
    body      = (c - df['o_eur']).abs()
    body_pct  = body / atr.replace(0, np.nan)

    # مومنتوم ۴ کندل اخیر
    mom4 = c - c.shift(4)

    sig = pd.Series(0, index=df.index)

    london_open = hour.between(7, 9) & weekday.between(0, 4)

    # Long momentum: بازار صعودی + کندل قوی + تایید
    long_cond = (
        london_open &
        (ema9  > ema21)   &
        (c     > ema9)    &
        (mom4  > atr * 0.5) &          # مومنتوم صعودی ۴ کندل
        (body_pct > 0.3)  &            # کندل با بدنه واقعی
        rsi.between(52, 75)  &
        (macd_hist > 0)      &
        (adx > 20)
    )

    # Short momentum
    short_cond = (
        london_open &
        (ema9  < ema21)   &
        (c     < ema9)    &
        (mom4  < -atr * 0.5) &
        (body_pct > 0.3)  &
        rsi.between(25, 48)  &
        (macd_hist < 0)      &
        (adx > 20)
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    # فقط اولین سیگنال هر روز در این ساعات
    nonzero   = sig[sig != 0]
    first_idx = nonzero.groupby(nonzero.index.date).head(1).index
    final     = pd.Series(0, index=df.index)
    final[first_idx] = sig[first_idx]

    return final, atr, adx


# ================================================================== #
#   Strategy 5: NEW - GBP Breakout (با GBPUSD)                      #
#                                                                    #
#  منطق:                                                             #
#  معامله روی GBPUSD که نوسان بیشتری دارد                           #
#  سیگنال: شکست BB + تایید RSI + حجم                               #
# ================================================================== #
def build_gbp_breakout_signals(df):
    """
    GBP نوسان بیشتری دارد → سود بالقوه بیشتر
    استراتژی: شکست باند بولینگر با تایید
    """
    c_g = df['c_gbp']
    h_g = df['h_gbp']
    l_g = df['l_gbp']

    bb_up, bb_mid, bb_lo = calc_bollinger(c_g, 20, 2.0)
    rsi    = calc_rsi(c_g, 14)
    atr    = calc_atr(h_g, l_g, c_g, 14)
    adx    = calc_adx(h_g, l_g, c_g, 14)
    _, _, macd_hist = calc_macd(c_g)

    # BB width (نرمال‌شده)
    bb_width   = (bb_up - bb_lo) / bb_mid.replace(0, np.nan)
    bb_width_m = bb_width.rolling(50).mean()

    # squeeze → breakout احتمالی
    bb_squeeze = bb_width < bb_width_m * 0.85

    hour    = pd.Series(df.index.hour,    index=df.index)
    weekday = pd.Series(df.index.weekday, index=df.index)
    active  = hour.between(7, 17) & weekday.between(0, 3)

    sig = pd.Series(0, index=df.index)

    # شکست بالای BB + تایید
    long_cond = (
        active &
        (c_g > bb_up.shift(1)) &          # کندل بالای BB بسته شد
        (df['o_gbp'] <= bb_up.shift(1)) &  # داخل BB باز شد
        (bb_squeeze.shift(3))   &          # squeeze قبل از breakout
        rsi.between(55, 78)     &
        (macd_hist > 0)         &
        (adx > 18)
    )

    # شکست پایین BB
    short_cond = (
        active &
        (c_g < bb_lo.shift(1)) &
        (df['o_gbp'] >= bb_lo.shift(1)) &
        (bb_squeeze.shift(3))   &
        rsi.between(22, 45)     &
        (macd_hist < 0)         &
        (adx > 18)
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    return sig, atr, bb_up, bb_lo


# ================================================================== #
#                       موتور Backtest                               #
# ================================================================== #
def run_backtest(df: pd.DataFrame):
    print("⚙️  محاسبه اندیکاتورها...")

    # ساخت سیگنال‌ها
    sig_arb,   z_score                    = build_corr_arb_signals(df)
    sig_ab,    ah, al, atr_ab             = build_asian_breakout_signals(df)
    sig_tp,    ema21, ema55, atr_tp, adx_tp = build_trend_pullback_signals(df)
    sig_lm,    atr_lm, adx_lm            = build_london_momentum_signals(df)
    sig_gb,    atr_gb, bb_up, bb_lo      = build_gbp_breakout_signals(df)

    atr_main = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)

    risk     = RiskManager()
    open_pos = {}
    trades   = []
    warmup   = 250

    print("⚙️  شروع شبیه‌سازی...\n")

    for i in range(warmup, len(df)):
        ts    = df.index[i]
        row   = df.iloc[i]
        c_eur = row['c_eur']
        c_gbp = row['c_gbp']
        h_eur = row['h_eur']
        l_eur = row['l_eur']
        h_gbp = row['h_gbp']
        l_gbp = row['l_gbp']
        atr   = atr_main.iloc[i]

        if np.isnan(atr) or atr <= 0:
            continue

        risk.new_bar(ts)

        # ---- اگر متوقف شده: فقط ببند ----
        if risk.halted:
            for key in list(open_pos.keys()):
                p     = open_pos.pop(key)
                sym   = p['symbol']
                ep    = c_eur if sym == 'EUR' else c_gbp
                p_pnl = calc_pnl(p['dir'], p['lot'], p['entry'], ep, sym)
                trades.append({**p, 'exit': ep, 'exit_ts': ts,
                               'pnl': p_pnl, 'status': 'halt_close'})
                risk.add_pnl(p_pnl, ts)
            continue

        # ============================================================
        #  مرحله ۱: بررسی SL/TP/Trailing پوزیشن‌های باز
        # ============================================================
        for key in list(open_pos.keys()):
            p   = open_pos[key]
            sym = p['symbol']

            if sym == 'EUR':
                hi, lo, cp = h_eur, l_eur, c_eur
            else:
                hi, lo, cp = h_gbp, l_gbp, c_gbp

            d, entry, sl, tp = p['dir'], p['entry'], p['sl'], p['tp']

            hit_sl = (d ==  1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d ==  1 and hi >= tp) or (d == -1 and lo <= tp)

            # ---- Trailing Stop بهینه‌شده ----
            if p['strategy'] in ('LondonMom', 'TrendPB'):
                move = d * (cp - entry)
                if move > atr * 1.0:     # بعد از ۱ ATR سود
                    new_sl = entry + d * atr * 0.5  # به break-even + 0.5ATR
                    if d == 1:
                        p['sl'] = max(p['sl'], new_sl)
                    else:
                        p['sl'] = min(p['sl'], new_sl)
                if move > atr * 2.0:    # بعد از ۲ ATR سود
                    new_sl = entry + d * atr * 1.2
                    if d == 1:
                        p['sl'] = max(p['sl'], new_sl)
                    else:
                        p['sl'] = min(p['sl'], new_sl)

            # ---- خروج CorrArb با Z ----
            if p['strategy'] == 'CorrArb':
                z_now = z_score.iloc[i]
                if pd.notna(z_now) and abs(z_now) < 0.35:
                    hit_tp = True

            # ---- Time Stop: اگر بیش از ۳ روز باز است ----
            if 'entry_ts' in p:
                duration_h = (ts - p['entry_ts']).total_seconds() / 3600
                max_hours  = {'CorrArb': 72, 'AsianBreak': 48,
                              'TrendPB': 72, 'LondonMom': 24,
                              'GBPBreak': 36}.get(p['strategy'], 48)
                if duration_h > max_hours and not hit_tp:
                    # با قیمت فعلی ببند
                    exit_reason = 'TimeStop'
                    exit_price  = cp
                    p_pnl = calc_pnl(d, p['lot'], entry, exit_price, sym)
                    trades.append({
                        **p, 'exit': exit_price, 'exit_ts': ts,
                        'pnl': p_pnl, 'status': exit_reason
                    })
                    risk.add_pnl(p_pnl, ts)
                    del open_pos[key]
                    continue

            exit_reason = None
            exit_price  = None
            if hit_sl:
                exit_reason = 'SL'
                exit_price  = sl
            elif hit_tp:
                exit_reason = 'TP'
                exit_price  = tp

            if exit_reason:
                p_pnl = calc_pnl(d, p['lot'], entry, exit_price, sym)
                trades.append({
                    **p, 'exit': exit_price, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': exit_reason
                })
                risk.add_pnl(p_pnl, ts)
                del open_pos[key]

        # ============================================================
        #  مرحله ۲: ورود → حداکثر max_open_positions پوزیشن
        # ============================================================
        if len(open_pos) >= Config.max_open_positions:
            continue

        # ---- 1: Correlation Arb ----
        if 'CorrArb' not in open_pos and sig_arb.iloc[i] != 0:
            sig_val = int(sig_arb.iloc[i])
            sl_pips = 18
            tp_pips = 30           # RR ≈ 1.67
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig_val * Config.spread_eur_pips * Config.pip / 2
            open_pos['CorrArb'] = {
                'strategy': 'CorrArb', 'symbol': 'EUR',
                'dir': sig_val, 'lot': lot, 'entry': ep,
                'sl':  ep - sig_val * sl_pips * Config.pip,
                'tp':  ep + sig_val * tp_pips * Config.pip,
                'entry_ts': ts,
            }

        # ---- 2: Asian Breakout ----
        if 'AsianBreak' not in open_pos and sig_ab.iloc[i] != 0:
            sig_val = int(sig_ab.iloc[i])
            rng_sz  = max((ah.iloc[i] - al.iloc[i]) / Config.pip, 12)
            sl_pips = max(12, rng_sz * 0.4)
            tp_pips = sl_pips * 2.5    # RR = 2.5
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig_val * Config.spread_eur_pips * Config.pip / 2
            open_pos['AsianBreak'] = {
                'strategy': 'AsianBreak', 'symbol': 'EUR',
                'dir': sig_val, 'lot': lot, 'entry': ep,
                'sl':  ep - sig_val * sl_pips * Config.pip,
                'tp':  ep + sig_val * tp_pips * Config.pip,
                'entry_ts': ts,
            }

        # ---- 3: Trend Pullback ----
        if 'TrendPB' not in open_pos and sig_tp.iloc[i] != 0:
            sig_val = int(sig_tp.iloc[i])
            sl_pips = max(15, atr / Config.pip * 1.2)
            tp_pips = sl_pips * 2.2    # RR = 2.2
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig_val * Config.spread_eur_pips * Config.pip / 2
            open_pos['TrendPB'] = {
                'strategy': 'TrendPB', 'symbol': 'EUR',
                'dir': sig_val, 'lot': lot, 'entry': ep,
                'sl':  ep - sig_val * sl_pips * Config.pip,
                'tp':  ep + sig_val * tp_pips * Config.pip,
                'entry_ts': ts,
            }

        # ---- 4: London Momentum ----
        if 'LondonMom' not in open_pos and sig_lm.iloc[i] != 0:
            sig_val = int(sig_lm.iloc[i])
            sl_pips = max(14, atr / Config.pip * 1.0)
            tp_pips = sl_pips * 3.0    # RR = 3.0 (momentum کوتاه‌مدت)
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig_val * Config.spread_eur_pips * Config.pip / 2
            open_pos['LondonMom'] = {
                'strategy': 'LondonMom', 'symbol': 'EUR',
                'dir': sig_val, 'lot': lot, 'entry': ep,
                'sl':  ep - sig_val * sl_pips * Config.pip,
                'tp':  ep + sig_val * tp_pips * Config.pip,
                'entry_ts': ts,
            }

        # ---- 5: GBP Breakout ----
        if 'GBPBreak' not in open_pos and sig_gb.iloc[i] != 0:
            sig_val = int(sig_gb.iloc[i])
            sl_pips = max(16, atr_gb.iloc[i] / Config.pip * 1.1)
            tp_pips = sl_pips * 2.8    # RR = 2.8
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_gbp + sig_val * Config.spread_gbp_pips * Config.pip / 2
            open_pos['GBPBreak'] = {
                'strategy': 'GBPBreak', 'symbol': 'GBP',
                'dir': sig_val, 'lot': lot, 'entry': ep,
                'sl':  ep - sig_val * sl_pips * Config.pip,
                'tp':  ep + sig_val * tp_pips * Config.pip,
                'entry_ts': ts,
            }

    # ---- بستن باقیمانده ----
    last_ts  = df.index[-1]
    last_eur = df['c_eur'].iloc[-1]
    last_gbp = df['c_gbp'].iloc[-1]
    for key, p in open_pos.items():
        ep    = last_eur if p['symbol'] == 'EUR' else last_gbp
        p_pnl = calc_pnl(p['dir'], p['lot'], p['entry'], ep, p['symbol'])
        trades.append({**p, 'exit': ep, 'exit_ts': last_ts,
                       'pnl': p_pnl, 'status': 'eod_close'})
        risk.add_pnl(p_pnl, last_ts)

    return trades, risk


# ================================================================== #
#              گزارش حرفه‌ای + ذخیره فایل                           #
# ================================================================== #
def report(trades: list, risk: RiskManager, df: pd.DataFrame):
    if not trades:
        print("❌ هیچ معامله‌ای ثبت نشد!")
        return

    t = pd.DataFrame(trades)
    t['pnl']          = pd.to_numeric(t['pnl'],      errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    # ── بازه زمانی ──
    start_date    = t['entry_ts'].min()
    end_date      = t['exit_ts'].max()
    total_days    = max((end_date - start_date).days, 1)
    total_weeks   = total_days / 7
    total_months  = total_days / 30.44
    total_years   = total_days / 365.25

    # ── آمار کلی ──
    final_eq    = risk.equity
    total_pnl   = final_eq - Config.initial_balance
    total_ret   = total_pnl / Config.initial_balance * 100
    ann_ret     = ((final_eq / Config.initial_balance) ** (365.25 / total_days) - 1) * 100

    win_t  = t[t['pnl'] > 0]
    loss_t = t[t['pnl'] < 0]
    be_t   = t[t['pnl'] == 0]

    win_rate    = len(win_t) / len(t) * 100
    avg_win     = win_t['pnl'].mean()   if len(win_t)   > 0 else 0
    avg_loss    = loss_t['pnl'].mean()  if len(loss_t)  > 0 else 0
    gross_win   = win_t['pnl'].sum()
    gross_loss  = abs(loss_t['pnl'].sum())
    pf          = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy  = t['pnl'].mean()
    rr_actual   = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    best_trade  = t['pnl'].max()
    worst_trade = t['pnl'].min()
    avg_dur_win = win_t['duration_min'].mean()  if len(win_t)  > 0 else 0
    avg_dur_los = loss_t['duration_min'].mean() if len(loss_t) > 0 else 0

    # consecutive wins/losses
    pnl_sign = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    max_consec_win  = 0
    max_consec_loss = 0
    cur_w, cur_l    = 0, 0
    for s in pnl_sign:
        if s > 0:
            cur_w += 1; cur_l = 0
            max_consec_win = max(max_consec_win, cur_w)
        elif s < 0:
            cur_l += 1; cur_w = 0
            max_consec_loss = max(max_consec_loss, cur_l)
        else:
            cur_w = cur_l = 0

    # آمار زمانی
    tpd = len(t) / total_days
    tpw = len(t) / total_weeks
    tpm = len(t) / total_months

    ppd = total_pnl / total_days
    ppw = total_pnl / total_weeks
    ppm = total_pnl / total_months
    ppy = total_pnl / total_years

    avg_dur = t['duration_min'].mean()

    # ── گزارش ماهانه ──
    t['ym'] = t['entry_ts'].dt.to_period('M')
    monthly = t.groupby('ym').agg(
        trades   =('pnl', 'count'),
        pnl      =('pnl', 'sum'),
        wins     =('pnl', lambda x: (x > 0).sum()),
        best     =('pnl', 'max'),
        worst    =('pnl', 'min'),
    ).reset_index()
    monthly['win_rate'] = monthly['wins'] / monthly['trades'] * 100
    monthly['ret_pct']  = monthly['pnl'] / Config.initial_balance * 100
    monthly['cum_pnl']  = monthly['pnl'].cumsum()
    monthly['cum_ret']  = monthly['cum_pnl'] / Config.initial_balance * 100

    # ── گزارش سالانه ──
    t['year'] = t['entry_ts'].dt.year
    yearly = t.groupby('year').agg(
        trades  =('pnl', 'count'),
        pnl     =('pnl', 'sum'),
        wins    =('pnl', lambda x: (x > 0).sum()),
        best_m  =('pnl', 'max'),
        worst_m =('pnl', 'min'),
    ).reset_index()
    yearly['win_rate']  = yearly['wins']  / yearly['trades'] * 100
    yearly['ret_pct']   = yearly['pnl']   / Config.initial_balance * 100
    yearly['ann_ret']   = [
        ((Config.initial_balance + r) / Config.initial_balance) ** (365.25 / total_days) - 1
        for r in yearly['pnl'].cumsum()
    ]

    # ── عملکرد هر استراتژی ──
    strat_stats = []
    for name in sorted(t['strategy'].unique()):
        sub   = t[t['strategy'] == name]
        sw    = sub[sub['pnl'] > 0]
        sl    = sub[sub['pnl'] < 0]
        wr    = (sub['pnl'] > 0).mean() * 100
        gw    = sw['pnl'].sum()
        gl    = abs(sl['pnl'].sum())
        spf   = gw / gl if gl > 0 else float('inf')
        aw    = sw['pnl'].mean() if len(sw) > 0 else 0
        al_   = sl['pnl'].mean() if len(sl) > 0 else 0
        tot   = sub['pnl'].sum()
        rr    = abs(aw / al_) if al_ != 0 else 0
        exp   = sub['pnl'].mean()
        dur   = sub['duration_min'].mean()
        strat_stats.append({
            'name': name, 'n': len(sub), 'wr': wr,
            'pf': spf, 'aw': aw, 'al': al_,
            'rr': rr, 'exp': exp, 'tot': tot, 'dur': dur
        })

    # ================================================================
    #  چاپ گزارش
    # ================================================================
    W   = 68
    SEP = "═" * W
    sep = "─" * W

    def hr(char="─"):
        return char * W

    def row(label, value, unit=""):
        lbl = f"  {label}"
        val = f"{value}{unit}"
        dots = "." * max(2, W - len(lbl) - len(val) - 2)
        return f"{lbl} {dots} {val}"

    lines = []
    lines += [
        SEP,
        " " * 10 + "▌ BACKTEST REPORT — Prop Trading System ▐",
        " " * 10 + f"▌ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M')}          ▐",
        SEP, "",
        "┌─ اطلاعات کلی " + "─" * (W - 15) + "┐",
        row("دوره بک‌تست",   f"{start_date.date()} → {end_date.date()}"),
        row("تعداد روزهای کاری", f"{total_days:,}"),
        row("جفت‌ارزها",     "EURUSD  +  GBPUSD"),
        row("تایم‌فریم",     "15 دقیقه"),
        row("تعداد استراتژی", f"{len(strat_stats)}"),
        "└" + "─" * (W - 1) + "┘", "",

        "┌─ نتایج مالی " + "─" * (W - 14) + "┐",
        row("موجودی اولیه",   f"${Config.initial_balance:>12,.2f}"),
        row("موجودی نهایی",   f"${final_eq:>12,.2f}"),
        row("سود/زیان خالص", f"${total_pnl:>+12,.2f}"),
        row("بازده کل",       f"{total_ret:>+.2f}%"),
        row("بازده سالانه",   f"{ann_ret:>+.2f}%"),
        row("بهترین معامله",  f"${best_trade:>+.2f}"),
        row("بدترین معامله",  f"${worst_trade:>+.2f}"),
        "└" + "─" * (W - 1) + "┘", "",

        "┌─ معیارهای ریسک " + "─" * (W - 17) + "┐",
        row("Max Drawdown",     f"{risk.max_dd:>.2f}%"),
        row("Max DD مطلق",     f"${risk.max_dd_abs:>+.2f}"),
        row("Sharpe Ratio",     f"{risk.sharpe:>.2f}"),
        row("Sortino Ratio",    f"{risk.sortino:>.2f}"),
        row("Calmar Ratio",     f"{risk.calmar:>.2f}"),
        row("Profit Factor",    f"{pf:>.2f}"),
        row("وضعیت توقف",      risk.halt_reason),
        "└" + "─" * (W - 1) + "┘", "",

        "┌─ آمار معاملات " + "─" * (W - 16) + "┐",
        row("تعداد کل معاملات",f"{len(t):,}"),
        row("Win Rate",         f"{win_rate:.1f}%"),
        row("معاملات سودده",   f"{len(win_t):,}"),
        row("معاملات ضررده",   f"{len(loss_t):,}"),
        row("Avg Win",          f"${avg_win:>+.2f}"),
        row("Avg Loss",         f"${avg_loss:>+.2f}"),
        row("RR واقعی",         f"{rr_actual:.2f}"),
        row("Expectancy/معامله",f"${expectancy:>+.2f}"),
        row("Max Consec. Wins", f"{max_consec_win}"),
        row("Max Consec. Loss", f"{max_consec_loss}"),
        row("مدت میانگین کل",  f"{avg_dur:.0f} min"),
        row("مدت میانگین Win", f"{avg_dur_win:.0f} min"),
        row("مدت میانگین Loss",f"{avg_dur_los:.0f} min"),
        "└" + "─" * (W - 1) + "┘", "",

        "┌─ آمار زمانی " + "─" * (W - 14) + "┐",
        row("معاملات / روز",   f"{tpd:.2f}"),
        row("معاملات / هفته",  f"{tpw:.2f}"),
        row("معاملات / ماه",   f"{tpm:.2f}"),
        row("سود / روز",       f"${ppd:>+.2f}"),
        row("سود / هفته",      f"${ppw:>+.2f}"),
        row("سود / ماه",       f"${ppm:>+.2f}"),
        row("سود / سال",       f"${ppy:>+.2f}"),
        "└" + "─" * (W - 1) + "┘", "",
    ]

    # ── جدول استراتژی‌ها ──
    lines += ["┌─ عملکرد استراتژی‌ها " + "─" * (W - 21) + "┐"]
    hdr = (f"  {'استراتژی':<12} {'#':>4} {'Win%':>6} {'PF':>5} "
           f"{'RR':>5} {'AvgW':>8} {'AvgL':>8} {'Exp':>7} "
           f"{'PnL':>10} {'مدت':>6}")
    lines.append(hdr)
    lines.append("  " + "─" * (W - 3))
    for s in strat_stats:
        flag = "✅" if s['tot'] > 0 else "❌"
        line = (f"  {s['name']:<12} {s['n']:>4} {s['wr']:>5.1f}% "
                f"{s['pf']:>5.2f} {s['rr']:>5.2f} "
                f"${s['aw']:>7.2f} ${s['al']:>7.2f} "
                f"${s['exp']:>6.2f} ${s['tot']:>9.2f} "
                f"{s['dur']:>5.0f}m {flag}")
        lines.append(line)
    lines += ["└" + "─" * (W - 1) + "┘", ""]

    # ── گزارش ماهانه ──
    lines += ["┌─ گزارش ماهانه " + "─" * (W - 16) + "┐"]
    lines.append(
        f"  {'ماه':>7} {'#':>4} {'Win%':>6} {'PnL':>10} "
        f"{'Ret%':>6} {'تجمعی':>10} {'CumRet%':>8}"
    )
    lines.append("  " + "─" * (W - 3))
    for _, mr in monthly.iterrows():
        arrow = "▲" if mr['pnl'] >= 0 else "▼"
        lines.append(
            f"  {str(mr['ym']):>7} {int(mr['trades']):>4} "
            f"{mr['win_rate']:>5.1f}% ${mr['pnl']:>9.2f} "
            f"{mr['ret_pct']:>+5.1f}% ${mr['cum_pnl']:>9.2f} "
            f"{mr['cum_ret']:>+6.1f}% {arrow}"
        )
    lines += ["└" + "─" * (W - 1) + "┘", ""]

    # ── گزارش سالانه ──
    lines += ["┌─ گزارش سالانه " + "─" * (W - 16) + "┐"]
    lines.append(
        f"  {'سال':>5} {'#':>5} {'Win%':>6} {'PnL':>10} "
        f"{'Ret%':>7} {'Ann Ret%':>9}"
    )
    lines.append("  " + "─" * (W - 3))
    for _, yr in yearly.iterrows():
        lines.append(
            f"  {int(yr['year']):>5} {int(yr['trades']):>5} "
            f"{yr['win_rate']:>5.1f}% ${yr['pnl']:>9.2f} "
            f"{yr['ret_pct']:>+6.1f}% {yr['ann_ret']*100:>+8.1f}%"
        )
    lines += ["└" + "─" * (W - 1) + "┘", ""]

    # ── توزیع خروج ──
    lines += ["┌─ توزیع نوع خروج " + "─" * (W - 18) + "┐"]
    total_exits = len(t)
    for status, cnt in t['status'].value_counts().items():
        pct  = cnt / total_exits * 100
        bar  = "█" * int(pct / 3)
        pnl_ = t.loc[t['status'] == status, 'pnl']
        avg_ = pnl_.mean()
        lines.append(
            f"  {status:<12} {cnt:>5} ({pct:>5.1f}%)  "
            f"{bar:<20}  avg=${avg_:>+.2f}"
        )
    lines += ["└" + "─" * (W - 1) + "┘", ""]
    lines.append(SEP)

    report_text = "\n".join(lines)
    print(report_text)

    # ── ذخیره فایل‌ها ──

    # ۱. گزارش متنی
    with open("Backtest_Report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    # ۲. گزارش CSV جامع (بدون ریز معاملات)
    summary_rows = []

    # بخش اطلاعات کلی
    summary_rows += [
        ["=== BACKTEST SUMMARY ===", "", ""],
        ["Parameter", "Value", "Unit"],
        ["Start Date",    str(start_date.date()),  ""],
        ["End Date",      str(end_date.date()),     ""],
        ["Total Days",    total_days,               "days"],
        ["Pairs",         "EURUSD+GBPUSD",          ""],
        ["Timeframe",     "15min",                  ""],
        ["", "", ""],
        ["=== FINANCIAL RESULTS ===", "", ""],
        ["Initial Balance",  Config.initial_balance, "USD"],
        ["Final Equity",     round(final_eq, 2),     "USD"],
        ["Total PnL",        round(total_pnl, 2),    "USD"],
        ["Total Return",     round(total_ret, 2),    "%"],
        ["Annualized Return",round(ann_ret, 2),      "%"],
        ["Best Trade",       round(best_trade, 2),   "USD"],
        ["Worst Trade",      round(worst_trade, 2),  "USD"],
        ["", "", ""],
        ["=== RISK METRICS ===", "", ""],
        ["Max Drawdown",     round(risk.max_dd, 2),  "%"],
        ["Max DD Absolute",  round(risk.max_dd_abs, 2), "USD"],
        ["Sharpe Ratio",     round(risk.sharpe, 2),  ""],
        ["Sortino Ratio",    round(risk.sortino, 2), ""],
        ["Calmar Ratio",     round(risk.calmar, 2),  ""],
        ["Profit Factor",    round(pf, 2),            ""],
        ["Halt Reason",      risk.halt_reason,        ""],
        ["", "", ""],
        ["=== TRADE STATS ===", "", ""],
        ["Total Trades",     len(t),                  ""],
        ["Win Rate",         round(win_rate, 1),      "%"],
        ["Winning Trades",   len(win_t),              ""],
        ["Losing Trades",    len(loss_t),             ""],
        ["Avg Win",          round(avg_win, 2),       "USD"],
        ["Avg Loss",         round(avg_loss, 2),      "USD"],
        ["Real RR",          round(rr_actual, 2),     ""],
        ["Expectancy",       round(expectancy, 2),    "USD"],
        ["Max Consec Wins",  max_consec_win,          ""],
        ["Max Consec Loss",  max_consec_loss,         ""],
        ["Avg Duration",     round(avg_dur, 0),       "min"],
        ["", "", ""],
        ["=== TIME STATS ===", "", ""],
        ["Trades/Day",       round(tpd, 2),  ""],
        ["Trades/Week",      round(tpw, 2),  ""],
        ["Trades/Month",     round(tpm, 2),  ""],
        ["Profit/Day",       round(ppd, 2),  "USD"],
        ["Profit/Week",      round(ppw, 2),  "USD"],
        ["Profit/Month",     round(ppm, 2),  "USD"],
        ["Profit/Year",      round(ppy, 2),  "USD"],
        ["", "", ""],
        ["=== STRATEGY PERFORMANCE ===", "", ""],
        ["Strategy", "Trades", "WinRate%", "PF", "RR",
         "AvgWin", "AvgLoss", "Expectancy", "TotalPnL", "AvgDuration_min"],
    ]
    for s in strat_stats:
        summary_rows.append([
            s['name'], s['n'], round(s['wr'], 1),
            round(s['pf'], 2), round(s['rr'], 2),
            round(s['aw'], 2), round(s['al'], 2),
            round(s['exp'], 2), round(s['tot'], 2),
            round(s['dur'], 0)
        ])

    summary_rows += [
        ["", "", ""],
        ["=== MONTHLY REPORT ===", "", ""],
        ["Month", "Trades", "WinRate%", "PnL_USD",
         "Return%", "CumPnL_USD", "CumReturn%"],
    ]
    for _, mr in monthly.iterrows():
        summary_rows.append([
            str(mr['ym']), int(mr['trades']),
            round(mr['win_rate'], 1), round(mr['pnl'], 2),
            round(mr['ret_pct'], 2),  round(mr['cum_pnl'], 2),
            round(mr['cum_ret'], 2)
        ])

    summary_rows += [
        ["", "", ""],
        ["=== YEARLY REPORT ===", "", ""],
        ["Year", "Trades", "WinRate%", "PnL_USD", "Return%", "AnnReturn%"],
    ]
    for _, yr in yearly.iterrows():
        summary_rows.append([
            int(yr['year']), int(yr['trades']),
            round(yr['win_rate'], 1), round(yr['pnl'], 2),
            round(yr['ret_pct'], 2), round(yr['ann_ret'] * 100, 2)
        ])

    summary_rows += [
        ["", "", ""],
        ["=== EXIT DISTRIBUTION ===", "", ""],
        ["ExitType", "Count", "Pct%", "AvgPnL_USD"],
    ]
    for status, cnt in t['status'].value_counts().items():
        pct_  = cnt / total_exits * 100
        avg_  = t.loc[t['status'] == status, 'pnl'].mean()
        summary_rows.append([
            status, cnt, round(pct_, 1), round(avg_, 2)
        ])

    pd.DataFrame(summary_rows).to_csv(
        "Backtest_Summary.csv", index=False, header=False, encoding="utf-8-sig"
    )

    # ۳. equity curve
    eq_df = pd.DataFrame({
        'timestamp': risk.curve_ts,
        'equity':    risk.curve,
    })
    eq_df['drawdown_pct'] = (
        (eq_df['equity'] - eq_df['equity'].cummax())
        / eq_df['equity'].cummax() * 100
    ).round(4)
    eq_df.to_csv("equity_curve.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ فایل‌های خروجی ذخیره شد:")
    print(f"   → Backtest_Report.txt   (گزارش کامل متنی)")
    print(f"   → Backtest_Summary.csv  (جداول کامل)")
    print(f"   → equity_curve.csv      ({len(eq_df)} نقطه + drawdown)")


# ================================================================== #
if __name__ == "__main__":
    df           = load_data()
    trades, risk = run_backtest(df)
    report(trades, risk, df)
