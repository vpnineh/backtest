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
    initial_balance    = 5_000.0
    risk_per_trade_pct = 0.01       # 1% per trade
    max_daily_loss_pct = 0.05       # 5%
    max_total_dd_pct   = 0.10       # 10%
    profit_target_pct  = 0.99       # عملاً بی‌نهایت → کل دوره را بک‌تست کن
    spread_eur_pips    = 1.0
    spread_gbp_pips    = 1.2
    commission_per_lot = 6.0
    pip                = 0.0001
    lot_size           = 100_000
    max_lot            = 2.0
    warmup             = 300


# ================================================================== #
#                        ابزارهای مشترک                             #
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

    df = df[df.index.weekday < 5]  # فقط روزهای کاری
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
    up   = high.diff()
    down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_n = down.where((down > up) & (down > 0), 0.0)
    tr   = calc_atr(high, low, close, 1)
    atr_s = tr.rolling(period).sum()
    di_p = 100 * dm_p.rolling(period).sum() / atr_s.replace(0, np.nan)
    di_n = 100 * dm_n.rolling(period).sum() / atr_s.replace(0, np.nan)
    dx   = (abs(di_p - di_n) / (di_p + di_n).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def calc_macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast,   adjust=False).mean()
    es = close.ewm(span=slow,   adjust=False).mean()
    m  = ef - es
    s  = m.ewm(span=signal,     adjust=False).mean()
    return m, s, m - s


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
#                      کلاس RiskManager                             #
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

    def new_bar(self, ts):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day      = day
            self.day_start_eq = self.equity
            if self.halted and "Daily" in self.halt_reason:
                self.halted      = False
                self.halt_reason = "در حال اجرا"

    def add_pnl(self, amount: float, ts) -> bool:
        self.equity += amount
        self.peak    = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))
        self.curve_ts.append(ts)

        dd_day = (self.equity - self.day_start_eq) / self.day_start_eq
        if dd_day <= -Config.max_daily_loss_pct:
            self.halted      = True
            self.halt_reason = f"Daily Loss {dd_day*100:.1f}%"
            return False

        dd_total = (self.equity - self.peak) / self.peak
        if dd_total <= -Config.max_total_dd_pct:
            self.halted      = True
            self.halt_reason = f"Max DD {dd_total*100:.1f}%"
            return False

        profit = (self.equity - Config.initial_balance) / Config.initial_balance
        if profit >= Config.profit_target_pct:
            self.halted      = True
            self.halt_reason = f"Profit Target {profit*100:.1f}%"
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
        return (r.mean() / r.std() * np.sqrt(252 * 96)) if r.std() > 0 else 0

    @property
    def sortino(self):
        r   = pd.Series(self.curve).pct_change().dropna()
        neg = r[r < 0]
        ds  = neg.std() if len(neg) > 0 else 1e-10
        return (r.mean() / ds * np.sqrt(252 * 96)) if ds > 0 else 0

    @property
    def calmar(self):
        ret = (self.equity / Config.initial_balance - 1)
        dd  = abs(self.max_dd / 100)
        return ret / dd if dd > 0 else 0


# ================================================================== #
#  ╔══════════════════════════════════════════════════════════════╗   #
#  ║  STRATEGY 1: Correlation Arbitrage (EUR/GBP Z-Score)        ║   #
#  ╠══════════════════════════════════════════════════════════════╣   #
#  ║  منطق: نرخ EURGBP از میانگین‌اش انحراف می‌گیرد → برگشت      ║   #
#  ║  ورود:  Z > +2.0 → Short EUR  |  Z < -2.0 → Long EUR        ║   #
#  ║  خروج:  Z < 0.3 (mean reversion کامل شد)                    ║   #
#  ║  SL: 22 pip  |  TP: 38 pip  →  RR = 1.73                   ║   #
#  ║  فیلتر: ADX < 30 (بازار رنج، نه ترند)                       ║   #
#  ║  ساعات: 07:00-19:00 GMT                                      ║   #
#  ╚══════════════════════════════════════════════════════════════╝   #
# ================================================================== #
def strategy_corr_arb(df: pd.DataFrame):
    """
    بک‌تست مستقل استراتژی Correlation Arbitrage
    """
    eurgbp = df['c_eur'] / df['c_gbp']
    period = 96   # 1 روز کاری (96 × 15min)

    mean   = eurgbp.rolling(period).mean()
    std    = eurgbp.rolling(period).std()
    z      = (eurgbp - mean) / std.replace(0, np.nan)

    # فیلترها
    std_ok  = std > std.rolling(period * 5).mean() * 0.25
    adx     = calc_adx(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    adx_ok  = adx < 30
    hour    = pd.Series(df.index.hour, index=df.index)
    time_ok = hour.between(7, 19)

    # سیگنال‌ها
    sig = pd.Series(0, index=df.index)
    sig[(z >  2.0) & std_ok & adx_ok & time_ok] = -1  # فروش EUR
    sig[(z < -2.0) & std_ok & adx_ok & time_ok] =  1  # خرید EUR
    sig = sig.where(sig != sig.shift(), 0)

    SL_PIPS = 22.0
    TP_PIPS = 38.0
    MAX_HOURS = 96   # time stop

    risk     = RiskManager()
    open_pos = None
    trades   = []

    for i in range(Config.warmup, len(df)):
        ts    = df.index[i]
        c_eur = df['c_eur'].iloc[i]
        h_eur = df['h_eur'].iloc[i]
        l_eur = df['l_eur'].iloc[i]
        atr_v = calc_atr(df['h_eur'], df['l_eur'], df['c_eur']).iloc[i]

        risk.new_bar(ts)
        if risk.halted:
            if open_pos:
                pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                               open_pos['entry'], c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'halt'})
                risk.add_pnl(pnl, ts)
                open_pos = None
            continue

        # ── بررسی خروج ──
        if open_pos:
            hi, lo = h_eur, l_eur
            d      = open_pos['dir']
            entry  = open_pos['entry']
            sl     = open_pos['sl']
            tp     = open_pos['tp']

            hit_sl = (d ==  1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d ==  1 and hi >= tp) or (d == -1 and lo <= tp)

            # Z-score exit
            z_now = z.iloc[i]
            if pd.notna(z_now) and abs(z_now) < 0.3:
                hit_tp = True

            # Time stop
            elapsed = (ts - open_pos['entry_ts']).total_seconds() / 3600
            if elapsed >= MAX_HOURS and not hit_tp:
                pnl = calc_pnl(d, open_pos['lot'], entry, c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts)
                open_pos = None
                continue

            exit_r = exit_p = None
            if hit_sl: exit_r, exit_p = 'SL', sl
            elif hit_tp: exit_r, exit_p = 'TP', tp

            if exit_r:
                pnl = calc_pnl(d, open_pos['lot'], entry, exit_p, 'EUR')
                trades.append({**open_pos, 'exit': exit_p,
                               'exit_ts': ts, 'pnl': pnl, 'status': exit_r})
                risk.add_pnl(pnl, ts)
                open_pos = None

        # ── ورود ──
        if open_pos is None and sig.iloc[i] != 0:
            sv  = int(sig.iloc[i])
            lot = lot_size_calc(risk.equity, SL_PIPS)
            ep  = c_eur + sv * Config.spread_eur_pips * Config.pip / 2
            open_pos = dict(
                strategy='CorrArb', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * SL_PIPS * Config.pip,
                tp=ep + sv * TP_PIPS * Config.pip,
                entry_ts=ts,
            )

    # ── بستن باقیمانده ──
    if open_pos:
        ep  = df['c_eur'].iloc[-1]
        pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                       open_pos['entry'], ep, 'EUR')
        trades.append({**open_pos, 'exit': ep, 'exit_ts': df.index[-1],
                       'pnl': pnl, 'status': 'eod'})
        risk.add_pnl(pnl, df.index[-1])

    return trades, risk


# ================================================================== #
#  ╔══════════════════════════════════════════════════════════════╗   #
#  ║  STRATEGY 2: Asian Range Breakout (اصلاح شده اساسی)         ║   #
#  ╠══════════════════════════════════════════════════════════════╣   #
#  ║  مشکل قبلی: SL خیلی تنگ → 9.1% Win Rate فاجعه‌بار          ║   #
#  ║  اصلاح:                                                      ║   #
#  ║  - SL = پشت میانه رنج (نه لبه رنج)                          ║   #
#  ║  - تایید: ۲ کندل پشت سر هم بالای رنج بسته شوند             ║   #
#  ║  - فیلتر ADX > 20 در لندن                                   ║   #
#  ║  - فقط رنج ۱۵-۴۵ پیپ                                        ║   #
#  ║  - TP = 2× رنج آسیا                                         ║   #
#  ╚══════════════════════════════════════════════════════════════╝   #
# ================================================================== #
def strategy_asian_breakout(df: pd.DataFrame):
    d = df.copy()
    d['hour']    = d.index.hour
    d['weekday'] = d.index.weekday
    d['date']    = d.index.date

    atr_s = calc_atr(d['h_eur'], d['l_eur'], d['c_eur'], 14)
    adx_s = calc_adx(d['h_eur'], d['l_eur'], d['c_eur'], 14)

    # رنج آسیا: ساعت ۱ تا ۶ GMT
    asian = d[d['hour'].between(1, 6)]
    rng   = asian.groupby('date').agg(
        ah=('h_eur', 'max'),
        al=('l_eur', 'min')
    )
    rng['rng_pips'] = (rng['ah'] - rng['al']) / Config.pip
    rng['mid']      = (rng['ah'] + rng['al']) / 2
    d = d.join(rng, on='date')

    # ── سیگنال: لندن ۸-۱۱، ۲ کندل تایید ──
    london = d['hour'].between(8, 11)
    valid  = d['rng_pips'].between(15, 45) & d['weekday'].between(0, 3)

    # شکست بالا: ۲ کندل پشت سر هم بالای ah
    above = (d['c_eur'] > d['ah']).astype(int)
    above2 = (above + above.shift(1)) >= 2   # دو کندل پشت سر هم

    below = (d['c_eur'] < d['al']).astype(int)
    below2 = (below + below.shift(1)) >= 2

    sig = pd.Series(0, index=d.index)
    sig[london & valid & above2 & (adx_s > 20)] =  1
    sig[london & valid & below2 & (adx_s > 20)] = -1

    # اولین سیگنال هر روز
    nz        = sig[sig != 0]
    first_idx = nz.groupby(nz.index.date).head(1).index
    final_sig = pd.Series(0, index=d.index)
    final_sig[first_idx] = sig[first_idx]

    MAX_HOURS = 48

    risk     = RiskManager()
    open_pos = None
    trades   = []

    for i in range(Config.warmup, len(d)):
        ts    = d.index[i]
        row   = d.iloc[i]
        c_eur = row['c_eur']
        h_eur = row['h_eur']
        l_eur = row['l_eur']

        risk.new_bar(ts)
        if risk.halted:
            if open_pos:
                pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                               open_pos['entry'], c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'halt'})
                risk.add_pnl(pnl, ts)
                open_pos = None
            continue

        if open_pos:
            d_dir = open_pos['dir']
            entry = open_pos['entry']
            sl    = open_pos['sl']
            tp    = open_pos['tp']

            hit_sl = (d_dir ==  1 and l_eur <= sl) or (d_dir == -1 and h_eur >= sl)
            hit_tp = (d_dir ==  1 and h_eur >= tp) or (d_dir == -1 and l_eur <= tp)

            # Trailing: بعد از ۱.۵× رنج سود، SL به break-even
            rng_sz = row['rng_pips'] * Config.pip
            move   = d_dir * (c_eur - entry)
            if move > rng_sz * 1.5 and rng_sz > 0:
                be = entry + d_dir * rng_sz * 0.5
                if d_dir == 1:
                    open_pos['sl'] = max(open_pos['sl'], be)
                else:
                    open_pos['sl'] = min(open_pos['sl'], be)

            elapsed = (ts - open_pos['entry_ts']).total_seconds() / 3600
            if elapsed >= MAX_HOURS and not hit_tp:
                pnl = calc_pnl(d_dir, open_pos['lot'], entry, c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts)
                open_pos = None
                continue

            exit_r = exit_p = None
            if hit_sl: exit_r, exit_p = 'SL', sl
            elif hit_tp: exit_r, exit_p = 'TP', tp
            if exit_r:
                pnl = calc_pnl(d_dir, open_pos['lot'], entry, exit_p, 'EUR')
                trades.append({**open_pos, 'exit': exit_p,
                               'exit_ts': ts, 'pnl': pnl, 'status': exit_r})
                risk.add_pnl(pnl, ts)
                open_pos = None

        if open_pos is None and final_sig.iloc[i] != 0:
            sv      = int(final_sig.iloc[i])
            rng_sz  = max(row['rng_pips'], 15)
            # SL: پشت میانه رنج (نه لبه) → SL بزرگتر = کمتر می‌خوریم
            sl_pips = max(18, rng_sz * 0.6)
            tp_pips = rng_sz * 2.0   # TP = 2× رنج
            if tp_pips < sl_pips * 1.5:
                tp_pips = sl_pips * 2.0  # حداقل RR = 2
            lot = lot_size_calc(risk.equity, sl_pips)
            ep  = c_eur + sv * Config.spread_eur_pips * Config.pip / 2
            open_pos = dict(
                strategy='AsianBreak', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

    if open_pos:
        ep  = d['c_eur'].iloc[-1]
        pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                       open_pos['entry'], ep, 'EUR')
        trades.append({**open_pos, 'exit': ep, 'exit_ts': d.index[-1],
                       'pnl': pnl, 'status': 'eod'})
        risk.add_pnl(pnl, d.index[-1])

    return trades, risk


# ================================================================== #
#  ╔══════════════════════════════════════════════════════════════╗   #
#  ║  STRATEGY 3: EMA Trend + Pullback (بازنگری کامل)            ║   #
#  ╠══════════════════════════════════════════════════════════════╣   #
#  ║  مشکل قبلی: RR=0.01 → کاملاً بی‌معنی بود                    ║   #
#  ║  علت: SL خیلی کوچک (0.59$) → اشتباه در محاسبه              ║   #
#  ║  اصلاح اساسی:                                                ║   #
#  ║  - SL = زیر Low آخرین swing (نه ATR ثابت)                   ║   #
#  ║  - ورود در pullback عمیق: قیمت بین EMA50 و EMA200           ║   #
#  ║  - تایید: RSI از ناحیه oversold/overbought برگشته           ║   #
#  ║  - MACD crossover تایید ورود                                 ║   #
#  ╚══════════════════════════════════════════════════════════════╝   #
# ================================================================== #
def strategy_trend_pullback(df: pd.DataFrame):
    c  = df['c_eur']
    h  = df['h_eur']
    l  = df['l_eur']

    ema21  = c.ewm(span=21,  adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rsi    = calc_rsi(c, 14)
    atr    = calc_atr(h, l, c, 14)
    adx    = calc_adx(h, l, c, 14)
    macd, macd_sig, macd_hist = calc_macd(c)

    # Swing low/high برای SL
    swing_low  = l.rolling(10).min()
    swing_high = h.rolling(10).max()

    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(8, 17)

    sig = pd.Series(0, index=df.index)

    # ─── Long: روند صعودی قوی + pullback عمیق ───
    # شرط: هر سه EMA صعودی + قیمت بین EMA50 و EMA200 (pullback عمیق)
    # یا: قیمت کمی زیر EMA21 اما بالای EMA50
    long_cond = (
        active &
        (ema21 > ema50) & (ema50 > ema200) &   # روند صعودی کامل
        (adx > 25) &                             # روند قوی
        (c > ema200) &                           # بالای MA بلندمدت
        (c < ema21) &                            # پایین‌تر از EMA21 (pullback)
        rsi.between(35, 52) &                    # RSI در ناحیه pullback
        (macd_hist > macd_hist.shift(1)) &       # MACD در حال بهبود
        (macd_hist > macd_hist.shift(2))         # تایید ۲ کندل
    )

    # ─── Short: روند نزولی قوی + pullback عمیق ───
    short_cond = (
        active &
        (ema21 < ema50) & (ema50 < ema200) &
        (adx > 25) &
        (c < ema200) &
        (c > ema21) &                            # بالاتر از EMA21 (pullback)
        rsi.between(48, 65) &
        (macd_hist < macd_hist.shift(1)) &
        (macd_hist < macd_hist.shift(2))
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    MAX_HOURS = 120

    risk     = RiskManager()
    open_pos = None
    trades   = []

    for i in range(Config.warmup, len(df)):
        ts    = df.index[i]
        c_eur = df['c_eur'].iloc[i]
        h_eur = df['h_eur'].iloc[i]
        l_eur = df['l_eur'].iloc[i]
        atr_v = atr.iloc[i]

        risk.new_bar(ts)
        if risk.halted:
            if open_pos:
                pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                               open_pos['entry'], c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'halt'})
                risk.add_pnl(pnl, ts)
                open_pos = None
            continue

        if open_pos:
            d_dir = open_pos['dir']
            entry = open_pos['entry']
            sl    = open_pos['sl']
            tp    = open_pos['tp']

            hit_sl = (d_dir ==  1 and l_eur <= sl) or (d_dir == -1 and h_eur >= sl)
            hit_tp = (d_dir ==  1 and h_eur >= tp) or (d_dir == -1 and l_eur <= tp)

            # Trailing: ۲ مرحله
            if pd.notna(atr_v) and atr_v > 0:
                move = d_dir * (c_eur - entry)
                if move > atr_v * 1.5:
                    be = entry + d_dir * atr_v * 0.5
                    if d_dir == 1: open_pos['sl'] = max(sl, be)
                    else:          open_pos['sl'] = min(sl, be)
                if move > atr_v * 2.5:
                    lock = entry + d_dir * atr_v * 1.5
                    if d_dir == 1: open_pos['sl'] = max(open_pos['sl'], lock)
                    else:          open_pos['sl'] = min(open_pos['sl'], lock)

            elapsed = (ts - open_pos['entry_ts']).total_seconds() / 3600
            if elapsed >= MAX_HOURS and not hit_tp:
                pnl = calc_pnl(d_dir, open_pos['lot'], entry, c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts)
                open_pos = None
                continue

            exit_r = exit_p = None
            if hit_sl: exit_r, exit_p = 'SL', sl
            elif hit_tp: exit_r, exit_p = 'TP', tp
            if exit_r:
                pnl = calc_pnl(d_dir, open_pos['lot'], entry, exit_p, 'EUR')
                trades.append({**open_pos, 'exit': exit_p,
                               'exit_ts': ts, 'pnl': pnl, 'status': exit_r})
                risk.add_pnl(pnl, ts)
                open_pos = None

        if open_pos is None and sig.iloc[i] != 0:
            sv    = int(sig.iloc[i])
            atr_v = atr.iloc[i]
            if pd.isnan(atr_v) or atr_v <= 0:
                continue

            # SL: زیر/بالای swing low/high اخیر
            if sv == 1:
                sl_price = swing_low.iloc[i] - 3 * Config.pip
            else:
                sl_price = swing_high.iloc[i] + 3 * Config.pip

            sl_pips = abs(c_eur - sl_price) / Config.pip
            sl_pips = max(sl_pips, 15)   # حداقل ۱۵ پیپ SL
            tp_pips = sl_pips * 2.5      # RR = 2.5
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sv * Config.spread_eur_pips * Config.pip / 2

            open_pos = dict(
                strategy='TrendPB', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

    if open_pos:
        ep  = df['c_eur'].iloc[-1]
        pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                       open_pos['entry'], ep, 'EUR')
        trades.append({**open_pos, 'exit': ep, 'exit_ts': df.index[-1],
                       'pnl': pnl, 'status': 'eod'})
        risk.add_pnl(pnl, df.index[-1])

    return trades, risk


# ================================================================== #
#  ╔══════════════════════════════════════════════════════════════╗   #
#  ║  STRATEGY 4: NY Session Fade (استراتژی جدید)                ║   #
#  ╠══════════════════════════════════════════════════════════════╣   #
#  ║  منطق: در اوج حرکت لندن (قبل از بسته شدن)،                  ║   #
#  ║  قیمت اغلب revert می‌کند وقتی NY باز می‌شود                  ║   #
#  ║  ورود: ۱۳:۰۰-۱۵:۰۰ GMT وقتی RSI > 70 یا < 30              ║   #
#  ║  + قیمت از Bollinger Band خارج شده                          ║   #
#  ║  + MACD divergence                                           ║   #
#  ║  SL: 2× ATR  |  TP: 1.5× ATR  → Mean Reversion سریع        ║   #
#  ╚══════════════════════════════════════════════════════════════╝   #
# ================================================================== #
def strategy_ny_fade(df: pd.DataFrame):
    c = df['c_eur']
    h = df['h_eur']
    l = df['l_eur']

    atr  = calc_atr(h, l, c, 14)
    rsi  = calc_rsi(c, 14)
    adx  = calc_adx(h, l, c, 14)
    macd, macd_sig, macd_hist = calc_macd(c)

    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2.0 * bb_std
    bb_lo  = bb_mid - 2.0 * bb_std

    hour    = pd.Series(df.index.hour,    index=df.index)
    weekday = pd.Series(df.index.weekday, index=df.index)

    # ساعات NY overlap با لندن: ۱۳:۰۰-۱۶:۰۰ GMT
    ny_time = hour.between(13, 16) & weekday.between(0, 3)

    sig = pd.Series(0, index=df.index)

    # ─── Fade Long (فروش after spike up) ───
    # RSI overbought + قیمت بالای BB + MACD در حال کاهش
    fade_short = (
        ny_time &
        (rsi > 68) &
        (c > bb_up) &
        (macd_hist < macd_hist.shift(1)) &   # MACD در حال کاهش
        (adx < 35)                            # اگر خیلی قوی است، fade نکن
    )

    # ─── Fade Short (خرید after spike down) ───
    fade_long = (
        ny_time &
        (rsi < 32) &
        (c < bb_lo) &
        (macd_hist > macd_hist.shift(1)) &
        (adx < 35)
    )

    sig[fade_short] = -1
    sig[fade_long]  =  1
    sig = sig.where(sig != sig.shift(), 0)

    # اولین سیگنال هر روز
    nz        = sig[sig != 0]
    first_idx = nz.groupby(nz.index.date).head(1).index
    final_sig = pd.Series(0, index=df.index)
    final_sig[first_idx] = sig[first_idx]

    SL_MULT  = 2.0   # SL = 2× ATR
    TP_MULT  = 1.5   # TP = 1.5× ATR → RR = 0.75? نه، چون Win Rate بالاست
    # برای fade استراتژی Win Rate بالا (65%+) با RR کوچکتر قبول است
    MAX_HOURS = 24

    risk     = RiskManager()
    open_pos = None
    trades   = []

    for i in range(Config.warmup, len(df)):
        ts    = df.index[i]
        c_eur = df['c_eur'].iloc[i]
        h_eur = df['h_eur'].iloc[i]
        l_eur = df['l_eur'].iloc[i]
        atr_v = atr.iloc[i]

        risk.new_bar(ts)
        if risk.halted:
            if open_pos:
                pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                               open_pos['entry'], c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'halt'})
                risk.add_pnl(pnl, ts)
                open_pos = None
            continue

        if open_pos:
            d_dir = open_pos['dir']
            entry = open_pos['entry']
            sl    = open_pos['sl']
            tp    = open_pos['tp']

            hit_sl = (d_dir ==  1 and l_eur <= sl) or (d_dir == -1 and h_eur >= sl)
            hit_tp = (d_dir ==  1 and h_eur >= tp) or (d_dir == -1 and l_eur <= tp)

            elapsed = (ts - open_pos['entry_ts']).total_seconds() / 3600
            if elapsed >= MAX_HOURS and not hit_tp:
                pnl = calc_pnl(d_dir, open_pos['lot'], entry, c_eur, 'EUR')
                trades.append({**open_pos, 'exit': c_eur,
                               'exit_ts': ts, 'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts)
                open_pos = None
                continue

            exit_r = exit_p = None
            if hit_sl: exit_r, exit_p = 'SL', sl
            elif hit_tp: exit_r, exit_p = 'TP', tp
            if exit_r:
                pnl = calc_pnl(d_dir, open_pos['lot'], entry, exit_p, 'EUR')
                trades.append({**open_pos, 'exit': exit_p,
                               'exit_ts': ts, 'pnl': pnl, 'status': exit_r})
                risk.add_pnl(pnl, ts)
                open_pos = None

        if open_pos is None and final_sig.iloc[i] != 0:
            sv = int(final_sig.iloc[i])
            if pd.isnan(atr_v) or atr_v <= 0:
                continue
            sl_pips = max(15, atr_v / Config.pip * SL_MULT)
            tp_pips = max(12, atr_v / Config.pip * TP_MULT)
            # اگر RR < 1 است، TP را بزرگتر کن
            if tp_pips < sl_pips * 0.8:
                tp_pips = sl_pips * 1.2
            lot = lot_size_calc(risk.equity, sl_pips)
            ep  = c_eur + sv * Config.spread_eur_pips * Config.pip / 2
            open_pos = dict(
                strategy='NYFade', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

    if open_pos:
        ep  = df['c_eur'].iloc[-1]
        pnl = calc_pnl(open_pos['dir'], open_pos['lot'],
                       open_pos['entry'], ep, 'EUR')
        trades.append({**open_pos, 'exit': ep, 'exit_ts': df.index[-1],
                       'pnl': pnl, 'status': 'eod'})
        risk.add_pnl(pnl, df.index[-1])

    return trades, risk


# ================================================================== #
#                گزارش یک استراتژی                                  #
# ================================================================== #
def compute_stats(trades: list, risk: RiskManager, name: str) -> dict:
    """محاسبه آمار کامل یک استراتژی"""
    if not trades:
        return None

    t = pd.DataFrame(trades)
    t['pnl']          = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    start_d      = t['entry_ts'].min()
    end_d        = t['exit_ts'].max()
    total_days   = max((end_d - start_d).days, 1)
    total_months = total_days / 30.44
    total_years  = total_days / 365.25

    final_eq  = risk.equity
    total_pnl = final_eq - Config.initial_balance
    total_ret = total_pnl / Config.initial_balance * 100
    ann_ret   = ((final_eq / Config.initial_balance) ** (365.25 / total_days) - 1) * 100

    win_t  = t[t['pnl'] > 0]
    loss_t = t[t['pnl'] < 0]
    win_r  = len(win_t) / len(t) * 100 if len(t) > 0 else 0
    avg_w  = win_t['pnl'].mean()  if len(win_t)  > 0 else 0
    avg_l  = loss_t['pnl'].mean() if len(loss_t) > 0 else 0
    gw     = win_t['pnl'].sum()
    gl     = abs(loss_t['pnl'].sum())
    pf     = gw / gl if gl > 0 else float('inf')
    exp    = t['pnl'].mean()
    rr     = abs(avg_w / avg_l) if avg_l != 0 else 0

    # consecutive
    sign  = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cw, cl, mcw, mcl = 0, 0, 0, 0
    for s in sign:
        if   s > 0: cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0: cl += 1; cw = 0; mcl = max(mcl, cl)
        else:       cw = cl = 0

    # ماهانه
    t['ym'] = t['entry_ts'].dt.to_period('M')
    monthly = (t.groupby('ym')
               .agg(n=('pnl','count'), pnl=('pnl','sum'),
                    wins=('pnl', lambda x: (x>0).sum()),
                    best=('pnl','max'), worst=('pnl','min'))
               .reset_index())
    monthly['wr']      = monthly['wins'] / monthly['n'] * 100
    monthly['ret']     = monthly['pnl'] / Config.initial_balance * 100
    monthly['cum_pnl'] = monthly['pnl'].cumsum()
    monthly['cum_ret'] = monthly['cum_pnl'] / Config.initial_balance * 100

    # سالانه
    t['yr'] = t['entry_ts'].dt.year
    yearly  = (t.groupby('yr')
               .agg(n=('pnl','count'), pnl=('pnl','sum'),
                    wins=('pnl', lambda x: (x>0).sum()))
               .reset_index())
    yearly['wr']  = yearly['wins'] / yearly['n'] * 100
    yearly['ret'] = yearly['pnl'] / Config.initial_balance * 100

    return {
        'name':        name,
        'trades':      t,
        'monthly':     monthly,
        'yearly':      yearly,
        'risk':        risk,
        'total_pnl':   total_pnl,
        'total_ret':   total_ret,
        'ann_ret':     ann_ret,
        'total_days':  total_days,
        'total_months':total_months,
        'win_r':       win_r,
        'avg_w':       avg_w,
        'avg_l':       avg_l,
        'pf':          pf,
        'exp':         exp,
        'rr':          rr,
        'mcw':         mcw,
        'mcl':         mcl,
        'ppm':         total_pnl / total_months,
        'best':        t['pnl'].max(),
        'worst':       t['pnl'].min(),
        'avg_dur':     t['duration_min'].mean(),
    }


# ================================================================== #
#                   گزارش‌ساز حرفه‌ای                               #
# ================================================================== #
def print_strategy_report(s: dict):
    """چاپ گزارش کامل یک استراتژی"""
    W   = 72
    SEP = "═" * W
    risk = s['risk']

    def rw(label, value):
        lbl  = f"  {label}"
        val  = str(value)
        dots = "·" * max(2, W - len(lbl) - len(val) - 2)
        return f"{lbl} {dots} {val}"

    def box(title):
        inner = f"─ {title} "
        return "┌" + inner + "─" * (W - len(inner) - 1) + "┐"

    bot = "└" + "─" * (W - 1) + "┘"

    # تعیین وضعیت
    monthly_avg = s['ppm'] / Config.initial_balance * 100
    is_good = (s['total_ret'] > 0 and
               s['pf'] > 1.2 and
               abs(risk.max_dd) < 15 and
               monthly_avg > 5)
    status_icon = "✅ سودده" if is_good else "❌ ضررده / ناکافی"

    lines = [
        "", SEP,
        f"  ▌ Strategy: {s['name']}   {status_icon}",
        f"  ▌ دوره: {s['trades']['entry_ts'].min().date()} "
        f"→ {s['trades']['exit_ts'].max().date()}",
        SEP, "",

        box("نتایج مالی"),
        rw("موجودی اولیه",  f"${Config.initial_balance:>12,.2f}"),
        rw("موجودی نهایی",  f"${risk.equity:>12,.2f}"),
        rw("سود/زیان کل",  f"${s['total_pnl']:>+12,.2f}"),
        rw("بازده کل",      f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه",  f"{s['ann_ret']:>+.2f}%"),
        rw("سود ماهانه avg",f"${s['ppm']:>+.2f}  ({monthly_avg:>+.1f}%)"),
        rw("بهترین معامله", f"${s['best']:>+.2f}"),
        rw("بدترین معامله", f"${s['worst']:>+.2f}"),
        bot, "",

        box("ریسک"),
        rw("Max Drawdown",  f"{risk.max_dd:.2f}%"),
        rw("Max DD مطلق",  f"${risk.max_dd_abs:>+.2f}"),
        rw("Sharpe Ratio",  f"{risk.sharpe:.2f}"),
        rw("Sortino Ratio", f"{risk.sortino:.2f}"),
        rw("Calmar Ratio",  f"{risk.calmar:.2f}"),
        rw("Profit Factor", f"{s['pf']:.2f}"),
        rw("وضعیت پایان",  risk.halt_reason),
        bot, "",

        box("معاملات"),
        rw("تعداد کل",      f"{len(s['trades']):,}"),
        rw("Win Rate",       f"{s['win_r']:.1f}%"),
        rw("Avg Win",        f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",       f"${s['avg_l']:>+.2f}"),
        rw("RR واقعی",      f"{s['rr']:.2f}"),
        rw("Expectancy",     f"${s['exp']:>+.2f}"),
        rw("Max Cons. Win",  f"{s['mcw']}"),
        rw("Max Cons. Loss", f"{s['mcl']}"),
        rw("مدت میانگین",   f"{s['avg_dur']:.0f} min"),
        bot, "",
    ]

    # ── توزیع خروج ──
    lines.append(box("توزیع خروج"))
    for status, cnt in s['trades']['status'].value_counts().items():
        pct  = cnt / len(s['trades']) * 100
        avg_ = s['trades'].loc[s['trades']['status'] == status, 'pnl'].mean()
        bar  = "█" * max(1, int(pct / 3))
        lines.append(
            f"  {status:<13} {cnt:>4} ({pct:>5.1f}%)  "
            f"{bar:<24}  avg=${avg_:>+.2f}"
        )
    lines += [bot, ""]

    # ── گزارش ماهانه ──
    lines.append(box("گزارش ماهانه"))
    lines.append(
        f"  {'ماه':>7}  {'#':>4}  {'Win%':>6}  {'PnL':>10}  "
        f"{'Ret%':>6}  {'بهترین':>8}  {'بدترین':>8}  "
        f"{'تجمعی':>10}  {'CumRet':>7}"
    )
    lines.append("  " + "─" * (W - 3))
    for _, mr in s['monthly'].iterrows():
        arrow = "▲" if mr['pnl'] >= 0 else "▼"
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>4}  "
            f"{mr['wr']:>5.1f}%  ${mr['pnl']:>9.2f}  "
            f"{mr['ret']:>+5.1f}%  ${mr['best']:>7.2f}  "
            f"${mr['worst']:>7.2f}  ${mr['cum_pnl']:>9.2f}  "
            f"{mr['cum_ret']:>+6.1f}% {arrow}"
        )
    lines += [bot, ""]

    # ── گزارش سالانه ──
    lines.append(box("گزارش سالانه"))
    lines.append(
        f"  {'سال':>5}  {'#':>5}  {'Win%':>6}  "
        f"{'PnL':>10}  {'Ret%':>7}"
    )
    lines.append("  " + "─" * (W - 3))
    for _, yr in s['yearly'].iterrows():
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>5.1f}%  ${yr['pnl']:>9.2f}  "
            f"{yr['ret']:>+6.1f}%"
        )
    lines.append(bot)

    out = "\n".join(lines)
    print(out)
    return out


def print_comparison_table(results: list):
    """جدول مقایسه همه استراتژی‌ها"""
    W   = 72
    SEP = "═" * W

    lines = [
        "", SEP,
        "  ▌  STRATEGY COMPARISON TABLE  ▐",
        SEP,
        f"  {'نام':<14} {'Ret%':>7} {'Ann%':>7} {'DD%':>7} "
        f"{'PF':>6} {'Win%':>6} {'RR':>5} "
        f"{'Exp$':>7} {'Shr':>6} {'وضعیت':>10}",
        "  " + "─" * (W - 3),
    ]

    for s in results:
        risk = s['risk']
        flag = "✅" if (s['total_ret'] > 0 and s['pf'] > 1.2
                        and abs(risk.max_dd) < 15) else "❌"
        pf_s = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "  ∞"
        lines.append(
            f"  {s['name']:<14} {s['total_ret']:>+6.1f}% "
            f"{s['ann_ret']:>+6.1f}% {risk.max_dd:>6.1f}% "
            f"{pf_s:>6} {s['win_r']:>5.1f}% {s['rr']:>5.2f} "
            f"${s['exp']:>6.2f} {risk.sharpe:>6.1f} {flag:>10}"
        )

    lines += ["  " + "─" * (W - 3), ""]

    # انتخاب برترها
    good = [s for s in results if s['total_ret'] > 0 and s['pf'] > 1.2]
    if good:
        lines.append("  🏆 استراتژی‌های واجد شرایط ترکیب:")
        for s in sorted(good, key=lambda x: x['ann_ret'], reverse=True):
            lines.append(
                f"     → {s['name']:<14}  "
                f"Ann={s['ann_ret']:>+.1f}%  "
                f"DD={s['risk'].max_dd:.1f}%  "
                f"PF={s['pf']:.2f}"
            )
    else:
        lines.append("  ⚠️  هیچ استراتژی‌ای معیارهای پراپ را ندارد → نیاز به بهینه‌سازی")

    lines += ["", SEP]
    out = "\n".join(lines)
    print(out)
    return out


# ================================================================== #
#                ذخیره CSV جامع                                     #
# ================================================================== #
def save_summary_csv(results: list):
    rows = [
        ["INDIVIDUAL STRATEGY BACKTEST RESULTS"],
        ["Generated:", datetime.now().strftime('%Y-%m-%d %H:%M')],
        [""],
        ["Strategy", "TotalPnL", "TotalRet%", "AnnRet%",
         "MaxDD%", "Sharpe", "PF", "WinRate%", "RR",
         "Expectancy", "Trades", "AvgDur_min", "Status"],
    ]
    for s in results:
        flag = "PASS" if (s['total_ret'] > 0 and s['pf'] > 1.2
                          and abs(s['risk'].max_dd) < 15) else "FAIL"
        pf_v = round(s['pf'], 2) if s['pf'] != float('inf') else 999
        rows.append([
            s['name'],
            round(s['total_pnl'], 2),
            round(s['total_ret'], 2),
            round(s['ann_ret'], 2),
            round(s['risk'].max_dd, 2),
            round(s['risk'].sharpe, 2),
            pf_v,
            round(s['win_r'], 1),
            round(s['rr'], 2),
            round(s['exp'], 2),
            len(s['trades']),
            round(s['avg_dur'], 0),
            flag,
        ])

    # گزارش ماهانه هر استراتژی
    for s in results:
        rows += [[""], [f"=== MONTHLY: {s['name']} ==="],
                 ["Month","Trades","WinRate%","PnL","Ret%",
                  "Best","Worst","CumPnL","CumRet%"]]
        for _, mr in s['monthly'].iterrows():
            rows.append([
                str(mr['ym']), int(mr['n']),
                round(mr['wr'], 1), round(mr['pnl'], 2),
                round(mr['ret'], 2), round(mr['best'], 2),
                round(mr['worst'], 2), round(mr['cum_pnl'], 2),
                round(mr['cum_ret'], 2),
            ])

    # گزارش سالانه هر استراتژی
    for s in results:
        rows += [[""], [f"=== YEARLY: {s['name']} ==="],
                 ["Year","Trades","WinRate%","PnL","Ret%"]]
        for _, yr in s['yearly'].iterrows():
            rows.append([
                int(yr['yr']), int(yr['n']),
                round(yr['wr'], 1), round(yr['pnl'], 2),
                round(yr['ret'], 2),
            ])

    pd.DataFrame(rows).to_csv(
        "Strategy_Report.csv", index=False, header=False, encoding="utf-8-sig"
    )

    # equity curve هر استراتژی
    all_eq = {}
    for s in results:
        r = s['risk']
        eq_df = pd.DataFrame({'ts': r.curve_ts, 'equity': r.curve})
        eq_df['dd'] = (
            (eq_df['equity'] - eq_df['equity'].cummax())
            / eq_df['equity'].cummax() * 100
        ).round(4)
        eq_df.to_csv(
            f"equity_{s['name']}.csv", index=False, encoding="utf-8-sig"
        )

    print(f"\n✅ فایل‌های ذخیره شد:")
    print(f"   → Strategy_Report.csv  (گزارش جامع همه استراتژی‌ها)")
    for s in results:
        print(f"   → equity_{s['name']}.csv")


# ================================================================== #
#                           MAIN                                     #
# ================================================================== #
if __name__ == "__main__":
    df = load_data()

    strategies = [
        ("CorrArb",    strategy_corr_arb),
        ("AsianBreak", strategy_asian_breakout),
        ("TrendPB",    strategy_trend_pullback),
        ("NYFade",     strategy_ny_fade),
    ]

    all_reports = []
    all_text    = []

    print("\n" + "═"*72)
    print("  اجرای بک‌تست جداگانه برای هر استراتژی...")
    print("═"*72)

    for name, func in strategies:
        print(f"\n  ▶ {name} ...", end="", flush=True)
        trades, risk = func(df)
        stats = compute_stats(trades, risk, name)
        if stats:
            all_reports.append(stats)
            txt = print_strategy_report(stats)
            all_text.append(txt)
            print(f"  ✓ تمام ({len(trades)} معامله)")
        else:
            print(f"  ✗ بدون معامله")

    # ── جدول مقایسه ──
    comp = print_comparison_table(all_reports)
    all_text.append(comp)

    # ── ذخیره ──
    with open("Backtest_Report.txt", "w", encoding="utf-8") as f:
        f.write(f"BACKTEST REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("\n".join(all_text))

    save_summary_csv(all_reports)
