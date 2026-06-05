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
    risk_per_trade_pct   = 0.008      # 0.8% per trade - محافظه‌کارانه‌تر
    max_daily_loss_pct   = 0.04       # 4% daily limit
    max_total_dd_pct     = 0.08       # 8% max DD (پراپ استاندارد)
    profit_target_pct    = 0.50       # 50% target برای بک‌تست کامل
    spread_eur_pips      = 1.0
    spread_gbp_pips      = 1.2
    commission_per_lot   = 6.0
    pip                  = 0.0001
    lot_size             = 100_000
    max_open_positions   = 4
    max_lot              = 1.5
    atr_period           = 14


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

    # فقط روزهای کاری
    df = df[df.index.weekday < 5]

    print(f"✅ {len(df):,} کندل | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── اندیکاتورها ──────────────────────────────────────────────────── #
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
    ema_f = close.ewm(span=fast,   adjust=False).mean()
    ema_s = close.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal,  adjust=False).mean()
    return macd, sig, macd - sig


def calc_bbands(close, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + mult * std, mid, mid - mult * std


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
        self.equity        = Config.initial_balance
        self.peak          = Config.initial_balance
        self.day_start_eq  = Config.initial_balance
        self.cur_day       = None
        self.halted        = False
        self.halt_reason   = "در حال اجرا"
        self.curve         = [Config.initial_balance]
        self.curve_ts      = [None]
        self.daily_log     = {}   # date → pnl
        self.monthly_log   = {}   # YYYY-MM → pnl

    def new_bar(self, ts: pd.Timestamp):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day      = day
            self.day_start_eq = self.equity
            if self.halted and "Daily" in self.halt_reason:
                self.halted      = False
                self.halt_reason = "در حال اجرا"

    def add_pnl(self, amount: float, ts: pd.Timestamp) -> bool:
        self.equity += amount
        self.peak    = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))
        self.curve_ts.append(ts)

        d_key = str(ts.date())
        m_key = ts.strftime('%Y-%m')
        self.daily_log[d_key]   = self.daily_log.get(d_key,   0) + amount
        self.monthly_log[m_key] = self.monthly_log.get(m_key, 0) + amount

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
#  Strategy 1 ─ Correlation Arbitrage  (اصلی‌ترین و سودده‌ترین)     #
#                                                                    #
#  منطق: Z-score نرخ EUR/GBP                                        #
#  ورود: |Z| > 2.0  →  mean-reversion                              #
#  خروج: |Z| < 0.3  یا  SL/TP                                      #
#  فیلتر: ADX < 28 (بازار رنج - نه ترند قوی)                       #
#  SL: 20 پیپ  |  TP: 35 پیپ  →  RR = 1.75                        #
# ================================================================== #
def build_corr_arb(df):
    eurgbp = df['c_eur'] / df['c_gbp']
    period = 96   # 1 روز کاری

    mean  = eurgbp.rolling(period).mean()
    std   = eurgbp.rolling(period).std()
    z     = (eurgbp - mean) / std.replace(0, np.nan)

    # فیلتر: std باید قابل توجه باشد
    std_ok  = std > std.rolling(period * 4).mean() * 0.3
    adx_eur = calc_adx(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    adx_ok  = adx_eur < 28    # رنج - نه ترند

    # ساعات مناسب: لندن + NY (نه آسیا که spread بالاست)
    hour    = pd.Series(df.index.hour, index=df.index)
    time_ok = hour.between(7, 19)

    sig = pd.Series(0, index=df.index)
    sig[(z >  2.0) & std_ok & adx_ok & time_ok] = -1
    sig[(z < -2.0) & std_ok & adx_ok & time_ok] =  1
    # فقط سیگنال تغییر
    sig = sig.where(sig != sig.shift(), 0)

    return sig, z


# ================================================================== #
#  Strategy 2 ─ Asian Range Breakout  (بازنگری کامل)                #
#                                                                    #
#  مشکل قبلی: SL خیلی تنگ، فیلترهای ضعیف                           #
#  حل: SL = 0.3 × range (داخل رنج)  |  TP = 2 × range              #
#  تایید: کندل کامل بالای/پایین رنج بسته شود                        #
#  فیلتر: ADX لندن > 20  +  ساعت ۸ به بعد (نه دقیقاً ۷)           #
# ================================================================== #
def build_asian_breakout(df):
    d = df.copy()
    d['hour']    = d.index.hour
    d['weekday'] = d.index.weekday
    d['date']    = d.index.date

    atr = calc_atr(d['h_eur'], d['l_eur'], d['c_eur'], 14)

    # رنج آسیا: ۱:۰۰ تا ۶:۵۹ GMT
    asian = d[d['hour'].between(1, 6)]
    rng   = asian.groupby('date').agg(
        ah=('h_eur', 'max'),
        al=('l_eur', 'min')
    )
    rng['rng_pips'] = (rng['ah'] - rng['al']) / Config.pip

    d = d.join(rng, on='date')

    # فیلتر range معقول: ۱۵ تا ۵۵ پیپ
    valid = (
        d['rng_pips'].between(15, 55) &
        d['weekday'].between(0, 3)   # دوشنبه تا پنجشنبه
    )

    # لندن ۸-۱۱ (نه ۷ که هنوز volatility کم است)
    london = d['hour'].between(8, 11)

    adx = calc_adx(d['h_eur'], d['l_eur'], d['c_eur'], 14)

    sig = pd.Series(0, index=d.index)

    # Long: کندل کامل بالای رنج بسته شد
    long_ok  = (
        london & valid &
        (d['c_eur'] > d['ah'] + 3 * Config.pip) &   # تایید: ۳ پیپ بالاتر
        (d['l_eur'] > d['al']) &                      # کل کندل خارج رنج
        (adx > 18)
    )
    # Short: کندل کامل پایین رنج بسته شد
    short_ok = (
        london & valid &
        (d['c_eur'] < d['al'] - 3 * Config.pip) &
        (d['h_eur'] < d['ah']) &
        (adx > 18)
    )

    sig[long_ok]  =  1
    sig[short_ok] = -1

    # فقط اولین سیگنال هر روز
    nz        = sig[sig != 0]
    first_idx = nz.groupby(nz.index.date).head(1).index
    final     = pd.Series(0, index=d.index)
    final[first_idx] = sig[first_idx]

    return final, d['ah'], d['al'], atr


# ================================================================== #
#  Strategy 3 ─ EMA Trend Pullback  (بازنگری کامل)                  #
#                                                                    #
#  مشکل قبلی: RR < 1، win rate پایین                                #
#  حل اصلی: فقط وقتی روند خیلی واضح است وارد شو                    #
#  شرط: EMA21 > EMA50 > EMA200  (هر سه هم‌راستا)                   #
#  Pullback: قیمت به EMA50 رسیده (نه EMA21)  → عمق بیشتر           #
#  RSI بین ۴۰-۵۵ (pullback zone واقعی)                              #
#  TP: 3x ATR  |  SL: 1.2x ATR  →  RR = 2.5                       #
# ================================================================== #
def build_trend_pullback(df):
    c  = df['c_eur']
    h  = df['h_eur']
    l  = df['l_eur']

    ema21  = c.ewm(span=21,  adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rsi    = calc_rsi(c, 14)
    atr    = calc_atr(h, l, c, 14)
    adx    = calc_adx(h, l, c, 14)
    _, _, macd_hist = calc_macd(c)

    # فاصله از EMA50 (نه EMA21)
    dist50 = (c - ema50) / atr.replace(0, np.nan)

    # ساعات فعال لندن + NY
    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(8, 17)

    sig = pd.Series(0, index=df.index)

    # ─── Long ───
    # هر سه EMA صعودی + pullback دقیق به EMA50 + RSI تایید
    long_cond = (
        active                      &
        (ema21  > ema50)            &   # روند کوتاه صعودی
        (ema50  > ema200)           &   # روند بلند صعودی
        (adx    > 22)               &   # روند قوی کافی
        dist50.between(-0.8, 0.15)  &   # pullback به EMA50
        rsi.between(38, 56)         &   # RSI پولبک
        (macd_hist > macd_hist.shift(2))  # MACD در حال بازگشت
    )

    # ─── Short ───
    short_cond = (
        active                      &
        (ema21  < ema50)            &
        (ema50  < ema200)           &
        (adx    > 22)               &
        dist50.between(-0.15, 0.8)  &
        rsi.between(44, 62)         &
        (macd_hist < macd_hist.shift(2))
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    return sig, ema50, atr


# ================================================================== #
#  Strategy 4 ─ London Session Breakout (جایگزین GBPBreak و LondonMom)
#                                                                    #
#  منطق: ساعت ۶:۴۵-۷:۱۵ GMT یک رنج کوچک ۳۰ دقیقه‌ای رسم کن        #
#  شکست این رنج در ابتدای لندن → ترید                               #
#  این استراتژی یکی از اثبات‌شده‌ترین‌ها در فارکس است               #
#  SL: پایین/بالای رنج  |  TP: 2× رنج  →  RR ≈ 2                  #
# ================================================================== #
def build_london_session_breakout(df):
    d = df.copy()
    d['hour']    = d.index.hour
    d['minute']  = d.index.minute
    d['weekday'] = d.index.weekday
    d['date']    = d.index.date

    atr = calc_atr(d['h_eur'], d['l_eur'], d['c_eur'], 14)
    adx = calc_adx(d['h_eur'], d['l_eur'], d['c_eur'], 14)

    # رنج پیش‌گشایش: ۶:۰۰ تا ۶:۵۹ GMT
    pre = d[d['hour'] == 6]
    pre_rng = pre.groupby('date').agg(
        pre_h=('h_eur', 'max'),
        pre_l=('l_eur', 'min')
    )
    pre_rng['pre_rng_pips'] = (pre_rng['pre_h'] - pre_rng['pre_l']) / Config.pip

    d = d.join(pre_rng, on='date')

    # فیلتر: رنج ۸ تا ۳۵ پیپ (نه خیلی کوچک/بزرگ)
    valid = (
        d['pre_rng_pips'].between(8, 35) &
        d['weekday'].between(0, 3)
    )

    # ساعات شکست: ۷:۰۰ تا ۱۰:۰۰ GMT
    breakout_time = d['hour'].between(7, 9)

    sig = pd.Series(0, index=d.index)

    long_cond = (
        breakout_time & valid &
        (d['c_eur'] > d['pre_h'] + 2 * Config.pip) &   # تایید شکست
        (d['o_eur'] <= d['pre_h'] + 1 * Config.pip) &   # کندل از داخل رنج شروع
        (adx > 15)
    )
    short_cond = (
        breakout_time & valid &
        (d['c_eur'] < d['pre_l'] - 2 * Config.pip) &
        (d['o_eur'] >= d['pre_l'] - 1 * Config.pip) &
        (adx > 15)
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1

    # اولین سیگنال هر روز
    nz        = sig[sig != 0]
    first_idx = nz.groupby(nz.index.date).head(1).index
    final     = pd.Series(0, index=d.index)
    final[first_idx] = sig[first_idx]

    return final, d['pre_h'], d['pre_l'], atr


# ================================================================== #
#                       موتور Backtest                               #
# ================================================================== #
def run_backtest(df: pd.DataFrame):
    print("⚙️  محاسبه اندیکاتورها...")

    sig_arb,  z_score         = build_corr_arb(df)
    sig_ab,   ah, al, atr_ab  = build_asian_breakout(df)
    sig_tp,   ema50, atr_tp   = build_trend_pullback(df)
    sig_lb,   pre_h, pre_l, atr_lb = build_london_session_breakout(df)

    atr_main = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    risk     = RiskManager()
    open_pos = {}
    trades   = []
    warmup   = 300

    print("⚙️  شروع شبیه‌سازی...\n")

    for i in range(warmup, len(df)):
        ts    = df.index[i]
        row   = df.iloc[i]
        c_eur = row['c_eur']
        c_gbp = row['c_gbp']
        h_eur = row['h_eur']
        l_eur = row['l_eur']
        atr   = atr_main.iloc[i]

        if np.isnan(atr) or atr <= 0:
            continue

        risk.new_bar(ts)

        # ── اگر متوقف شده ──
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
        #  مرحله ۱: بررسی خروج پوزیشن‌های باز
        # ============================================================
        for key in list(open_pos.keys()):
            p   = open_pos[key]
            sym = p['symbol']

            if sym == 'EUR':
                hi, lo, cp = h_eur, l_eur, c_eur
            else:
                hi, lo, cp = row['h_gbp'], row['l_gbp'], c_gbp

            d_dir  = p['dir']
            entry  = p['entry']
            sl     = p['sl']
            tp     = p['tp']

            hit_sl = (d_dir ==  1 and lo <= sl) or (d_dir == -1 and hi >= sl)
            hit_tp = (d_dir ==  1 and hi >= tp) or (d_dir == -1 and lo <= tp)

            # ── Trailing Stop (فقط برای TrendPB و LondonBreak) ──
            if p['strategy'] in ('TrendPB', 'LondonBreak'):
                move = d_dir * (cp - entry)
                atr_v = atr_ab.iloc[i]
                if pd.notna(atr_v) and atr_v > 0:
                    # مرحله ۱: به break-even بعد از 1 ATR
                    if move > atr_v * 1.0:
                        be = entry + d_dir * atr_v * 0.3
                        if d_dir == 1:
                            p['sl'] = max(p['sl'], be)
                        else:
                            p['sl'] = min(p['sl'], be)
                    # مرحله ۲: lock profit بعد از 2 ATR
                    if move > atr_v * 1.8:
                        lock = entry + d_dir * atr_v * 1.0
                        if d_dir == 1:
                            p['sl'] = max(p['sl'], lock)
                        else:
                            p['sl'] = min(p['sl'], lock)

            # ── خروج CorrArb با Z-score ──
            if p['strategy'] == 'CorrArb':
                z_now = z_score.iloc[i]
                if pd.notna(z_now) and abs(z_now) < 0.3:
                    hit_tp = True

            # ── Time Stop ──
            if 'entry_ts' in p:
                elapsed_h = (ts - p['entry_ts']).total_seconds() / 3600
                time_limits = {
                    'CorrArb':     96,   # 4 روز
                    'AsianBreak':  48,   # 2 روز
                    'TrendPB':     96,
                    'LondonBreak': 48,
                }
                max_h = time_limits.get(p['strategy'], 72)
                if elapsed_h >= max_h and not hit_tp:
                    ep    = cp
                    p_pnl = calc_pnl(d_dir, p['lot'], entry, ep, sym)
                    trades.append({
                        **p, 'exit': ep, 'exit_ts': ts,
                        'pnl': p_pnl, 'status': 'TimeStop'
                    })
                    risk.add_pnl(p_pnl, ts)
                    del open_pos[key]
                    continue

            exit_r = None
            exit_p = None
            if hit_sl:
                exit_r = 'SL'
                exit_p = sl
            elif hit_tp:
                exit_r = 'TP'
                exit_p = tp

            if exit_r:
                p_pnl = calc_pnl(d_dir, p['lot'], entry, exit_p, sym)
                trades.append({
                    **p, 'exit': exit_p, 'exit_ts': ts,
                    'pnl': p_pnl, 'status': exit_r
                })
                risk.add_pnl(p_pnl, ts)
                del open_pos[key]

        # ============================================================
        #  مرحله ۲: ورود به پوزیشن
        # ============================================================
        if len(open_pos) >= Config.max_open_positions:
            continue

        # ── 1: Correlation Arb ──────────────────────────────────── #
        if 'CorrArb' not in open_pos and sig_arb.iloc[i] != 0:
            sv      = int(sig_arb.iloc[i])
            sl_pips = 20.0
            tp_pips = 35.0          # RR = 1.75
            lot     = lot_size_calc(risk.equity, sl_pips)
            half_sp = Config.spread_eur_pips * Config.pip / 2
            ep      = c_eur + sv * half_sp
            open_pos['CorrArb'] = dict(
                strategy='CorrArb', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

        # ── 2: Asian Breakout ────────────────────────────────────── #
        if 'AsianBreak' not in open_pos and sig_ab.iloc[i] != 0:
            sv       = int(sig_ab.iloc[i])
            rng_sz   = max((ah.iloc[i] - al.iloc[i]) / Config.pip, 15)
            sl_pips  = max(12, rng_sz * 0.35)   # SL داخل رنج
            tp_pips  = sl_pips * 2.5             # RR = 2.5
            lot      = lot_size_calc(risk.equity, sl_pips)
            half_sp  = Config.spread_eur_pips * Config.pip / 2
            ep       = c_eur + sv * half_sp
            open_pos['AsianBreak'] = dict(
                strategy='AsianBreak', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

        # ── 3: Trend Pullback ────────────────────────────────────── #
        if 'TrendPB' not in open_pos and sig_tp.iloc[i] != 0:
            sv      = int(sig_tp.iloc[i])
            atr_v   = atr_tp.iloc[i]
            sl_pips = max(15, atr_v / Config.pip * 1.2)
            tp_pips = sl_pips * 2.5    # RR = 2.5 (بهبود از 2.2)
            lot     = lot_size_calc(risk.equity, sl_pips)
            half_sp = Config.spread_eur_pips * Config.pip / 2
            ep      = c_eur + sv * half_sp
            open_pos['TrendPB'] = dict(
                strategy='TrendPB', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

        # ── 4: London Session Breakout ───────────────────────────── #
        if 'LondonBreak' not in open_pos and sig_lb.iloc[i] != 0:
            sv       = int(sig_lb.iloc[i])
            rng_sz   = max((pre_h.iloc[i] - pre_l.iloc[i]) / Config.pip, 8)
            sl_pips  = max(10, rng_sz * 0.8)   # SL پشت رنج
            tp_pips  = sl_pips * 2.2            # RR = 2.2
            lot      = lot_size_calc(risk.equity, sl_pips)
            half_sp  = Config.spread_eur_pips * Config.pip / 2
            ep       = c_eur + sv * half_sp
            open_pos['LondonBreak'] = dict(
                strategy='LondonBreak', symbol='EUR',
                dir=sv, lot=lot, entry=ep,
                sl=ep - sv * sl_pips * Config.pip,
                tp=ep + sv * tp_pips * Config.pip,
                entry_ts=ts,
            )

    # ── بستن باقیمانده ──
    last_ts  = df.index[-1]
    last_eur = df['c_eur'].iloc[-1]
    last_gbp = df['c_gbp'].iloc[-1]
    for key, p in open_pos.items():
        ep    = last_eur if p['symbol'] == 'EUR' else last_gbp
        p_pnl = calc_pnl(p['dir'], p['lot'], p['entry'], ep, p['symbol'])
        trades.append({
            **p, 'exit': ep, 'exit_ts': last_ts,
            'pnl': p_pnl, 'status': 'eod_close'
        })
        risk.add_pnl(p_pnl, last_ts)

    return trades, risk


# ================================================================== #
#              گزارش حرفه‌ای                                         #
# ================================================================== #
def report(trades: list, risk: RiskManager):
    if not trades:
        print("❌ هیچ معامله‌ای ثبت نشد!")
        return

    t = pd.DataFrame(trades)
    t['pnl']          = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    # ── بازه زمانی ──
    start_d      = t['entry_ts'].min()
    end_d        = t['exit_ts'].max()
    total_days   = max((end_d - start_d).days, 1)
    total_weeks  = total_days / 7
    total_months = total_days / 30.44
    total_years  = total_days / 365.25

    # ── آمار کلی ──
    final_eq   = risk.equity
    total_pnl  = final_eq - Config.initial_balance
    total_ret  = total_pnl / Config.initial_balance * 100
    ann_ret    = ((final_eq / Config.initial_balance)
                  ** (365.25 / total_days) - 1) * 100

    win_t  = t[t['pnl'] > 0]
    loss_t = t[t['pnl'] < 0]
    win_r  = len(win_t) / len(t) * 100
    avg_w  = win_t['pnl'].mean()  if len(win_t)  > 0 else 0
    avg_l  = loss_t['pnl'].mean() if len(loss_t) > 0 else 0
    gw     = win_t['pnl'].sum()
    gl     = abs(loss_t['pnl'].sum())
    pf     = gw / gl if gl > 0 else float('inf')
    exp    = t['pnl'].mean()
    rr     = abs(avg_w / avg_l) if avg_l != 0 else 0

    best_t  = t['pnl'].max()
    worst_t = t['pnl'].min()

    # consecutive
    sign  = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cw, cl, mcw, mcl = 0, 0, 0, 0
    for s in sign:
        if   s > 0: cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0: cl += 1; cw = 0; mcl = max(mcl, cl)
        else:       cw = cl = 0

    ppd = total_pnl / total_days
    ppw = total_pnl / total_weeks
    ppm = total_pnl / total_months
    ppy = total_pnl / total_years

    # ── ماهانه ──
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

    # ── سالانه ──
    t['yr'] = t['entry_ts'].dt.year
    yearly  = (t.groupby('yr')
               .agg(n=('pnl','count'), pnl=('pnl','sum'),
                    wins=('pnl', lambda x: (x>0).sum()))
               .reset_index())
    yearly['wr']   = yearly['wins'] / yearly['n'] * 100
    yearly['ret']  = yearly['pnl'] / Config.initial_balance * 100
    yearly['annr'] = [
        (((Config.initial_balance + r) / Config.initial_balance)
         ** (365.25 / total_days) - 1) * 100
        for r in yearly['pnl'].cumsum()
    ]

    # ── استراتژی ──
    strats = []
    for nm in sorted(t['strategy'].unique()):
        s  = t[t['strategy'] == nm]
        sw = s[s['pnl'] > 0]
        sl = s[s['pnl'] < 0]
        wr_ = (s['pnl'] > 0).mean() * 100
        gw_ = sw['pnl'].sum()
        gl_ = abs(sl['pnl'].sum())
        pf_ = gw_ / gl_ if gl_ > 0 else float('inf')
        aw_ = sw['pnl'].mean() if len(sw) > 0 else 0
        al_ = sl['pnl'].mean() if len(sl) > 0 else 0
        rr_ = abs(aw_ / al_) if al_ != 0 else 0
        strats.append(dict(
            name=nm, n=len(s), wr=wr_, pf=pf_,
            rr=rr_, aw=aw_, al=al_,
            exp=s['pnl'].mean(), tot=s['pnl'].sum(),
            dur=s['duration_min'].mean()
        ))

    # ================================================================
    #  چاپ
    # ================================================================
    W   = 70
    SEP = "═" * W

    def rw(label, value):
        lbl  = f"  {label}"
        val  = str(value)
        dots = "·" * max(2, W - len(lbl) - len(val) - 2)
        return f"{lbl} {dots} {val}"

    def box_top(title):
        inner = f"─ {title} "
        return "┌" + inner + "─" * (W - len(inner) - 1) + "┐"

    def box_bot():
        return "└" + "─" * (W - 1) + "┘"

    lines = [
        SEP,
        " " * 8 + "▌  BACKTEST REPORT  —  Prop Trading System  ▐",
        " " * 8 + f"▌  اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ▐",
        SEP, "",

        box_top("اطلاعات کلی"),
        rw("دوره بک‌تست",    f"{start_d.date()} → {end_d.date()}"),
        rw("تعداد روزها",    f"{total_days:,}"),
        rw("جفت‌ارزها",      "EURUSD + GBPUSD"),
        rw("تایم‌فریم",      "15 دقیقه"),
        rw("استراتژی‌ها",    f"{len(strats)} عدد"),
        box_bot(), "",

        box_top("نتایج مالی"),
        rw("موجودی اولیه",   f"${Config.initial_balance:>12,.2f}"),
        rw("موجودی نهایی",   f"${final_eq:>12,.2f}"),
        rw("سود/زیان کل",   f"${total_pnl:>+12,.2f}"),
        rw("بازده کل",       f"{total_ret:>+.2f}%"),
        rw("بازده سالانه",   f"{ann_ret:>+.2f}%"),
        rw("بهترین معامله", f"${best_t:>+.2f}"),
        rw("بدترین معامله", f"${worst_t:>+.2f}"),
        box_bot(), "",

        box_top("معیارهای ریسک"),
        rw("Max Drawdown",   f"{risk.max_dd:.2f}%"),
        rw("Max DD مطلق",   f"${risk.max_dd_abs:>+.2f}"),
        rw("Sharpe Ratio",   f"{risk.sharpe:.2f}"),
        rw("Sortino Ratio",  f"{risk.sortino:.2f}"),
        rw("Calmar Ratio",   f"{risk.calmar:.2f}"),
        rw("Profit Factor",  f"{pf:.2f}"),
        rw("وضعیت پایان",   risk.halt_reason),
        box_bot(), "",

        box_top("آمار معاملات"),
        rw("تعداد کل",       f"{len(t):,}"),
        rw("Win Rate",        f"{win_r:.1f}%"),
        rw("معاملات سودده", f"{len(win_t):,}"),
        rw("معاملات ضررده", f"{len(loss_t):,}"),
        rw("Avg Win",         f"${avg_w:>+.2f}"),
        rw("Avg Loss",        f"${avg_l:>+.2f}"),
        rw("RR واقعی",       f"{rr:.2f}"),
        rw("Expectancy",      f"${exp:>+.2f}"),
        rw("Max Consec Win",  f"{mcw}"),
        rw("Max Consec Loss", f"{mcl}"),
        rw("مدت میانگین",    f"{t['duration_min'].mean():.0f} min"),
        box_bot(), "",

        box_top("آمار زمانی"),
        rw("معاملات/روز",    f"{len(t)/total_days:.2f}"),
        rw("معاملات/هفته",  f"{len(t)/total_weeks:.2f}"),
        rw("معاملات/ماه",   f"{len(t)/total_months:.2f}"),
        rw("سود/روز",        f"${ppd:>+.2f}"),
        rw("سود/هفته",      f"${ppw:>+.2f}"),
        rw("سود/ماه",       f"${ppm:>+.2f}"),
        rw("سود/سال (proj.)",f"${ppy:>+.2f}"),
        box_bot(), "",
    ]

    # ── جدول استراتژی‌ها ──
    lines.append(box_top("عملکرد استراتژی‌ها"))
    lines.append(
        f"  {'نام':<13} {'#':>4}  {'Win%':>6}  {'PF':>5}  "
        f"{'RR':>5}  {'AvgW':>8}  {'AvgL':>8}  "
        f"{'Exp':>7}  {'PnL':>10}  {'مدت':>6}"
    )
    lines.append("  " + "─" * (W - 3))
    for s in strats:
        flag = "✅" if s['tot'] > 0 else "❌"
        pf_s = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "  ∞ "
        lines.append(
            f"  {s['name']:<13} {s['n']:>4}  {s['wr']:>5.1f}%  "
            f"{pf_s:>5}  {s['rr']:>5.2f}  "
            f"${s['aw']:>7.2f}  ${s['al']:>7.2f}  "
            f"${s['exp']:>6.2f}  ${s['tot']:>9.2f}  "
            f"{s['dur']:>5.0f}m  {flag}"
        )
    lines += [box_bot(), ""]

    # ── گزارش ماهانه ──
    lines.append(box_top("گزارش ماهانه"))
    lines.append(
        f"  {'ماه':>7}  {'#':>4}  {'Win%':>6}  {'PnL':>10}  "
        f"{'Ret%':>6}  {'بهترین':>8}  {'بدترین':>8}  "
        f"{'تجمعی':>10}  {'CumRet':>7}"
    )
    lines.append("  " + "─" * (W - 3))
    for _, mr in monthly.iterrows():
        arrow = "▲" if mr['pnl'] >= 0 else "▼"
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>4}  "
            f"{mr['wr']:>5.1f}%  ${mr['pnl']:>9.2f}  "
            f"{mr['ret']:>+5.1f}%  ${mr['best']:>7.2f}  "
            f"${mr['worst']:>7.2f}  ${mr['cum_pnl']:>9.2f}  "
            f"{mr['cum_ret']:>+6.1f}% {arrow}"
        )
    lines += [box_bot(), ""]

    # ── گزارش سالانه ──
    lines.append(box_top("گزارش سالانه"))
    lines.append(
        f"  {'سال':>5}  {'#':>5}  {'Win%':>6}  "
        f"{'PnL':>10}  {'Ret%':>7}  {'Ann Ret%':>9}"
    )
    lines.append("  " + "─" * (W - 3))
    for _, yr in yearly.iterrows():
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>5.1f}%  ${yr['pnl']:>9.2f}  "
            f"{yr['ret']:>+6.1f}%  {yr['annr']:>+8.1f}%"
        )
    lines += [box_bot(), ""]

    # ── توزیع خروج ──
    lines.append(box_top("توزیع نوع خروج"))
    for status, cnt in t['status'].value_counts().items():
        pct_  = cnt / len(t) * 100
        avg_  = t.loc[t['status'] == status, 'pnl'].mean()
        bar   = "█" * max(1, int(pct_ / 2.5))
        lines.append(
            f"  {status:<13} {cnt:>5} ({pct_:>5.1f}%)  "
            f"{bar:<28}  avg=${avg_:>+.2f}"
        )
    lines += [box_bot(), "", SEP]

    out = "\n".join(lines)
    print(out)

    # ── ذخیره فایل‌ها ──
    with open("Backtest_Report.txt", "w", encoding="utf-8") as f:
        f.write(out)

    # CSV جامع
    rows = [
        ["BACKTEST SUMMARY"], [""],
        ["Parameter", "Value", "Unit"],
        ["Period",     f"{start_d.date()} → {end_d.date()}", ""],
        ["Days",       total_days, ""],
        ["Initial",    Config.initial_balance, "USD"],
        ["Final",      round(final_eq, 2), "USD"],
        ["PnL",        round(total_pnl, 2), "USD"],
        ["Return",     round(total_ret, 2), "%"],
        ["Ann Return", round(ann_ret, 2), "%"],
        ["Max DD",     round(risk.max_dd, 2), "%"],
        ["Sharpe",     round(risk.sharpe, 2), ""],
        ["Sortino",    round(risk.sortino, 2), ""],
        ["Calmar",     round(risk.calmar, 2), ""],
        ["PF",         round(pf, 2), ""],
        ["Win Rate",   round(win_r, 1), "%"],
        ["RR",         round(rr, 2), ""],
        ["Expectancy", round(exp, 2), "USD"],
        ["Trades",     len(t), ""],
        [""], ["STRATEGY PERFORMANCE"],
        ["Name","N","WinRate%","PF","RR","AvgWin","AvgLoss",
         "Expectancy","TotalPnL","AvgDur_min"],
    ]
    for s in strats:
        pf_v = round(s['pf'], 2) if s['pf'] != float('inf') else 999
        rows.append([s['name'], s['n'], round(s['wr'],1), pf_v,
                     round(s['rr'],2), round(s['aw'],2), round(s['al'],2),
                     round(s['exp'],2), round(s['tot'],2), round(s['dur'],0)])

    rows += [[""], ["MONTHLY REPORT"],
             ["Month","Trades","WinRate%","PnL","Ret%",
              "Best","Worst","CumPnL","CumRet%"]]
    for _, mr in monthly.iterrows():
        rows.append([str(mr['ym']), int(mr['n']), round(mr['wr'],1),
                     round(mr['pnl'],2), round(mr['ret'],2),
                     round(mr['best'],2), round(mr['worst'],2),
                     round(mr['cum_pnl'],2), round(mr['cum_ret'],2)])

    rows += [[""], ["YEARLY REPORT"],
             ["Year","Trades","WinRate%","PnL","Ret%","AnnRet%"]]
    for _, yr in yearly.iterrows():
        rows.append([int(yr['yr']), int(yr['n']),
                     round(yr['wr'],1), round(yr['pnl'],2),
                     round(yr['ret'],2), round(yr['annr'],2)])

    rows += [[""], ["EXIT DISTRIBUTION"],
             ["Type","Count","Pct%","AvgPnL"]]
    for status, cnt in t['status'].value_counts().items():
        rows.append([status, cnt,
                     round(cnt/len(t)*100, 1),
                     round(t.loc[t['status']==status,'pnl'].mean(), 2)])

    pd.DataFrame(rows).to_csv(
        "Backtest_Summary.csv", index=False, header=False, encoding="utf-8-sig"
    )

    # Equity curve
    eq_df = pd.DataFrame({'ts': risk.curve_ts, 'equity': risk.curve})
    eq_df['dd_pct'] = (
        (eq_df['equity'] - eq_df['equity'].cummax())
        / eq_df['equity'].cummax() * 100
    ).round(4)
    eq_df.to_csv("equity_curve.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ فایل‌های ذخیره شد:")
    print(f"   → Backtest_Report.txt")
    print(f"   → Backtest_Summary.csv")
    print(f"   → equity_curve.csv  ({len(eq_df)} نقطه)")


# ================================================================== #
if __name__ == "__main__":
    df           = load_data()
    trades, risk = run_backtest(df)
    report(trades, risk)
