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
    risk_per_trade_pct = 0.01
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10
    profit_target_pct  = 0.99
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

    df = df[df.index.weekday < 5]
    print(f"✅ {len(df):,} کندل | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── اندیکاتورها (vectorized) ────────────────────────────────────── #
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

def lot_size_calc(equity: float, sl_pips: float) -> float:
    if sl_pips <= 0:
        return 0.01
    risk_usd = equity * Config.risk_per_trade_pct
    lot = risk_usd / (sl_pips * Config.pip * Config.lot_size)
    return round(float(np.clip(lot, 0.01, Config.max_lot)), 2)


# ================================================================== #
#   موتور بک‌تست سریع (vectorized signal + numpy loop فشرده)        #
# ================================================================== #
def fast_backtest(
    signals: pd.Series,       # +1 / -1 / 0
    sl_pips_arr: np.ndarray,  # SL به پیپ برای هر کندل
    tp_pips_arr: np.ndarray,  # TP به پیپ برای هر کندل
    high: np.ndarray,
    low:  np.ndarray,
    close: np.ndarray,
    symbol: str,
    strat_name: str,
    time_stop_bars: int = 192,   # ۴۸ ساعت = ۱۹۲ × ۱۵min
) -> tuple:
    """
    موتور بک‌تست سریع با numpy arrays.
    هیچ Python loop روی تمام کندل‌ها نیست - فقط روی معاملات.
    """
    pip      = Config.pip
    ls       = Config.lot_size
    sp       = Config.spread_eur_pips if symbol == 'EUR' else Config.spread_gbp_pips
    comm     = Config.commission_per_lot

    eq       = Config.initial_balance
    peak     = Config.initial_balance
    day_eq   = Config.initial_balance

    curve    = [eq]
    trades   = []

    sig_idx  = np.where(signals.values != 0)[0]
    sig_idx  = sig_idx[sig_idx >= Config.warmup]

    i_ptr    = 0          # اشاره‌گر به sig_idx
    n_sigs   = len(sig_idx)
    n_bars   = len(close)

    in_trade    = False
    entry_bar   = 0
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    direction   = 0
    lot         = 0.01

    halted      = False
    halt_reason = "در حال اجرا"

    # برای equity curve با timestamp نیاز داریم
    # فقط نقاط تغییر را ذخیره می‌کنیم
    curve_ts = [None]

    ts_arr = signals.index  # DatetimeIndex

    # daily tracking
    cur_day  = None

    bar = Config.warmup
    while bar < n_bars:
        day = ts_arr[bar].date()
        if day != cur_day:
            cur_day  = day
            day_eq   = eq
            if halted and "Daily" in halt_reason:
                halted      = False
                halt_reason = "در حال اجرا"

        if halted:
            if in_trade:
                # ببند با قیمت فعلی
                raw  = direction * (close[bar] - entry_price) * lot * ls
                cost = sp * pip * lot * ls + comm * lot
                pnl  = raw - cost
                eq  += pnl
                peak = max(peak, eq)
                curve.append(round(eq, 4))
                curve_ts.append(ts_arr[bar])
                trades.append({
                    'strategy': strat_name, 'symbol': symbol,
                    'dir': direction, 'lot': lot,
                    'entry': entry_price, 'exit': close[bar],
                    'entry_ts': ts_arr[entry_bar],
                    'exit_ts':  ts_arr[bar],
                    'pnl': pnl, 'status': 'halt',
                    'sl': sl_price, 'tp': tp_price,
                })
                in_trade = False
            bar += 1
            continue

        # ── بررسی خروج پوزیشن باز ──
        if in_trade:
            hi = high[bar]
            lo = low[bar]

            hit_sl = (direction ==  1 and lo <= sl_price) or \
                     (direction == -1 and hi >= sl_price)
            hit_tp = (direction ==  1 and hi >= tp_price) or \
                     (direction == -1 and lo <= tp_price)

            # time stop
            if (bar - entry_bar) >= time_stop_bars and not hit_tp:
                ep   = close[bar]
                raw  = direction * (ep - entry_price) * lot * ls
                cost = sp * pip * lot * ls + comm * lot
                pnl  = raw - cost
                eq  += pnl
                peak = max(peak, eq)
                curve.append(round(eq, 4))
                curve_ts.append(ts_arr[bar])

                # risk checks
                dd_d = (eq - day_eq) / day_eq
                dd_t = (eq - peak)   / peak
                if dd_d <= -Config.max_daily_loss_pct:
                    halted = True; halt_reason = f"Daily Loss {dd_d*100:.1f}%"
                elif dd_t <= -Config.max_total_dd_pct:
                    halted = True; halt_reason = f"Max DD {dd_t*100:.1f}%"

                trades.append({
                    'strategy': strat_name, 'symbol': symbol,
                    'dir': direction, 'lot': lot,
                    'entry': entry_price, 'exit': ep,
                    'entry_ts': ts_arr[entry_bar],
                    'exit_ts':  ts_arr[bar],
                    'pnl': pnl, 'status': 'TimeStop',
                    'sl': sl_price, 'tp': tp_price,
                })
                in_trade = False
                bar += 1
                continue

            exit_r = exit_p = None
            if hit_sl: exit_r, exit_p = 'SL', sl_price
            elif hit_tp: exit_r, exit_p = 'TP', tp_price

            if exit_r:
                raw  = direction * (exit_p - entry_price) * lot * ls
                cost = sp * pip * lot * ls + comm * lot
                pnl  = raw - cost
                eq  += pnl
                peak = max(peak, eq)
                curve.append(round(eq, 4))
                curve_ts.append(ts_arr[bar])

                dd_d = (eq - day_eq) / day_eq
                dd_t = (eq - peak)   / peak
                pr   = (eq - Config.initial_balance) / Config.initial_balance
                if dd_d <= -Config.max_daily_loss_pct:
                    halted = True; halt_reason = f"Daily Loss {dd_d*100:.1f}%"
                elif dd_t <= -Config.max_total_dd_pct:
                    halted = True; halt_reason = f"Max DD {dd_t*100:.1f}%"
                elif pr >= Config.profit_target_pct:
                    halted = True; halt_reason = f"Target {pr*100:.1f}%"

                trades.append({
                    'strategy': strat_name, 'symbol': symbol,
                    'dir': direction, 'lot': lot,
                    'entry': entry_price, 'exit': exit_p,
                    'entry_ts': ts_arr[entry_bar],
                    'exit_ts':  ts_arr[bar],
                    'pnl': pnl, 'status': exit_r,
                    'sl': sl_price, 'tp': tp_price,
                })
                in_trade = False

        # ── ورود ──
        if not in_trade and not halted:
            # پیدا کن اولین سیگنال >= bar
            while i_ptr < n_sigs and sig_idx[i_ptr] < bar:
                i_ptr += 1
            if i_ptr < n_sigs and sig_idx[i_ptr] == bar:
                sv      = int(signals.values[bar])
                sl_p    = float(sl_pips_arr[bar])
                tp_p    = float(tp_pips_arr[bar])
                if sl_p > 0 and tp_p > 0:
                    lot         = lot_size_calc(eq, sl_p)
                    half_sp     = sp * pip / 2
                    entry_price = close[bar] + sv * half_sp
                    sl_price    = entry_price - sv * sl_p * pip
                    tp_price    = entry_price + sv * tp_p * pip
                    entry_bar   = bar
                    direction   = sv
                    in_trade    = True

        bar += 1

    # ── بستن باقیمانده ──
    if in_trade:
        ep   = close[-1]
        raw  = direction * (ep - entry_price) * lot * ls
        cost = sp * pip * lot * ls + comm * lot
        pnl  = raw - cost
        eq  += pnl
        curve.append(round(eq, 4))
        curve_ts.append(ts_arr[-1])
        trades.append({
            'strategy': strat_name, 'symbol': symbol,
            'dir': direction, 'lot': lot,
            'entry': entry_price, 'exit': ep,
            'entry_ts': ts_arr[entry_bar],
            'exit_ts':  ts_arr[-1],
            'pnl': pnl, 'status': 'eod',
            'sl': sl_price, 'tp': tp_price,
        })

    # ── ساخت RiskManager-like object ──
    class RiskResult:
        def __init__(self, curve_, curve_ts_, halt_r):
            self.curve       = curve_
            self.curve_ts    = curve_ts_
            self.equity      = curve_[-1]
            self.halt_reason = halt_r

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
            return (r.mean() / r.std() * np.sqrt(252*96)) if r.std() > 0 else 0

        @property
        def sortino(self):
            r   = pd.Series(self.curve).pct_change().dropna()
            neg = r[r < 0]
            ds  = neg.std() if len(neg) > 0 else 1e-10
            return (r.mean() / ds * np.sqrt(252*96)) if ds > 0 else 0

        @property
        def calmar(self):
            ret = (self.curve[-1] / Config.initial_balance - 1)
            dd  = abs(self.max_dd / 100)
            return ret / dd if dd > 0 else 0

    risk_result = RiskResult(curve, curve_ts, halt_reason)
    return trades, risk_result


# ================================================================== #
#   سیگنال‌سازی vectorized برای هر استراتژی                         #
# ================================================================== #

def signals_corr_arb(df: pd.DataFrame):
    """CorrArb: Z-score EURGBP"""
    eurgbp = df['c_eur'] / df['c_gbp']
    period = 96

    mean  = eurgbp.rolling(period).mean()
    std   = eurgbp.rolling(period).std()
    z     = (eurgbp - mean) / std.replace(0, np.nan)

    std_ok = std > std.rolling(period * 5).mean() * 0.25
    adx    = calc_adx(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    hour   = pd.Series(df.index.hour, index=df.index)

    cond_long  = (z < -2.0) & std_ok & (adx < 30) & hour.between(7, 19)
    cond_short = (z >  2.0) & std_ok & (adx < 30) & hour.between(7, 19)

    sig = pd.Series(0, index=df.index)
    sig[cond_long]  =  1
    sig[cond_short] = -1
    sig = sig.where(sig != sig.shift(), 0)

    SL = np.where(sig != 0, 22.0, 0.0)
    TP = np.where(sig != 0, 38.0, 0.0)

    # Z-score exit را جداگانه handle می‌کنیم در موتور خاص
    return sig, SL, TP, z


def signals_asian_breakout(df: pd.DataFrame):
    """Asian Breakout: رنج آسیا + شکست لندن"""
    d = df.copy()
    d['hour']    = d.index.hour
    d['weekday'] = d.index.weekday
    d['date']    = d.index.date

    adx = calc_adx(d['h_eur'], d['l_eur'], d['c_eur'], 14)

    asian = d[d['hour'].between(1, 6)]
    rng   = asian.groupby('date').agg(
        ah=('h_eur','max'), al=('l_eur','min'))
    rng['rng_pips'] = (rng['ah'] - rng['al']) / Config.pip
    d = d.join(rng, on='date')

    london = d['hour'].between(8, 11)
    valid  = d['rng_pips'].between(15, 45) & d['weekday'].between(0, 3)

    above2 = ((d['c_eur'] > d['ah']).astype(int) +
              (d['c_eur'].shift(1) > d['ah'].shift(1)).astype(int)) >= 2
    below2 = ((d['c_eur'] < d['al']).astype(int) +
              (d['c_eur'].shift(1) < d['al'].shift(1)).astype(int)) >= 2

    raw_sig = pd.Series(0, index=d.index)
    raw_sig[london & valid & above2 & (adx > 20)] =  1
    raw_sig[london & valid & below2 & (adx > 20)] = -1

    # اولین سیگنال هر روز
    nz  = raw_sig[raw_sig != 0]
    fi  = nz.groupby(nz.index.date).head(1).index
    sig = pd.Series(0, index=d.index)
    sig[fi] = raw_sig[fi]

    rng_pips = d['rng_pips'].fillna(20)
    sl_pips  = np.maximum(18, rng_pips.values * 0.6)
    tp_pips  = np.maximum(sl_pips * 2.0, rng_pips.values * 2.0)

    SL = np.where(sig.values != 0, sl_pips, 0.0)
    TP = np.where(sig.values != 0, tp_pips, 0.0)

    return sig, SL, TP


def signals_trend_pullback(df: pd.DataFrame):
    """TrendPB: ۳ EMA هم‌راستا + pullback + MACD"""
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

    swing_low  = l.rolling(10).min()
    swing_high = h.rolling(10).max()

    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(8, 17)

    long_cond = (
        active &
        (ema21 > ema50) & (ema50 > ema200) &
        (adx > 25) & (c > ema200) & (c < ema21) &
        rsi.between(35, 52) &
        (macd_hist > macd_hist.shift(1)) &
        (macd_hist > macd_hist.shift(2))
    )
    short_cond = (
        active &
        (ema21 < ema50) & (ema50 < ema200) &
        (adx > 25) & (c < ema200) & (c > ema21) &
        rsi.between(48, 65) &
        (macd_hist < macd_hist.shift(1)) &
        (macd_hist < macd_hist.shift(2))
    )

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    # SL از swing
    sl_from_swing = np.where(
        sig.values ==  1,
        np.maximum(15, (c.values - swing_low.values)  / Config.pip),
        np.where(
        sig.values == -1,
        np.maximum(15, (swing_high.values - c.values) / Config.pip),
        0.0)
    )
    tp_pips = sl_from_swing * 2.5

    SL = np.where(sig.values != 0, sl_from_swing, 0.0)
    TP = np.where(sig.values != 0, tp_pips,       0.0)

    return sig, SL, TP


def signals_ny_fade(df: pd.DataFrame):
    """NYFade: RSI extreme + Bollinger در NY overlap"""
    c = df['c_eur']
    h = df['h_eur']
    l = df['l_eur']

    atr  = calc_atr(h, l, c, 14)
    rsi  = calc_rsi(c, 14)
    adx  = calc_adx(h, l, c, 14)
    _, _, macd_hist = calc_macd(c)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up  = bb_mid + 2.0 * bb_std
    bb_lo  = bb_mid - 2.0 * bb_std

    hour    = pd.Series(df.index.hour,    index=df.index)
    weekday = pd.Series(df.index.weekday, index=df.index)
    ny_time = hour.between(13, 16) & weekday.between(0, 3)

    fade_short = ny_time & (rsi > 68) & (c > bb_up) & \
                 (macd_hist < macd_hist.shift(1)) & (adx < 35)
    fade_long  = ny_time & (rsi < 32) & (c < bb_lo) & \
                 (macd_hist > macd_hist.shift(1)) & (adx < 35)

    raw_sig = pd.Series(0, index=df.index)
    raw_sig[fade_short] = -1
    raw_sig[fade_long]  =  1

    nz  = raw_sig[raw_sig != 0]
    fi  = nz.groupby(nz.index.date).head(1).index
    sig = pd.Series(0, index=df.index)
    sig[fi] = raw_sig[fi]

    sl_pips = np.maximum(15, atr.values / Config.pip * 1.8)
    tp_pips = np.maximum(12, atr.values / Config.pip * 1.3)
    # اگر TP < SL × 0.8 → TP را بزرگتر کن
    tp_pips = np.where(tp_pips < sl_pips * 0.8, sl_pips * 1.2, tp_pips)

    SL = np.where(sig.values != 0, sl_pips, 0.0)
    TP = np.where(sig.values != 0, tp_pips, 0.0)

    return sig, SL, TP


# ── CorrArb موتور خاص (با Z-exit) ─────────────────────────────── #
def run_corr_arb_special(df: pd.DataFrame, sig, SL, TP, z_series):
    """
    CorrArb نیاز به Z-score exit دارد → موتور جداگانه
    اما هنوز سریع: فقط روی معاملات loop می‌زنیم
    """
    pip  = Config.pip
    ls   = Config.lot_size
    sp   = Config.spread_eur_pips
    comm = Config.commission_per_lot

    close_arr = df['c_eur'].values
    high_arr  = df['h_eur'].values
    low_arr   = df['l_eur'].values
    z_arr     = z_series.values
    sig_vals  = sig.values
    sl_arr    = SL
    tp_arr    = TP
    ts_arr    = df.index

    eq    = Config.initial_balance
    peak  = Config.initial_balance
    day_eq = Config.initial_balance
    curve  = [eq]
    curve_ts = [None]
    trades = []
    halted = False
    halt_r = "در حال اجرا"

    in_trade    = False
    entry_bar   = 0
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    direction   = 0
    lot         = 0.01
    cur_day     = None
    TIME_STOP   = 96 * 4   # 4 روز

    sig_positions = np.where(sig_vals != 0)[0]
    sig_positions = sig_positions[sig_positions >= Config.warmup]
    sig_set       = set(sig_positions.tolist())

    for bar in range(Config.warmup, len(close_arr)):
        day = ts_arr[bar].date()
        if day != cur_day:
            cur_day = day
            day_eq  = eq
            if halted and "Daily" in halt_r:
                halted = False; halt_r = "در حال اجرا"

        if halted:
            if in_trade:
                raw  = direction * (close_arr[bar] - entry_price) * lot * ls
                cost = sp * pip * lot * ls + comm * lot
                pnl  = raw - cost
                eq  += pnl
                peak = max(peak, eq)
                curve.append(round(eq, 4)); curve_ts.append(ts_arr[bar])
                trades.append({'strategy':'CorrArb','symbol':'EUR',
                    'dir':direction,'lot':lot,'entry':entry_price,
                    'exit':close_arr[bar],'entry_ts':ts_arr[entry_bar],
                    'exit_ts':ts_arr[bar],'pnl':pnl,'status':'halt',
                    'sl':sl_price,'tp':tp_price})
                in_trade = False
            continue

        if in_trade:
            hi = high_arr[bar]; lo = low_arr[bar]
            hit_sl = (direction== 1 and lo<=sl_price) or (direction==-1 and hi>=sl_price)
            hit_tp = (direction== 1 and hi>=tp_price) or (direction==-1 and lo<=tp_price)

            # Z-score exit
            z_now = z_arr[bar]
            if not np.isnan(z_now) and abs(z_now) < 0.3:
                hit_tp = True

            # time stop
            if (bar - entry_bar) >= TIME_STOP and not hit_tp:
                ep   = close_arr[bar]
                raw  = direction * (ep - entry_price) * lot * ls
                cost = sp * pip * lot * ls + comm * lot
                pnl  = raw - cost
                eq  += pnl; peak = max(peak, eq)
                curve.append(round(eq,4)); curve_ts.append(ts_arr[bar])
                dd_d=(eq-day_eq)/day_eq; dd_t=(eq-peak)/peak
                if dd_d<=-Config.max_daily_loss_pct: halted=True;halt_r=f"Daily {dd_d*100:.1f}%"
                elif dd_t<=-Config.max_total_dd_pct: halted=True;halt_r=f"MaxDD {dd_t*100:.1f}%"
                trades.append({'strategy':'CorrArb','symbol':'EUR',
                    'dir':direction,'lot':lot,'entry':entry_price,'exit':ep,
                    'entry_ts':ts_arr[entry_bar],'exit_ts':ts_arr[bar],
                    'pnl':pnl,'status':'TimeStop','sl':sl_price,'tp':tp_price})
                in_trade = False
                continue

            exit_r = exit_p = None
            if hit_sl: exit_r,exit_p='SL',sl_price
            elif hit_tp: exit_r,exit_p='TP',tp_price
            if exit_r:
                raw  = direction*(exit_p-entry_price)*lot*ls
                cost = sp*pip*lot*ls+comm*lot
                pnl  = raw-cost
                eq  += pnl; peak=max(peak,eq)
                curve.append(round(eq,4)); curve_ts.append(ts_arr[bar])
                dd_d=(eq-day_eq)/day_eq; dd_t=(eq-peak)/peak
                pr=(eq-Config.initial_balance)/Config.initial_balance
                if dd_d<=-Config.max_daily_loss_pct: halted=True;halt_r=f"Daily {dd_d*100:.1f}%"
                elif dd_t<=-Config.max_total_dd_pct: halted=True;halt_r=f"MaxDD {dd_t*100:.1f}%"
                elif pr>=Config.profit_target_pct:   halted=True;halt_r=f"Target {pr*100:.1f}%"
                trades.append({'strategy':'CorrArb','symbol':'EUR',
                    'dir':direction,'lot':lot,'entry':entry_price,'exit':exit_p,
                    'entry_ts':ts_arr[entry_bar],'exit_ts':ts_arr[bar],
                    'pnl':pnl,'status':exit_r,'sl':sl_price,'tp':tp_price})
                in_trade = False

        if not in_trade and not halted and bar in sig_set:
            sv = int(sig_vals[bar])
            sp_ = float(sl_arr[bar]); tp_ = float(tp_arr[bar])
            if sp_ > 0 and tp_ > 0:
                lot         = lot_size_calc(eq, sp_)
                half_sp     = sp * pip / 2
                entry_price = close_arr[bar] + sv * half_sp
                sl_price    = entry_price - sv * sp_ * pip
                tp_price    = entry_price + sv * tp_ * pip
                entry_bar   = bar
                direction   = sv
                in_trade    = True

    if in_trade:
        ep   = close_arr[-1]
        raw  = direction*(ep-entry_price)*lot*ls
        cost = sp*pip*lot*ls+comm*lot
        pnl  = raw-cost
        eq  += pnl
        curve.append(round(eq,4)); curve_ts.append(ts_arr[-1])
        trades.append({'strategy':'CorrArb','symbol':'EUR',
            'dir':direction,'lot':lot,'entry':entry_price,'exit':ep,
            'entry_ts':ts_arr[entry_bar],'exit_ts':ts_arr[-1],
            'pnl':pnl,'status':'eod','sl':sl_price,'tp':tp_price})

    class RR:
        def __init__(self, c, ct, hr):
            self.curve=c; self.curve_ts=ct; self.equity=c[-1]; self.halt_reason=hr
        @property
        def max_dd(self):
            s=pd.Series(self.curve); return ((s-s.cummax())/s.cummax()*100).min()
        @property
        def max_dd_abs(self):
            s=pd.Series(self.curve); return (s-s.cummax()).min()
        @property
        def sharpe(self):
            r=pd.Series(self.curve).pct_change().dropna()
            return (r.mean()/r.std()*np.sqrt(252*96)) if r.std()>0 else 0
        @property
        def sortino(self):
            r=pd.Series(self.curve).pct_change().dropna()
            neg=r[r<0]; ds=neg.std() if len(neg)>0 else 1e-10
            return (r.mean()/ds*np.sqrt(252*96)) if ds>0 else 0
        @property
        def calmar(self):
            ret=(self.curve[-1]/Config.initial_balance-1)
            dd=abs(self.max_dd/100); return ret/dd if dd>0 else 0

    return trades, RR(curve, curve_ts, halt_r)


# ================================================================== #
#               آمار + گزارش                                        #
# ================================================================== #
def compute_stats(trades: list, risk, name: str) -> dict:
    if not trades:
        return None

    t = pd.DataFrame(trades)
    t['pnl']          = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts']-t['entry_ts']).dt.total_seconds()/60

    start_d      = t['entry_ts'].min()
    end_d        = t['exit_ts'].max()
    total_days   = max((end_d-start_d).days, 1)
    total_months = total_days/30.44
    total_years  = total_days/365.25

    final_eq  = risk.equity
    total_pnl = final_eq - Config.initial_balance
    total_ret = total_pnl / Config.initial_balance * 100
    ann_ret   = ((final_eq/Config.initial_balance)**(365.25/total_days)-1)*100

    win_t  = t[t['pnl']>0]; loss_t = t[t['pnl']<0]
    win_r  = len(win_t)/len(t)*100 if len(t)>0 else 0
    avg_w  = win_t['pnl'].mean()  if len(win_t)>0  else 0
    avg_l  = loss_t['pnl'].mean() if len(loss_t)>0 else 0
    gw     = win_t['pnl'].sum()
    gl     = abs(loss_t['pnl'].sum())
    pf     = gw/gl if gl>0 else float('inf')
    exp_v  = t['pnl'].mean()
    rr     = abs(avg_w/avg_l) if avg_l!=0 else 0

    sign   = t['pnl'].apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    cw,cl,mcw,mcl=0,0,0,0
    for s in sign:
        if s>0: cw+=1;cl=0;mcw=max(mcw,cw)
        elif s<0: cl+=1;cw=0;mcl=max(mcl,cl)
        else: cw=cl=0

    t['ym'] = t['entry_ts'].dt.to_period('M')
    monthly = (t.groupby('ym')
               .agg(n=('pnl','count'),pnl=('pnl','sum'),
                    wins=('pnl',lambda x:(x>0).sum()),
                    best=('pnl','max'),worst=('pnl','min'))
               .reset_index())
    monthly['wr']      = monthly['wins']/monthly['n']*100
    monthly['ret']     = monthly['pnl']/Config.initial_balance*100
    monthly['cum_pnl'] = monthly['pnl'].cumsum()
    monthly['cum_ret'] = monthly['cum_pnl']/Config.initial_balance*100

    t['yr'] = t['entry_ts'].dt.year
    yearly  = (t.groupby('yr')
               .agg(n=('pnl','count'),pnl=('pnl','sum'),
                    wins=('pnl',lambda x:(x>0).sum()))
               .reset_index())
    yearly['wr']  = yearly['wins']/yearly['n']*100
    yearly['ret'] = yearly['pnl']/Config.initial_balance*100

    return dict(
        name=name, trades=t, monthly=monthly, yearly=yearly, risk=risk,
        total_pnl=total_pnl, total_ret=total_ret, ann_ret=ann_ret,
        total_days=total_days, total_months=total_months,
        win_r=win_r, avg_w=avg_w, avg_l=avg_l, pf=pf,
        exp=exp_v, rr=rr, mcw=mcw, mcl=mcl,
        ppm=total_pnl/total_months,
        best=t['pnl'].max(), worst=t['pnl'].min(),
        avg_dur=t['duration_min'].mean(),
    )


def print_report(s: dict) -> str:
    W   = 72
    SEP = "═" * W
    risk = s['risk']

    def rw(label, value):
        lbl  = f"  {label}"
        val  = str(value)
        dots = "·" * max(2, W-len(lbl)-len(val)-2)
        return f"{lbl} {dots} {val}"

    def box(title):
        inner = f"─ {title} "
        return "┌" + inner + "─"*(W-len(inner)-1) + "┐"

    bot = "└" + "─"*(W-1) + "┘"

    ppm_pct = s['ppm']/Config.initial_balance*100
    is_good = (s['total_ret']>0 and s['pf']>1.2
               and abs(risk.max_dd)<12 and ppm_pct>5)
    flag = "✅ واجد شرایط پراپ" if is_good else "❌ نیاز به بهینه‌سازی"

    lines = [
        "", SEP,
        f"  ▌ Strategy: {s['name']}   {flag}",
        f"  ▌ دوره: {s['trades']['entry_ts'].min().date()} "
        f"→ {s['trades']['exit_ts'].max().date()}  "
        f"({s['total_days']} روز)",
        SEP, "",
        box("نتایج مالی"),
        rw("موجودی اولیه",   f"${Config.initial_balance:>12,.2f}"),
        rw("موجودی نهایی",   f"${risk.equity:>12,.2f}"),
        rw("سود/زیان کل",   f"${s['total_pnl']:>+12,.2f}"),
        rw("بازده کل",       f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه",   f"{s['ann_ret']:>+.2f}%"),
        rw("سود ماهانه avg", f"${s['ppm']:>+.2f}  ({ppm_pct:>+.1f}%/month)"),
        rw("بهترین معامله",  f"${s['best']:>+.2f}"),
        rw("بدترین معامله",  f"${s['worst']:>+.2f}"),
        bot, "",
        box("ریسک"),
        rw("Max Drawdown",   f"{risk.max_dd:.2f}%"),
        rw("Max DD مطلق",   f"${risk.max_dd_abs:>+.2f}"),
        rw("Sharpe Ratio",   f"{risk.sharpe:.2f}"),
        rw("Sortino Ratio",  f"{risk.sortino:.2f}"),
        rw("Calmar Ratio",   f"{risk.calmar:.2f}"),
        rw("Profit Factor",  f"{s['pf']:.2f}"),
        rw("وضعیت پایان",   risk.halt_reason),
        bot, "",
        box("معاملات"),
        rw("تعداد کل",       f"{len(s['trades']):,}"),
        rw("Win Rate",        f"{s['win_r']:.1f}%"),
        rw("Avg Win",         f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",        f"${s['avg_l']:>+.2f}"),
        rw("RR واقعی",       f"{s['rr']:.2f}"),
        rw("Expectancy",      f"${s['exp']:>+.2f}"),
        rw("Max Cons. Win",   f"{s['mcw']}"),
        rw("Max Cons. Loss",  f"{s['mcl']}"),
        rw("مدت میانگین",    f"{s['avg_dur']:.0f} min"),
        bot, "",
    ]

    lines.append(box("توزیع خروج"))
    for status, cnt in s['trades']['status'].value_counts().items():
        pct  = cnt/len(s['trades'])*100
        avg_ = s['trades'].loc[s['trades']['status']==status,'pnl'].mean()
        bar_ = "█"*max(1,int(pct/3))
        lines.append(
            f"  {status:<13} {cnt:>4} ({pct:>5.1f}%)  "
            f"{bar_:<24}  avg=${avg_:>+.2f}")
    lines += [bot, ""]

    lines.append(box("گزارش ماهانه"))
    lines.append(
        f"  {'ماه':>7}  {'#':>4}  {'Win%':>6}  {'PnL':>10}  "
        f"{'Ret%':>6}  {'بهترین':>8}  {'بدترین':>8}  "
        f"{'تجمعی':>10}  {'CumRet':>7}")
    lines.append("  "+"─"*(W-3))
    for _, mr in s['monthly'].iterrows():
        arrow = "▲" if mr['pnl']>=0 else "▼"
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>4}  "
            f"{mr['wr']:>5.1f}%  ${mr['pnl']:>9.2f}  "
            f"{mr['ret']:>+5.1f}%  ${mr['best']:>7.2f}  "
            f"${mr['worst']:>7.2f}  ${mr['cum_pnl']:>9.2f}  "
            f"{mr['cum_ret']:>+6.1f}% {arrow}")
    lines += [bot, ""]

    lines.append(box("گزارش سالانه"))
    lines.append(
        f"  {'سال':>5}  {'#':>5}  {'Win%':>6}  "
        f"{'PnL':>10}  {'Ret%':>7}")
    lines.append("  "+"─"*(W-3))
    for _, yr in s['yearly'].iterrows():
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>5.1f}%  ${yr['pnl']:>9.2f}  "
            f"{yr['ret']:>+6.1f}%")
    lines.append(bot)

    out = "\n".join(lines)
    print(out)
    return out


def print_comparison(results: list) -> str:
    W   = 72
    SEP = "═" * W
    lines = [
        "", SEP,
        "  ▌  STRATEGY COMPARISON  ▐",
        SEP,
        f"  {'نام':<14} {'Ret%':>7} {'Ann%':>8} {'DD%':>7} "
        f"{'PF':>6} {'Win%':>6} {'RR':>5} "
        f"{'Shr':>6}  {'وضعیت'}",
        "  "+"─"*(W-3),
    ]
    for s in results:
        r    = s['risk']
        flag = "✅ PASS" if (s['total_ret']>0 and s['pf']>1.2
                             and abs(r.max_dd)<12) else "❌ FAIL"
        pf_s = f"{s['pf']:.2f}" if s['pf']!=float('inf') else "  ∞"
        lines.append(
            f"  {s['name']:<14} {s['total_ret']:>+6.1f}% "
            f"{s['ann_ret']:>+7.1f}% {r.max_dd:>6.1f}% "
            f"{pf_s:>6} {s['win_r']:>5.1f}% {s['rr']:>5.2f} "
            f"{r.sharpe:>6.1f}  {flag}")

    good = [s for s in results
            if s['total_ret']>0 and s['pf']>1.2 and abs(s['risk'].max_dd)<12]
    lines += ["  "+"─"*(W-3), ""]
    if good:
        lines.append("  🏆 استراتژی‌های واجد شرایط پراپ:")
        for s in sorted(good, key=lambda x: x['ann_ret'], reverse=True):
            lines.append(
                f"     ✅ {s['name']:<14}  "
                f"سالانه={s['ann_ret']:>+.1f}%  "
                f"DD={s['risk'].max_dd:.1f}%  "
                f"PF={s['pf']:.2f}  "
                f"WR={s['win_r']:.1f}%")
    else:
        lines.append("  ⚠️  هیچ استراتژی‌ای معیار پراپ را ندارد")

    lines += ["", SEP]
    out = "\n".join(lines)
    print(out)
    return out


def save_csv(results: list):
    rows = [
        ["STRATEGY BACKTEST REPORT"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [""],
        ["Strategy","TotalPnL","TotalRet%","AnnRet%","MaxDD%",
         "Sharpe","PF","WinRate%","RR","Expectancy","Trades","Status"],
    ]
    for s in results:
        r    = s['risk']
        flag = "PASS" if (s['total_ret']>0 and s['pf']>1.2
                          and abs(r.max_dd)<12) else "FAIL"
        pf_v = round(s['pf'],2) if s['pf']!=float('inf') else 999
        rows.append([s['name'], round(s['total_pnl'],2), round(s['total_ret'],2),
                     round(s['ann_ret'],2), round(r.max_dd,2),
                     round(r.sharpe,2), pf_v, round(s['win_r'],1),
                     round(s['rr'],2), round(s['exp'],2),
                     len(s['trades']), flag])

    for s in results:
        rows += [[""], [f"=== MONTHLY: {s['name']} ==="],
                 ["Month","N","WinRate%","PnL","Ret%",
                  "Best","Worst","CumPnL","CumRet%"]]
        for _, mr in s['monthly'].iterrows():
            rows.append([str(mr['ym']),int(mr['n']),round(mr['wr'],1),
                         round(mr['pnl'],2),round(mr['ret'],2),
                         round(mr['best'],2),round(mr['worst'],2),
                         round(mr['cum_pnl'],2),round(mr['cum_ret'],2)])

    for s in results:
        rows += [[""], [f"=== YEARLY: {s['name']} ==="],
                 ["Year","N","WinRate%","PnL","Ret%"]]
        for _, yr in s['yearly'].iterrows():
            rows.append([int(yr['yr']),int(yr['n']),round(yr['wr'],1),
                         round(yr['pnl'],2),round(yr['ret'],2)])

    pd.DataFrame(rows).to_csv(
        "Strategy_Report.csv", index=False, header=False, encoding="utf-8-sig")

    for s in results:
        r     = s['risk']
        eq_df = pd.DataFrame({'ts':r.curve_ts,'equity':r.curve})
        eq_df['dd'] = (
            (eq_df['equity']-eq_df['equity'].cummax())
            /eq_df['equity'].cummax()*100).round(4)
        eq_df.to_csv(f"equity_{s['name']}.csv",
                     index=False, encoding="utf-8-sig")

    print(f"\n✅ فایل‌ها ذخیره شد:")
    print(f"   → Strategy_Report.csv")
    for s in results:
        print(f"   → equity_{s['name']}.csv")


# ================================================================== #
#                           MAIN                                     #
# ================================================================== #
if __name__ == "__main__":
    df = load_data()

    print("\n" + "═"*72)
    print("  محاسبه اندیکاتورها (vectorized)...")

    # ── محاسبه همه سیگنال‌ها یکجا ──
    sig_arb, SL_arb, TP_arb, z_ser = signals_corr_arb(df)
    sig_ab,  SL_ab,  TP_ab         = signals_asian_breakout(df)
    sig_tp,  SL_tp,  TP_tp         = signals_trend_pullback(df)
    sig_ny,  SL_ny,  TP_ny         = signals_ny_fade(df)

    print("  شروع بک‌تست‌ها...")
    print("═"*72)

    h_e = df['h_eur'].values
    l_e = df['l_eur'].values
    c_e = df['c_eur'].values

    runs = [
        # (name, sig, SL, TP, special)
        ("CorrArb",    sig_arb, SL_arb, TP_arb, True),
        ("AsianBreak", sig_ab,  SL_ab,  TP_ab,  False),
        ("TrendPB",    sig_tp,  SL_tp,  TP_tp,  False),
        ("NYFade",     sig_ny,  SL_ny,  TP_ny,  False),
    ]

    all_results = []
    all_texts   = []

    for name, sig, SL, TP, special in runs:
        t0 = datetime.now()
        print(f"\n  ▶ {name} ...", end="", flush=True)

        if special:
            trades, risk = run_corr_arb_special(df, sig, SL, TP, z_ser)
        else:
            trades, risk = fast_backtest(
                sig, SL, TP, h_e, l_e, c_e, 'EUR', name)

        dt = (datetime.now()-t0).total_seconds()
        print(f" {dt:.1f}s | {len(trades)} معامله")

        stats = compute_stats(trades, risk, name)
        if stats:
            all_results.append(stats)
            txt = print_report(stats)
            all_texts.append(txt)

    comp = print_comparison(all_results)
    all_texts.append(comp)

    with open("Backtest_Report.txt","w",encoding="utf-8") as f:
        f.write(f"BACKTEST — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("\n".join(all_texts))

    save_csv(all_results)
