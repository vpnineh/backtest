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
    risk_per_trade_pct   = 0.015       # 1.5% - aggressive but controlled
    max_daily_loss_pct   = 0.045       # 4.5% daily (prop standard)
    max_total_dd_pct     = 0.09        # 9% total (under prop 10%)
    profit_target_pct    = 50.0        # no practical cap
    spread_eur_pips      = 1.0
    spread_gbp_pips      = 1.2
    commission_per_lot   = 6.0
    pip                  = 0.0001
    lot_size             = 100_000
    max_lot              = 3.0
    atr_period           = 14
    min_rr               = 1.5        # minimum reward:risk


# ================================================================== #
#                        DATA LOADER                                 #
# ================================================================== #
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur or not files_gbp:
        raise FileNotFoundError("CSV not found in data/")

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
    print(f"✅ {len(df):,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ================================================================== #
#                      INDICATORS                                    #
# ================================================================== #
def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rsi(close, period=14):
    d    = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100/(1+rs)

def calc_adx_full(high, low, close, period=14):
    """Returns ADX, +DI, -DI"""
    up   = high.diff()
    down = -low.diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_n = down.where((down > up) & (down > 0), 0.0)
    tr   = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_s = tr.rolling(period).sum()
    di_p  = 100 * dm_p.rolling(period).sum() / atr_s.replace(0, np.nan)
    di_n  = 100 * dm_n.rolling(period).sum() / atr_s.replace(0, np.nan)
    dx    = (abs(di_p - di_n) / (di_p + di_n).replace(0, np.nan)) * 100
    adx   = dx.rolling(period).mean()
    return adx, di_p, di_n

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast,   adjust=False).mean()
    ema_s = close.ewm(span=slow,   adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal,  adjust=False).mean()
    return macd, sig, macd - sig

def calc_bbands(close, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + mult*std, mid, mid - mult*std

def calc_stoch(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    k_val   = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d_val   = k_val.rolling(d).mean()
    return k_val, d_val

def calc_vwap_like(close, volume, period=20):
    """Pseudo-VWAP on rolling window"""
    return (close * volume).rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)

def calc_ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


# ================================================================== #
#                     UTILITIES                                      #
# ================================================================== #
def trade_cost(lot, symbol):
    sp = Config.spread_eur_pips if symbol == 'EUR' else Config.spread_gbp_pips
    return sp * Config.pip * lot * Config.lot_size + Config.commission_per_lot * lot

def calc_pnl(direction, lot, entry, exit_p, symbol):
    raw = direction * (exit_p - entry) * lot * Config.lot_size
    return raw - trade_cost(lot, symbol)

def lot_size_calc(equity, sl_pips):
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
        self.name          = name
        self.equity        = Config.initial_balance
        self.peak          = Config.initial_balance
        self.day_start_eq  = Config.initial_balance
        self.cur_day       = None
        self.halted        = False
        self.halt_reason   = "Running"
        self.curve         = [Config.initial_balance]
        self.curve_ts      = [None]
        self.daily_pnl     = {}

    def new_bar(self, ts):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day      = day
            self.day_start_eq = self.equity

    def add_pnl(self, amount, ts):
        self.equity += amount
        self.peak    = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))
        self.curve_ts.append(ts)

        d_key = str(ts.date())
        self.daily_pnl[d_key] = self.daily_pnl.get(d_key, 0) + amount

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
        return True

    def can_trade(self):
        if self.halted:
            return False
        daily_dd = (self.equity - self.day_start_eq) / self.day_start_eq
        return daily_dd > -Config.max_daily_loss_pct * 0.7  # margin

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
        ret = self.equity / Config.initial_balance - 1
        dd  = abs(self.max_dd / 100)
        return ret / dd if dd > 0 else 0


# ================================================================== #
# STRATEGY 1: EMA Crossover + RSI Filter (EUR)                      #
#                                                                    #
# EMA8/EMA21 cross + RSI confirmation + ADX trend filter             #
# High frequency, trend-following                                    #
# ================================================================== #
def strategy_ema_cross(df):
    c = df['c_eur']; h = df['h_eur']; l = df['l_eur']

    ema8   = calc_ema(c, 8)
    ema21  = calc_ema(c, 21)
    ema50  = calc_ema(c, 50)
    rsi    = calc_rsi(c, 10)
    atr    = calc_atr(h, l, c, 14)
    adx, di_p, di_n = calc_adx_full(h, l, c, 14)
    _, _, macd_hist  = calc_macd(c, 8, 21, 5)

    hour = pd.Series(df.index.hour, index=df.index)
    active = hour.between(7, 17)  # London + NY

    # Detect fresh crossover
    cross_up   = (ema8 > ema21) & (ema8.shift(1) <= ema21.shift(1))
    cross_down = (ema8 < ema21) & (ema8.shift(1) >= ema21.shift(1))

    sigs = pd.DataFrame(index=df.index)
    sigs['signal']  = 0
    sigs['sl_pips'] = 0.0
    sigs['tp_pips'] = 0.0

    # ── Long conditions ──
    long_cond = (
        cross_up &
        active &
        (c > ema50) &                    # above major trend
        (rsi > 45) & (rsi < 72) &        # not overbought
        (adx > 20) &                     # trending
        (di_p > di_n) &                  # bulls dominant
        (macd_hist > 0)                  # MACD confirms
    )

    # ── Short conditions ──
    short_cond = (
        cross_down &
        active &
        (c < ema50) &
        (rsi > 28) & (rsi < 55) &
        (adx > 20) &
        (di_n > di_p) &
        (macd_hist < 0)
    )

    for idx in df.index:
        i = df.index.get_loc(idx)
        if i < 200:
            continue
        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        atr_pips = atr_v / Config.pip

        if long_cond.iloc[i]:
            sl_p = max(10, min(atr_pips * 1.0, 22))
            tp_p = sl_p * 2.0
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p

        elif short_cond.iloc[i]:
            sl_p = max(10, min(atr_pips * 1.0, 22))
            tp_p = sl_p * 2.0
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p

    # Min 2h gap between signals
    nz = sigs[sigs['signal'] != 0]
    if len(nz) > 1:
        keep = [nz.index[0]]
        for idx in nz.index[1:]:
            if (idx - keep[-1]).total_seconds() >= 2*3600:
                keep.append(idx)
        drop = [i for i in nz.index if i not in keep]
        sigs.loc[drop, 'signal'] = 0

    return sigs


# ================================================================== #
# STRATEGY 2: Bollinger Band Mean Reversion (EUR)                    #
#                                                                    #
# Touch lower band in uptrend → buy | upper band in downtrend → sell #
# RSI confirmation + volume spike                                    #
# ================================================================== #
def strategy_bb_reversion(df):
    c = df['c_eur']; h = df['h_eur']; l = df['l_eur']; o = df['o_eur']
    v = df['v_eur']

    bb_up, bb_mid, bb_lo = calc_bbands(c, 20, 2.0)
    rsi  = calc_rsi(c, 14)
    atr  = calc_atr(h, l, c, 14)
    ema100 = calc_ema(c, 100)
    stk, std = calc_stoch(h, l, c, 14, 3)

    # Volume above average
    vol_ma = v.rolling(50).mean()
    vol_ok = v > vol_ma * 0.8

    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(8, 17)

    sigs = pd.DataFrame(index=df.index)
    sigs['signal']  = 0
    sigs['sl_pips'] = 0.0
    sigs['tp_pips'] = 0.0

    # ── Long: price touches/pierces lower BB in uptrend ──
    long_cond = (
        active & vol_ok &
        (l <= bb_lo) &                   # touched lower band
        (c > bb_lo) &                    # closed above it (rejection)
        (c > o) &                        # bullish candle
        (c > ema100) &                   # major uptrend
        (rsi > 25) & (rsi < 45) &        # oversold zone
        (stk < 25)                       # stoch oversold
    )

    # ── Short: price touches upper BB in downtrend ──
    short_cond = (
        active & vol_ok &
        (h >= bb_up) &
        (c < bb_up) &
        (c < o) &
        (c < ema100) &
        (rsi > 55) & (rsi < 75) &
        (stk > 75)
    )

    for idx in df.index:
        i = df.index.get_loc(idx)
        if i < 200:
            continue
        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        atr_pips = atr_v / Config.pip

        if long_cond.iloc[i]:
            sl_p = max(8, min(atr_pips * 0.8, 18))
            tp_p = sl_p * 2.2
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p

        elif short_cond.iloc[i]:
            sl_p = max(8, min(atr_pips * 0.8, 18))
            tp_p = sl_p * 2.2
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p

    # Min 3h gap
    nz = sigs[sigs['signal'] != 0]
    if len(nz) > 1:
        keep = [nz.index[0]]
        for idx in nz.index[1:]:
            if (idx - keep[-1]).total_seconds() >= 3*3600:
                keep.append(idx)
        drop = [i for i in nz.index if i not in keep]
        sigs.loc[drop, 'signal'] = 0

    return sigs


# ================================================================== #
# STRATEGY 3: RSI Divergence Enhanced (EUR) - Proven from V2         #
#                                                                    #
# Same core logic that produced PF=4.44, WR=76.8%                   #
# Enhanced: tighter SL, better timing, compound sizing               #
# ================================================================== #
def strategy_rsi_div(df):
    c = df['c_eur']; h = df['h_eur']; l = df['l_eur']; o = df['o_eur']

    rsi    = calc_rsi(c, 14)
    atr    = calc_atr(h, l, c, 14)
    ema50  = calc_ema(c, 50)
    adx, _, _ = calc_adx_full(h, l, c, 14)

    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(7, 18)

    lb = 10
    swing_low  = l.rolling(lb*2+1, center=True).min()
    swing_high = h.rolling(lb*2+1, center=True).max()

    is_sw_lo = (l == swing_low) & (l < l.shift(1)) & (l < l.shift(-1))
    is_sw_hi = (h == swing_high) & (h > h.shift(1)) & (h > h.shift(-1))

    last_swl_p = l.where(is_sw_lo).ffill()
    last_swl_r = rsi.where(is_sw_lo).ffill()
    prev_swl_p = last_swl_p.shift(lb)
    prev_swl_r = last_swl_r.shift(lb)

    last_swh_p = h.where(is_sw_hi).ffill()
    last_swh_r = rsi.where(is_sw_hi).ffill()
    prev_swh_p = last_swh_p.shift(lb)
    prev_swh_r = last_swh_r.shift(lb)

    bull_div = (
        (last_swl_p < prev_swl_p) &
        (last_swl_r > prev_swl_r + 3) &
        (rsi < 40) & (rsi > rsi.shift(1)) &
        (c > o) & active
    )
    bear_div = (
        (last_swh_p > prev_swh_p) &
        (last_swh_r < prev_swh_r - 3) &
        (rsi > 60) & (rsi < rsi.shift(1)) &
        (c < o) & active
    )

    sigs = pd.DataFrame(index=df.index)
    sigs['signal']  = 0
    sigs['sl_pips'] = 0.0
    sigs['tp_pips'] = 0.0

    for idx in df.index:
        i = df.index.get_loc(idx)
        if i < 250:
            continue
        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        atr_pips = atr_v / Config.pip

        if bull_div.iloc[i]:
            sl_p = max(12, min(atr_pips * 1.2, 25))
            tp_p = sl_p * 2.0
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p
        elif bear_div.iloc[i]:
            sl_p = max(12, min(atr_pips * 1.2, 25))
            tp_p = sl_p * 2.0
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p

    # 4h gap
    nz = sigs[sigs['signal'] != 0]
    if len(nz) > 1:
        keep = [nz.index[0]]
        for idx in nz.index[1:]:
            if (idx - keep[-1]).total_seconds() >= 4*3600:
                keep.append(idx)
        drop = [i for i in nz.index if i not in keep]
        sigs.loc[drop, 'signal'] = 0

    return sigs


# ================================================================== #
# STRATEGY 4: GBP Momentum Breakout                                  #
#                                                                    #
# GBP is more volatile → exploit momentum breakouts                  #
# ADX rising + DI cross + volume confirmation                        #
# ================================================================== #
def strategy_gbp_momentum(df):
    c = df['c_gbp']; h = df['h_gbp']; l = df['l_gbp']; o = df['o_gbp']
    v = df['v_gbp']

    ema13  = calc_ema(c, 13)
    ema34  = calc_ema(c, 34)
    rsi    = calc_rsi(c, 10)
    atr    = calc_atr(h, l, c, 14)
    adx, di_p, di_n = calc_adx_full(h, l, c, 14)
    _, _, macd_hist  = calc_macd(c, 12, 26, 9)

    vol_ma = v.rolling(50).mean()
    vol_ok = v > vol_ma * 1.0  # above average volume

    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(8, 16)  # London session primarily

    # ADX trending up
    adx_rising = adx > adx.shift(2)

    sigs = pd.DataFrame(index=df.index)
    sigs['signal']  = 0
    sigs['sl_pips'] = 0.0
    sigs['tp_pips'] = 0.0

    # ── Long: DI+ crosses above DI-, momentum confirmed ──
    di_cross_up   = (di_p > di_n) & (di_p.shift(1) <= di_n.shift(1))
    di_cross_down = (di_n > di_p) & (di_n.shift(1) <= di_p.shift(1))

    long_cond = (
        active & vol_ok &
        (di_cross_up | ((di_p > di_n + 5) & (ema13 > ema34) &
                        (ema13.shift(1) <= ema34.shift(1)))) &
        (adx > 22) & adx_rising &
        (rsi > 50) & (rsi < 75) &
        (macd_hist > 0) &
        (c > ema34)
    )

    short_cond = (
        active & vol_ok &
        (di_cross_down | ((di_n > di_p + 5) & (ema13 < ema34) &
                          (ema13.shift(1) >= ema34.shift(1)))) &
        (adx > 22) & adx_rising &
        (rsi > 25) & (rsi < 50) &
        (macd_hist < 0) &
        (c < ema34)
    )

    for idx in df.index:
        i = df.index.get_loc(idx)
        if i < 200:
            continue
        atr_v = atr.iloc[i]
        if pd.isna(atr_v) or atr_v <= 0:
            continue
        atr_pips = atr_v / Config.pip

        if long_cond.iloc[i]:
            sl_p = max(12, min(atr_pips * 1.0, 25))
            tp_p = sl_p * 2.0
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p
        elif short_cond.iloc[i]:
            sl_p = max(12, min(atr_pips * 1.0, 25))
            tp_p = sl_p * 2.0
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl_p
            sigs.at[idx, 'tp_pips'] = tp_p

    # 3h gap
    nz = sigs[sigs['signal'] != 0]
    if len(nz) > 1:
        keep = [nz.index[0]]
        for idx in nz.index[1:]:
            if (idx - keep[-1]).total_seconds() >= 3*3600:
                keep.append(idx)
        drop = [i for i in nz.index if i not in keep]
        sigs.loc[drop, 'signal'] = 0

    return sigs


# ================================================================== #
#                    BACKTEST ENGINE                                  #
# ================================================================== #
def run_single(df, name, signals, symbol='EUR',
               trailing=True, time_stop_h=48):
    risk     = RiskManager(name)
    trades   = []
    position = None
    warmup   = 300

    if symbol == 'EUR':
        h_col, l_col, c_col = 'h_eur', 'l_eur', 'c_eur'
    else:
        h_col, l_col, c_col = 'h_gbp', 'l_gbp', 'c_gbp'

    atr = calc_atr(df[h_col], df[l_col], df[c_col], 14)

    for i in range(warmup, len(df)):
        ts  = df.index[i]
        hi  = df[h_col].iloc[i]
        lo  = df[l_col].iloc[i]
        cp  = df[c_col].iloc[i]

        risk.new_bar(ts)

        if risk.halted:
            if position:
                pnl = calc_pnl(position['dir'], position['lot'],
                               position['entry'], cp, symbol)
                trades.append({**position, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'halt_close'})
                risk.add_pnl(pnl, ts)
                position = None
            continue

        # ── EXIT ──
        if position:
            d  = position['dir']
            ep = position['entry']
            sl = position['sl']
            tp = position['tp']

            # Trailing stop
            if trailing:
                atr_v = atr.iloc[i]
                if pd.notna(atr_v) and atr_v > 0:
                    move = d * (cp - ep)
                    if move > atr_v * 1.2:
                        be = ep + d * atr_v * 0.3
                        if d == 1:
                            position['sl'] = max(position['sl'], be)
                        else:
                            position['sl'] = min(position['sl'], be)
                    if move > atr_v * 2.0:
                        lk = ep + d * atr_v * 1.0
                        if d == 1:
                            position['sl'] = max(position['sl'], lk)
                        else:
                            position['sl'] = min(position['sl'], lk)
                    sl = position['sl']

            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            # Time stop
            elapsed = (ts - position['entry_ts']).total_seconds() / 3600
            if elapsed >= time_stop_h and not hit_tp:
                pnl = calc_pnl(d, position['lot'], ep, cp, symbol)
                trades.append({**position, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts)
                position = None
                continue

            # Weekend
            if ts.weekday() == 4 and ts.hour >= 20:
                pnl = calc_pnl(d, position['lot'], ep, cp, symbol)
                trades.append({**position, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'WeekEnd'})
                risk.add_pnl(pnl, ts)
                position = None
                continue

            if hit_sl or hit_tp:
                ex_p = sl if hit_sl else tp
                st   = 'SL' if hit_sl else 'TP'
                pnl  = calc_pnl(d, position['lot'], ep, ex_p, symbol)
                trades.append({**position, 'exit': ex_p, 'exit_ts': ts,
                               'pnl': pnl, 'status': st})
                risk.add_pnl(pnl, ts)
                position = None

        # ── ENTRY ──
        if position is None and risk.can_trade():
            sv = signals['signal'].iloc[i]
            if sv != 0:
                sv     = int(sv)
                sl_p   = signals['sl_pips'].iloc[i]
                tp_p   = signals['tp_pips'].iloc[i]
                if sl_p <= 0 or tp_p <= 0:
                    continue
                if tp_p / sl_p < Config.min_rr:
                    continue

                lot    = lot_size_calc(risk.equity, sl_p)
                spread = (Config.spread_eur_pips if symbol == 'EUR'
                          else Config.spread_gbp_pips)
                half   = spread * Config.pip / 2
                entry  = cp + sv * half

                position = dict(
                    strategy=name, symbol=symbol,
                    dir=sv, lot=lot, entry=entry,
                    sl=entry - sv * sl_p * Config.pip,
                    tp=entry + sv * tp_p * Config.pip,
                    entry_ts=ts,
                )

    # Close remaining
    if position:
        last_p = df[c_col].iloc[-1]
        pnl = calc_pnl(position['dir'], position['lot'],
                        position['entry'], last_p, symbol)
        trades.append({**position, 'exit': last_p,
                       'exit_ts': df.index[-1],
                       'pnl': pnl, 'status': 'eod_close'})
        risk.add_pnl(pnl, df.index[-1])

    return trades, risk


# ================================================================== #
#                    COMBINED ENGINE                                  #
# ================================================================== #
def run_combined(df, strat_dict):
    """strat_dict: {name: (signals_df, symbol)}"""
    risk     = RiskManager("Combined")
    trades   = []
    open_pos = {}
    warmup   = 300

    atr_eur = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    atr_gbp = calc_atr(df['h_gbp'], df['l_gbp'], df['c_gbp'], 14)

    max_pos = min(len(strat_dict), 4)

    for i in range(warmup, len(df)):
        ts = df.index[i]
        risk.new_bar(ts)

        if risk.halted:
            for k in list(open_pos.keys()):
                p   = open_pos.pop(k)
                sym = p['symbol']
                cp  = df[f'c_{sym.lower()}'].iloc[i]
                pnl = calc_pnl(p['dir'], p['lot'], p['entry'], cp, sym)
                trades.append({**p, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'halt_close'})
                risk.add_pnl(pnl, ts)
            continue

        # ── EXIT ──
        for k in list(open_pos.keys()):
            p   = open_pos[k]
            sym = p['symbol']
            s_  = sym.lower()
            hi  = df[f'h_{s_}'].iloc[i]
            lo  = df[f'l_{s_}'].iloc[i]
            cp  = df[f'c_{s_}'].iloc[i]
            d   = p['dir']

            # Trailing
            atr_v = (atr_eur if sym == 'EUR' else atr_gbp).iloc[i]
            if pd.notna(atr_v) and atr_v > 0:
                move = d * (cp - p['entry'])
                if move > atr_v * 1.2:
                    be = p['entry'] + d * atr_v * 0.3
                    if d == 1:
                        p['sl'] = max(p['sl'], be)
                    else:
                        p['sl'] = min(p['sl'], be)
                if move > atr_v * 2.0:
                    lk = p['entry'] + d * atr_v * 1.0
                    if d == 1:
                        p['sl'] = max(p['sl'], lk)
                    else:
                        p['sl'] = min(p['sl'], lk)

            sl = p['sl']
            tp = p['tp']
            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            elapsed = (ts - p['entry_ts']).total_seconds() / 3600
            max_h   = {'EMA_Cross': 36, 'BB_Reversion': 36,
                       'RSI_Div': 48, 'GBP_Momentum': 36}
            if elapsed >= max_h.get(p['strategy'], 48):
                pnl = calc_pnl(d, p['lot'], p['entry'], cp, sym)
                trades.append({**p, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts)
                del open_pos[k]; continue

            if ts.weekday() == 4 and ts.hour >= 20:
                pnl = calc_pnl(d, p['lot'], p['entry'], cp, sym)
                trades.append({**p, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'WeekEnd'})
                risk.add_pnl(pnl, ts)
                del open_pos[k]; continue

            if hit_sl or hit_tp:
                ex_p = sl if hit_sl else tp
                st   = 'SL' if hit_sl else 'TP'
                pnl  = calc_pnl(d, p['lot'], p['entry'], ex_p, sym)
                trades.append({**p, 'exit': ex_p, 'exit_ts': ts,
                               'pnl': pnl, 'status': st})
                risk.add_pnl(pnl, ts)
                del open_pos[k]

        # ── ENTRY ──
        if not risk.can_trade() or len(open_pos) >= max_pos:
            continue

        for sname, (sigs, sym) in strat_dict.items():
            if sname in open_pos or len(open_pos) >= max_pos:
                continue

            sv = sigs['signal'].iloc[i]
            if sv == 0:
                continue
            sv   = int(sv)
            sl_p = sigs['sl_pips'].iloc[i]
            tp_p = sigs['tp_pips'].iloc[i]
            if sl_p <= 0 or tp_p <= 0:
                continue
            if tp_p / sl_p < Config.min_rr:
                continue

            cp     = df[f'c_{sym.lower()}'].iloc[i]
            lot    = lot_size_calc(risk.equity, sl_p)
            spread = (Config.spread_eur_pips if sym == 'EUR'
                      else Config.spread_gbp_pips)
            entry  = cp + sv * spread * Config.pip / 2

            open_pos[sname] = dict(
                strategy=sname, symbol=sym,
                dir=sv, lot=lot, entry=entry,
                sl=entry - sv * sl_p * Config.pip,
                tp=entry + sv * tp_p * Config.pip,
                entry_ts=ts,
            )

    # Close rest
    for k, p in open_pos.items():
        sym = p['symbol']
        cp  = df[f'c_{sym.lower()}'].iloc[-1]
        pnl = calc_pnl(p['dir'], p['lot'], p['entry'], cp, sym)
        trades.append({**p, 'exit': cp, 'exit_ts': df.index[-1],
                       'pnl': pnl, 'status': 'eod_close'})
        risk.add_pnl(pnl, df.index[-1])

    return trades, risk


# ================================================================== #
#                    REPORT GENERATOR                                #
# ================================================================== #
def report(trades, risk, title=""):
    if not trades:
        print(f"\n❌ [{title}] No trades!")
        return None

    t = pd.DataFrame(trades)
    t['pnl']      = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts'] = pd.to_datetime(t['entry_ts'])
    t['exit_ts']  = pd.to_datetime(t['exit_ts'])
    t['dur_min']  = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    start_d     = t['entry_ts'].min()
    end_d       = t['exit_ts'].max()
    total_days  = max((end_d - start_d).days, 1)
    total_mo    = total_days / 30.44
    total_yr    = total_days / 365.25

    final_eq   = risk.equity
    total_pnl  = final_eq - Config.initial_balance
    total_ret  = total_pnl / Config.initial_balance * 100
    ann_ret    = (((final_eq/Config.initial_balance)**(365.25/total_days)-1)
                  *100) if total_days > 1 else 0

    wt = t[t['pnl'] > 0]; lt = t[t['pnl'] < 0]
    wr   = len(wt)/len(t)*100 if len(t) > 0 else 0
    avgw = wt['pnl'].mean() if len(wt) > 0 else 0
    avgl = lt['pnl'].mean() if len(lt) > 0 else 0
    gw   = wt['pnl'].sum(); gl = abs(lt['pnl'].sum())
    pf   = gw/gl if gl > 0 else float('inf')
    exp  = t['pnl'].mean()
    rr   = abs(avgw/avgl) if avgl != 0 else 0
    mo_r = total_ret / total_mo if total_mo > 0 else 0

    sign = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cw=cl=mcw=mcl=0
    for s in sign:
        if   s > 0: cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0: cl += 1; cw = 0; mcl = max(mcl, cl)
        else:       cw = cl = 0

    W   = 72
    SEP = "═" * W

    def rw(label, value):
        lbl  = f"  {label}"
        val  = str(value)
        dots = "·" * max(2, W - len(lbl) - len(val) - 2)
        return f"{lbl} {dots} {val}"

    def bt(t_):
        inner = f"─ {t_} "
        return "┌" + inner + "─"*(W-len(inner)-1) + "┐"

    bb = lambda: "└" + "─"*(W-1) + "┘"

    # Prop score
    ps = 0
    if wr >= 40:  ps += 20
    if pf >= 1.3: ps += 20
    if abs(risk.max_dd) <= 5: ps += 25
    elif abs(risk.max_dd) <= 8: ps += 15
    if mo_r >= 10: ps += 25
    elif mo_r >= 5: ps += 15
    if rr >= 1.5: ps += 10

    pg = "F"
    if   ps >= 90: pg = "A+"
    elif ps >= 80: pg = "A"
    elif ps >= 70: pg = "B+"
    elif ps >= 60: pg = "B"
    elif ps >= 50: pg = "C"
    elif ps >= 40: pg = "D"

    lines = [
        "", SEP, f"  ▌  {title}  ▐", SEP, "",
        bt("General"),
        rw("Period",       f"{start_d.date()} → {end_d.date()}"),
        rw("Total Days",   f"{total_days:,}"),
        rw("Total Trades", f"{len(t):,}"),
        rw("Trades/Week",  f"{len(t)/(total_days/7):.1f}"),
        bb(), "",

        bt("Financial"),
        rw("Initial",       f"${Config.initial_balance:,.2f}"),
        rw("Final",         f"${final_eq:,.2f}"),
        rw("PnL",           f"${total_pnl:+,.2f}"),
        rw("Total Return",  f"{total_ret:+.2f}%"),
        rw("Monthly Return",f"{mo_r:+.2f}%"),
        rw("Annualized",    f"{ann_ret:+.2f}%"),
        rw("Best Trade",    f"${t['pnl'].max():+.2f}"),
        rw("Worst Trade",   f"${t['pnl'].min():+.2f}"),
        bb(), "",

        bt("Risk"),
        rw("Max DD %",    f"{risk.max_dd:.2f}%"),
        rw("Max DD $",    f"${risk.max_dd_abs:+.2f}"),
        rw("Sharpe",      f"{risk.sharpe:.2f}"),
        rw("Sortino",     f"{risk.sortino:.2f}"),
        rw("Calmar",      f"{risk.calmar:.2f}"),
        rw("Profit Factor",f"{pf:.2f}"),
        rw("Status",       risk.halt_reason),
        bb(), "",

        bt("Statistics"),
        rw("Win Rate",        f"{wr:.1f}%"),
        rw("Winners",         f"{len(wt):,}"),
        rw("Losers",          f"{len(lt):,}"),
        rw("Avg Win",         f"${avgw:+.2f}"),
        rw("Avg Loss",        f"${avgl:+.2f}"),
        rw("R:R",             f"{rr:.2f}"),
        rw("Expectancy",      f"${exp:+.2f}"),
        rw("Max Consec Win",  f"{mcw}"),
        rw("Max Consec Loss", f"{mcl}"),
        rw("Avg Duration",    f"{t['dur_min'].mean():.0f} min"),
        bb(), "",

        bt("Prop Fitness"),
        rw("Score",        f"{ps}/100"),
        rw("Grade",        pg),
        rw("DD < 5%",     "✅" if abs(risk.max_dd) <= 5 else "❌"),
        rw("DD < 10%",    "✅" if abs(risk.max_dd) <= 10 else "❌"),
        rw("Monthly>10%", "✅" if mo_r >= 10 else "❌"),
        rw("PF > 1.3",    "✅" if pf >= 1.3 else "❌"),
        rw("WR > 40%",    "✅" if wr >= 40 else "❌"),
        bb(), "",
    ]

    # Exit dist
    lines.append(bt("Exit Distribution"))
    for st, cnt in t['status'].value_counts().items():
        pct = cnt/len(t)*100
        ap  = t.loc[t['status']==st, 'pnl'].mean()
        bar = "█" * max(1, int(pct/2.5))
        lines.append(f"  {st:<13} {cnt:>5} ({pct:>5.1f}%)  "
                     f"{bar:<28}  avg=${ap:>+.2f}")
    lines.append(bb())

    # Monthly
    t['ym'] = t['entry_ts'].dt.to_period('M')
    mo = (t.groupby('ym')
          .agg(n=('pnl','count'), pnl=('pnl','sum'),
               wins=('pnl', lambda x: (x>0).sum()))
          .reset_index())
    mo['wr']  = mo['wins']/mo['n']*100
    mo['ret'] = mo['pnl']/Config.initial_balance*100
    mo['cum'] = mo['pnl'].cumsum()
    mo['cr']  = mo['cum']/Config.initial_balance*100

    pm = (mo['pnl'] >= 0).sum()

    lines += ["", bt("Monthly")]
    lines.append(f"  {'Month':>7}  {'#':>4}  {'WR%':>5}  "
                 f"{'PnL':>10}  {'Ret%':>6}  {'CumPnL':>10}  {'CumR':>7}")
    lines.append("  " + "─"*(W-3))
    for _, r in mo.iterrows():
        a = "▲" if r['pnl'] >= 0 else "▼"
        lines.append(
            f"  {str(r['ym']):>7}  {int(r['n']):>4}  "
            f"{r['wr']:>4.0f}%  ${r['pnl']:>9.2f}  "
            f"{r['ret']:>+5.1f}%  ${r['cum']:>9.2f}  "
            f"{r['cr']:>+6.1f}% {a}")
    lines.append("  " + "─"*(W-3))
    lines.append(f"  Profitable Months: {pm}/{len(mo)} ({pm/len(mo)*100:.0f}%)")
    lines.append(bb())

    # Yearly
    t['yr'] = t['entry_ts'].dt.year
    yr = (t.groupby('yr')
          .agg(n=('pnl','count'), pnl=('pnl','sum'),
               wins=('pnl', lambda x: (x>0).sum()))
          .reset_index())
    yr['wr']  = yr['wins']/yr['n']*100
    yr['ret'] = yr['pnl']/Config.initial_balance*100

    lines += ["", bt("Yearly")]
    lines.append(f"  {'Year':>5}  {'#':>5}  {'WR%':>5}  "
                 f"{'PnL':>10}  {'Ret%':>7}")
    lines.append("  " + "─"*(W-3))
    for _, r in yr.iterrows():
        lines.append(
            f"  {int(r['yr']):>5}  {int(r['n']):>5}  "
            f"{r['wr']:>4.0f}%  ${r['pnl']:>9.2f}  "
            f"{r['ret']:>+6.1f}%")
    lines += [bb(), "", SEP]

    output = "\n".join(lines)
    print(output)

    return {
        'name': title, 'trades': len(t),
        'total_pnl': total_pnl, 'total_ret': total_ret,
        'monthly_ret': mo_r, 'win_rate': wr, 'pf': pf, 'rr': rr,
        'max_dd': risk.max_dd, 'sharpe': risk.sharpe,
        'sortino': risk.sortino, 'calmar': risk.calmar,
        'exp': exp, 'prop_score': ps, 'prop_grade': pg,
        'pos_months': pm, 'tot_months': len(mo),
        'output': output, 'risk': risk, 'trades_df': t, 'monthly_df': mo,
    }


# ================================================================== #
#                    SAVE RESULTS                                    #
# ================================================================== #
def save_all(results, combined=None):
    with open("Backtest_Report.txt", "w", encoding="utf-8") as f:
        f.write("="*72 + "\n")
        f.write("  PROFESSIONAL PROP TRADING BACKTEST\n")
        f.write(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*72 + "\n\n")

        for r in results:
            if r: f.write(r['output'] + "\n\n")

        if combined:
            f.write("\n" + "="*72 + "\n")
            f.write("  COMBINED PORTFOLIO\n")
            f.write("="*72 + "\n")
            f.write(combined['output'] + "\n")

        # Comparison table
        f.write("\n\n" + "="*72 + "\n")
        f.write("  COMPARISON\n")
        f.write("="*72 + "\n\n")
        f.write(f"  {'Strategy':<20} {'#':>5} {'WR%':>6} "
                f"{'PF':>6} {'RR':>5} {'DD%':>7} "
                f"{'Mo%':>7} {'PnL':>10} {'Sc':>4}\n")
        f.write("  " + "─"*66 + "\n")
        for r in results:
            if r:
                pfs = f"{r['pf']:.2f}" if r['pf']!=float('inf') else "  ∞"
                f.write(
                    f"  {r['name']:<20} {r['trades']:>5} "
                    f"{r['win_rate']:>5.1f}% {pfs:>6} "
                    f"{r['rr']:>5.2f} {r['max_dd']:>6.2f}% "
                    f"{r['monthly_ret']:>+6.2f}% "
                    f"${r['total_pnl']:>9.2f} "
                    f"{r['prop_score']:>3}\n")
        if combined:
            r = combined
            pfs = f"{r['pf']:.2f}" if r['pf']!=float('inf') else "  ∞"
            f.write("  " + "─"*66 + "\n")
            f.write(
                f"  {'COMBINED':<20} {r['trades']:>5} "
                f"{r['win_rate']:>5.1f}% {pfs:>6} "
                f"{r['rr']:>5.2f} {r['max_dd']:>6.2f}% "
                f"{r['monthly_ret']:>+6.2f}% "
                f"${r['total_pnl']:>9.2f} "
                f"{r['prop_score']:>3}\n")

    # CSV
    rows = [["Strategy","Trades","WR%","PF","RR","MaxDD%",
             "MonthlyRet%","TotalPnL","Score","Grade"]]
    for r in results:
        if r:
            rows.append([r['name'], r['trades'],
                         round(r['win_rate'],1),
                         round(r['pf'],2), round(r['rr'],2),
                         round(r['max_dd'],2), round(r['monthly_ret'],2),
                         round(r['total_pnl'],2),
                         r['prop_score'], r['prop_grade']])
    if combined:
        r = combined
        rows.append(["COMBINED", r['trades'],
                     round(r['win_rate'],1),
                     round(r['pf'],2), round(r['rr'],2),
                     round(r['max_dd'],2), round(r['monthly_ret'],2),
                     round(r['total_pnl'],2),
                     r['prop_score'], r['prop_grade']])
    pd.DataFrame(rows).to_csv("Backtest_Summary.csv",
                              index=False, header=False, encoding="utf-8-sig")

    # Equity curves
    for r in results:
        if r and r['risk']:
            eq = pd.DataFrame({'ts': r['risk'].curve_ts,
                               'equity': r['risk'].curve})
            eq['dd'] = ((eq['equity'] - eq['equity'].cummax())
                        / eq['equity'].cummax() * 100).round(4)
            nm = r['name'].replace(' ','_').replace(':','').replace('/','')
            eq.to_csv(f"equity_{nm}.csv", index=False, encoding="utf-8-sig")

    if combined and combined['risk']:
        eq = pd.DataFrame({'ts': combined['risk'].curve_ts,
                           'equity': combined['risk'].curve})
        eq['dd'] = ((eq['equity'] - eq['equity'].cummax())
                    / eq['equity'].cummax() * 100).round(4)
        eq.to_csv("equity_Combined.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved:")
    print(f"   → Backtest_Report.txt")
    print(f"   → Backtest_Summary.csv")
    for r in results:
        if r:
            nm = r['name'].replace(' ','_').replace(':','').replace('/','')
            print(f"   → equity_{nm}.csv")
    if combined:
        print(f"   → equity_Combined.csv")


# ================================================================== #
#                         MAIN                                       #
# ================================================================== #
if __name__ == "__main__":
    print("="*72)
    print("  PROFESSIONAL PROP TRADING BACKTEST v3")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*72)

    df = load_data()

    # ═══════════════════════════════════════════════════
    #  BUILD SIGNALS
    # ═══════════════════════════════════════════════════
    print("\n⚙️  Strategy 1: EMA Crossover + RSI...")
    sig1 = strategy_ema_cross(df)
    n1   = (sig1['signal'] != 0).sum()
    print(f"   → {n1} signals")

    print("⚙️  Strategy 2: Bollinger Band Reversion...")
    sig2 = strategy_bb_reversion(df)
    n2   = (sig2['signal'] != 0).sum()
    print(f"   → {n2} signals")

    print("⚙️  Strategy 3: RSI Divergence (proven)...")
    sig3 = strategy_rsi_div(df)
    n3   = (sig3['signal'] != 0).sum()
    print(f"   → {n3} signals")

    print("⚙️  Strategy 4: GBP Momentum...")
    sig4 = strategy_gbp_momentum(df)
    n4   = (sig4['signal'] != 0).sum()
    print(f"   → {n4} signals")

    # ═══════════════════════════════════════════════════
    #  INDIVIDUAL BACKTESTS
    # ═══════════════════════════════════════════════════
    print("\n" + "="*72)
    print("  INDIVIDUAL BACKTESTS")
    print("="*72)

    all_results   = []
    all_strat_map = {}

    configs = [
        ("EMA_Cross",     sig1, 'EUR', True, 36),
        ("BB_Reversion",  sig2, 'EUR', True, 36),
        ("RSI_Div",       sig3, 'EUR', True, 48),
        ("GBP_Momentum",  sig4, 'GBP', True, 36),
    ]

    for name, sigs, sym, trail, tsh in configs:
        n_sig = (sigs['signal'] != 0).sum()
        if n_sig == 0:
            print(f"\n⚠️  {name}: 0 signals, skipping")
            all_results.append(None)
            continue

        print(f"\n🔄 {name} ({sym})...")
        tr, rk = run_single(df, name, sigs, sym, trail, tsh)
        r = report(tr, rk, name)
        all_results.append(r)
        if r:
            all_strat_map[name] = (sigs, sym, r)

    # ═══════════════════════════════════════════════════
    #  FILTER PROFITABLE ONLY
    # ═══════════════════════════════════════════════════
    print("\n" + "="*72)
    print("  STRATEGY SELECTION")
    print("="*72)

    profitable = {}
    for name, (sigs, sym, r) in all_strat_map.items():
        ok   = True
        fail = []
        if r['total_pnl'] <= 0:
            ok = False; fail.append(f"Losing ${r['total_pnl']:.0f}")
        if r['pf'] < 1.0:
            ok = False; fail.append(f"PF={r['pf']:.2f}")
        if r['win_rate'] < 35:
            ok = False; fail.append(f"WR={r['win_rate']:.0f}%")
        if abs(r['max_dd']) > 8:
            ok = False; fail.append(f"DD={r['max_dd']:.1f}%")

        if ok:
            print(f"  ✅ {name}: PnL=${r['total_pnl']:+,.0f}  "
                  f"WR={r['win_rate']:.0f}%  PF={r['pf']:.2f}  "
                  f"DD={r['max_dd']:.1f}%  Mo={r['monthly_ret']:+.1f}%")
            profitable[name] = (sigs, sym)
        else:
            print(f"  ❌ {name}: {', '.join(fail)}")

    # ═══════════════════════════════════════════════════
    #  COMBINED PORTFOLIO
    # ═══════════════════════════════════════════════════
    combined_r = None
    if profitable:
        print(f"\n{'='*72}")
        print(f"  COMBINED PORTFOLIO ({len(profitable)} strategies)")
        print("="*72)

        tr_c, rk_c = run_combined(df, profitable)
        combined_r  = report(tr_c, rk_c,
                            f"COMBINED ({len(profitable)} strats)")
    else:
        print("\n⚠️  No profitable strategies!")

    # ═══════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════
    print("\n" + "="*72)
    print("  FINAL SUMMARY")
    print("="*72)

    print(f"\n  {'Strategy':<20} {'#':>5} {'WR%':>6} "
          f"{'PF':>6} {'DD%':>7} {'Mo%':>7} "
          f"{'PnL':>10} {'Gr':>3}")
    print("  " + "─"*66)

    for r in all_results:
        if r:
            st = "✅" if r['total_pnl'] > 0 else "❌"
            pfs = f"{r['pf']:.2f}" if r['pf']!=float('inf') else "  ∞"
            print(
                f"  {r['name']:<20} {r['trades']:>5} "
                f"{r['win_rate']:>5.1f}% {pfs:>6} "
                f"{r['max_dd']:>6.2f}% "
                f"{r['monthly_ret']:>+6.2f}% "
                f"${r['total_pnl']:>9.2f} "
                f"{r['prop_grade']:>3} {st}")

    if combined_r:
        r = combined_r
        pfs = f"{r['pf']:.2f}" if r['pf']!=float('inf') else "  ∞"
        print("  " + "─"*66)
        print(
            f"  {'COMBINED':<20} {r['trades']:>5} "
            f"{r['win_rate']:>5.1f}% {pfs:>6} "
            f"{r['max_dd']:>6.2f}% "
            f"{r['monthly_ret']:>+6.2f}% "
            f"${r['total_pnl']:>9.2f} "
            f"{r['prop_grade']:>3} "
            f"{'✅' if r['total_pnl']>0 else '❌'}")

    save_all(all_results, combined_r)

    print("\n" + "="*72)
    print("  ✅ BACKTEST COMPLETE")
    print("="*72)
