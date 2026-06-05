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
    risk_per_trade_pct   = 0.015
    max_daily_loss_pct   = 0.045
    max_total_dd_pct     = 0.09
    spread_eur_pips      = 1.0
    spread_gbp_pips      = 1.2
    commission_per_lot   = 6.0
    pip                  = 0.0001
    lot_size             = 100_000
    max_lot              = 3.0
    atr_period           = 14
    min_rr               = 1.5


# ================================================================== #
#                        DATA LOADER                                 #
# ================================================================== #
def load_data():
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur or not files_gbp:
        raise FileNotFoundError("CSV not found in data/")

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
    print(f"✅ {len(df):,} candles | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


# ================================================================== #
#                      INDICATORS                                    #
# ================================================================== #
def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(close, period=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def calc_adx_full(high, low, close, period=14):
    up = high.diff()
    dn = -low.diff()
    dmp = up.where((up > dn) & (up > 0), 0.0)
    dmn = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_s = tr.rolling(period).sum()
    dip = 100 * dmp.rolling(period).sum() / atr_s.replace(0, np.nan)
    din = 100 * dmn.rolling(period).sum() / atr_s.replace(0, np.nan)
    dx = (abs(dip - din) / (dip + din).replace(0, np.nan)) * 100
    adx = dx.rolling(period).mean()
    return adx, dip, din


def calc_macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    m = ef - es
    s = m.ewm(span=signal, adjust=False).mean()
    return m, s, m - s


def calc_bbands(close, period=20, mult=2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + mult*std, mid, mid - mult*std


def calc_stoch(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    kv = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    return kv, kv.rolling(d).mean()


def ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


# ================================================================== #
#                     UTILITIES                                      #
# ================================================================== #
def trade_cost(lot, symbol):
    sp = Config.spread_eur_pips if symbol == 'EUR' else Config.spread_gbp_pips
    return (sp * Config.pip * lot * Config.lot_size
            + Config.commission_per_lot * lot)


def calc_pnl(direction, lot, entry, exit_p, symbol):
    raw = direction * (exit_p - entry) * lot * Config.lot_size
    return raw - trade_cost(lot, symbol)


def lot_size_calc(equity, sl_pips):
    if sl_pips <= 0:
        return 0.01
    risk = equity * Config.risk_per_trade_pct
    lot = risk / (sl_pips * Config.pip * Config.lot_size)
    return round(np.clip(lot, 0.01, Config.max_lot), 2)


def make_signals(index):
    return pd.DataFrame({
        'signal': 0, 'sl_pips': 0.0, 'tp_pips': 0.0
    }, index=index, dtype=float)


def throttle_signals(sigs, min_gap_hours=3):
    nz = sigs[sigs['signal'] != 0]
    if len(nz) <= 1:
        return sigs
    keep = [nz.index[0]]
    for idx in nz.index[1:]:
        if (idx - keep[-1]).total_seconds() >= min_gap_hours * 3600:
            keep.append(idx)
    drop = [i for i in nz.index if i not in keep]
    sigs.loc[drop, 'signal'] = 0
    return sigs


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

    def new_bar(self, ts):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day = day
            self.day_start_eq = self.equity

    def add_pnl(self, amount, ts):
        self.equity += amount
        self.peak = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))
        self.curve_ts.append(ts)
        dk = str(ts.date())
        self.daily_pnl[dk] = self.daily_pnl.get(dk, 0) + amount
        dd = (self.equity - self.day_start_eq) / self.day_start_eq
        if dd <= -Config.max_daily_loss_pct:
            self.halted = True
            self.halt_reason = f"Daily Loss {dd*100:.1f}%"
            return False
        tdd = (self.equity - self.peak) / self.peak
        if tdd <= -Config.max_total_dd_pct:
            self.halted = True
            self.halt_reason = f"Max DD {tdd*100:.1f}%"
            return False
        return True

    def can_trade(self):
        if self.halted:
            return False
        dd = (self.equity - self.day_start_eq) / self.day_start_eq
        return dd > -Config.max_daily_loss_pct * 0.7

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
        return r.mean() / r.std() * np.sqrt(252*96) if r.std() > 0 else 0

    @property
    def sortino(self):
        r = pd.Series(self.curve).pct_change().dropna()
        n = r[r < 0]
        ds = n.std() if len(n) > 0 else 1e-10
        return r.mean() / ds * np.sqrt(252*96) if ds > 0 else 0

    @property
    def calmar(self):
        ret = self.equity / Config.initial_balance - 1
        dd = abs(self.max_dd / 100)
        return ret / dd if dd > 0 else 0


# ================================================================== #
# ★ STRATEGY 1: Correlation Arbitrage (V2 proven)                   #
#   Z-score EUR/GBP → mean reversion                                #
#   V2 result: WR=75%, PF=4.48                                      #
# ================================================================== #
def strategy_corr_arb(df):
    eurgbp = df['c_eur'] / df['c_gbp']
    period = 96

    mean = eurgbp.rolling(period).mean()
    std  = eurgbp.rolling(period).std()
    z    = (eurgbp - mean) / std.replace(0, np.nan)

    std_ok  = std > std.rolling(period*4).mean() * 0.3
    adx_eur = calc_adx_full(df['h_eur'], df['l_eur'], df['c_eur'], 14)[0]
    adx_ok  = adx_eur < 28
    hour    = pd.Series(df.index.hour, index=df.index)
    time_ok = hour.between(7, 19)

    sigs = make_signals(df.index)
    sigs.loc[(z > 2.0) & std_ok & adx_ok & time_ok, 'signal']  = -1
    sigs.loc[(z < -2.0) & std_ok & adx_ok & time_ok, 'signal'] =  1
    sigs.loc[sigs['signal'] != 0, 'sl_pips'] = 20.0
    sigs.loc[sigs['signal'] != 0, 'tp_pips'] = 35.0

    mask = sigs['signal'] != sigs['signal'].shift()
    sigs.loc[~mask, 'signal'] = 0
    sigs = throttle_signals(sigs, 4)
    return sigs, z


# ================================================================== #
# ★ STRATEGY 2: RSI Divergence (proven PF=3.46, WR=76%)            #
#   Price lower-low + RSI higher-low → reversal                     #
# ================================================================== #
def strategy_rsi_div(df):
    c = df['c_eur']; h = df['h_eur']
    l = df['l_eur']; o = df['o_eur']

    rsi = calc_rsi(c, 14)
    atr = calc_atr(h, l, c, 14)
    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(7, 18)

    lb = 10
    sw_lo = l.rolling(lb*2+1, center=True).min()
    sw_hi = h.rolling(lb*2+1, center=True).max()
    is_lo = (l == sw_lo) & (l < l.shift(1)) & (l < l.shift(-1))
    is_hi = (h == sw_hi) & (h > h.shift(1)) & (h > h.shift(-1))

    lp = l.where(is_lo).ffill(); lr = rsi.where(is_lo).ffill()
    pp = lp.shift(lb);           pr = lr.shift(lb)
    hp = h.where(is_hi).ffill(); hr_ = rsi.where(is_hi).ffill()
    php = hp.shift(lb);          phr = hr_.shift(lb)

    bull = ((lp < pp) & (lr > pr+3) & (rsi < 40)
            & (rsi > rsi.shift(1)) & (c > o) & active)
    bear = ((hp > php) & (hr_ < phr-3) & (rsi > 60)
            & (rsi < rsi.shift(1)) & (c < o) & active)

    sigs = make_signals(df.index)
    for i in range(250, len(df)):
        av = atr.iloc[i]
        if pd.isna(av) or av <= 0:
            continue
        ap = av / Config.pip
        idx = df.index[i]
        if bull.iloc[i]:
            sl = max(12, min(ap*1.2, 25))
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.0
        elif bear.iloc[i]:
            sl = max(12, min(ap*1.2, 25))
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.0

    sigs = throttle_signals(sigs, 4)
    return sigs


# ================================================================== #
# ★ STRATEGY 3: MACD Histogram Reversal + EMA Filter               #
#   Histogram direction change + trend filter                        #
# ================================================================== #
def strategy_macd_reversal(df):
    c = df['c_eur']; h = df['h_eur']; l = df['l_eur']

    _, _, hist = calc_macd(c, 12, 26, 9)
    ema50      = ema(c, 50)
    rsi_v      = calc_rsi(c, 14)
    atr        = calc_atr(h, l, c, 14)
    adx, dip, din = calc_adx_full(h, l, c, 14)

    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(8, 17)

    # Histogram turns up from below
    hist_up = (hist > hist.shift(1)) & (hist.shift(1) <= hist.shift(2))
    hist_dn = (hist < hist.shift(1)) & (hist.shift(1) >= hist.shift(2))

    long_c = (active & hist_up & (c > ema50)
               & (rsi_v > 40) & (rsi_v < 68)
               & (adx > 18) & (dip > din))
    short_c = (active & hist_dn & (c < ema50)
                & (rsi_v > 32) & (rsi_v < 60)
                & (adx > 18) & (din > dip))

    sigs = make_signals(df.index)
    for i in range(200, len(df)):
        av = atr.iloc[i]
        if pd.isna(av) or av <= 0:
            continue
        ap = av / Config.pip
        idx = df.index[i]
        if long_c.iloc[i]:
            sl = max(10, min(ap*1.0, 20))
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.0
        elif short_c.iloc[i]:
            sl = max(10, min(ap*1.0, 20))
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.0

    sigs = throttle_signals(sigs, 3)
    return sigs


# ================================================================== #
# ★ STRATEGY 4: BB Squeeze + Stochastic                            #
#   Low volatility squeeze → breakout direction from Stoch          #
#   FIX: use rolling().quantile(0.25) instead of .percentile(25)    #
# ================================================================== #
def strategy_bb_stoch(df):
    c = df['c_eur']; h = df['h_eur']
    l = df['l_eur']; o = df['o_eur']
    v = df['v_eur']

    bb_up, bb_mid, bb_lo = calc_bbands(c, 20, 2.0)
    bb_width = (bb_up - bb_lo) / bb_mid.replace(0, np.nan) * 100

    # ✅ FIX: use quantile(0.25) instead of percentile(25)
    bb_width_q25 = bb_width.rolling(96).quantile(0.25)
    bb_squeeze   = bb_width < bb_width_q25

    stk, std_d = calc_stoch(h, l, c, 14, 3)
    rsi_v       = calc_rsi(c, 14)
    atr         = calc_atr(h, l, c, 14)

    ema20  = ema(c, 20)
    vol_ma = v.rolling(50).mean()
    vol_ok = v > vol_ma * 1.2

    hour   = pd.Series(df.index.hour, index=df.index)
    active = hour.between(8, 17)

    # Long: squeeze releases upward
    long_c = (active & vol_ok
              & (bb_squeeze.shift(1) | bb_squeeze.shift(2))
              & (c > bb_mid) & (c > ema20)
              & (stk > 50) & (stk < 85) & (stk > std_d)
              & (rsi_v > 45) & (rsi_v < 70))

    # Short: squeeze releases downward
    short_c = (active & vol_ok
               & (bb_squeeze.shift(1) | bb_squeeze.shift(2))
               & (c < bb_mid) & (c < ema20)
               & (stk < 50) & (stk > 15) & (stk < std_d)
               & (rsi_v > 30) & (rsi_v < 55))

    sigs = make_signals(df.index)
    for i in range(200, len(df)):
        av = atr.iloc[i]
        if pd.isna(av) or av <= 0:
            continue
        ap  = av / Config.pip
        idx = df.index[i]
        if long_c.iloc[i]:
            sl = max(10, min(ap*1.0, 22))
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.2
        elif short_c.iloc[i]:
            sl = max(10, min(ap*1.0, 22))
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.2

    sigs = throttle_signals(sigs, 4)
    return sigs


# ================================================================== #
# ★ STRATEGY 5: London Killzone Momentum                            #
#   7AM direction → pullback entry 8-10AM                           #
# ================================================================== #
def strategy_london_killzone(df):
    c = df['c_eur']; h = df['h_eur']; l = df['l_eur']

    ema21 = ema(c, 21)
    ema50 = ema(c, 50)
    rsi_v = calc_rsi(c, 14)
    atr   = calc_atr(h, l, c, 14)
    adx, dip, din = calc_adx_full(h, l, c, 14)

    hour    = pd.Series(df.index.hour, index=df.index)
    date_s  = pd.Series(df.index.date, index=df.index)
    weekday = pd.Series(df.index.weekday, index=df.index)

    # 7AM candle group per day
    mask_7  = (hour == 7)
    h7      = df[mask_7]
    dates_7 = pd.Series(h7.index.date, index=h7.index)

    daily_7 = h7.groupby(dates_7).agg(
        open_7 =('o_eur', 'first'),
        close_7=('c_eur', 'last'),
        high_7 =('h_eur', 'max'),
        low_7  =('l_eur', 'min')
    )
    daily_7['dir_7']   = np.sign(daily_7['close_7'] - daily_7['open_7'])
    daily_7['range_7'] = (daily_7['high_7'] - daily_7['low_7']) / Config.pip

    d = df.copy()
    d['_date'] = d.index.date
    d = d.join(daily_7, on='_date')

    sigs        = make_signals(df.index)
    entry_time  = hour.between(8, 10)
    day_ok      = weekday.between(0, 3)

    for i in range(300, len(df)):
        idx = df.index[i]
        if not entry_time.iloc[i] or not day_ok.iloc[i]:
            continue

        d7  = d['dir_7'].iloc[i]
        r7  = d['range_7'].iloc[i]
        if pd.isna(d7) or d7 == 0 or pd.isna(r7):
            continue
        if r7 < 8 or r7 > 40:
            continue

        av = atr.iloc[i]
        if pd.isna(av) or av <= 0:
            continue
        ap  = av / Config.pip
        adv = adx.iloc[i]
        if pd.isna(adv) or adv < 18:
            continue

        cv  = c.iloc[i]
        e21 = ema21.iloc[i]
        e50 = ema50.iloc[i]
        rv  = rsi_v.iloc[i]
        h7v = d['high_7'].iloc[i]
        l7v = d['low_7'].iloc[i]

        if d7 > 0:   # Bullish 7AM
            if not (e21 > e50): continue
            if not (40 < rv < 65): continue
            if pd.notna(h7v) and cv > h7v: continue
            sl = max(10, min(ap*1.0, 20))
            sigs.at[idx, 'signal']  =  1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.0

        elif d7 < 0:  # Bearish 7AM
            if not (e21 < e50): continue
            if not (35 < rv < 60): continue
            if pd.notna(l7v) and cv < l7v: continue
            sl = max(10, min(ap*1.0, 20))
            sigs.at[idx, 'signal']  = -1
            sigs.at[idx, 'sl_pips'] = sl
            sigs.at[idx, 'tp_pips'] = sl * 2.0

    # One per day
    nz = sigs[sigs['signal'] != 0]
    if len(nz) > 0:
        first = nz.groupby(nz.index.date).head(1).index
        drop  = [i for i in nz.index if i not in first]
        sigs.loc[drop, 'signal'] = 0

    return sigs


# ================================================================== #
#                    BACKTEST ENGINE                                  #
# ================================================================== #
def run_single(df, name, signals, symbol='EUR',
               trailing=True, time_stop_h=48):
    risk   = RiskManager(name)
    trades = []
    pos    = None
    warmup = 300

    s_  = symbol.lower()
    hc  = f'h_{s_}'; lc = f'l_{s_}'; cc = f'c_{s_}'
    atr = calc_atr(df[hc], df[lc], df[cc], 14)

    for i in range(warmup, len(df)):
        ts = df.index[i]
        hi = df[hc].iloc[i]
        lo = df[lc].iloc[i]
        cp = df[cc].iloc[i]

        risk.new_bar(ts)

        # Force-close on halt
        if risk.halted:
            if pos:
                pnl = calc_pnl(pos['dir'], pos['lot'],
                               pos['entry'], cp, symbol)
                trades.append({**pos, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'halt_close'})
                risk.add_pnl(pnl, ts)
                pos = None
            continue

        # ── EXIT ──
        if pos:
            d = pos['dir']; ep = pos['entry']

            # Trailing stop
            if trailing:
                av = atr.iloc[i]
                if pd.notna(av) and av > 0:
                    mv = d * (cp - ep)
                    if mv > av * 1.2:
                        be = ep + d * av * 0.3
                        pos['sl'] = (max(pos['sl'], be) if d == 1
                                     else min(pos['sl'], be))
                    if mv > av * 2.0:
                        lk = ep + d * av * 1.0
                        pos['sl'] = (max(pos['sl'], lk) if d == 1
                                     else min(pos['sl'], lk))

            sl = pos['sl']; tp = pos['tp']
            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            # Time stop
            elapsed = (ts - pos['entry_ts']).total_seconds() / 3600
            if elapsed >= time_stop_h and not hit_tp:
                pnl = calc_pnl(d, pos['lot'], ep, cp, symbol)
                trades.append({**pos, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts); pos = None; continue

            # Weekend close
            if ts.weekday() == 4 and ts.hour >= 20:
                pnl = calc_pnl(d, pos['lot'], ep, cp, symbol)
                trades.append({**pos, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'WeekEnd'})
                risk.add_pnl(pnl, ts); pos = None; continue

            if hit_sl or hit_tp:
                xp  = sl if hit_sl else tp
                xs  = 'SL' if hit_sl else 'TP'
                pnl = calc_pnl(d, pos['lot'], ep, xp, symbol)
                trades.append({**pos, 'exit': xp, 'exit_ts': ts,
                               'pnl': pnl, 'status': xs})
                risk.add_pnl(pnl, ts); pos = None

        # ── ENTRY ──
        if pos is None and risk.can_trade():
            sv = int(signals['signal'].iloc[i])
            if sv != 0:
                slp = signals['sl_pips'].iloc[i]
                tpp = signals['tp_pips'].iloc[i]
                if slp <= 0 or tpp <= 0: continue
                if tpp / slp < Config.min_rr: continue
                lot   = lot_size_calc(risk.equity, slp)
                sp    = (Config.spread_eur_pips if symbol == 'EUR'
                         else Config.spread_gbp_pips)
                entry = cp + sv * sp * Config.pip / 2
                pos   = dict(strategy=name, symbol=symbol,
                             dir=sv, lot=lot, entry=entry,
                             sl=entry - sv*slp*Config.pip,
                             tp=entry + sv*tpp*Config.pip,
                             entry_ts=ts)

    # Close remaining
    if pos:
        cp  = df[cc].iloc[-1]
        pnl = calc_pnl(pos['dir'], pos['lot'],
                        pos['entry'], cp, symbol)
        trades.append({**pos, 'exit': cp,
                       'exit_ts': df.index[-1],
                       'pnl': pnl, 'status': 'eod_close'})
        risk.add_pnl(pnl, df.index[-1])

    return trades, risk


# ================================================================== #
#                    COMBINED ENGINE                                  #
# ================================================================== #
def run_combined(df, strat_dict):
    """strat_dict = {name: (signals_df, symbol)}"""
    risk     = RiskManager("Combined")
    trades   = []
    open_pos = {}
    warmup   = 300
    max_p    = min(len(strat_dict), 5)

    atr_e = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)
    atr_g = calc_atr(df['h_gbp'], df['l_gbp'], df['c_gbp'], 14)

    time_limits = {
        'CorrArb': 72, 'RSI_Div': 48,
        'MACD_Rev': 36, 'BB_Stoch': 36,
        'London_KZ': 24
    }

    for i in range(warmup, len(df)):
        ts = df.index[i]
        risk.new_bar(ts)

        # Halt: force close all
        if risk.halted:
            for k in list(open_pos.keys()):
                p  = open_pos.pop(k)
                s_ = p['symbol'].lower()
                cp = df[f'c_{s_}'].iloc[i]
                pnl = calc_pnl(p['dir'], p['lot'],
                               p['entry'], cp, p['symbol'])
                trades.append({**p, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'halt_close'})
                risk.add_pnl(pnl, ts)
            continue

        # ── EXIT ──
        for k in list(open_pos.keys()):
            p   = open_pos[k]
            s_  = p['symbol'].lower()
            hi  = df[f'h_{s_}'].iloc[i]
            lo  = df[f'l_{s_}'].iloc[i]
            cp  = df[f'c_{s_}'].iloc[i]
            d   = p['dir']

            av = (atr_e if p['symbol'] == 'EUR' else atr_g).iloc[i]
            if pd.notna(av) and av > 0:
                mv = d * (cp - p['entry'])
                if mv > av * 1.2:
                    be = p['entry'] + d * av * 0.3
                    p['sl'] = (max(p['sl'], be) if d == 1
                               else min(p['sl'], be))
                if mv > av * 2.0:
                    lk = p['entry'] + d * av * 1.0
                    p['sl'] = (max(p['sl'], lk) if d == 1
                               else min(p['sl'], lk))

            sl = p['sl']; tp = p['tp']
            hit_sl = (d==1 and lo<=sl) or (d==-1 and hi>=sl)
            hit_tp = (d==1 and hi>=tp) or (d==-1 and lo<=tp)

            elapsed = (ts - p['entry_ts']).total_seconds() / 3600
            if elapsed >= time_limits.get(p['strategy'], 48):
                pnl = calc_pnl(d, p['lot'], p['entry'], cp, p['symbol'])
                trades.append({**p, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'TimeStop'})
                risk.add_pnl(pnl, ts)
                del open_pos[k]; continue

            if ts.weekday() == 4 and ts.hour >= 20:
                pnl = calc_pnl(d, p['lot'], p['entry'], cp, p['symbol'])
                trades.append({**p, 'exit': cp, 'exit_ts': ts,
                               'pnl': pnl, 'status': 'WeekEnd'})
                risk.add_pnl(pnl, ts)
                del open_pos[k]; continue

            if hit_sl or hit_tp:
                xp  = sl if hit_sl else tp
                xs  = 'SL' if hit_sl else 'TP'
                pnl = calc_pnl(d, p['lot'], p['entry'], xp, p['symbol'])
                trades.append({**p, 'exit': xp, 'exit_ts': ts,
                               'pnl': pnl, 'status': xs})
                risk.add_pnl(pnl, ts)
                del open_pos[k]

        # ── ENTRY ──
        if not risk.can_trade() or len(open_pos) >= max_p:
            continue

        for sn, (sg, sym) in strat_dict.items():
            if sn in open_pos or len(open_pos) >= max_p:
                continue
            sv = int(sg['signal'].iloc[i])
            if sv == 0: continue
            slp = sg['sl_pips'].iloc[i]
            tpp = sg['tp_pips'].iloc[i]
            if slp <= 0 or tpp <= 0: continue
            if tpp / slp < Config.min_rr: continue
            s_    = sym.lower()
            cp    = df[f'c_{s_}'].iloc[i]
            lot   = lot_size_calc(risk.equity, slp)
            sp    = (Config.spread_eur_pips if sym == 'EUR'
                     else Config.spread_gbp_pips)
            entry = cp + sv * sp * Config.pip / 2
            open_pos[sn] = dict(
                strategy=sn, symbol=sym, dir=sv, lot=lot,
                entry=entry,
                sl=entry - sv*slp*Config.pip,
                tp=entry + sv*tpp*Config.pip,
                entry_ts=ts)

    # Close remaining
    for k, p in open_pos.items():
        s_  = p['symbol'].lower()
        cp  = df[f'c_{s_}'].iloc[-1]
        pnl = calc_pnl(p['dir'], p['lot'], p['entry'], cp, p['symbol'])
        trades.append({**p, 'exit': cp, 'exit_ts': df.index[-1],
                       'pnl': pnl, 'status': 'eod_close'})
        risk.add_pnl(pnl, df.index[-1])

    return trades, risk


# ================================================================== #
#                    REPORT                                          #
# ================================================================== #
def report(trades, risk, title=""):
    if not trades:
        print(f"\n❌ [{title}] No trades!")
        return None

    t = pd.DataFrame(trades)
    t['pnl']      = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts'] = pd.to_datetime(t['entry_ts'])
    t['exit_ts']  = pd.to_datetime(t['exit_ts'])
    t['dur']      = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    sd = t['entry_ts'].min(); ed = t['exit_ts'].max()
    td = max((ed - sd).days, 1)
    tm = td / 30.44

    fe   = risk.equity
    tp_  = fe - Config.initial_balance
    tr_  = tp_ / Config.initial_balance * 100
    ar   = (((fe/Config.initial_balance)**(365.25/td)-1)*100
            if td > 1 else 0)
    mr   = tr_ / tm if tm > 0 else 0

    wt   = t[t['pnl'] > 0]; lt_ = t[t['pnl'] < 0]
    wr   = len(wt)/len(t)*100 if len(t) > 0 else 0
    aw   = wt['pnl'].mean()  if len(wt)  > 0 else 0
    al   = lt_['pnl'].mean() if len(lt_) > 0 else 0
    gw   = wt['pnl'].sum();   gl = abs(lt_['pnl'].sum())
    pf   = gw/gl if gl > 0 else float('inf')
    ex   = t['pnl'].mean()
    rr   = abs(aw/al) if al != 0 else 0

    sgn = t['pnl'].apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))
    cw=cl=mcw=mcl=0
    for s in sgn:
        if   s > 0: cw+=1; cl=0; mcw=max(mcw,cw)
        elif s < 0: cl+=1; cw=0; mcl=max(mcl,cl)
        else:       cw=cl=0

    W   = 72
    SEP = "═"*W

    def rw(lb, v):
        lb2 = f"  {lb}"; vs = str(v)
        d   = "·"*max(2, W-len(lb2)-len(vs)-2)
        return f"{lb2} {d} {vs}"

    def bt(t_):
        inner = f"─ {t_} "
        return "┌" + inner + "─"*(W-len(inner)-1) + "┐"

    bb = lambda: "└"+"─"*(W-1)+"┘"

    # Prop score
    ps = 0
    if wr >= 40:  ps += 20
    if pf >= 1.3: ps += 20
    if   abs(risk.max_dd) <= 5: ps += 25
    elif abs(risk.max_dd) <= 8: ps += 15
    if   mr >= 10: ps += 25
    elif mr >= 5:  ps += 15
    if rr >= 1.5:  ps += 10

    pg = ("A+" if ps>=90 else "A" if ps>=80 else "B+" if ps>=70
          else "B" if ps>=60 else "C" if ps>=50
          else "D" if ps>=40 else "F")

    lines = [
        "", SEP, f"  ▌  {title}  ▐", SEP, "",
        bt("General"),
        rw("Period",       f"{sd.date()} → {ed.date()}"),
        rw("Days",         f"{td:,}"),
        rw("Trades",       f"{len(t):,}"),
        rw("Trades/Week",  f"{len(t)/(td/7):.1f}"),
        bb(), "",
        bt("Financial"),
        rw("Initial",      f"${Config.initial_balance:,.2f}"),
        rw("Final",        f"${fe:,.2f}"),
        rw("PnL",          f"${tp_:+,.2f}"),
        rw("Return",       f"{tr_:+.2f}%"),
        rw("Monthly",      f"{mr:+.2f}%"),
        rw("Annualized",   f"{ar:+.2f}%"),
        rw("Best",         f"${t['pnl'].max():+.2f}"),
        rw("Worst",        f"${t['pnl'].min():+.2f}"),
        bb(), "",
        bt("Risk"),
        rw("Max DD%",      f"{risk.max_dd:.2f}%"),
        rw("Max DD$",      f"${risk.max_dd_abs:+.2f}"),
        rw("Sharpe",       f"{risk.sharpe:.2f}"),
        rw("Sortino",      f"{risk.sortino:.2f}"),
        rw("Calmar",       f"{risk.calmar:.2f}"),
        rw("PF",           f"{pf:.2f}"),
        rw("Status",       risk.halt_reason),
        bb(), "",
        bt("Statistics"),
        rw("Win Rate",     f"{wr:.1f}%"),
        rw("Winners",      f"{len(wt):,}"),
        rw("Losers",       f"{len(lt_):,}"),
        rw("Avg Win",      f"${aw:+.2f}"),
        rw("Avg Loss",     f"${al:+.2f}"),
        rw("R:R",          f"{rr:.2f}"),
        rw("Expectancy",   f"${ex:+.2f}"),
        rw("Max ConsecW",  f"{mcw}"),
        rw("Max ConsecL",  f"{mcl}"),
        rw("Avg Dur",      f"{t['dur'].mean():.0f} min"),
        bb(), "",
        bt("Prop Fitness"),
        rw("Score",        f"{ps}/100"),
        rw("Grade",        pg),
        rw("DD<5%",        "✅" if abs(risk.max_dd)<=5 else "❌"),
        rw("DD<10%",       "✅" if abs(risk.max_dd)<=10 else "❌"),
        rw("Mo>10%",       "✅" if mr>=10 else "❌"),
        rw("PF>1.3",       "✅" if pf>=1.3 else "❌"),
        rw("WR>40%",       "✅" if wr>=40 else "❌"),
        bb(), "",
    ]

    # Exits
    lines.append(bt("Exits"))
    for st, cnt in t['status'].value_counts().items():
        p_  = cnt/len(t)*100
        a_  = t.loc[t['status']==st, 'pnl'].mean()
        bar = "█"*max(1, int(p_/2.5))
        lines.append(f"  {st:<13} {cnt:>5} ({p_:>5.1f}%)  "
                     f"{bar:<28}  avg=${a_:>+.2f}")
    lines.append(bb())

    # Monthly
    t['ym'] = t['entry_ts'].dt.to_period('M')
    mo = (t.groupby('ym')
          .agg(n=('pnl','count'), pnl=('pnl','sum'),
               wins=('pnl', lambda x:(x>0).sum()))
          .reset_index())
    mo['wr']  = mo['wins']/mo['n']*100
    mo['ret'] = mo['pnl']/Config.initial_balance*100
    mo['cum'] = mo['pnl'].cumsum()
    mo['cr']  = mo['cum']/Config.initial_balance*100
    pm = (mo['pnl'] >= 0).sum()

    lines += ["", bt("Monthly")]
    lines.append(f"  {'Mo':>7} {'#':>4} {'WR':>5} "
                 f"{'PnL':>10} {'Ret%':>6} "
                 f"{'CumPnL':>10} {'CumR%':>7}")
    lines.append("  "+"─"*(W-3))
    for _, r in mo.iterrows():
        a = "▲" if r['pnl'] >= 0 else "▼"
        lines.append(
            f"  {str(r['ym']):>7} {int(r['n']):>4} "
            f"{r['wr']:>4.0f}% ${r['pnl']:>9.2f} "
            f"{r['ret']:>+5.1f}% ${r['cum']:>9.2f} "
            f"{r['cr']:>+6.1f}% {a}")
    lines.append("  "+"─"*(W-3))
    lines.append(f"  Profitable: {pm}/{len(mo)} "
                 f"({pm/len(mo)*100:.0f}%)")
    lines.append(bb())

    # Yearly
    t['yr'] = t['entry_ts'].dt.year
    yr = (t.groupby('yr')
          .agg(n=('pnl','count'), pnl=('pnl','sum'),
               wins=('pnl', lambda x:(x>0).sum()))
          .reset_index())
    yr['wr']  = yr['wins']/yr['n']*100
    yr['ret'] = yr['pnl']/Config.initial_balance*100

    lines += ["", bt("Yearly")]
    lines.append(f"  {'Year':>5} {'#':>5} {'WR':>5} "
                 f"{'PnL':>10} {'Ret%':>7}")
    lines.append("  "+"─"*(W-3))
    for _, r in yr.iterrows():
        lines.append(
            f"  {int(r['yr']):>5} {int(r['n']):>5} "
            f"{r['wr']:>4.0f}% ${r['pnl']:>9.2f} "
            f"{r['ret']:>+6.1f}%")
    lines += [bb(), "", SEP]

    output = "\n".join(lines)
    print(output)

    return {
        'name': title, 'trades': len(t),
        'total_pnl': tp_, 'total_ret': tr_,
        'monthly_ret': mr, 'win_rate': wr,
        'pf': pf, 'rr': rr, 'max_dd': risk.max_dd,
        'sharpe': risk.sharpe, 'sortino': risk.sortino,
        'calmar': risk.calmar, 'exp': ex,
        'prop_score': ps, 'prop_grade': pg,
        'pos_months': pm, 'tot_months': len(mo),
        'output': output, 'risk': risk,
        'trades_df': t, 'monthly_df': mo,
    }


# ================================================================== #
#                    SAVE                                             #
# ================================================================== #
def save_all(results, combined=None):
    with open("Backtest_Report.txt", "w", encoding="utf-8") as f:
        f.write("="*72+"\n")
        f.write("  PROFESSIONAL PROP TRADING BACKTEST v4\n")
        f.write(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*72+"\n\n")
        for r in results:
            if r: f.write(r['output']+"\n\n")
        if combined:
            f.write("\n"+"="*72+"\n")
            f.write("  COMBINED PORTFOLIO\n"+"="*72+"\n")
            f.write(combined['output']+"\n")
        f.write("\n\n"+"="*72+"\n  COMPARISON\n"+"="*72+"\n\n")
        f.write(f"  {'Strategy':<18}{'#':>5}{'WR%':>7}"
                f"{'PF':>7}{'RR':>6}{'DD%':>8}"
                f"{'Mo%':>8}{'PnL':>11}{'Sc':>5}\n")
        f.write("  "+"─"*68+"\n")
        for r in results:
            if r:
                pfs = f"{r['pf']:.2f}" if r['pf']!=float('inf') else "∞"
                f.write(
                    f"  {r['name']:<18}{r['trades']:>5}"
                    f"{r['win_rate']:>6.1f}%{pfs:>7}"
                    f"{r['rr']:>6.2f}{r['max_dd']:>7.2f}%"
                    f"{r['monthly_ret']:>+7.2f}%"
                    f"${r['total_pnl']:>10.2f}"
                    f"{r['prop_score']:>4}\n")
        if combined:
            r=combined
            pfs=f"{r['pf']:.2f}" if r['pf']!=float('inf') else "∞"
            f.write("  "+"─"*68+"\n")
            f.write(
                f"  {'COMBINED':<18}{r['trades']:>5}"
                f"{r['win_rate']:>6.1f}%{pfs:>7}"
                f"{r['rr']:>6.2f}{r['max_dd']:>7.2f}%"
                f"{r['monthly_ret']:>+7.2f}%"
                f"${r['total_pnl']:>10.2f}"
                f"{r['prop_score']:>4}\n")

    rows = [["Strategy","Trades","WR%","PF","RR","MaxDD%",
             "Mo%","PnL","Score","Grade"]]
    for r in results:
        if r:
            rows.append([r['name'], r['trades'],
                         round(r['win_rate'],1), round(r['pf'],2),
                         round(r['rr'],2), round(r['max_dd'],2),
                         round(r['monthly_ret'],2),
                         round(r['total_pnl'],2),
                         r['prop_score'], r['prop_grade']])
    if combined:
        r=combined
        rows.append(["COMBINED", r['trades'],
                     round(r['win_rate'],1), round(r['pf'],2),
                     round(r['rr'],2), round(r['max_dd'],2),
                     round(r['monthly_ret'],2),
                     round(r['total_pnl'],2),
                     r['prop_score'], r['prop_grade']])
    pd.DataFrame(rows).to_csv("Backtest_Summary.csv",
                              index=False, header=False,
                              encoding="utf-8-sig")

    for r in results:
        if r and r['risk']:
            eq = pd.DataFrame({'ts': r['risk'].curve_ts,
                               'equity': r['risk'].curve})
            eq['dd'] = ((eq['equity']-eq['equity'].cummax())
                        /eq['equity'].cummax()*100).round(4)
            nm = (r['name'].replace(' ','_')
                  .replace(':','').replace('/',''))
            eq.to_csv(f"equity_{nm}.csv",
                      index=False, encoding="utf-8-sig")
    if combined and combined['risk']:
        eq = pd.DataFrame({'ts': combined['risk'].curve_ts,
                           'equity': combined['risk'].curve})
        eq['dd'] = ((eq['equity']-eq['equity'].cummax())
                    /eq['equity'].cummax()*100).round(4)
        eq.to_csv("equity_Combined.csv",
                  index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved: Backtest_Report.txt | Backtest_Summary.csv")
    for r in results:
        if r:
            nm = (r['name'].replace(' ','_')
                  .replace(':','').replace('/',''))
            print(f"   equity_{nm}.csv")
    if combined:
        print(f"   equity_Combined.csv")


# ================================================================== #
#                         MAIN                                       #
# ================================================================== #
if __name__ == "__main__":
    print("="*72)
    print("  PROP TRADING BACKTEST v4 — 5 Strategies")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*72)

    df = load_data()

    # BUILD SIGNALS
    print("\n⚙️  Building signals...")
    print("  1. Correlation Arbitrage (V2 proven)...")
    sig1, z = strategy_corr_arb(df)
    print(f"     → {(sig1['signal']!=0).sum()} signals")

    print("  2. RSI Divergence (proven PF=3.46)...")
    sig2 = strategy_rsi_div(df)
    print(f"     → {(sig2['signal']!=0).sum()} signals")

    print("  3. MACD Histogram Reversal...")
    sig3 = strategy_macd_reversal(df)
    print(f"     → {(sig3['signal']!=0).sum()} signals")

    print("  4. BB Squeeze + Stochastic...")
    sig4 = strategy_bb_stoch(df)
    print(f"     → {(sig4['signal']!=0).sum()} signals")

    print("  5. London Killzone Momentum...")
    sig5 = strategy_london_killzone(df)
    print(f"     → {(sig5['signal']!=0).sum()} signals")

    # INDIVIDUAL
    print("\n"+"="*72)
    print("  INDIVIDUAL BACKTESTS")
    print("="*72)

    configs = [
        ("CorrArb",    sig1, 'EUR', True, 72),
        ("RSI_Div",    sig2, 'EUR', True, 48),
        ("MACD_Rev",   sig3, 'EUR', True, 36),
        ("BB_Stoch",   sig4, 'EUR', True, 36),
        ("London_KZ",  sig5, 'EUR', True, 24),
    ]

    all_res   = []
    strat_map = {}

    for name, sigs, sym, trail, tsh in configs:
        ns = (sigs['signal']!=0).sum()
        if ns == 0:
            print(f"\n⚠️  {name}: 0 signals, skipped")
            all_res.append(None); continue
        print(f"\n🔄 {name} ({ns} signals)...")
        tr, rk = run_single(df, name, sigs, sym, trail, tsh)
        r = report(tr, rk, name)
        all_res.append(r)
        if r: strat_map[name] = (sigs, sym, r)

    # FILTER
    print("\n"+"="*72)
    print("  STRATEGY SELECTION")
    print("="*72)

    profitable = {}
    for name, (sigs, sym, r) in strat_map.items():
        ok = True; fail = []
        if r['total_pnl'] <= 0:
            ok=False; fail.append(f"Loss ${r['total_pnl']:.0f}")
        if r['pf'] < 1.0:
            ok=False; fail.append(f"PF={r['pf']:.2f}")
        if r['win_rate'] < 35:
            ok=False; fail.append(f"WR={r['win_rate']:.0f}%")
        if abs(r['max_dd']) > 8:
            ok=False; fail.append(f"DD={r['max_dd']:.1f}%")
        if ok:
            print(f"  ✅ {name}: "
                  f"PnL=${r['total_pnl']:+,.0f}  "
                  f"WR={r['win_rate']:.0f}%  "
                  f"PF={r['pf']:.2f}  "
                  f"DD={r['max_dd']:.1f}%  "
                  f"Mo={r['monthly_ret']:+.1f}%")
            profitable[name] = (sigs, sym)
        else:
            print(f"  ❌ {name}: {', '.join(fail)}")

    # COMBINED
    combined_r = None
    if profitable:
        print(f"\n{'='*72}")
        print(f"  COMBINED ({len(profitable)} profitable strategies)")
        print("="*72)
        tr_c, rk_c = run_combined(df, profitable)
        combined_r  = report(tr_c, rk_c,
                            f"COMBINED ({len(profitable)} strats)")

    # SUMMARY
    print("\n"+"="*72)
    print("  FINAL SUMMARY")
    print("="*72)
    print(f"\n  {'Strategy':<18}{'#':>5}{'WR%':>7}"
          f"{'PF':>7}{'DD%':>8}{'Mo%':>8}{'PnL':>11}{'Gr':>4}")
    print("  "+"─"*68)
    for r in all_res:
        if r:
            st  = "✅" if r['total_pnl']>0 else "❌"
            pfs = f"{r['pf']:.2f}" if r['pf']!=float('inf') else "∞"
            print(f"  {r['name']:<18}{r['trades']:>5}"
                  f"{r['win_rate']:>6.1f}%{pfs:>7}"
                  f"{r['max_dd']:>7.2f}%"
                  f"{r['monthly_ret']:>+7.2f}%"
                  f"${r['total_pnl']:>10.2f}"
                  f"{r['prop_grade']:>4} {st}")
    if combined_r:
        r   = combined_r
        pfs = f"{r['pf']:.2f}" if r['pf']!=float('inf') else "∞"
        print("  "+"─"*68)
        print(f"  {'COMBINED':<18}{r['trades']:>5}"
              f"{r['win_rate']:>6.1f}%{pfs:>7}"
              f"{r['max_dd']:>7.2f}%"
              f"{r['monthly_ret']:>+7.2f}%"
              f"${r['total_pnl']:>10.2f}"
              f"{r['prop_grade']:>4} "
              f"{'✅' if r['total_pnl']>0 else '❌'}")

    save_all(all_res, combined_r)
    print("\n"+"="*72)
    print("  ✅ BACKTEST COMPLETE v4")
    print("="*72)
