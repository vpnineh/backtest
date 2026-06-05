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
    initial_balance     = 5_000.0
    risk_per_trade_pct  = 0.005
    max_daily_loss_pct  = 0.04
    max_total_dd_pct    = 0.08
    profit_target_pct   = 0.10
    spread_eur_pips     = 1.2
    spread_gbp_pips     = 1.5
    commission_per_lot  = 7.0
    pip                 = 0.0001
    lot_size            = 100_000


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
        'o_gbp': raw['o_gbp'].resample('15min').first(),
        'h_gbp': raw['h_gbp'].resample('15min').max(),
        'l_gbp': raw['l_gbp'].resample('15min').min(),
        'c_gbp': raw['c_gbp'].resample('15min').last(),
    }).dropna()

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
    return round(np.clip(lot, 0.01, 1.0), 2)


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

    def new_bar(self, ts: pd.Timestamp):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day      = day
            self.day_start_eq = self.equity
            if "Daily" in self.halt_reason:
                self.halted      = False
                self.halt_reason = "در حال اجرا"

    def add_pnl(self, amount: float, ts: pd.Timestamp) -> bool:
        self.equity += amount
        self.peak    = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))
        self.curve_ts.append(ts)

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
        return ((s - s.cummax()) / s.cummax()).min() * 100

    @property
    def sharpe(self):
        r = pd.Series(self.curve).pct_change().dropna()
        return (r.mean() / r.std() * np.sqrt(252 * 96)) if r.std() > 0 else 0


# ================================================================== #
#          Strategy A: Correlation Pair Trading (نگه داشته شد)      #
# ================================================================== #
def build_corr_arb_signals(df):
    eurgbp = df['c_eur'] / df['c_gbp']
    period = 96
    mean   = eurgbp.rolling(period).mean()
    std    = eurgbp.rolling(period).std()
    z      = (eurgbp - mean) / std.replace(0, np.nan)
    std_filter = std > std.rolling(period * 5).mean() * 0.5

    sig = pd.Series(0, index=df.index)
    sig[(z >  2.0) & std_filter] = -1
    sig[(z < -2.0) & std_filter] =  1
    sig = sig.where(sig != sig.shift(), 0)
    return sig, z


# ================================================================== #
#     Strategy B: Asian Range Breakout (جایگزین London Breakout)    #
#                                                                    #
#  منطق:                                                             #
#  Range ساعت ۰۰:۰۰ تا ۰۷:۰۰ GMT (سشن آسیا) را بساز               #
#  شکست در لندن → ورود با تایید volume spike                        #
#  فیلتر روز: دوشنبه تا پنجشنبه (جمعه حذف)                         #
#  فیلتر: Range حداقل ۱۵ پیپ و حداکثر ۶۰ پیپ                      #
# ================================================================== #
def build_asian_breakout_signals(df):
    d = df.copy()
    d['hour']    = d.index.hour
    d['weekday'] = d.index.weekday   # 0=Mon ... 4=Fri
    d['date']    = d.index.date

    atr = calc_atr(d['h_eur'], d['l_eur'], d['c_eur'], 14)

    # Asian session range
    asian = d[d['hour'].between(0, 6)]
    rng   = asian.groupby('date').agg(
        ah=('h_eur', 'max'),
        al=('l_eur', 'min')
    )
    rng['rng_pips'] = (rng['ah'] - rng['al']) / Config.pip

    d = d.join(rng, on='date')

    valid = (
        d['rng_pips'].between(15, 60) &
        d['weekday'].between(0, 3)          # دوشنبه تا پنجشنبه
    )

    sig = pd.Series(0, index=d.index)
    london = d['hour'].between(7, 12)

    sig[london & valid & (d['c_eur'] > d['ah']) & (d['o_eur'] <= d['ah'])] =  1
    sig[london & valid & (d['c_eur'] < d['al']) & (d['o_eur'] >= d['al'])] = -1

    # فقط اولین سیگنال هر روز
    first_idx = sig[sig != 0].groupby(sig[sig != 0].index.date).head(1).index
    final     = pd.Series(0, index=d.index)
    final[first_idx] = sig[first_idx]

    return final, d['ah'], d['al'], atr


# ================================================================== #
#     Strategy C: EMA Trend Pullback (بهینه‌شده)                    #
# ================================================================== #
def build_trend_pullback_signals(df):
    c      = df['c_eur']
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rsi    = calc_rsi(c, 14)
    atr    = calc_atr(df['h_eur'], df['l_eur'], c, 14)

    dist = (c - ema50) / atr.replace(0, np.nan)

    sig = pd.Series(0, index=df.index)

    sig[
        (ema50 > ema200) &
        dist.between(-0.8, 0.2) &
        rsi.between(42, 58)
    ] = 1

    sig[
        (ema50 < ema200) &
        dist.between(-0.2, 0.8) &
        rsi.between(42, 58)
    ] = -1

    sig = sig.where(sig != sig.shift(), 0)
    return sig, ema50, atr


# ================================================================== #
#     Strategy D: NEW - London/NY Overlap Momentum                   #
#                                                                    #
#  منطق:                                                             #
#  ساعت ۱۳:۰۰ تا ۱۶:۰۰ GMT (تداخل لندن و نیویورک)                  #
#  بیشترین نقدینگی و ترند قوی                                        #
#  سیگنال: EMA20 > EMA50 + قیمت بالای هر دو + RSI بین ۵۰-۷۰        #
#  SL: زیر کمترین ۳ کندل اخیر                                       #
# ================================================================== #
def build_overlap_momentum_signals(df):
    c     = df['c_eur']
    h     = df['h_eur']
    l     = df['l_eur']
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    rsi   = calc_rsi(c, 14)
    atr   = calc_atr(h, l, c, 14)

    hour  = df.index.hour

    # کمترین/بیشترین ۳ کندل اخیر برای SL داینامیک
    recent_low  = l.rolling(3).min()
    recent_high = h.rolling(3).max()

    sig = pd.Series(0, index=df.index)
    overlap = pd.Series(hour, index=df.index).between(13, 16)

    # Long momentum
    sig[
        overlap &
        (ema20 > ema50) &
        (c > ema20) &
        rsi.between(52, 72) &
        (c > c.shift(1)) &          # کندل صعودی
        (atr > atr.rolling(20).mean() * 0.8)    # بازار فعال
    ] = 1

    # Short momentum
    sig[
        overlap &
        (ema20 < ema50) &
        (c < ema20) &
        rsi.between(28, 48) &
        (c < c.shift(1)) &
        (atr > atr.rolling(20).mean() * 0.8)
    ] = -1

    sig = sig.where(sig != sig.shift(), 0)
    return sig, recent_low, recent_high, atr


# ================================================================== #
#                       موتور Backtest                               #
# ================================================================== #
def run_backtest(df: pd.DataFrame):
    print("⚙️  محاسبه اندیکاتورها...")

    sig_arb,  z_score             = build_corr_arb_signals(df)
    sig_ab,   ah, al, atr_ab      = build_asian_breakout_signals(df)
    sig_tp,   ema50, atr_tp       = build_trend_pullback_signals(df)
    sig_om,   r_low, r_high, atr_om = build_overlap_momentum_signals(df)

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

        # ================================================================
        #  مرحله ۱: بررسی SL/TP پوزیشن‌های باز
        # ================================================================
        for key in list(open_pos.keys()):
            p   = open_pos[key]
            sym = p['symbol']

            if sym == 'EUR':
                hi, lo, cp = h_eur, l_eur, c_eur
            else:
                hi, lo, cp = row['h_gbp'], row['l_gbp'], c_gbp

            d, entry, sl, tp = p['dir'], p['entry'], p['sl'], p['tp']

            hit_sl = (d ==  1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d ==  1 and hi >= tp) or (d == -1 and lo <= tp)

            # Trailing برای Momentum
            if p['strategy'] == 'OverlapMom':
                if d == 1 and cp > entry + atr * 0.8:
                    p['sl'] = max(p['sl'], entry + atr * 0.3)
                elif d == -1 and cp < entry - atr * 0.8:
                    p['sl'] = min(p['sl'], entry - atr * 0.3)

            # خروج CorrArb با Z
            if p['strategy'] == 'CorrArb':
                z_now = z_score.iloc[i]
                if pd.notna(z_now) and abs(z_now) < 0.4:
                    hit_tp = True

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
                    **p,
                    'exit':     exit_price,
                    'exit_ts':  ts,
                    'pnl':      p_pnl,
                    'status':   exit_reason
                })
                risk.add_pnl(p_pnl, ts)
                del open_pos[key]

        # ================================================================
        #  مرحله ۲: ورود به پوزیشن جدید
        # ================================================================

        # ---- A: Correlation Arb ----
        if 'CorrArb' not in open_pos and sig_arb.iloc[i] != 0:
            sig     = int(sig_arb.iloc[i])
            sl_pips = 22
            tp_pips = 33
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig * Config.spread_eur_pips * Config.pip / 2
            open_pos['CorrArb'] = {
                'strategy': 'CorrArb', 'symbol': 'EUR',
                'dir': sig, 'lot': lot, 'entry': ep,
                'sl': ep - sig * sl_pips * Config.pip,
                'tp': ep + sig * tp_pips * Config.pip,
                'entry_ts': ts,
            }

        # ---- B: Asian Breakout ----
        if 'AsianBreak' not in open_pos and sig_ab.iloc[i] != 0:
            sig     = int(sig_ab.iloc[i])
            rng_sz  = (ah.iloc[i] - al.iloc[i]) / Config.pip
            sl_pips = max(15, rng_sz * 0.5)
            tp_pips = sl_pips * 2.2
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig * Config.spread_eur_pips * Config.pip / 2
            open_pos['AsianBreak'] = {
                'strategy': 'AsianBreak', 'symbol': 'EUR',
                'dir': sig, 'lot': lot, 'entry': ep,
                'sl': ep - sig * sl_pips * Config.pip,
                'tp': ep + sig * tp_pips * Config.pip,
                'entry_ts': ts,
            }

        # ---- C: Trend Pullback ----
        if 'TrendPB' not in open_pos and sig_tp.iloc[i] != 0:
            sig     = int(sig_tp.iloc[i])
            sl_pips = max(18, atr / Config.pip * 1.3)
            tp_pips = sl_pips * 2.0
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig * Config.spread_eur_pips * Config.pip / 2
            open_pos['TrendPB'] = {
                'strategy': 'TrendPB', 'symbol': 'EUR',
                'dir': sig, 'lot': lot, 'entry': ep,
                'sl': ep - sig * sl_pips * Config.pip,
                'tp': ep + sig * tp_pips * Config.pip,
                'entry_ts': ts,
            }

        # ---- D: Overlap Momentum ----
        if 'OverlapMom' not in open_pos and sig_om.iloc[i] != 0:
            sig     = int(sig_om.iloc[i])
            if sig == 1:
                sl_price = r_low.iloc[i] - 3 * Config.pip
            else:
                sl_price = r_high.iloc[i] + 3 * Config.pip
            sl_pips = abs(c_eur - sl_price) / Config.pip
            sl_pips = max(12, sl_pips)
            tp_pips = sl_pips * 2.5
            lot     = lot_size_calc(risk.equity, sl_pips)
            ep      = c_eur + sig * Config.spread_eur_pips * Config.pip / 2
            open_pos['OverlapMom'] = {
                'strategy': 'OverlapMom', 'symbol': 'EUR',
                'dir': sig, 'lot': lot, 'entry': ep,
                'sl': ep - sig * sl_pips * Config.pip,
                'tp': ep + sig * tp_pips * Config.pip,
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
#                گزارش جامع + ذخیره فایل                            #
# ================================================================== #
def report(trades: list, risk: RiskManager, df: pd.DataFrame):
    if not trades:
        print("❌ هیچ معامله‌ای ثبت نشد!")
        return

    t = pd.DataFrame(trades)
    t['pnl']      = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts'] = pd.to_datetime(t['entry_ts'])
    t['exit_ts']  = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    # ---- بازه زمانی واقعی ----
    start_date = t['entry_ts'].min()
    end_date   = t['exit_ts'].max()
    total_days = (end_date - start_date).days
    total_weeks  = max(total_days / 7, 1)
    total_months = max(total_days / 30.44, 1)
    total_years  = max(total_days / 365.25, 1)

    # ---- آمار کلی ----
    final_eq    = risk.equity
    total_ret   = (final_eq - Config.initial_balance) / Config.initial_balance * 100
    win_trades  = t[t['pnl'] > 0]
    loss_trades = t[t['pnl'] < 0]
    win_rate    = len(win_trades) / len(t) * 100
    avg_win     = win_trades['pnl'].mean()  if len(win_trades)  > 0 else 0
    avg_loss    = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
    gross_win   = win_trades['pnl'].sum()
    gross_loss  = abs(loss_trades['pnl'].sum())
    pf          = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy  = t['pnl'].mean()
    rr_actual   = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # ---- آمار زمانی ----
    trades_per_day   = len(t) / total_days   if total_days   > 0 else 0
    trades_per_week  = len(t) / total_weeks
    trades_per_month = len(t) / total_months

    profit_per_day   = t['pnl'].sum() / total_days   if total_days   > 0 else 0
    profit_per_week  = t['pnl'].sum() / total_weeks
    profit_per_month = t['pnl'].sum() / total_months
    profit_per_year  = t['pnl'].sum() / total_years

    avg_duration = t['duration_min'].mean()

    # ---- گزارش ماهانه ----
    t['year_month'] = t['entry_ts'].dt.to_period('M')
    monthly = t.groupby('year_month').agg(
        trades    = ('pnl', 'count'),
        pnl       = ('pnl', 'sum'),
        win_count = ('pnl', lambda x: (x > 0).sum()),
    ).reset_index()
    monthly['win_rate'] = monthly['win_count'] / monthly['trades'] * 100
    monthly['cum_pnl']  = monthly['pnl'].cumsum()

    # ---- گزارش سالانه ----
    t['year'] = t['entry_ts'].dt.year
    yearly = t.groupby('year').agg(
        trades    = ('pnl', 'count'),
        pnl       = ('pnl', 'sum'),
        win_count = ('pnl', lambda x: (x > 0).sum()),
    ).reset_index()
    yearly['win_rate']  = yearly['win_count'] / yearly['trades'] * 100
    yearly['return_pct'] = yearly['pnl'] / Config.initial_balance * 100

    # ================================================================
    #  ساخت گزارش CSV جامع
    # ================================================================
    lines = []
    SEP   = "=" * 65

    lines += [
        SEP,
        "         BACKTEST REPORT  |  Prop Trading System",
        f"         تاریخ اجرا: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        SEP,
        "",
        "[ اطلاعات کلی ]",
        f"  دوره بک‌تست        : {start_date.date()} → {end_date.date()}",
        f"  تعداد روزها        : {total_days:,}",
        f"  جفت‌ارزها          : EURUSD + GBPUSD",
        f"  تایم‌فریم          : 15 دقیقه",
        "",
        "[ نتایج مالی ]",
        f"  موجودی اولیه      : ${Config.initial_balance:>10,.2f}",
        f"  موجودی نهایی      : ${final_eq:>10,.2f}",
        f"  سود/زیان کل       : ${final_eq - Config.initial_balance:>+10,.2f}",
        f"  بازده کل          : {total_ret:>+10.2f}%",
        "",
        "[ معیارهای ریسک ]",
        f"  Max Drawdown       : {risk.max_dd:>10.2f}%",
        f"  Sharpe Ratio       : {risk.sharpe:>10.2f}",
        f"  Profit Factor      : {pf:>10.2f}",
        f"  توقف به دلیل       : {risk.halt_reason}",
        "",
        "[ آمار معاملات ]",
        f"  تعداد کل معاملات  : {len(t):>10,}",
        f"  Win Rate           : {win_rate:>10.1f}%",
        f"  Avg Win            : ${avg_win:>10.2f}",
        f"  Avg Loss           : ${avg_loss:>10.2f}",
        f"  RR واقعی           : {rr_actual:>10.2f}",
        f"  Expectancy/trade   : ${expectancy:>10.2f}",
        f"  میانگین مدت معامله : {avg_duration:>10.1f} دقیقه",
        "",
        "[ آمار زمانی ]",
        f"  معاملات در روز    : {trades_per_day:>10.2f}",
        f"  معاملات در هفته   : {trades_per_week:>10.2f}",
        f"  معاملات در ماه    : {trades_per_month:>10.2f}",
        f"  سود روزانه (میانگین): ${profit_per_day:>8.2f}",
        f"  سود هفتگی (میانگین): ${profit_per_week:>8.2f}",
        f"  سود ماهانه (میانگین): ${profit_per_month:>8.2f}",
        f"  سود سالانه (میانگین): ${profit_per_year:>8.2f}",
        "",
    ]

    # ---- عملکرد هر استراتژی ----
    lines.append("[ عملکرد هر استراتژی ]")
    lines.append(f"  {'استراتژی':<14} {'#':>4} {'Win%':>6} {'PF':>6} "
                 f"{'AvgW':>7} {'AvgL':>8} {'PnL':>10} {'وضعیت':>6}")
    lines.append("  " + "-" * 62)

    for name in t['strategy'].unique():
        sub  = t[t['strategy'] == name]
        wr   = (sub['pnl'] > 0).mean() * 100
        gw   = sub.loc[sub['pnl'] > 0, 'pnl'].sum()
        gl   = abs(sub.loc[sub['pnl'] < 0, 'pnl'].sum())
        spf  = gw / gl if gl > 0 else float('inf')
        aw   = sub.loc[sub['pnl'] > 0, 'pnl'].mean() if len(sub[sub['pnl']>0]) > 0 else 0
        al   = sub.loc[sub['pnl'] < 0, 'pnl'].mean() if len(sub[sub['pnl']<0]) > 0 else 0
        tot  = sub['pnl'].sum()
        flag = "✅" if tot > 0 else "❌"
        lines.append(
            f"  {name:<14} {len(sub):>4,} {wr:>5.1f}% {spf:>6.2f} "
            f"${aw:>6.2f} ${al:>7.2f} ${tot:>9.2f}  {flag}"
        )

    lines += ["", "[ گزارش سالانه ]"]
    lines.append(f"  {'سال':>5} {'معاملات':>8} {'Win%':>6} {'PnL':>10} {'بازده%':>8}")
    lines.append("  " + "-" * 42)
    for _, row in yearly.iterrows():
        lines.append(
            f"  {int(row['year']):>5} {int(row['trades']):>8,} "
            f"{row['win_rate']:>5.1f}% ${row['pnl']:>9.2f} "
            f"{row['return_pct']:>+7.1f}%"
        )

    lines += ["", "[ گزارش ماهانه ]"]
    lines.append(f"  {'ماه':>8} {'معاملات':>8} {'Win%':>6} {'PnL':>10} {'تجمعی':>10}")
    lines.append("  " + "-" * 48)
    for _, row in monthly.iterrows():
        lines.append(
            f"  {str(row['year_month']):>8} {int(row['trades']):>8,} "
            f"{row['win_rate']:>5.1f}% ${row['pnl']:>9.2f} ${row['cum_pnl']:>9.2f}"
        )

    lines += ["", "[ توزیع خروج معاملات ]"]
    for status, cnt in t['status'].value_counts().items():
        lines.append(f"  {status:<15}: {cnt:>5,}")

    lines += ["", SEP]

    report_text = "\n".join(lines)
    print(report_text)

    # ================================================================
    #  ذخیره فایل‌ها
    # ================================================================

    # ۱. گزارش متنی کامل
    with open("Equity_Curve_Report.csv", "w", encoding="utf-8") as f:
        f.write(report_text)

    # ۲. جزئیات معاملات
    t.to_csv("trades_detail.csv", index=False, encoding="utf-8")

    # ۳. equity curve با timestamp
    eq_df = pd.DataFrame({
        'timestamp': risk.curve_ts,
        'equity':    risk.curve
    })
    eq_df.to_csv("equity_curve.csv", index=False, encoding="utf-8")

    # ۴. گزارش ماهانه
    monthly.to_csv("monthly_report.csv", index=False, encoding="utf-8")

    # ۵. گزارش سالانه
    yearly.to_csv("yearly_report.csv", index=False, encoding="utf-8")

    print(f"\n✅ فایل‌های خروجی:")
    print(f"   → Equity_Curve_Report.csv  (گزارش کامل)")
    print(f"   → trades_detail.csv        ({len(t)} معامله)")
    print(f"   → equity_curve.csv         ({len(eq_df)} نقطه)")
    print(f"   → monthly_report.csv       ({len(monthly)} ماه)")
    print(f"   → yearly_report.csv        ({len(yearly)} سال)")


# ================================================================== #
if __name__ == "__main__":
    df             = load_data()
    trades, risk   = run_backtest(df)
    report(trades, risk, df)
