import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

# ================================================================== #
#                         CONFIG                                     #
# ================================================================== #
class Config:
    initial_balance     = 5_000.0
    risk_per_trade_pct  = 0.005      # 0.5% ریسک هر معامله (محافظه‌کارانه)
    max_daily_loss_pct  = 0.04       # 4%
    max_total_dd_pct    = 0.08       # 8%
    profit_target_pct   = 0.10       # 10%
    spread_eur_pips     = 1.2        # پیپ
    spread_gbp_pips     = 1.5
    commission_per_lot  = 7.0        # دلار round-trip
    pip                 = 0.0001
    lot_size            = 100_000


# ================================================================== #
#                      ابزارهای کمکی                                #
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
    d     = close.diff()
    gain  = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def trade_cost(lot: float, symbol: str) -> float:
    """هزینه کل یک معامله: اسپرد + کمیسیون"""
    spread_pips = (Config.spread_eur_pips 
                   if symbol == 'EUR' 
                   else Config.spread_gbp_pips)
    spread_cost = spread_pips * Config.pip * lot * Config.lot_size
    commission  = Config.commission_per_lot * lot
    return spread_cost + commission


def pnl(direction, lot, entry, exit_p, symbol) -> float:
    raw = direction * (exit_p - entry) * lot * Config.lot_size
    return raw - trade_cost(lot, symbol)


def lot_size(equity, sl_pips) -> float:
    """حجم بر اساس ریسک ثابت"""
    if sl_pips <= 0:
        return 0.01
    risk_usd = equity * Config.risk_per_trade_pct
    lot = risk_usd / (sl_pips * Config.pip * Config.lot_size)
    return round(np.clip(lot, 0.01, 1.0), 2)


# ================================================================== #
#                   کلاس مرکزی Risk Manager                         #
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

    def new_bar(self, ts: pd.Timestamp):
        day = ts.date()
        if day != self.cur_day:
            self.cur_day      = day
            self.day_start_eq = self.equity
            # فقط daily halt را ریست می‌کنیم
            if "Daily" in self.halt_reason:
                self.halted      = False
                self.halt_reason = "در حال اجرا"

    def add_pnl(self, amount: float, ts: pd.Timestamp) -> bool:
        self.equity += amount
        self.peak    = max(self.peak, self.equity)
        self.curve.append(round(self.equity, 4))

        # چک روزانه
        daily_dd = (self.equity - self.day_start_eq) / self.day_start_eq
        if daily_dd <= -Config.max_daily_loss_pct:
            self.halted      = True
            self.halt_reason = f"Daily Loss {daily_dd*100:.1f}%"
            return False

        # چک کل
        total_dd = (self.equity - self.peak) / self.peak
        if total_dd <= -Config.max_total_dd_pct:
            self.halted      = True
            self.halt_reason = f"Max DD {total_dd*100:.1f}%"
            return False

        # هدف سود
        profit_pct = (self.equity - Config.initial_balance) / Config.initial_balance
        if profit_pct >= Config.profit_target_pct:
            self.halted      = True
            self.halt_reason = f"✅ Profit Target {profit_pct*100:.1f}%"
            return False

        return True

    @property
    def max_dd(self):
        s       = pd.Series(self.curve)
        roll_mx = s.cummax()
        return ((s - roll_mx) / roll_mx).min() * 100

    @property
    def sharpe(self):
        r = pd.Series(self.curve).pct_change().dropna()
        return (r.mean() / r.std() * np.sqrt(252 * 96)) if r.std() > 0 else 0


# ================================================================== #
#           Strategy A: Correlation Pair Trading                     #
#                                                                    #
#  منطق واقعی:                                                       #
#  EUR و GBP معمولاً با هم حرکت می‌کنند.                            #
#  وقتی EUR/GBP (کراس) از میانگین انحراف پیدا کرد                  #
#  → معامله در جهت بازگشت                                           #
# ================================================================== #
def build_corr_arb_signals(df: pd.DataFrame) -> pd.Series:
    # نسبت EUR به GBP ≈ قیمت EURGBP
    eurgbp = df['c_eur'] / df['c_gbp']

    period = 96          # ۲۴ ساعت
    mean   = eurgbp.rolling(period).mean()
    std    = eurgbp.rolling(period).std()
    z      = (eurgbp - mean) / std.replace(0, np.nan)

    # فیلتر: فقط وقتی std به اندازه کافی بزرگ است (بازار فعال)
    std_filter = std > std.rolling(period * 5).mean() * 0.5

    sig = pd.Series(0, index=df.index)

    # Z > +2 → EUR نسبت به GBP گران است → EURUSD بفروش
    sig[(z >  2.0) & std_filter] = -1
    # Z < -2 → EUR نسبت به GBP ارزان است → EURUSD بخر
    sig[(z < -2.0) & std_filter] =  1

    # فیلتر: سیگنال تکراری نگیر (فقط اولین کندل تغییر)
    sig = sig.where(sig != sig.shift(), 0)

    return sig, z


# ================================================================== #
#        Strategy B: London Open Breakout (بهینه‌شده)               #
#                                                                    #
#  منطق:                                                             #
#  Range ساعت ۵ و ۶ صبح GMT را بساز                                 #
#  در ساعت ۷ (شروع لندن) منتظر شکست باش                            #
#  فیلتر ADR: اگر روز قبل حرکت کمی داشت → رد کن                    #
# ================================================================== #
def build_breakout_signals(df: pd.DataFrame) -> pd.Series:
    df  = df.copy()
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    atr = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)

    # Range پیش از لندن: ساعت ۵ و ۶ GMT
    pre = df[df['hour'].isin([5, 6])]
    rng = pre.groupby('date').agg(
        rh=('h_eur', 'max'),
        rl=('l_eur', 'min')
    )
    rng['rng_size'] = rng['rh'] - rng['rl']

    df = df.join(rng, on='date')

    # فیلتر: range باید منطقی باشد (نه خیلی کوچک، نه خیلی بزرگ)
    atr_avg       = atr.rolling(20).mean()
    valid_range   = (
        (df['rng_size'] > atr_avg * 0.3) &   # حداقل ۳۰٪ ATR
        (df['rng_size'] < atr_avg * 2.0)      # حداکثر ۲۰۰٪ ATR
    )

    sig = pd.Series(0, index=df.index)

    # فقط ساعت ۷ تا ۱۱ GMT (لندن)
    london_hours = df['hour'].between(7, 11)

    # شکست بالا
    sig[
        london_hours &
        valid_range &
        (df['c_eur'] > df['rh']) &
        (df['o_eur'] < df['rh'])    # شکست در همین کندل اتفاق افتاده
    ] = 1

    # شکست پایین
    sig[
        london_hours &
        valid_range &
        (df['c_eur'] < df['rl']) &
        (df['o_eur'] > df['rl'])
    ] = -1

    # فقط اولین سیگنال هر روز
    daily_first = sig[sig != 0].groupby(sig[sig != 0].index.date).head(1).index
    final_sig   = pd.Series(0, index=df.index)
    final_sig[daily_first] = sig[daily_first]

    return final_sig, df['rh'], df['rl'], atr


# ================================================================== #
#        Strategy C: EMA Trend + RSI Pullback                        #
#                                                                    #
#  منطق (جایگزین Mean Reversion که شکست خورد):                      #
#  ترند را با EMA200 تشخیص بده                                       #
#  منتظر پولبک به EMA50 باش (RSI بین ۴۰-۶۰)                        #
#  در جهت ترند اصلی وارد شو                                          #
#  این استراتژی با ترند کار می‌کند نه ضد آن                          #
# ================================================================== #
def build_trend_pullback_signals(df: pd.DataFrame) -> pd.Series:
    c = df['c_eur']

    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rsi    = calc_rsi(c, 14)
    atr    = calc_atr(df['h_eur'], df['l_eur'], c, 14)

    # ترند: EMA50 بالای EMA200 → uptrend
    uptrend   = ema50 > ema200
    downtrend = ema50 < ema200

    # فاصله قیمت از EMA50 (نسبت به ATR)
    dist = (c - ema50) / atr.replace(0, np.nan)

    sig = pd.Series(0, index=df.index)

    # خرید: uptrend + قیمت نزدیک به EMA50 (پولبک) + RSI در ناحیه خنثی
    sig[
        uptrend &
        (dist > -1.0) & (dist < 0.3) &    # نزدیک EMA50 از پایین
        (rsi > 40) & (rsi < 60)
    ] = 1

    # فروش: downtrend + قیمت نزدیک به EMA50 (پولبک) + RSI خنثی
    sig[
        downtrend &
        (dist < 1.0) & (dist > -0.3) &
        (rsi > 40) & (rsi < 60)
    ] = -1

    # فیلتر: سیگنال تکراری نگیر
    sig = sig.where(sig != sig.shift(), 0)

    return sig, ema50, atr


# ================================================================== #
#                      موتور Backtest                                #
# ================================================================== #
def run_backtest(df: pd.DataFrame):

    print("\n⚙️  محاسبه اندیکاتورها...")

    # سیگنال‌ها
    sig_arb, z_score           = build_corr_arb_signals(df)
    sig_brk, rh, rl, atr_brk  = build_breakout_signals(df)
    sig_tp,  ema50, atr_tp     = build_trend_pullback_signals(df)

    atr_main = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14)

    risk = RiskManager()

    # ساختار هر پوزیشن باز
    # { key: {dir, lot, entry, sl, tp, symbol, strategy, entry_ts} }
    open_pos = {}

    trades = []   # لیست معاملات بسته‌شده

    warmup = 250  # کندل warmup برای اندیکاتورها

    print("⚙️  شروع شبیه‌سازی کندل به کندل...\n")

    for i in range(warmup, len(df)):
        ts  = df.index[i]
        row = df.iloc[i]

        c_eur = row['c_eur']
        c_gbp = row['c_gbp']
        h_eur = row['h_eur']
        l_eur = row['l_eur']
        atr   = atr_main.iloc[i]

        if np.isnan(atr) or atr <= 0:
            continue

        risk.new_bar(ts)

        # ---- اگر trading متوقف شده: فقط پوزیشن‌های باز را ببند ----
        if risk.halted:
            for key in list(open_pos.keys()):
                p       = open_pos.pop(key)
                sym     = p['symbol']
                ep      = c_eur if sym == 'EUR' else c_gbp
                p_pnl   = pnl(p['dir'], p['lot'], p['entry'], ep, sym)
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
                hi = row['h_gbp']
                lo = row['l_gbp']
                cp = c_gbp

            d     = p['dir']
            entry = p['entry']
            sl    = p['sl']
            tp    = p['tp']

            hit_sl = (d ==  1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d ==  1 and hi >= tp) or (d == -1 and lo <= tp)

            # Trailing Stop برای Breakout (بعد از ۱ ATR سود)
            if p['strategy'] == 'Breakout':
                if d == 1 and cp > entry + atr:
                    p['sl'] = max(p['sl'], entry + atr * 0.3)
                elif d == -1 and cp < entry - atr:
                    p['sl'] = min(p['sl'], entry - atr * 0.3)

            # خروج Correlation Arb: وقتی Z به صفر برگشت
            if p['strategy'] == 'CorrArb':
                z_now = z_score.iloc[i]
                if pd.notna(z_now) and abs(z_now) < 0.5:
                    hit_tp = True

            exit_reason = None
            exit_price  = None

            if hit_sl:
                exit_reason = 'SL'
                exit_price  = sl    # بدبینانه: در قیمت SL بسته می‌شود
            elif hit_tp:
                exit_reason = 'TP'
                exit_price  = tp

            if exit_reason:
                p_pnl = pnl(d, p['lot'], entry, exit_price, sym)
                trades.append({
                    **p,
                    'exit':      exit_price,
                    'exit_ts':   ts,
                    'pnl':       p_pnl,
                    'status':    exit_reason
                })
                risk.add_pnl(p_pnl, ts)
                del open_pos[key]

        # ================================================================
        #  مرحله ۲: باز کردن پوزیشن جدید
        # ================================================================

        # ---- A: Correlation Arbitrage ----
        if 'CorrArb' not in open_pos:
            sig = sig_arb.iloc[i]
            if sig != 0:
                sl_pips = 20          # SL ثابت ۲۰ پیپ
                tp_pips = 30          # TP ثابت ۳۰ پیپ  (RR=1.5)
                lot     = lot_size(risk.equity, sl_pips)
                ep      = c_eur + sig * Config.spread_eur_pips * Config.pip / 2

                open_pos['CorrArb'] = {
                    'strategy':  'CorrArb',
                    'symbol':    'EUR',
                    'dir':       sig,
                    'lot':       lot,
                    'entry':     ep,
                    'sl':        ep - sig * sl_pips * Config.pip,
                    'tp':        ep + sig * tp_pips * Config.pip,
                    'entry_ts':  ts,
                }

        # ---- B: London Breakout ----
        if 'Breakout' not in open_pos:
            sig = sig_brk.iloc[i]
            if sig != 0:
                sl_pips = max(12, (rh.iloc[i] - rl.iloc[i]) / Config.pip)
                tp_pips = sl_pips * 2.0     # RR = 2:1
                lot     = lot_size(risk.equity, sl_pips)
                ep      = c_eur + sig * Config.spread_eur_pips * Config.pip / 2

                open_pos['Breakout'] = {
                    'strategy': 'Breakout',
                    'symbol':   'EUR',
                    'dir':      sig,
                    'lot':      lot,
                    'entry':    ep,
                    'sl':       ep - sig * sl_pips * Config.pip,
                    'tp':       ep + sig * tp_pips * Config.pip,
                    'entry_ts': ts,
                }

        # ---- C: Trend Pullback ----
        if 'TrendPB' not in open_pos:
            sig = sig_tp.iloc[i]
            if sig != 0:
                sl_pips = max(15, atr / Config.pip * 1.2)
                tp_pips = sl_pips * 2.0     # RR = 2:1
                lot     = lot_size(risk.equity, sl_pips)
                ep      = c_eur + sig * Config.spread_eur_pips * Config.pip / 2

                open_pos['TrendPB'] = {
                    'strategy': 'TrendPB',
                    'symbol':   'EUR',
                    'dir':      sig,
                    'lot':      lot,
                    'entry':    ep,
                    'sl':       ep - sig * sl_pips * Config.pip,
                    'tp':       ep + sig * tp_pips * Config.pip,
                    'entry_ts': ts,
                }

    # ---- بستن پوزیشن‌های باقی‌مانده در پایان ----
    last_ts  = df.index[-1]
    last_eur = df['c_eur'].iloc[-1]
    last_gbp = df['c_gbp'].iloc[-1]

    for key, p in open_pos.items():
        ep    = last_eur if p['symbol'] == 'EUR' else last_gbp
        p_pnl = pnl(p['dir'], p['lot'], p['entry'], ep, p['symbol'])
        trades.append({**p, 'exit': ep, 'exit_ts': last_ts,
                       'pnl': p_pnl, 'status': 'eod_close'})
        risk.add_pnl(p_pnl, last_ts)

    return trades, risk


# ================================================================== #
#                          گزارش                                     #
# ================================================================== #
def report(trades: list, risk: RiskManager, df: pd.DataFrame):
    if not trades:
        print("❌ هیچ معامله‌ای ثبت نشد!")
        return

    t = pd.DataFrame(trades)
    t['pnl'] = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)

    # ---- آمار کلی ----
    final_eq    = risk.equity
    total_ret   = (final_eq - Config.initial_balance) / Config.initial_balance * 100
    win_rate    = (t['pnl'] > 0).mean() * 100
    avg_win     = t.loc[t['pnl'] > 0, 'pnl'].mean()
    avg_loss    = t.loc[t['pnl'] < 0, 'pnl'].mean()
    gross_win   = t.loc[t['pnl'] > 0, 'pnl'].sum()
    gross_loss  = t.loc[t['pnl'] < 0, 'pnl'].sum().abs()
    pf          = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy  = t['pnl'].mean()

    rr_actual = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    print("\n" + "=" * 60)
    print("       📊 گزارش نهایی Backtest  (2020 → 2025)")
    print("=" * 60)
    print(f"  {'موجودی اولیه':<22}: ${Config.initial_balance:>10,.2f}")
    print(f"  {'موجودی نهایی':<22}: ${final_eq:>10,.2f}")
    print(f"  {'بازده کل':<22}: {total_ret:>+10.2f}%")
    print(f"  {'Max Drawdown':<22}: {risk.max_dd:>10.2f}%")
    print(f"  {'Sharpe Ratio':<22}: {risk.sharpe:>10.2f}")
    print(f"  {'Profit Factor':<22}: {pf:>10.2f}")
    print(f"  {'Win Rate':<22}: {win_rate:>10.1f}%")
    print(f"  {'Avg Win':<22}: ${avg_win:>10.2f}")
    print(f"  {'Avg Loss':<22}: ${avg_loss:>10.2f}")
    print(f"  {'RR واقعی':<22}: {rr_actual:>10.2f}")
    print(f"  {'Expectancy / trade':<22}: ${expectancy:>10.2f}")
    print(f"  {'تعداد معاملات':<22}: {len(t):>10,}")
    print(f"  {'توقف به دلیل':<22}: {risk.halt_reason}")
    print("-" * 60)

    # ---- آمار هر استراتژی ----
    print(f"\n  {'استراتژی':<15} {'#':>5} {'Win%':>6} {'PF':>6} "
          f"{'AvgWin':>8} {'AvgLoss':>9} {'PnL':>10}")
    print("  " + "-" * 60)

    for name in t['strategy'].unique():
        sub  = t[t['strategy'] == name]
        wr   = (sub['pnl'] > 0).mean() * 100
        gw   = sub.loc[sub['pnl'] > 0, 'pnl'].sum()
        gl   = sub.loc[sub['pnl'] < 0, 'pnl'].sum().abs()
        spf  = gw / gl if gl > 0 else float('inf')
        aw   = sub.loc[sub['pnl'] > 0, 'pnl'].mean()
        al   = sub.loc[sub['pnl'] < 0, 'pnl'].mean()
        tot  = sub['pnl'].sum()
        print(f"  {name:<15} {len(sub):>5,} {wr:>5.1f}% {spf:>6.2f} "
              f"${aw:>7.2f} ${al:>8.2f} ${tot:>10.2f}")

    print("=" * 60)

    # ---- ذخیره ----
    t.to_csv("trades_report.csv", index=False)

    eq_df = pd.DataFrame({
        'equity': risk.curve
    })
    eq_df.to_csv("equity_curve.csv", index=False)

    # ---- خلاصه دیباگ ----
    print(f"\n  📋 توزیع وضعیت خروج:")
    print(t['status'].value_counts().to_string())

    print(f"\n✅ trades_report.csv  ({len(t)} ردیف)")
    print(f"✅ equity_curve.csv   ({len(eq_df)} نقطه)")


# ================================================================== #
if __name__ == "__main__":
    df              = load_data()
    trades, risk    = run_backtest(df)
    report(trades, risk, df)
