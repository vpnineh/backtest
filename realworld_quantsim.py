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
    risk_per_trade_pct = 0.010      # 1% per trade
    max_daily_loss_pct = 0.04       # 4% daily
    max_total_dd_pct   = 0.08       # 8% max DD ماهانه
    monthly_target_pct = 0.10       # هدف 10% ماهانه
    spread_eur_pips    = 1.0
    spread_gbp_pips    = 1.2
    commission_per_lot = 6.0
    pip                = 0.0001
    lot_size           = 100_000
    max_lot            = 2.0
    warmup             = 300


# ================================================================== #
#                     داده و اندیکاتورها                            #
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
    print(f"✅ {len(df):,} کندل | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def calc_atr(h, l, c, p=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(),
                    (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def calc_rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))

def calc_adx(h, l, c, p=14):
    up  = h.diff(); dn = -l.diff()
    dmp = up.where((up>dn)&(up>0), 0.0)
    dmn = dn.where((dn>up)&(dn>0), 0.0)
    tr  = calc_atr(h, l, c, 1)
    s   = tr.rolling(p).sum()
    dip = 100*dmp.rolling(p).sum()/s.replace(0,np.nan)
    din = 100*dmn.rolling(p).sum()/s.replace(0,np.nan)
    dx  = (abs(dip-din)/(dip+din).replace(0,np.nan))*100
    return dx.rolling(p).mean()

def calc_macd(c, f=12, s=26, sig=9):
    m = c.ewm(span=f,adjust=False).mean() - c.ewm(span=s,adjust=False).mean()
    return m, m.ewm(span=sig,adjust=False).mean(), m-m.ewm(span=sig,adjust=False).mean()

def lot_size_calc(equity, sl_pips):
    if sl_pips <= 0: return 0.01
    lot = equity*Config.risk_per_trade_pct / (sl_pips*Config.pip*Config.lot_size)
    return round(float(np.clip(lot, 0.01, Config.max_lot)), 2)


# ================================================================== #
#   REGIME DETECTOR                                                  #
#   ────────────────────────────────────────────────────────────── #
#   رژیم بازار را تشخیص می‌دهد:                                     #
#   RANGE  → CorrArb فعال                                           #
#   TREND  → TrendFollow فعال                                       #
#   MIXED  → هر دو با نصف ریسک                                      #
# ================================================================== #
def compute_regime(df: pd.DataFrame) -> pd.Series:
    """
    ترکیب چند شاخص برای تشخیص رژیم:
    1. ADX (200 کندل = ~2 روز): روند یا رنج؟
    2. Bollinger Width: نوسان رژیم
    3. EMA slope: جهت روند

    خروجی:
    +1 = TREND (ترند)
     0 = MIXED
    -1 = RANGE (رنج)
    """
    c   = df['c_eur']
    h   = df['h_eur']
    l   = df['l_eur']

    # ADX بلند‌مدت (بازه 200 کندل ≈ 2 روز)
    adx_slow  = calc_adx(h, l, c, 50)
    adx_fast  = calc_adx(h, l, c, 14)

    # Bollinger Band Width (نرمال‌شده)
    bb_mid    = c.rolling(96).mean()    # 1 روز
    bb_std    = c.rolling(96).std()
    bb_width  = (bb_std / bb_mid.replace(0, np.nan)) * 100
    bb_wm     = bb_width.rolling(480).mean()   # 5 روز میانگین

    # EMA slope (شیب EMA50)
    ema50     = c.ewm(span=50, adjust=False).mean()
    ema200    = c.ewm(span=200, adjust=False).mean()
    slope50   = (ema50 - ema50.shift(48)) / ema50.shift(48) * 100  # تغییر در 12 ساعت

    # ─── تعریف رژیم ───
    # RANGE: ADX پایین + BB فشرده
    is_range = (adx_slow < 22) & (adx_fast < 25) & (bb_width < bb_wm * 0.9)

    # TREND: ADX بالا + BB گشاد + slope قوی
    is_trend = (adx_slow > 28) & (adx_fast > 25) & \
               (bb_width > bb_wm * 1.1) & (slope50.abs() > 0.05)

    regime = pd.Series(0, index=df.index)
    regime[is_range] = -1   # RANGE
    regime[is_trend] =  1   # TREND

    # smooth: از جهش ناگهانی جلوگیری کن
    regime = regime.rolling(8).median().fillna(0)
    regime = regime.apply(lambda x: -1 if x <= -0.5 else (1 if x >= 0.5 else 0))

    return regime


# ================================================================== #
#   محاسبه همه سیگنال‌ها (یک‌بار)                                  #
# ================================================================== #
def compute_all_signals(df: pd.DataFrame) -> dict:
    print("  محاسبه اندیکاتورها...", end="", flush=True)

    c_e = df['c_eur']; h_e = df['h_eur']
    l_e = df['l_eur']; c_g = df['c_gbp']

    atr    = calc_atr(h_e, l_e, c_e, 14)
    rsi    = calc_rsi(c_e, 14)
    adx    = calc_adx(h_e, l_e, c_e, 14)
    ema9   = c_e.ewm(span=9,   adjust=False).mean()
    ema21  = c_e.ewm(span=21,  adjust=False).mean()
    ema50  = c_e.ewm(span=50,  adjust=False).mean()
    ema200 = c_e.ewm(span=200, adjust=False).mean()
    _, _, macd_h = calc_macd(c_e)

    hour    = pd.Series(df.index.hour,    index=df.index)
    weekday = pd.Series(df.index.weekday, index=df.index)

    # ── Regime ──
    regime = compute_regime(df)

    # ════════════════════════════════════════════
    #  S1: CorrArb — فقط در رژیم RANGE
    # ════════════════════════════════════════════
    eurgbp  = c_e / c_g
    z_mean  = eurgbp.rolling(96).mean()
    z_std   = eurgbp.rolling(96).std()
    z_score = (eurgbp - z_mean) / z_std.replace(0, np.nan)

    std_ok   = z_std > z_std.rolling(480).mean() * 0.3
    time_ok  = hour.between(7, 19)
    # فقط در رژیم RANGE فعال
    range_ok = (regime == -1)

    sig1 = pd.Series(0, index=df.index)
    sig1[(z_score >  2.2) & std_ok & time_ok & range_ok & (rsi>53)] = -1
    sig1[(z_score < -2.2) & std_ok & time_ok & range_ok & (rsi<47)] =  1
    sig1 = sig1.where(sig1 != sig1.shift(), 0)

    sl1 = np.where(sig1 != 0, 20.0, 0.0)
    tp1 = np.where(sig1 != 0, 38.0, 0.0)

    # ════════════════════════════════════════════
    #  S2: Trend Follow EMA — فقط در رژیم TREND
    # ════════════════════════════════════════════
    # ساده و مستقیم: EMA21 crossover با تایید EMA50 و EMA200
    # در جهت روند قوی
    trend_ok = (regime == 1)
    active   = hour.between(7, 18)

    cross_up   = (ema21 > ema50) & (ema21.shift(1) <= ema50.shift(1))
    cross_dn   = (ema21 < ema50) & (ema21.shift(1) >= ema50.shift(1))

    sig2 = pd.Series(0, index=df.index)
    sig2[active & trend_ok & cross_up &
         (c_e > ema200) & (adx > 22) &
         rsi.between(48, 72) & (macd_h > 0)] =  1
    sig2[active & trend_ok & cross_dn &
         (c_e < ema200) & (adx > 22) &
         rsi.between(28, 52) & (macd_h < 0)] = -1

    nz2 = sig2[sig2 != 0]
    fi2 = nz2.groupby(nz2.index.date).head(1).index
    s2  = pd.Series(0, index=df.index); s2[fi2] = sig2[fi2]

    dist21   = (c_e - ema21).abs().values / Config.pip
    sl2      = np.where(s2 != 0,
                        np.maximum(18, dist21 + atr.values/Config.pip*0.8), 0.0)
    tp2      = np.where(s2 != 0, sl2 * 2.5, 0.0)

    # ════════════════════════════════════════════
    #  S3: Asian Breakout — در هر رژیم (با فیلتر)
    # ════════════════════════════════════════════
    # در رژیم RANGE: شکست کوتاه‌مدت
    # در رژیم TREND: شکست در جهت ترند
    d_temp        = df.copy()
    d_temp['date'] = d_temp.index.date
    asian_df      = d_temp[hour.between(2, 6)]
    rng_day       = asian_df.groupby('date').agg(
        ah=('h_eur','max'), al=('l_eur','min'))
    rng_day['rng_pips'] = (rng_day['ah']-rng_day['al'])/Config.pip
    d_temp = d_temp.join(rng_day, on='date')

    london   = hour.between(8, 10)
    rng_ok   = d_temp['rng_pips'].between(12, 45)
    day_ok   = weekday.between(0, 3)

    above2 = ((c_e > d_temp['ah']).astype(int) +
              (c_e.shift(1) > d_temp['ah'].shift(1)).astype(int)) >= 2
    below2 = ((c_e < d_temp['al']).astype(int) +
              (c_e.shift(1) < d_temp['al'].shift(1)).astype(int)) >= 2

    # در ترند: فقط در جهت ترند
    trend_up = c_e > ema200; trend_dn = c_e < ema200
    # در رنج: هر دو جهت
    any_dir  = pd.Series(True, index=df.index)

    long_ok  = (above2 & (adx > 18) & london & rng_ok & day_ok &
                ((trend_ok & trend_up) | (~trend_ok)))
    short_ok = (below2 & (adx > 18) & london & rng_ok & day_ok &
                ((trend_ok & trend_dn) | (~trend_ok)))

    sig3 = pd.Series(0, index=df.index)
    sig3[long_ok]  =  1
    sig3[short_ok] = -1

    nz3 = sig3[sig3 != 0]
    fi3 = nz3.groupby(nz3.index.date).head(1).index
    s3  = pd.Series(0, index=df.index); s3[fi3] = sig3[fi3]

    rp3 = d_temp['rng_pips'].fillna(20).values
    sl3 = np.where(s3!=0, np.maximum(15, rp3+3), 0.0)
    tp3 = np.where(s3!=0, np.maximum(sl3*2, rp3*3.0), 0.0)

    # ════════════════════════════════════════════
    #  S4: Mean Reversion RSI — رژیم RANGE
    # ════════════════════════════════════════════
    # در رنج عمیق: RSI extreme + Bollinger extreme
    bb_mid2 = c_e.rolling(20).mean()
    bb_std2 = c_e.rolling(20).std()
    bb_up   = bb_mid2 + 2.2 * bb_std2
    bb_lo   = bb_mid2 - 2.2 * bb_std2

    # فاصله از BB میانه نرمال‌شده
    bb_pos  = (c_e - bb_mid2) / (2.2 * bb_std2.replace(0, np.nan))

    active4 = hour.between(8, 18) & weekday.between(0, 3)

    sig4 = pd.Series(0, index=df.index)
    # خرید: RSI < 30 + پایین BB + رژیم رنج
    sig4[active4 & range_ok &
         (rsi < 30) & (c_e < bb_lo) &
         (macd_h > macd_h.shift(1)) &
         (adx < 28)] =  1
    # فروش: RSI > 70 + بالای BB + رژیم رنج
    sig4[active4 & range_ok &
         (rsi > 70) & (c_e > bb_up) &
         (macd_h < macd_h.shift(1)) &
         (adx < 28)] = -1

    nz4 = sig4[sig4 != 0]
    fi4 = nz4.groupby(nz4.index.date).head(1).index
    s4  = pd.Series(0, index=df.index); s4[fi4] = sig4[fi4]

    sl4 = np.where(s4!=0, np.maximum(15, atr.values/Config.pip*1.5), 0.0)
    tp4 = np.where(s4!=0, sl4*1.8, 0.0)  # RR کمتر ولی WR بالاتر

    print(" ✓")
    print(f"  سیگنال‌ها: CorrArb={int((sig1!=0).sum())} | "
          f"TrendEMA={int((s2!=0).sum())} | "
          f"AsianBreak={int((s3!=0).sum())} | "
          f"MeanRev={int((s4!=0).sum())}")

    # آمار رژیم
    r_range = (regime == -1).sum()
    r_trend = (regime ==  1).sum()
    r_mixed = (regime ==  0).sum()
    total   = len(regime)
    print(f"  رژیم بازار: Range={r_range/total*100:.0f}% | "
          f"Trend={r_trend/total*100:.0f}% | "
          f"Mixed={r_mixed/total*100:.0f}%")

    return {
        'CorrArb':    (sig1,  sl1,  tp1,  z_score),
        'TrendEMA':   (s2,    sl2,  tp2,  None),
        'AsianBreak': (s3,    sl3,  tp3,  None),
        'MeanRev':    (s4,    sl4,  tp4,  None),
        'regime':     regime,
        'atr':        atr,
        'ema21':      ema21,
    }


# ================================================================== #
#   موتور بک‌تست ماهانه (بهینه‌شده)                                 #
# ================================================================== #
def run_monthly_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    sig_series: pd.Series,
    sl_arr: np.ndarray,
    tp_arr: np.ndarray,
    z_series=None,
) -> tuple:
    pip  = Config.pip
    ls   = Config.lot_size
    sp   = Config.spread_eur_pips
    comm = Config.commission_per_lot

    close_a = df['c_eur'].values
    high_a  = df['h_eur'].values
    low_a   = df['l_eur'].values
    sig_a   = sig_series.values
    ts_a    = df.index
    z_a     = z_series.values if z_series is not None else None

    # ماه‌ها
    periods   = pd.Series(ts_a).dt.to_period('M')
    months    = periods.unique()

    equity       = Config.initial_balance
    all_trades   = []
    monthly_log  = []
    eq_curve     = [equity]
    eq_curve_ts  = [None]
    open_pos     = None

    for month_period in months:
        mask      = (periods == month_period).values
        m_bars    = np.where(mask)[0]
        if len(m_bars) == 0: continue
        bar_start = m_bars[0]
        bar_end   = m_bars[-1]
        if bar_end < Config.warmup: continue

        month_start_eq = equity
        month_peak     = equity
        month_day_eq   = equity
        month_halted   = False
        month_halt_r   = ""
        month_trades   = []
        cur_day        = None

        sig_set = {x for x in np.where(sig_a != 0)[0]
                   if bar_start <= x <= bar_end
                   and x >= Config.warmup}

        for bar in range(max(bar_start, Config.warmup), bar_end + 1):
            day = ts_a[bar].date()
            if day != cur_day:
                cur_day = day; month_day_eq = equity
                if month_halted and "Daily" in month_halt_r:
                    month_halted = False; month_halt_r = ""

            # ── Halt: ببند پوزیشن و از ماه خارج شو ──
            if month_halted:
                if open_pos is not None:
                    cp  = close_a[bar]
                    raw = open_pos['dir']*(cp-open_pos['entry'])*open_pos['lot']*ls
                    pnl = raw - (sp*pip*open_pos['lot']*ls+comm*open_pos['lot'])
                    equity += pnl
                    month_peak = max(month_peak, equity)
                    eq_curve.append(round(equity,4))
                    eq_curve_ts.append(ts_a[bar])
                    rec = {**open_pos,'exit':cp,'exit_ts':ts_a[bar],
                           'pnl':pnl,'status':'month_halt'}
                    month_trades.append(rec); all_trades.append(rec)
                    open_pos = None
                break

            # ── مدیریت پوزیشن باز ──
            if open_pos is not None:
                hi=high_a[bar]; lo=low_a[bar]; cp=close_a[bar]
                d=open_pos['dir']; ep=open_pos['entry']
                sl=open_pos['sl']; tp=open_pos['tp']

                hit_sl = (d==1 and lo<=sl) or (d==-1 and hi>=sl)
                hit_tp = (d==1 and hi>=tp) or (d==-1 and lo<=tp)

                # Z-exit
                if z_a is not None:
                    zn = z_a[bar]
                    if not np.isnan(zn) and abs(zn) < 0.25:
                        hit_tp = True

                # Trailing SL (دو مرحله)
                move    = d*(cp-ep)
                tp_dist = abs(tp-ep)
                if tp_dist > 0:
                    pct = move/tp_dist
                    if pct > 0.55:
                        be = ep + d*tp_dist*0.12
                        open_pos['sl'] = (max(sl,be) if d==1 else min(sl,be))
                    if pct > 0.80:
                        lock = ep + d*tp_dist*0.45
                        open_pos['sl'] = (max(open_pos['sl'],lock)
                                          if d==1 else min(open_pos['sl'],lock))

                # Time stop (4 روز = 384 کندل)
                if (bar-open_pos['entry_bar']) >= 384 and not hit_tp:
                    raw = d*(cp-ep)*open_pos['lot']*ls
                    pnl = raw-(sp*pip*open_pos['lot']*ls+comm*open_pos['lot'])
                    equity+=pnl; month_peak=max(month_peak,equity)
                    eq_curve.append(round(equity,4))
                    eq_curve_ts.append(ts_a[bar])
                    rec={**open_pos,'exit':cp,'exit_ts':ts_a[bar],
                         'pnl':pnl,'status':'TimeStop'}
                    month_trades.append(rec); all_trades.append(rec)
                    open_pos=None
                    # بررسی DD
                    dd_d=(equity-month_day_eq)/month_day_eq
                    dd_p=(equity-month_peak)/month_peak
                    if dd_d<=-Config.max_daily_loss_pct:
                        month_halted=True; month_halt_r=f"Daily {dd_d*100:.1f}%"
                    elif dd_p<=-Config.max_total_dd_pct:
                        month_halted=True; month_halt_r=f"MaxDD {dd_p*100:.1f}%"
                    continue

                ep2=exit_p=None
                if hit_sl: ep2,exit_p='SL',open_pos['sl']
                elif hit_tp: ep2,exit_p='TP',open_pos['tp']
                if ep2:
                    raw=d*(exit_p-ep)*open_pos['lot']*ls
                    pnl=raw-(sp*pip*open_pos['lot']*ls+comm*open_pos['lot'])
                    equity+=pnl; month_peak=max(month_peak,equity)
                    eq_curve.append(round(equity,4))
                    eq_curve_ts.append(ts_a[bar])
                    rec={**open_pos,'exit':exit_p,'exit_ts':ts_a[bar],
                         'pnl':pnl,'status':ep2}
                    month_trades.append(rec); all_trades.append(rec)
                    open_pos=None
                    dd_d=(equity-month_day_eq)/month_day_eq
                    dd_p=(equity-month_peak)/month_peak
                    if dd_d<=-Config.max_daily_loss_pct:
                        month_halted=True; month_halt_r=f"Daily {dd_d*100:.1f}%"
                    elif dd_p<=-Config.max_total_dd_pct:
                        month_halted=True; month_halt_r=f"MaxDD {dd_p*100:.1f}%"

            # ── ورود ──
            if open_pos is None and not month_halted and bar in sig_set:
                sv=int(sig_a[bar])
                slp=float(sl_arr[bar]); tpp=float(tp_arr[bar])
                if slp>0 and tpp>0 and not np.isnan(slp) and not np.isnan(tpp):
                    lot=lot_size_calc(equity, slp)
                    ep=close_a[bar]+sv*sp*pip/2
                    open_pos=dict(
                        strategy=strategy_name, symbol='EUR',
                        dir=sv, lot=lot, entry=ep,
                        sl=ep-sv*slp*pip, tp=ep+sv*tpp*pip,
                        entry_ts=ts_a[bar], entry_bar=bar,
                    )

        # ── ثبت ماه ──
        month_pnl = equity - month_start_eq
        month_ret = month_pnl / month_start_eq * 100
        wins      = sum(1 for t in month_trades if t.get('pnl',0)>0)
        wr        = wins/len(month_trades)*100 if month_trades else 0

        if month_halted:
            st = f"🛑 HALTED ({month_halt_r})"
        elif month_ret >= Config.monthly_target_pct*100:
            st = f"🎯 TARGET ({month_ret:>+.1f}%)"
        elif month_ret > 0:
            st = f"✅ +{month_ret:.1f}%"
        elif month_ret == 0 and len(month_trades)==0:
            st = "⏸  بدون معامله"
        else:
            st = f"❌ {month_ret:.1f}%"

        dd_m = min(0, (equity-month_peak)/month_peak*100)

        monthly_log.append(dict(
            period=str(month_period),
            start_eq=round(month_start_eq,2),
            end_eq=round(equity,2),
            pnl=round(month_pnl,2),
            ret_pct=round(month_ret,2),
            trades=len(month_trades),
            wins=wins, wr=round(wr,1),
            max_dd=round(dd_m,2),
            halted=month_halted,
            status=st,
        ))

    return all_trades, monthly_log, eq_curve, eq_curve_ts


# ================================================================== #
#               آمار کامل                                           #
# ================================================================== #
def compute_stats(trades, monthly_log, eq_curve, eq_curve_ts, name):
    if not trades: return None

    t = pd.DataFrame(trades)
    t['pnl']          = pd.to_numeric(t['pnl'],errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts']-t['entry_ts']).dt.total_seconds()/60

    ml = pd.DataFrame(monthly_log)

    final_eq   = eq_curve[-1]
    total_pnl  = final_eq - Config.initial_balance
    total_ret  = total_pnl/Config.initial_balance*100
    start_d    = t['entry_ts'].min(); end_d = t['exit_ts'].max()
    total_days = max((end_d-start_d).days,1)
    ann_ret    = ((final_eq/Config.initial_balance)**(365.25/total_days)-1)*100

    win_t=t[t['pnl']>0]; loss_t=t[t['pnl']<0]
    win_r=len(win_t)/len(t)*100 if len(t)>0 else 0
    avg_w=win_t['pnl'].mean() if len(win_t)>0 else 0
    avg_l=loss_t['pnl'].mean() if len(loss_t)>0 else 0
    gw=win_t['pnl'].sum(); gl=abs(loss_t['pnl'].sum())
    pf=gw/gl if gl>0 else float('inf')
    exp_v=t['pnl'].mean()
    rr=abs(avg_w/avg_l) if avg_l!=0 else 0

    eq_s   = pd.Series(eq_curve)
    max_dd = ((eq_s-eq_s.cummax())/eq_s.cummax()*100).min()
    r_ret  = eq_s.pct_change().dropna()
    sharpe = (r_ret.mean()/r_ret.std()*np.sqrt(252*96)) if r_ret.std()>0 else 0
    neg    = r_ret[r_ret<0]
    ds     = neg.std() if len(neg)>0 else 1e-10
    sortino= r_ret.mean()/ds*np.sqrt(252*96)
    calmar = (final_eq/Config.initial_balance-1)/abs(max_dd/100) if max_dd!=0 else 0

    sign=t['pnl'].apply(lambda x:1 if x>0 else(-1 if x<0 else 0))
    cw=cl=mcw=mcl=0
    for s in sign:
        if s>0: cw+=1;cl=0;mcw=max(mcw,cw)
        elif s<0: cl+=1;cw=0;mcl=max(mcl,cl)
        else: cw=cl=0

    # آمار ماهانه
    prof_m   = (ml['pnl']>0).sum()
    loss_m   = (ml['pnl']<=0).sum()
    halt_m   = ml['halted'].sum()
    target_m = (ml['ret_pct']>=Config.monthly_target_pct*100).sum()
    no_trade = (ml['trades']==0).sum()
    avg_mret = ml[ml['trades']>0]['ret_pct'].mean()  # فقط ماه‌های با معامله
    best_m   = ml['pnl'].max(); worst_m=ml['pnl'].min()

    return dict(
        name=name, trades=t, monthly=ml,
        eq_curve=eq_curve, eq_curve_ts=eq_curve_ts,
        final_eq=final_eq, total_pnl=total_pnl,
        total_ret=total_ret, ann_ret=ann_ret, total_days=total_days,
        win_r=win_r, avg_w=avg_w, avg_l=avg_l, pf=pf,
        exp=exp_v, rr=rr, mcw=mcw, mcl=mcl,
        max_dd=max_dd, sharpe=sharpe, sortino=sortino, calmar=calmar,
        prof_m=int(prof_m), loss_m=int(loss_m),
        halt_m=int(halt_m), target_m=int(target_m),
        no_trade_m=int(no_trade),
        avg_mret=round(avg_mret,2),
        best_m=best_m, worst_m=worst_m,
        best_t=t['pnl'].max(), worst_t=t['pnl'].min(),
        avg_dur=t['duration_min'].mean(),
    )


# ================================================================== #
#               گزارش‌ساز                                           #
# ================================================================== #
def print_report(s: dict) -> str:
    W=74; SEP="═"*W

    def rw(lbl, val, ok=None):
        l=f"  {lbl}"; v=str(val)
        mk="" if ok is None else("  ✅" if ok else"  ❌")
        d="·"*max(2,W-len(l)-len(v)-len(mk)-2)
        return f"{l} {d} {v}{mk}"

    def box(t):
        i=f"─ {t} "; return"┌"+i+"─"*(W-len(i)-1)+"┐"
    bot="└"+"─"*(W-1)+"┘"

    ppm     = s['avg_mret']
    prop_ok = (s['total_ret']>0 and s['pf']>1.3
               and abs(s['max_dd'])<8 and ppm>8
               and s['prof_m']>s['loss_m'])
    status  = "✅ PROP READY" if prop_ok else "⚠️  در حال بهینه‌سازی"

    lines=[
        "",SEP,
        f"  ▌  {s['name']}   {status}  ▐",
        f"  ▌  {s['trades']['entry_ts'].min().date()} → "
        f"{s['trades']['exit_ts'].max().date()}  ({s['total_days']} روز)  ▐",
        SEP,"",

        box("نتایج مالی"),
        rw("موجودی اولیه",   f"${Config.initial_balance:>12,.2f}"),
        rw("موجودی نهایی",   f"${s['final_eq']:>12,.2f}"),
        rw("سود کل",        f"${s['total_pnl']:>+12,.2f}"),
        rw("بازده کل",      f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه",  f"{s['ann_ret']:>+.2f}%"),
        rw("ماهانه avg*",   f"{ppm:>+.2f}%",ok=(ppm>8)),
        rw("* فقط ماه‌های دارای معامله",""),
        rw("بهترین ماه",    f"${s['best_m']:>+.2f}"),
        rw("بدترین ماه",    f"${s['worst_m']:>+.2f}"),
        bot,"",

        box("ریسک"),
        rw("Max Drawdown",  f"{s['max_dd']:.2f}%",ok=(abs(s['max_dd'])<8)),
        rw("Sharpe",        f"{s['sharpe']:.2f}"),
        rw("Sortino",       f"{s['sortino']:.2f}"),
        rw("Calmar",        f"{s['calmar']:.2f}"),
        rw("Profit Factor", f"{s['pf']:.2f}",ok=(s['pf']>1.3)),
        bot,"",

        box("معاملات"),
        rw("تعداد کل",      f"{len(s['trades']):,}"),
        rw("Win Rate",       f"{s['win_r']:.1f}%"),
        rw("Avg Win",        f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",       f"${s['avg_l']:>+.2f}"),
        rw("RR",             f"{s['rr']:.2f}"),
        rw("Expectancy",     f"${s['exp']:>+.2f}"),
        rw("Max Cons Win",   f"{s['mcw']}"),
        rw("Max Cons Loss",  f"{s['mcl']}"),
        rw("مدت میانگین",   f"{s['avg_dur']:.0f} min"),
        bot,"",

        box("آمار ماهانه"),
        rw("کل ماه‌ها",     f"{s['prof_m']+s['loss_m']+s['no_trade_m']}"),
        rw("سودده",         f"{s['prof_m']}",ok=(s['prof_m']>s['loss_m'])),
        rw("ضررده",         f"{s['loss_m']}"),
        rw("بدون معامله",   f"{s['no_trade_m']}"),
        rw("Halted (DD)",   f"{s['halt_m']}",ok=(s['halt_m']<3)),
        rw("رسیده به ۱۰%", f"{s['target_m']}"),
        bot,"",
    ]

    # ── جدول ماه به ماه ──
    lines.append(box("جدول ماه به ماه"))
    lines.append(
        f"  {'ماه':>7}  {'موجودی':>10}  {'PnL':>9}  "
        f"{'Ret%':>6}  {'#':>3}  {'WR%':>5}  "
        f"{'DD%':>6}  وضعیت")
    lines.append("  "+"─"*(W-3))

    cumret = 0
    for _, mr in s['monthly'].iterrows():
        cumret += mr['ret_pct']
        lines.append(
            f"  {mr['period']:>7}  ${mr['start_eq']:>9,.0f}  "
            f"${mr['pnl']:>+8,.2f}  {mr['ret_pct']:>+5.1f}%  "
            f"{mr['trades']:>3}  {mr['wr']:>4.0f}%  "
            f"{mr['max_dd']:>5.1f}%  {mr['status']}")
    lines+=[bot,""]

    # ── سالانه ──
    s['trades']['yr'] = s['trades']['entry_ts'].dt.year
    yr_g = (s['trades'].groupby('yr')
            .agg(n=('pnl','count'),pnl=('pnl','sum'),
                 wins=('pnl',lambda x:(x>0).sum()))
            .reset_index())
    yr_g['wr']  = yr_g['wins']/yr_g['n']*100
    yr_g['ret'] = yr_g['pnl']/Config.initial_balance*100

    # سود سالانه واقعی (بر اساس موجودی شروع سال)
    eq_start_year = {}
    prev_yr = None
    for _, mr in s['monthly'].iterrows():
        yr = int(mr['period'][:4])
        if yr != prev_yr:
            eq_start_year[yr] = mr['start_eq']
            prev_yr = yr

    lines.append(box("گزارش سالانه"))
    lines.append(f"  {'سال':>5}  {'#':>5}  {'WR%':>6}  "
                 f"{'PnL':>10}  {'Ret%':>7}  {'واقعی%':>8}")
    lines.append("  "+"─"*(W-3))
    for _,yr in yr_g.iterrows():
        y    = int(yr['yr'])
        base = eq_start_year.get(y, Config.initial_balance)
        real_ret = yr['pnl']/base*100
        lines.append(
            f"  {y:>5}  {int(yr['n']):>5}  {yr['wr']:>5.1f}%  "
            f"${yr['pnl']:>9.2f}  {yr['ret']:>+6.1f}%  "
            f"{real_ret:>+7.1f}%")
    lines.append(bot)

    out="\n".join(lines)
    print(out)
    return out


def print_comparison(results):
    W=74; SEP="═"*W
    lines=["",SEP,
           "  ▌  COMPARISON — Monthly Regime-Aware Simulation  ▐",SEP,
           f"  {'نام':<14} {'Ann%':>7} {'AvgM%':>6} {'DD%':>7} "
           f"{'PF':>5} {'WR%':>5} {'M+':>4} {'M-':>4} "
           f"{'Halt':>5} {'10%+':>5}  نتیجه",
           "  "+"─"*(W-3)]

    for s in results:
        ppm  = s['avg_mret']
        ok   = (s['total_ret']>0 and s['pf']>1.3
                and abs(s['max_dd'])<8 and ppm>8
                and s['prof_m']>s['loss_m'])
        flag = "✅ PASS" if ok else "❌ FAIL"
        pf_s = f"{s['pf']:.2f}" if s['pf']!=float('inf') else "  ∞"
        lines.append(
            f"  {s['name']:<14} {s['ann_ret']:>+6.1f}% {ppm:>+5.1f}% "
            f"{s['max_dd']:>6.1f}% {pf_s:>5} {s['win_r']:>4.1f}% "
            f"{s['prof_m']:>4} {s['loss_m']:>4} "
            f"{s['halt_m']:>5} {s['target_m']:>5}  {flag}")

    lines+=["  "+"─"*(W-3),""]
    good=[s for s in results
          if s['total_ret']>0 and s['pf']>1.3
          and abs(s['max_dd'])<8 and s['avg_mret']>8
          and s['prof_m']>s['loss_m']]
    if good:
        lines.append("  🏆 PROP READY:")
        for s in sorted(good,key=lambda x:x['ann_ret'],reverse=True):
            lines.append(
                f"     ✅ {s['name']:<14}  "
                f"سالانه={s['ann_ret']:>+.1f}%  "
                f"ماهانه={s['avg_mret']:>+.1f}%  "
                f"DD={s['max_dd']:.1f}%  "
                f"ماه‌سودده={s['prof_m']}")
    else:
        # بهترین کاندید
        best = max(results, key=lambda x: x['ann_ret'])
        lines.append("  ⚠️  هیچ استراتژی هنوز کاملاً PROP READY نیست")
        lines.append(f"  📊 بهترین: {best['name']} "
                     f"(Ann={best['ann_ret']:>+.1f}%, "
                     f"AvgMonth={best['avg_mret']:>+.1f}%)")
    lines+=["",SEP]
    out="\n".join(lines); print(out); return out


def save_outputs(results):
    rows=[
        ["REGIME-AWARE MONTHLY SIMULATION"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [""],
        ["Strategy","FinalEq","TotalRet%","AnnRet%","AvgMonth%",
         "MaxDD%","PF","WinRate%","RR","Exp$","Trades",
         "ProfMonths","LossMonths","HaltedM","TargetM","Status"],
    ]
    for s in results:
        ok=(s['total_ret']>0 and s['pf']>1.3
            and abs(s['max_dd'])<8 and s['avg_mret']>8)
        pf_v=round(s['pf'],2) if s['pf']!=float('inf') else 999
        rows.append([s['name'],round(s['final_eq'],2),round(s['total_ret'],2),
                     round(s['ann_ret'],2),round(s['avg_mret'],2),
                     round(s['max_dd'],2),pf_v,round(s['win_r'],1),
                     round(s['rr'],2),round(s['exp'],2),len(s['trades']),
                     s['prof_m'],s['loss_m'],s['halt_m'],s['target_m'],
                     "PASS" if ok else "FAIL"])

    for s in results:
        rows+=[[""],
               [f"=== MONTHLY: {s['name']} ==="],
               ["Month","StartEq","EndEq","PnL","Ret%",
                "Trades","WinRate%","MaxDD%","Status"]]
        for _,mr in s['monthly'].iterrows():
            rows.append([mr['period'],round(mr['start_eq'],2),
                         round(mr['end_eq'],2),round(mr['pnl'],2),
                         round(mr['ret_pct'],2),mr['trades'],
                         round(mr['wr'],1),round(mr['max_dd'],2),
                         mr['status']])

    pd.DataFrame(rows).to_csv("Report.csv",index=False,
                               header=False,encoding="utf-8-sig")

    for s in results:
        eq_df=pd.DataFrame({'ts':s['eq_curve_ts'],'equity':s['eq_curve']})
        eq_df['dd']=(
            (eq_df['equity']-eq_df['equity'].cummax())
            /eq_df['equity'].cummax()*100).round(4)
        eq_df.to_csv(f"equity_{s['name']}.csv",
                     index=False,encoding="utf-8-sig")

    print(f"\n✅ فایل‌ها:")
    print(f"   → Report.csv")
    for s in results:
        print(f"   → equity_{s['name']}.csv")


# ================================================================== #
if __name__ == "__main__":
    df = load_data()
    print("\n"+"═"*74)
    print("  REGIME-AWARE MONTHLY SIMULATION")
    print("═"*74)

    signals = compute_all_signals(df)

    strats=[
        ('CorrArb',    *signals['CorrArb']),
        ('TrendEMA',   *signals['TrendEMA']),
        ('AsianBreak', *signals['AsianBreak']),
        ('MeanRev',    *signals['MeanRev']),
    ]

    all_results=[]; all_texts=[]
    for name,sig,sl,tp,z in strats:
        t0=datetime.now()
        print(f"\n  ▶ {name}...",end="",flush=True)
        trades,ml,eqc,eqts=run_monthly_backtest(df,name,sig,sl,tp,z)
        dt=(datetime.now()-t0).total_seconds()
        print(f" {dt:.1f}s | {len(trades)} معامله")
        if not trades: continue
        st=compute_stats(trades,ml,eqc,eqts,name)
        if st:
            all_results.append(st)
            all_texts.append(print_report(st))

    if all_results:
        all_texts.append(print_comparison(all_results))
        with open("Report.txt","w",encoding="utf-8") as f:
            f.write("\n".join(all_texts))
        save_outputs(all_results)
