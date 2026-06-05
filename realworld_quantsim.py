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
    risk_per_trade_pct = 0.008      # 0.8% per trade
    max_daily_loss_pct = 0.04       # 4% daily → پراپ استاندارد
    max_total_dd_pct   = 0.08       # 8% max DD → پراپ استاندارد
    monthly_target_pct = 0.10       # 10% هدف ماهانه
    spread_eur_pips    = 1.0
    spread_gbp_pips    = 1.2
    commission_per_lot = 6.0
    pip                = 0.0001
    lot_size           = 100_000
    max_lot            = 1.5
    warmup             = 300        # کندل‌های warmup


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
    up    = high.diff()
    down  = -low.diff()
    dm_p  = up.where((up > down) & (up > 0), 0.0)
    dm_n  = down.where((down > up) & (down > 0), 0.0)
    tr    = calc_atr(high, low, close, 1)
    atr_s = tr.rolling(period).sum()
    di_p  = 100 * dm_p.rolling(period).sum() / atr_s.replace(0, np.nan)
    di_n  = 100 * dm_n.rolling(period).sum() / atr_s.replace(0, np.nan)
    dx    = (abs(di_p - di_n) / (di_p + di_n).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()

def calc_macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast,   adjust=False).mean()
    es = close.ewm(span=slow,   adjust=False).mean()
    m  = ef - es
    s  = m.ewm(span=signal,     adjust=False).mean()
    return m, s, m - s

def lot_size_calc(equity: float, sl_pips: float) -> float:
    if sl_pips <= 0:
        return 0.01
    risk_usd = equity * Config.risk_per_trade_pct
    lot = risk_usd / (sl_pips * Config.pip * Config.lot_size)
    return round(float(np.clip(lot, 0.01, Config.max_lot)), 2)


# ================================================================== #
#   محاسبه همه سیگنال‌ها روی کل دیتا (یک‌بار - سریع)              #
# ================================================================== #
def compute_all_signals(df: pd.DataFrame) -> dict:
    """
    همه اندیکاتورها و سیگنال‌ها را یک‌بار روی کل دیتا حساب می‌کنیم.
    این مهم است چون:
    1. EMA200 برای محاسبه درست نیاز به تاریخچه کامل دارد
    2. سریع‌تر است
    """
    print("  محاسبه اندیکاتورها...", end="", flush=True)

    c_e = df['c_eur']
    h_e = df['h_eur']
    l_e = df['l_eur']
    c_g = df['c_gbp']

    # ── اندیکاتورهای مشترک ──
    atr    = calc_atr(h_e, l_e, c_e, 14)
    rsi    = calc_rsi(c_e, 14)
    adx    = calc_adx(h_e, l_e, c_e, 14)
    ema9   = c_e.ewm(span=9,   adjust=False).mean()
    ema21  = c_e.ewm(span=21,  adjust=False).mean()
    ema50  = c_e.ewm(span=50,  adjust=False).mean()
    ema200 = c_e.ewm(span=200, adjust=False).mean()
    _, _, macd_hist = calc_macd(c_e)

    hour    = pd.Series(df.index.hour,    index=df.index)
    weekday = pd.Series(df.index.weekday, index=df.index)
    date_s  = pd.Series(df.index.date,    index=df.index)

    # ════════════════════════════════════════════
    #  SIGNAL 1: CorrArb
    # ════════════════════════════════════════════
    eurgbp  = c_e / c_g
    period  = 96
    z_mean  = eurgbp.rolling(period).mean()
    z_std   = eurgbp.rolling(period).std()
    z_score = (eurgbp - z_mean) / z_std.replace(0, np.nan)
    std_ok  = z_std > z_std.rolling(period * 5).mean() * 0.3
    adx_ok  = adx < 25
    time_ok = hour.between(7, 19)
    rsi_ok_s = rsi > 55
    rsi_ok_l = rsi < 45

    sig_arb = pd.Series(0, index=df.index)
    sig_arb[(z_score >  2.3) & std_ok & adx_ok & time_ok & rsi_ok_s] = -1
    sig_arb[(z_score < -2.3) & std_ok & adx_ok & time_ok & rsi_ok_l] =  1
    sig_arb = sig_arb.where(sig_arb != sig_arb.shift(), 0)

    # SL/TP ثابت برای CorrArb
    sl_arb = np.where(sig_arb != 0, 20.0, 0.0)
    tp_arb = np.where(sig_arb != 0, 40.0, 0.0)

    # ════════════════════════════════════════════
    #  SIGNAL 2: Asian Breakout
    # ════════════════════════════════════════════
    asian_mask = hour.between(2, 6)
    asian_df   = df[asian_mask].copy()
    asian_df['date'] = asian_df.index.date
    rng_day = asian_df.groupby('date').agg(
        ah=('h_eur','max'), al=('l_eur','min'))
    rng_day['rng_pips'] = (rng_day['ah'] - rng_day['al']) / Config.pip

    d_temp = df.copy()
    d_temp['date'] = d_temp.index.date
    d_temp = d_temp.join(rng_day, on='date')

    london   = hour.between(8, 10)
    rng_ok   = d_temp['rng_pips'].between(12, 40)
    day_ok   = weekday.between(0, 3)
    trend_up = c_e > ema200
    trend_dn = c_e < ema200

    above = (c_e > d_temp['ah']).astype(int)
    below = (c_e < d_temp['al']).astype(int)
    above2 = (above + above.shift(1)) >= 2
    below2 = (below + below.shift(1)) >= 2

    raw_ab = pd.Series(0, index=df.index)
    raw_ab[london & rng_ok & day_ok & (adx>18) & above2 & trend_up] =  1
    raw_ab[london & rng_ok & day_ok & (adx>18) & below2 & trend_dn] = -1

    nz_ab = raw_ab[raw_ab != 0]
    fi_ab = nz_ab.groupby(nz_ab.index.date).head(1).index
    sig_ab = pd.Series(0, index=df.index)
    sig_ab[fi_ab] = raw_ab[fi_ab]

    rng_pips_arr = d_temp['rng_pips'].fillna(20).values
    sl_ab = np.where(sig_ab != 0,
                     np.maximum(15, rng_pips_arr + 3), 0.0)
    tp_ab = np.where(sig_ab != 0,
                     np.maximum(sl_ab * 2, rng_pips_arr * 3.0), 0.0)

    # ════════════════════════════════════════════
    #  SIGNAL 3: EMA Crossover
    # ════════════════════════════════════════════
    cross_up   = (ema9 > ema21) & (ema9.shift(1) <= ema21.shift(1))
    cross_down = (ema9 < ema21) & (ema9.shift(1) >= ema21.shift(1))
    active     = hour.between(8, 16)

    raw_ec = pd.Series(0, index=df.index)
    raw_ec[active & cross_up   & (ema21>ema50) & (c_e>ema200)
           & (adx>20) & rsi.between(45,70) & (macd_hist>0)] =  1
    raw_ec[active & cross_down & (ema21<ema50) & (c_e<ema200)
           & (adx>20) & rsi.between(30,55) & (macd_hist<0)] = -1

    nz_ec = raw_ec[raw_ec != 0]
    fi_ec = nz_ec.groupby(nz_ec.index.date).head(1).index
    sig_ec = pd.Series(0, index=df.index)
    sig_ec[fi_ec] = raw_ec[fi_ec]

    dist_ema21 = (c_e - ema21).abs() / Config.pip
    sl_ec = np.where(sig_ec != 0,
                     np.maximum(15, dist_ema21.values +
                                atr.values / Config.pip * 0.5), 0.0)
    tp_ec = np.where(sig_ec != 0, sl_ec * 2.5, 0.0)

    print(" ✓")

    return {
        'CorrArb':    (sig_arb, sl_arb, tp_arb, z_score),
        'AsianBreak': (sig_ab,  sl_ab,  tp_ab,  None),
        'EMACross':   (sig_ec,  sl_ec,  tp_ec,  None),
        'ema21':      ema21,
        'atr':        atr,
    }


# ================================================================== #
#   موتور بک‌تست ماهانه                                             #
#   ─────────────────────────────────────────────────────────────── #
#   منطق کار:                                                        #
#   1. کل دیتا را به ماه‌ها تقسیم می‌کنیم                           #
#   2. هر ماه با موجودی پایان ماه قبل شروع می‌شه                    #
#   3. DD و Daily Loss نسبت به شروع همان ماه حساب می‌شه             #
#   4. اگر ماهی به DD 8% رسید → اون ماه بسته می‌شه                 #
#      ماه بعد با همان موجودی (کاهش یافته) شروع می‌شه              #
#   5. اگر ماهی به 10% سود رسید → همچنان ادامه می‌ده               #
#      (فقط DD را reset می‌کنه، چون پول واقعی سودت است)            #
# ================================================================== #
def run_monthly_backtest(
    df: pd.DataFrame,
    signals: dict,
    strategy_name: str,
    sig_series: pd.Series,
    sl_arr: np.ndarray,
    tp_arr: np.ndarray,
    z_series=None,          # فقط برای CorrArb
) -> tuple:
    """
    بک‌تست ماهانه واقعی‌محور:
    - هر ماه مستقل است از نظر DD tracking
    - موجودی از ماه قبل منتقل می‌شود
    - معاملات باز از ماه قبل به ماه بعد منتقل می‌شوند
    """
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

    # ── شناسایی ماه‌ها ──
    months = pd.Series(ts_a).dt.to_period('M').unique()

    # ── وضعیت کلی ──
    equity       = Config.initial_balance
    all_trades   = []
    monthly_log  = []   # یک dict برای هر ماه

    # ── پوزیشن باز (بین ماه‌ها منتقل می‌شود) ──
    open_pos = None   # dict یا None

    # ── برای equity curve ──
    eq_curve    = [equity]
    eq_curve_ts = [None]

    for month_period in months:
        # ── بازه این ماه ──
        month_mask = (pd.Series(ts_a).dt.to_period('M') == month_period).values
        month_bars = np.where(month_mask)[0]

        if len(month_bars) == 0:
            continue

        bar_start = month_bars[0]
        bar_end   = month_bars[-1]

        # ── اگر warmup هنوز تمام نشده این ماه را رد کن ──
        if bar_end < Config.warmup:
            continue

        # ── شروع ماه: ثبت موجودی ──
        month_start_eq  = equity
        month_peak      = equity        # peak این ماه
        month_day_eq    = equity        # برای daily loss
        month_halted    = False
        month_halt_r    = "ادامه"
        month_trades    = []
        cur_day         = None

        # ── سیگنال‌های این ماه ──
        sig_set = {
            x for x in np.where(sig_a != 0)[0]
            if bar_start <= x <= bar_end and x >= Config.warmup
        }

        for bar in range(max(bar_start, Config.warmup), bar_end + 1):
            day = ts_a[bar].date()

            # ── تغییر روز ──
            if day != cur_day:
                cur_day      = day
                month_day_eq = equity
                # ریست daily halt (اگر daily بود)
                if month_halted and "Daily" in month_halt_r:
                    month_halted = False
                    month_halt_r = "ادامه"

            # ── اگر این ماه متوقف شده ──
            if month_halted:
                # پوزیشن باز را ببند
                if open_pos is not None:
                    cp   = close_a[bar]
                    raw  = open_pos['dir'] * (cp - open_pos['entry']) * open_pos['lot'] * ls
                    cost = sp * pip * open_pos['lot'] * ls + comm * open_pos['lot']
                    pnl  = raw - cost
                    equity += pnl
                    month_peak = max(month_peak, equity)
                    eq_curve.append(round(equity, 4))
                    eq_curve_ts.append(ts_a[bar])
                    rec = {**open_pos, 'exit': cp, 'exit_ts': ts_a[bar],
                           'pnl': pnl, 'status': 'month_halt'}
                    month_trades.append(rec)
                    all_trades.append(rec)
                    open_pos = None
                break  # از این ماه خارج شو

            # ── بررسی پوزیشن باز ──
            if open_pos is not None:
                hi = high_a[bar]
                lo = low_a[bar]
                cp = close_a[bar]
                d  = open_pos['dir']
                ep = open_pos['entry']
                sl = open_pos['sl']
                tp = open_pos['tp']

                hit_sl = (d ==  1 and lo <= sl) or (d == -1 and hi >= sl)
                hit_tp = (d ==  1 and hi >= tp) or (d == -1 and lo <= tp)

                # Z-exit برای CorrArb
                if z_a is not None:
                    z_now = z_a[bar]
                    if not np.isnan(z_now) and abs(z_now) < 0.25:
                        hit_tp = True

                # Trailing SL
                move    = d * (cp - ep)
                tp_dist = abs(tp - ep)
                if tp_dist > 0:
                    if move > tp_dist * 0.6:
                        be = ep + d * tp_dist * 0.15
                        if d == 1:
                            open_pos['sl'] = max(open_pos['sl'], be)
                        else:
                            open_pos['sl'] = min(open_pos['sl'], be)
                    if move > tp_dist * 0.85:
                        lock = ep + d * tp_dist * 0.5
                        if d == 1:
                            open_pos['sl'] = max(open_pos['sl'], lock)
                        else:
                            open_pos['sl'] = min(open_pos['sl'], lock)

                # Time stop: ۴ روز
                elapsed_bars = bar - open_pos['entry_bar']
                if elapsed_bars >= 384 and not hit_tp:  # 96×4
                    raw  = d * (cp - ep) * open_pos['lot'] * ls
                    cost = sp * pip * open_pos['lot'] * ls + comm * open_pos['lot']
                    pnl  = raw - cost
                    equity += pnl
                    month_peak = max(month_peak, equity)
                    eq_curve.append(round(equity, 4))
                    eq_curve_ts.append(ts_a[bar])
                    rec = {**open_pos, 'exit': cp, 'exit_ts': ts_a[bar],
                           'pnl': pnl, 'status': 'TimeStop'}
                    month_trades.append(rec)
                    all_trades.append(rec)
                    open_pos = None

                    # بررسی DD
                    dd_d = (equity - month_day_eq) / month_day_eq
                    dd_m = (equity - month_start_eq) / month_start_eq
                    dd_pk = (equity - month_peak) / month_peak
                    if dd_d <= -Config.max_daily_loss_pct:
                        month_halted = True
                        month_halt_r = f"Daily {dd_d*100:.1f}%"
                    elif dd_pk <= -Config.max_total_dd_pct:
                        month_halted = True
                        month_halt_r = f"MaxDD {dd_pk*100:.1f}%"
                    continue

                exit_r = exit_p = None
                if hit_sl: exit_r, exit_p = 'SL', open_pos['sl']
                elif hit_tp: exit_r, exit_p = 'TP', open_pos['tp']

                if exit_r:
                    raw  = d * (exit_p - ep) * open_pos['lot'] * ls
                    cost = sp * pip * open_pos['lot'] * ls + comm * open_pos['lot']
                    pnl  = raw - cost
                    equity += pnl
                    month_peak = max(month_peak, equity)
                    eq_curve.append(round(equity, 4))
                    eq_curve_ts.append(ts_a[bar])
                    rec = {**open_pos, 'exit': exit_p, 'exit_ts': ts_a[bar],
                           'pnl': pnl, 'status': exit_r}
                    month_trades.append(rec)
                    all_trades.append(rec)
                    open_pos = None

                    # بررسی DD
                    dd_d  = (equity - month_day_eq)   / month_day_eq
                    dd_pk = (equity - month_peak)      / month_peak
                    if dd_d <= -Config.max_daily_loss_pct:
                        month_halted = True
                        month_halt_r = f"Daily {dd_d*100:.1f}%"
                    elif dd_pk <= -Config.max_total_dd_pct:
                        month_halted = True
                        month_halt_r = f"MaxDD {dd_pk*100:.1f}%"

            # ── ورود ──
            if open_pos is None and not month_halted and bar in sig_set:
                sv   = int(sig_a[bar])
                slp  = float(sl_arr[bar])
                tpp  = float(tp_arr[bar])
                if slp > 0 and tpp > 0 and not np.isnan(slp) and not np.isnan(tpp):
                    lot         = lot_size_calc(equity, slp)
                    half_sp     = sp * pip / 2
                    entry_price = close_a[bar] + sv * half_sp
                    open_pos = dict(
                        strategy=strategy_name, symbol='EUR',
                        dir=sv, lot=lot,
                        entry=entry_price,
                        sl=entry_price - sv * slp * pip,
                        tp=entry_price + sv * tpp * pip,
                        entry_ts=ts_a[bar],
                        entry_bar=bar,
                    )

        # ── پایان ماه: ثبت آمار ──
        month_pnl = equity - month_start_eq
        month_ret = month_pnl / month_start_eq * 100

        # وضعیت ماه
        if month_halted:
            status = f"HALTED ({month_halt_r})"
        elif month_ret >= Config.monthly_target_pct * 100:
            status = f"TARGET ✅ ({month_ret:>+.1f}%)"
        else:
            status = f"Normal ({month_ret:>+.1f}%)"

        monthly_log.append({
            'period':     str(month_period),
            'start_eq':   round(month_start_eq, 2),
            'end_eq':     round(equity, 2),
            'pnl':        round(month_pnl, 2),
            'ret_pct':    round(month_ret, 2),
            'trades':     len(month_trades),
            'wins':       sum(1 for t in month_trades if t.get('pnl',0) > 0),
            'status':     status,
            'halted':     month_halted,
            'halt_r':     month_halt_r if month_halted else '',
            'max_dd_month': round(
                min((equity - month_peak) / month_peak * 100, 0), 2),
        })

    return all_trades, monthly_log, eq_curve, eq_curve_ts


# ================================================================== #
#               آمار کامل                                           #
# ================================================================== #
def compute_full_stats(
    all_trades: list,
    monthly_log: list,
    eq_curve: list,
    eq_curve_ts: list,
    strategy_name: str,
) -> dict:

    if not all_trades:
        return None

    t = pd.DataFrame(all_trades)
    t['pnl']          = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts']-t['entry_ts']).dt.total_seconds()/60

    ml = pd.DataFrame(monthly_log)

    final_eq  = eq_curve[-1]
    total_pnl = final_eq - Config.initial_balance
    total_ret = total_pnl / Config.initial_balance * 100

    start_d    = t['entry_ts'].min()
    end_d      = t['exit_ts'].max()
    total_days = max((end_d - start_d).days, 1)
    ann_ret    = ((final_eq/Config.initial_balance)**(365.25/total_days)-1)*100

    win_t  = t[t['pnl'] > 0]
    loss_t = t[t['pnl'] < 0]
    win_r  = len(win_t)/len(t)*100 if len(t) > 0 else 0
    avg_w  = win_t['pnl'].mean()  if len(win_t)  > 0 else 0
    avg_l  = loss_t['pnl'].mean() if len(loss_t) > 0 else 0
    gw     = win_t['pnl'].sum()
    gl     = abs(loss_t['pnl'].sum())
    pf     = gw / gl if gl > 0 else float('inf')
    exp_v  = t['pnl'].mean()
    rr     = abs(avg_w/avg_l) if avg_l != 0 else 0

    # equity curve stats
    eq_s   = pd.Series(eq_curve)
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()
    r_ret  = eq_s.pct_change().dropna()
    sharpe = (r_ret.mean()/r_ret.std()*np.sqrt(252*96)) if r_ret.std()>0 else 0
    neg    = r_ret[r_ret<0]
    ds     = neg.std() if len(neg)>0 else 1e-10
    sortino= (r_ret.mean()/ds*np.sqrt(252*96)) if ds>0 else 0
    calmar = (final_eq/Config.initial_balance-1)/abs(max_dd/100) if max_dd!=0 else 0

    # آمار ماهانه
    profitable_m = (ml['pnl'] > 0).sum()
    losing_m     = (ml['pnl'] <= 0).sum()
    halted_m     = ml['halted'].sum()
    target_m     = (ml['ret_pct'] >= Config.monthly_target_pct*100).sum()
    avg_monthly_ret = ml['ret_pct'].mean()
    best_month   = ml['pnl'].max()
    worst_month  = ml['pnl'].min()

    return dict(
        name=strategy_name,
        trades=t, monthly=ml,
        eq_curve=eq_curve, eq_curve_ts=eq_curve_ts,
        final_eq=final_eq, total_pnl=total_pnl,
        total_ret=total_ret, ann_ret=ann_ret,
        total_days=total_days,
        win_r=win_r, avg_w=avg_w, avg_l=avg_l,
        pf=pf, exp=exp_v, rr=rr,
        max_dd=max_dd, sharpe=sharpe, sortino=sortino, calmar=calmar,
        profitable_months=int(profitable_m),
        losing_months=int(losing_m),
        halted_months=int(halted_m),
        target_months=int(target_m),
        avg_monthly_ret=avg_monthly_ret,
        best_month=best_month, worst_month=worst_month,
        best_trade=t['pnl'].max(), worst_trade=t['pnl'].min(),
        avg_dur=t['duration_min'].mean(),
    )


# ================================================================== #
#               گزارش‌ساز حرفه‌ای                                   #
# ================================================================== #
def print_report(s: dict) -> str:
    W   = 74
    SEP = "═" * W

    def rw(label, value, ok=None):
        lbl  = f"  {label}"
        val  = str(value)
        mark = ""
        if ok is True:  mark = "  ✅"
        if ok is False: mark = "  ❌"
        dots = "·" * max(2, W-len(lbl)-len(val)-len(mark)-2)
        return f"{lbl} {dots} {val}{mark}"

    def box(title):
        inner = f"─ {title} "
        return "┌" + inner + "─"*(W-len(inner)-1) + "┐"

    bot = "└" + "─"*(W-1) + "┘"

    # آیا Prop Ready است؟
    ppm_pct  = s['avg_monthly_ret']
    prop_ok  = (
        s['total_ret'] > 0 and
        s['pf'] > 1.3 and
        abs(s['max_dd']) < 8 and
        ppm_pct > 8 and
        s['profitable_months'] > s['losing_months']
    )
    status = "✅ PROP READY" if prop_ok else "⚠️  نیاز به بهبود"

    lines = [
        "", SEP,
        f"  ▌  Strategy: {s['name']}   {status}  ▐",
        f"  ▌  دوره: {s['trades']['entry_ts'].min().date()} "
        f"→ {s['trades']['exit_ts'].max().date()}  ({s['total_days']} روز)  ▐",
        SEP, "",

        box("نتایج مالی کل دوره"),
        rw("موجودی اولیه",        f"${Config.initial_balance:>12,.2f}"),
        rw("موجودی نهایی",        f"${s['final_eq']:>12,.2f}"),
        rw("سود/زیان کل",        f"${s['total_pnl']:>+12,.2f}"),
        rw("بازده کل",            f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه",        f"{s['ann_ret']:>+.2f}%"),
        rw("میانگین سود ماهانه",  f"{s['avg_monthly_ret']:>+.2f}%",
           ok=(s['avg_monthly_ret'] > 8)),
        rw("بهترین ماه",          f"${s['best_month']:>+.2f}"),
        rw("بدترین ماه",          f"${s['worst_month']:>+.2f}"),
        bot, "",

        box("معیارهای ریسک (پراپ)"),
        rw("Max Drawdown",         f"{s['max_dd']:.2f}%",
           ok=(abs(s['max_dd']) < 8)),
        rw("Sharpe Ratio",         f"{s['sharpe']:.2f}"),
        rw("Sortino Ratio",        f"{s['sortino']:.2f}"),
        rw("Calmar Ratio",         f"{s['calmar']:.2f}"),
        rw("Profit Factor",        f"{s['pf']:.2f}",
           ok=(s['pf'] > 1.3)),
        bot, "",

        box("معاملات"),
        rw("تعداد کل",             f"{len(s['trades']):,}"),
        rw("Win Rate",              f"{s['win_r']:.1f}%"),
        rw("Avg Win",               f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",              f"${s['avg_l']:>+.2f}"),
        rw("RR واقعی",             f"{s['rr']:.2f}"),
        rw("Expectancy/trade",      f"${s['exp']:>+.2f}"),
        rw("بهترین معامله",        f"${s['best_trade']:>+.2f}"),
        rw("بدترین معامله",        f"${s['worst_trade']:>+.2f}"),
        rw("مدت میانگین",          f"{s['avg_dur']:.0f} min"),
        bot, "",

        box("آمار ماهانه"),
        rw("کل ماه‌ها",            f"{s['profitable_months']+s['losing_months']}"),
        rw("ماه‌های سودده",        f"{s['profitable_months']}",
           ok=(s['profitable_months'] > s['losing_months'])),
        rw("ماه‌های ضررده",        f"{s['losing_months']}"),
        rw("ماه‌های Halted (DD)",   f"{s['halted_months']}"),
        rw("ماه‌های رسیده به ۱۰%", f"{s['target_months']}"),
        bot, "",
    ]

    # ── جدول ماهانه دقیق ──
    lines.append(box("جدول ماه به ماه (سیمولاسیون واقعی)"))
    lines.append(
        f"  {'ماه':>7}  {'موجودی شروع':>13}  {'PnL':>9}  "
        f"{'Ret%':>6}  {'#T':>3}  {'Win%':>5}  "
        f"{'MaxDD%':>7}  وضعیت")
    lines.append("  " + "─"*(W-3))

    for _, mr in s['monthly'].iterrows():
        trades_m = len(s['trades'][
            s['trades']['entry_ts'].dt.to_period('M').astype(str) == mr['period']
        ])
        wins_m   = mr['wins']
        wr_m     = wins_m / mr['trades'] * 100 if mr['trades'] > 0 else 0
        arrow    = "▲" if mr['pnl'] >= 0 else "▼"
        # رنگ وضعیت
        st = mr['status']
        if 'TARGET' in st:    st_mark = "🎯"
        elif 'HALTED' in st:  st_mark = "🛑"
        elif mr['pnl'] >= 0:  st_mark = "✅"
        else:                  st_mark = "❌"

        lines.append(
            f"  {mr['period']:>7}  ${mr['start_eq']:>12,.2f}  "
            f"${mr['pnl']:>+8,.2f}  {mr['ret_pct']:>+5.1f}%  "
            f"{mr['trades']:>3}  {wr_m:>4.0f}%  "
            f"{mr['max_dd_month']:>6.1f}%  "
            f"{st_mark} {arrow}")

    lines += [bot, ""]

    # ── گزارش سالانه ──
    s['trades']['yr'] = s['trades']['entry_ts'].dt.year
    yearly = (s['trades'].groupby('yr')
              .agg(n=('pnl','count'), pnl=('pnl','sum'),
                   wins=('pnl', lambda x: (x>0).sum()))
              .reset_index())
    yearly['wr']  = yearly['wins']/yearly['n']*100
    yearly['ret'] = yearly['pnl']/Config.initial_balance*100

    lines.append(box("گزارش سالانه"))
    lines.append(
        f"  {'سال':>5}  {'#':>5}  {'Win%':>6}  "
        f"{'PnL':>10}  {'Ret%':>7}")
    lines.append("  " + "─"*(W-3))
    for _, yr in yearly.iterrows():
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>5.1f}%  ${yr['pnl']:>9.2f}  "
            f"{yr['ret']:>+6.1f}%")
    lines.append(bot)

    out = "\n".join(lines)
    print(out)
    return out


def print_comparison(results: list) -> str:
    W   = 74
    SEP = "═" * W

    lines = [
        "", SEP,
        "  ▌  STRATEGY COMPARISON — Monthly Simulation  ▐",
        SEP,
        f"  {'نام':<16} {'Ann%':>8} {'AvgM%':>6} {'DD%':>7} "
        f"{'PF':>5} {'Win%':>6} {'M+':>4} {'M-':>4} "
        f"{'Halt':>5} {'10%+':>5}  وضعیت",
        "  " + "─"*(W-3),
    ]

    for s in results:
        ppm = s['avg_monthly_ret']
        flag = "✅" if (s['total_ret']>0 and s['pf']>1.3
                        and abs(s['max_dd'])<8 and ppm>8
                        and s['profitable_months']>s['losing_months']) else "❌"
        pf_s = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "  ∞"
        lines.append(
            f"  {s['name']:<16} {s['ann_ret']:>+7.1f}% {ppm:>+5.1f}% "
            f"{s['max_dd']:>6.1f}% {pf_s:>5} {s['win_r']:>5.1f}% "
            f"{s['profitable_months']:>4} {s['losing_months']:>4} "
            f"{s['halted_months']:>5} {s['target_months']:>5}  {flag}")

    lines += ["  " + "─"*(W-3), ""]

    good = [s for s in results
            if s['total_ret']>0 and s['pf']>1.3
            and abs(s['max_dd'])<8
            and s['avg_monthly_ret']>8
            and s['profitable_months']>s['losing_months']]

    if good:
        lines.append("  🏆 PROP READY:")
        for s in sorted(good, key=lambda x: x['ann_ret'], reverse=True):
            lines.append(
                f"     ✅ {s['name']:<16}  "
                f"سالانه={s['ann_ret']:>+.1f}%  "
                f"ماهانه avg={s['avg_monthly_ret']:>+.1f}%  "
                f"DD={s['max_dd']:.1f}%  "
                f"ماه‌های سودده={s['profitable_months']}")
    else:
        lines.append("  ⚠️  هنوز هیچ استراتژی کاملاً آماده پراپ نیست")

    lines += ["", SEP]
    out = "\n".join(lines)
    print(out)
    return out


def save_outputs(results: list):
    # CSV اصلی
    rows = [
        ["MONTHLY SIMULATION BACKTEST"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [f"Risk/trade: {Config.risk_per_trade_pct*100}%  "
         f"MaxDD: {Config.max_total_dd_pct*100}%  "
         f"MaxLot: {Config.max_lot}"],
        [""],
        ["Strategy","FinalEq","TotalPnL","TotalRet%","AnnRet%",
         "MaxDD%","Sharpe","PF","WinRate%","RR","Exp$",
         "Trades","AvgMonthly%","ProfMonths","LossMonths",
         "HaltedMonths","TargetMonths","Status"],
    ]
    for s in results:
        ppm  = s['avg_monthly_ret']
        flag = ("PROP_READY"
                if (s['total_ret']>0 and s['pf']>1.3
                    and abs(s['max_dd'])<8 and ppm>8)
                else "NEEDS_WORK")
        pf_v = round(s['pf'],2) if s['pf']!=float('inf') else 999
        rows.append([
            s['name'], round(s['final_eq'],2), round(s['total_pnl'],2),
            round(s['total_ret'],2), round(s['ann_ret'],2),
            round(s['max_dd'],2), round(s['sharpe'],2), pf_v,
            round(s['win_r'],1), round(s['rr'],2), round(s['exp'],2),
            len(s['trades']), round(ppm,2),
            s['profitable_months'], s['losing_months'],
            s['halted_months'], s['target_months'], flag,
        ])

    # جدول ماهانه هر استراتژی
    for s in results:
        rows += [[""], [f"=== MONTHLY TABLE: {s['name']} ==="],
                 ["Month","StartEq","EndEq","PnL","Ret%",
                  "Trades","Wins","WinRate%","MaxDD%","Status"]]
        for _, mr in s['monthly'].iterrows():
            wr_m = mr['wins']/mr['trades']*100 if mr['trades']>0 else 0
            rows.append([
                mr['period'], round(mr['start_eq'],2),
                round(mr['end_eq'],2), round(mr['pnl'],2),
                round(mr['ret_pct'],2), mr['trades'],
                mr['wins'], round(wr_m,1),
                round(mr['max_dd_month'],2), mr['status'],
            ])

    pd.DataFrame(rows).to_csv(
        "Monthly_Simulation_Report.csv",
        index=False, header=False, encoding="utf-8-sig")

    # equity curve هر استراتژی
    for s in results:
        eq_df = pd.DataFrame({
            'ts':     s['eq_curve_ts'],
            'equity': s['eq_curve'],
        })
        eq_df['dd_pct'] = (
            (eq_df['equity'] - eq_df['equity'].cummax())
            / eq_df['equity'].cummax() * 100
        ).round(4)
        eq_df.to_csv(f"equity_{s['name']}.csv",
                     index=False, encoding="utf-8-sig")

    print(f"\n✅ فایل‌ها:")
    print(f"   → Monthly_Simulation_Report.csv")
    for s in results:
        print(f"   → equity_{s['name']}.csv")


# ================================================================== #
#                           MAIN                                     #
# ================================================================== #
if __name__ == "__main__":
    df = load_data()

    print("\n" + "═"*74)
    print("  MONTHLY SIMULATION BACKTEST")
    print("  منطق: هر ماه مستقل | موجودی از ماه قبل منتقل می‌شه")
    print("  DD و Daily Loss نسبت به شروع همان ماه حساب می‌شه")
    print("═"*74)

    # محاسبه یک‌باره سیگنال‌ها
    signals = compute_all_signals(df)

    strategies = [
        ('CorrArb',    signals['CorrArb'][0],    signals['CorrArb'][1],
                       signals['CorrArb'][2],    signals['CorrArb'][3]),
        ('AsianBreak', signals['AsianBreak'][0], signals['AsianBreak'][1],
                       signals['AsianBreak'][2], None),
        ('EMACross',   signals['EMACross'][0],   signals['EMACross'][1],
                       signals['EMACross'][2],   None),
    ]

    all_results = []
    all_texts   = []

    for name, sig, sl, tp, z in strategies:
        t0 = datetime.now()
        print(f"\n  ▶ {name} ...", end="", flush=True)

        trades, monthly_log, eq_curve, eq_curve_ts = run_monthly_backtest(
            df, signals, name, sig, sl, tp, z_series=z)

        dt = (datetime.now()-t0).total_seconds()
        print(f" {dt:.1f}s | {len(trades)} معامله | {len(monthly_log)} ماه")

        if not trades:
            print(f"     ❌ بدون معامله")
            continue

        stats = compute_full_stats(
            trades, monthly_log, eq_curve, eq_curve_ts, name)
        if stats:
            all_results.append(stats)
            txt = print_report(stats)
            all_texts.append(txt)

    if all_results:
        comp = print_comparison(all_results)
        all_texts.append(comp)

        with open("Backtest_Report.txt","w",encoding="utf-8") as f:
            f.write(f"MONTHLY SIMULATION — "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("\n".join(all_texts))

        save_outputs(all_results)
