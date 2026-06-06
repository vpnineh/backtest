"""
CorrArb MTF Simulator v6 — اصلاح‌شده
"""

import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.04
    max_total_dd_pct   = 0.08

    # ── ریسک ──
    risk_base_pct = 0.008
    risk_min_pct  = 0.004

    # ── هزینه‌ها ──
    spread_pips        = 1.2
    commission_per_lot = 7.0
    slippage_pips      = 0.3

    # ── بازار ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 2.0
    min_lot  = 0.01
    warmup   = 500

    # ── 15min ──
    z_fast_period   = 96
    z_slow_period   = 384
    z_entry         = 1.8
    z_exit          = 0.5
    z_slow_confirm  = 0.6
    adx_max         = 28
    rsi_long_max    = 45
    rsi_short_min   = 55
    atr_period      = 14
    atr_ma_period   = 96
    atr_max_mult    = 2.5
    atr_min_mult    = 0.4
    corr_window     = 48
    corr_min        = 0.65
    std_min_pct     = 0.20
    hour_start      = 7
    hour_end        = 18
    trade_days      = [0, 1, 2, 3]

    # ── 1min تایید ──
    confirm_window_min  = 15
    confirm_rsi_period  = 7
    confirm_mom_bars    = 5
    confirm_vol_ma_bars = 20
    confirm_vol_mult    = 1.1    # کمتر از قبل
    confirm_body_min    = 0.25   # کمتر از قبل

    # ── SL/TP ──
    sl_lookback_bars = 10
    sl_buffer_pips   = 2.0
    sl_min_pips      = 8.0
    sl_max_pips      = 20.0
    tp_rr            = 2.2

    # ── مدیریت ──
    max_trades_day  = 2
    time_stop_bars  = 160
    trail_be_prog   = 0.50
    trail_be_pct    = 0.08
    trail_lock_prog = 0.75
    trail_lock_pct  = 0.45
    consec_loss_n   = 3
    risk_reduce     = 0.65


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده — اصلاح‌شده با دیباگ کامل
# ═══════════════════════════════════════════════════════════════════════════
def load_data() -> tuple:
    """
    بارگذاری داده با مدیریت خطای کامل.
    اگر داده 1min موجود نبود، از resample 15min استفاده می‌کند.
    """
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))

    print(f"  EURUSD files: {len(files_eur)}")
    print(f"  GBPUSD files: {len(files_gbp)}")

    if not files_eur:
        raise FileNotFoundError("❌ EURUSD CSV not found in data/")
    if not files_gbp:
        raise FileNotFoundError("❌ GBPUSD CSV not found in data/")

    def read_raw(paths: list, suffix: str) -> pd.DataFrame:
        frames = []
        for p in paths:
            try:
                df = pd.read_csv(
                    p, sep=';', header=None,
                    names=['ts', 'o', 'h', 'l', 'c', 'v']
                )
                df['ts'] = pd.to_datetime(
                    df['ts'], format='%Y%m%d %H%M%S', errors='coerce'
                )
                df = df.dropna(subset=['ts'])
                df = df.set_index('ts')
                df = df[~df.index.duplicated(keep='last')]
                # تبدیل ستون‌های عددی
                for col in ['o', 'h', 'l', 'c', 'v']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                df.columns = [f'{col}_{suffix}' for col in df.columns]
                frames.append(df)
                print(f"    ✓ {p}: {len(df):,} ردیف")
            except Exception as e:
                print(f"    ✗ {p}: {e}")
        if not frames:
            raise ValueError(f"هیچ فایل {suffix} معتبری بارگذاری نشد")
        return pd.concat(frames).sort_index()

    print("  بارگذاری EURUSD...")
    eur_raw = read_raw(files_eur, 'eur')
    print("  بارگذاری GBPUSD...")
    gbp_raw = read_raw(files_gbp, 'gbp')

    # join روی index مشترک
    raw = eur_raw.join(gbp_raw, how='inner').dropna()
    print(f"  داده مشترک: {len(raw):,} ردیف | "
          f"{raw.index[0].date()} → {raw.index[-1].date()}")

    # تشخیص تایم‌فریم اصلی داده
    if len(raw) > 1:
        diffs = raw.index.to_series().diff().dropna()
        median_diff = diffs.median()
        base_tf_min = int(median_diff.total_seconds() / 60)
        print(f"  تایم‌فریم پایه داده: {base_tf_min} دقیقه")
    else:
        base_tf_min = 15

    def make_tf(raw_df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """ساخت DataFrame با تایم‌فریم مشخص"""
        df = pd.DataFrame({
            'o_eur': raw_df['o_eur'].resample(freq).first(),
            'h_eur': raw_df['h_eur'].resample(freq).max(),
            'l_eur': raw_df['l_eur'].resample(freq).min(),
            'c_eur': raw_df['c_eur'].resample(freq).last(),
            'v_eur': raw_df['v_eur'].resample(freq).sum(),
            'o_gbp': raw_df['o_gbp'].resample(freq).first(),
            'h_gbp': raw_df['h_gbp'].resample(freq).max(),
            'l_gbp': raw_df['l_gbp'].resample(freq).min(),
            'c_gbp': raw_df['c_gbp'].resample(freq).last(),
            'v_gbp': raw_df['v_gbp'].resample(freq).sum(),
        }).dropna()

        # فقط روزهای هفته
        df = df[df.index.weekday < 5]
        # حذف کندل‌های کاملاً خالی (نه فیلتر حجم)
        df = df[df['c_eur'] > 0]
        return df

    # ── 15min ──
    df15 = make_tf(raw, '15min')
    if len(df15) == 0:
        raise ValueError("❌ DataFrame 15min خالی است")
    print(f"✅ 15min: {len(df15):,} کندل | "
          f"{df15.index[0].date()} → {df15.index[-1].date()}")

    # ── 1min ──
    if base_tf_min <= 1:
        # داده اصلی 1min است
        df1 = make_tf(raw, '1min')
        if len(df1) == 0:
            print("⚠️  DataFrame 1min خالی — استفاده از 15min به عنوان جایگزین")
            df1 = df15.copy()
            use_1min = False
        else:
            use_1min = True
            print(f"✅  1min: {len(df1):,} کندل | "
                  f"{df1.index[0].date()} → {df1.index[-1].date()}")
    else:
        # داده اصلی 15min یا بیشتر است — نمی‌توان 1min ساخت
        print(f"⚠️  داده پایه {base_tf_min}min است — "
              f"1min از resample قابل ساخت نیست")
        print("  → استراتژی: فقط 15min (بدون تایید 1min)")
        df1 = df15.copy()
        use_1min = False

    return df15, df1, use_1min


# ═══════════════════════════════════════════════════════════════════════════
#  اندیکاتورها
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(c, period=14):
    d    = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_adx(h, l, c, period=14):
    up   = h.diff()
    dn   = -l.diff()
    dmp  = up.where((up > dn) & (up > 0), 0.0)
    dmn  = dn.where((dn > up) & (dn > 0), 0.0)
    atr1 = calc_atr(h, l, c, 1)
    s    = atr1.rolling(period).sum().replace(0, np.nan)
    dip  = 100 * dmp.rolling(period).sum() / s
    din  = 100 * dmn.rolling(period).sum() / s
    dx   = (abs(dip - din) / (dip + din).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  سیگنال‌های 15min
# ═══════════════════════════════════════════════════════════════════════════
def compute_15min_signals(df15: pd.DataFrame) -> dict:
    print("  [15min] محاسبه سیگنال‌ها...", end="", flush=True)
    C   = Config
    c_e = df15['c_eur']
    h_e = df15['h_eur']
    l_e = df15['l_eur']
    c_g = df15['c_gbp']

    rsi_e = calc_rsi(c_e, 14)
    rsi_g = calc_rsi(c_g, 14)
    adx   = calc_adx(h_e, l_e, c_e, 14)
    atr   = calc_atr(h_e, l_e, c_e, C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()

    ratio  = c_e / c_g
    z_mf   = ratio.rolling(C.z_fast_period).mean()
    z_sf   = ratio.rolling(C.z_fast_period).std()
    z_fast = (ratio - z_mf) / z_sf.replace(0, np.nan)
    z_ms   = ratio.rolling(C.z_slow_period).mean()
    z_ss   = ratio.rolling(C.z_slow_period).std()
    z_slow = (ratio - z_ms) / z_ss.replace(0, np.nan)

    corr   = c_e.pct_change().rolling(C.corr_window).corr(c_g.pct_change())

    std_hist = z_sf.rolling(C.z_slow_period).mean()
    std_ok   = z_sf > std_hist * C.std_min_pct
    vol_ok   = (
        (atr > atr_ma * C.atr_min_mult) &
        (atr < atr_ma * C.atr_max_mult)
    )
    hour    = pd.Series(df15.index.hour, index=df15.index)
    dow     = pd.Series(df15.index.dayofweek, index=df15.index)
    time_ok = (
        hour.between(C.hour_start, C.hour_end) &
        dow.isin(C.trade_days)
    )
    adx_ok  = adx < C.adx_max
    corr_ok = corr > C.corr_min
    div_12h = c_e.pct_change(48) - c_g.pct_change(48)

    sig = pd.Series(0, index=df15.index)
    sig[
        (z_fast < -C.z_entry) & (z_slow < -C.z_slow_confirm) &
        (div_12h < -0.0005) & std_ok & vol_ok & time_ok &
        adx_ok & corr_ok &
        (rsi_e < C.rsi_long_max) & (rsi_e < rsi_g - 5)
    ] = 1
    sig[
        (z_fast > C.z_entry) & (z_slow > C.z_slow_confirm) &
        (div_12h > 0.0005) & std_ok & vol_ok & time_ok &
        adx_ok & corr_ok &
        (rsi_e > C.rsi_short_min) & (rsi_e > rsi_g + 5)
    ] = -1
    sig = sig.where(sig != sig.shift(), 0)

    print(f" ✓  {int((sig!=0).sum())} فرصت "
          f"(L:{int((sig==1).sum())}, S:{int((sig==-1).sum())})")
    return {'sig': sig, 'z_fast': z_fast, 'atr15': atr}


# ═══════════════════════════════════════════════════════════════════════════
#  اندیکاتورهای تایم‌فریم کوچک (1min یا همان 15min)
# ═══════════════════════════════════════════════════════════════════════════
def compute_ltf_indicators(df_ltf: pd.DataFrame) -> pd.DataFrame:
    """محاسبه اندیکاتورهای تایم‌فریم کوچک — causal"""
    print("  [ LTF] محاسبه اندیکاتورها...", end="", flush=True)
    C   = Config
    c_e = df_ltf['c_eur']
    h_e = df_ltf['h_eur']
    l_e = df_ltf['l_eur']
    o_e = df_ltf['o_eur']
    v_e = df_ltf['v_eur']

    atr_ltf  = calc_atr(h_e, l_e, c_e, 14)
    rsi_ltf  = calc_rsi(c_e, Config.confirm_rsi_period)
    mom_ltf  = c_e.pct_change(Config.confirm_mom_bars)

    vol_ma  = v_e.rolling(Config.confirm_vol_ma_bars).mean()
    vol_r   = v_e / vol_ma.replace(0, np.nan)

    rng     = (h_e - l_e).replace(0, np.nan)
    body_r  = (c_e - o_e) / rng

    out = pd.DataFrame({
        'c':    c_e,
        'h':    h_e,
        'l':    l_e,
        'o':    o_e,
        'v':    v_e,
        'atr':  atr_ltf,
        'rsi':  rsi_ltf,
        'mom':  mom_ltf,
        'vol_r':vol_r,
        'body': body_r,
    }, index=df_ltf.index)

    print(f" ✓  {len(out):,} کندل")
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  پیدا کردن ورود در تایم‌فریم کوچک
# ═══════════════════════════════════════════════════════════════════════════
def find_ltf_entry(
        direction:      int,
        ts_signal:      pd.Timestamp,
        ind_ltf:        pd.DataFrame,
        confirm_window: int,
        use_1min:       bool,
) -> tuple:
    """
    جستجوی تایید ورود در کندل‌های LTF بعد از سیگنال 15min.

    ✅ Causal:
       - فقط کندل‌های بعد از ts_signal بررسی می‌شوند
       - ورود روی open کندل بعد از تایید

    Returns: (confirmed, sl_pips, entry_ts | None)
    """
    C = Config

    # کندل‌های بعد از بسته شدن کندل 15min
    mask   = ind_ltf.index > ts_signal
    window = ind_ltf[mask].head(confirm_window + 1)

    if len(window) < 2:
        return False, C.sl_min_pips, None

    # برای هر کندل تایید (به جز آخری که برای ورود است)
    for i in range(len(window) - 1):
        row      = window.iloc[i]
        next_row = window.iloc[i + 1]

        # اگر NaN بود، skip
        if any(pd.isna([row['rsi'], row['mom'], row['vol_r'], row['body']])):
            continue

        # شرط تایید
        if direction == 1:   # Long
            ok = (
                row['rsi']   > 35 and
                row['mom']   > 0  and
                (not use_1min or row['vol_r'] > C.confirm_vol_mult) and
                row['body']  > C.confirm_body_min
            )
        else:                # Short
            ok = (
                row['rsi']   < 65 and
                row['mom']   < 0  and
                (not use_1min or row['vol_r'] > C.confirm_vol_mult) and
                row['body']  < -C.confirm_body_min
            )

        if ok:
            # SL از swing high/low در N کندل قبل از تایید
            prev = ind_ltf[ind_ltf.index <= row.name].tail(C.sl_lookback_bars)
            if len(prev) > 0:
                if direction == 1:
                    swing    = prev['l'].min()
                    sl_dist  = (float(next_row['o']) - swing) / C.pip
                else:
                    swing    = prev['h'].max()
                    sl_dist  = (swing - float(next_row['o'])) / C.pip
                sl_pips = float(np.clip(
                    sl_dist + C.sl_buffer_pips,
                    C.sl_min_pips, C.sl_max_pips
                ))
            else:
                sl_pips = C.sl_min_pips

            return True, sl_pips, next_row.name

    return False, C.sl_min_pips, None


# ═══════════════════════════════════════════════════════════════════════════
#  توابع کمکی پراپ
# ═══════════════════════════════════════════════════════════════════════════
def trade_cost(lot: float) -> float:
    C = Config
    return (C.spread_pips * 2 * C.pip * lot * C.lot_size
            + C.commission_per_lot * lot)


def calc_lot(equity: float, sl_pips: float, consec_loss: int) -> float:
    C    = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
    if sl_pips <= 0:
        return C.min_lot
    raw = equity * risk / (sl_pips * C.pip * C.lot_size)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def check_prop(equity: float, day_start: float,
               prop_floor: float) -> tuple:
    C = Config
    if day_start > 0:
        dd = (equity - day_start) / day_start
        if dd <= -C.max_daily_loss_pct:
            return True, f"DailyDD {dd*100:.2f}%"
    if equity <= prop_floor:
        dd = (equity - C.initial_balance) / C.initial_balance
        return True, f"TotalDD {dd*100:.2f}%"
    return False, ""


def new_acc(ts) -> dict:
    C = Config
    return dict(
        equity      = C.initial_balance,
        start_ts    = ts,
        trades      = [],
        open_pos    = None,
        blown       = False,
        blown_rsn   = "",
        peak        = C.initial_balance,
        min_eq      = C.initial_balance,
        max_dd_pct  = 0.0,
        consec_loss = 0,
        consec_win  = 0,
    )


def upd_dd(acc: dict):
    eq = acc['equity']
    if eq > acc['peak']:   acc['peak']   = eq
    if eq < acc['min_eq']: acc['min_eq'] = eq
    if acc['peak'] > 0:
        dd = (eq - acc['peak']) / acc['peak'] * 100
        if dd < acc['max_dd_pct']:
            acc['max_dd_pct'] = dd


def reg_acc(acc: dict, end_ts, tw: float,
            num: int, reason: str, logs: list):
    C   = Config
    pnl = acc['equity'] - C.initial_balance
    w   = sum(1 for t in acc['trades'] if t.get('pnl', 0) > 0)
    wr  = w / len(acc['trades']) * 100 if acc['trades'] else 0
    logs.append(dict(
        account         = num,
        start_ts        = acc['start_ts'],
        end_ts          = end_ts,
        final           = round(acc['equity'], 2),
        pnl             = round(pnl, 2),
        ret_pct         = round(pnl / C.initial_balance * 100, 2),
        trades          = len(acc['trades']),
        wins            = w,
        wr              = round(wr, 1),
        reason          = reason,
        total_withdrawn = round(tw, 2),
        max_dd_pct      = round(acc['max_dd_pct'], 4),
    ))


# ═══════════════════════════════════════════════════════════════════════════
#  موتور بک‌تست
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(
        df15:      pd.DataFrame,
        signals15: dict,
        ind_ltf:   pd.DataFrame,
        use_1min:  bool,
) -> dict:
    C    = Config
    pip  = C.pip
    ls   = C.lot_size

    open15  = df15['o_eur'].values
    close15 = df15['c_eur'].values
    high15  = df15['h_eur'].values
    low15   = df15['l_eur'].values
    ts15    = df15.index
    sig_a   = signals15['sig'].values
    z_a     = signals15['z_fast'].values

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    # window تایید: اگر 1min واقعی → 15 کندل، اگر 15min → 1 کندل
    confirm_window = C.confirm_window_min if use_1min else 1

    total_withdrawn = 0.0
    acc_num         = 1
    acc_logs        = []
    all_trades      = []
    eq_curve        = []
    eq_ts_list      = []
    tot_curve       = []

    acc          = new_acc(ts15[C.warmup])
    cur_day      = None
    day_start_eq = C.initial_balance
    trades_today = 0
    pending      = None

    sig_bars = {
        i: int(sig_a[i])
        for i in range(C.warmup, len(ts15) - 1)
        if sig_a[i] != 0
    }

    n_confirmed = 0
    n_rejected  = 0

    mode_str = "15min+1min" if use_1min else "15min only"
    print(f"\n  حالت: {mode_str}")
    print(f"  FLOOR=${PROP_FLOOR:,.0f} | هدف=${PROFIT_LEVEL:,.0f}")

    for bar in range(C.warmup, len(ts15)):
        ts  = ts15[bar]
        day = ts.date()
        eq  = acc['equity']

        eq_curve.append(round(eq, 4))
        eq_ts_list.append(ts)
        tot_curve.append(round(eq + total_withdrawn, 4))
        upd_dd(acc)

        if day != cur_day:
            cur_day      = day
            day_start_eq = eq
            trades_today = 0

        # ── Blown ──
        if acc['blown']:
            if acc['open_pos'] is not None:
                pos = acc['open_pos']
                cp  = close15[bar]
                raw = pos['dir'] * (cp - pos['entry']) * pos['lot'] * ls
                pnl = raw - trade_cost(pos['lot'])
                acc['equity'] += pnl
                rec = {**pos, 'exit': cp, 'exit_ts': ts,
                       'pnl': pnl, 'status': 'blown_close'}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
            reg_acc(acc, ts, total_withdrawn, acc_num,
                    acc['blown_rsn'], acc_logs)
            print(f"    💥 #{acc_num:>3} | {ts.date()} | "
                  f"${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            acc_num     += 1
            acc          = new_acc(ts)
            day_start_eq = acc['equity']
            trades_today = 0
            pending      = None
            PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ── بررسی pending → تایید LTF ──
        if (pending is not None
                and acc['open_pos'] is None
                and not acc['blown']
                and trades_today < C.max_trades_day):

            sv        = pending['dir']
            ts_signal = pending['ts_signal']

            confirmed, sl_pips, entry_ts = find_ltf_entry(
                sv, ts_signal, ind_ltf, confirm_window, use_1min
            )

            if confirmed and entry_ts is not None and entry_ts < ts:
                tp_pips = sl_pips * C.tp_rr
                lot     = calc_lot(acc['equity'], sl_pips, acc['consec_loss'])

                # ورود: close کندل تایید (= قیمت واقعی موجود)
                entry_price = float(ind_ltf.loc[entry_ts, 'c'])
                ep  = entry_price + sv * (C.slippage_pips + C.spread_pips/2) * pip
                sl  = ep - sv * sl_pips * pip
                tp  = ep + sv * tp_pips * pip

                # بررسی immediate SL روی 15min
                hi15 = high15[bar]
                lo15 = low15[bar]
                imm  = (sv == 1 and lo15 <= sl) or (sv == -1 and hi15 >= sl)

                if not imm:
                    acc['open_pos'] = dict(
                        account   = acc_num,
                        dir       = sv,
                        lot       = lot,
                        entry     = ep,
                        sl        = sl,
                        tp        = tp,
                        sl_pips   = round(sl_pips, 1),
                        tp_pips   = round(tp_pips, 1),
                        entry_ts  = entry_ts,
                        entry_bar = bar,
                    )
                    trades_today += 1
                    n_confirmed  += 1
                else:
                    n_rejected += 1
            else:
                n_rejected += 1

            pending = None

        # ── مدیریت پوزیشن (روی 15min) ──
        pos = acc['open_pos']
        if pos is not None:
            hi = high15[bar]
            lo = low15[bar]
            cp = close15[bar]
            d  = pos['dir']
            ep = pos['entry']
            sl = pos['sl']
            tp = pos['tp']

            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            # Z-exit
            zn = z_a[bar]
            if not np.isnan(zn) and abs(zn) < C.z_exit:
                hit_tp = True
            if hit_sl and hit_tp:
                hit_tp = False

            # Intra-candle blown check
            if not hit_sl:
                w_pnl = (d*(sl-ep)*pos['lot']*ls
                         - trade_cost(pos['lot']))
                blown, rsn = check_prop(
                    acc['equity'] + w_pnl, day_start_eq, PROP_FLOOR
                )
                if blown:
                    pnl = d*(sl-ep)*pos['lot']*ls - trade_cost(pos['lot'])
                    acc['equity'] += pnl
                    rec = {**pos, 'exit': sl, 'exit_ts': ts,
                           'pnl': pnl, 'status': 'blown_SL'}
                    acc['trades'].append(rec)
                    all_trades.append(rec)
                    acc['open_pos']    = None
                    acc['blown']       = True
                    acc['blown_rsn']   = rsn
                    acc['consec_loss'] += 1
                    acc['consec_win']   = 0
                    upd_dd(acc)
                    continue

            # Trailing Stop
            tp_dist = abs(tp - ep)
            if tp_dist > 0:
                prog = d * (cp - ep) / tp_dist
                if prog >= C.trail_be_prog:
                    be = ep + d * tp_dist * C.trail_be_pct
                    if d == 1 and be > pos['sl']:   pos['sl'] = be
                    elif d == -1 and be < pos['sl']: pos['sl'] = be
                if prog >= C.trail_lock_prog:
                    lock = ep + d * tp_dist * C.trail_lock_pct
                    if d == 1 and lock > pos['sl']:   pos['sl'] = lock
                    elif d == -1 and lock < pos['sl']: pos['sl'] = lock

            # Time Stop
            if (bar - pos['entry_bar'] >= C.time_stop_bars
                    and not hit_tp and not hit_sl):
                raw = d * (cp - ep) * pos['lot'] * ls
                pnl = raw - trade_cost(pos['lot'])
                acc['equity'] += pnl
                st  = 'TP_time' if pnl > 0 else 'SL_time'
                rec = {**pos, 'exit': cp, 'exit_ts': ts,
                       'pnl': pnl, 'status': st}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
                if pnl > 0:
                    acc['consec_win']  += 1; acc['consec_loss']  = 0
                else:
                    acc['consec_loss'] += 1; acc['consec_win']   = 0
                upd_dd(acc)
                blown, rsn = check_prop(
                    acc['equity'], day_start_eq, PROP_FLOOR
                )
                acc['blown'] = blown; acc['blown_rsn'] = rsn
                continue

            # SL / TP
            if hit_sl or hit_tp:
                exit_px = sl if hit_sl else tp
                st      = 'SL' if hit_sl else 'TP'
                raw     = d * (exit_px - ep) * pos['lot'] * ls
                pnl     = raw - trade_cost(pos['lot'])
                acc['equity'] += pnl
                rec = {**pos, 'exit': exit_px, 'exit_ts': ts,
                       'pnl': pnl, 'status': st}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
                if pnl > 0:
                    acc['consec_win']  += 1; acc['consec_loss']  = 0
                else:
                    acc['consec_loss'] += 1; acc['consec_win']   = 0
                upd_dd(acc)
                blown, rsn = check_prop(
                    acc['equity'], day_start_eq, PROP_FLOOR
                )
                acc['blown'] = blown; acc['blown_rsn'] = rsn

        # ── هدف برداشت ──
        if (acc['equity'] >= PROFIT_LEVEL
                and acc['open_pos'] is None
                and not acc['blown']):
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            reg_acc(acc, ts, total_withdrawn, acc_num,
                    "TARGET_HIT", acc_logs)
            print(f"    💰 #{acc_num:>3} | {ts.date()} | "
                  f"برداشت: ${w:>7.2f} | کل: ${total_withdrawn:>9.2f}")
            acc_num     += 1
            acc          = new_acc(ts)
            day_start_eq = acc['equity']
            trades_today = 0
            pending      = None
            PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ── سیگنال جدید ──
        if (acc['open_pos'] is None
                and pending is None
                and not acc['blown']
                and bar in sig_bars
                and trades_today < C.max_trades_day):
            pending = {
                'dir':       sig_bars[bar],
                'ts_signal': ts,
                'bar15':     bar,
            }

    # پایان داده
    if acc['open_pos'] is not None:
        pos = acc['open_pos']
        cp  = close15[-1]
        raw = pos['dir'] * (cp - pos['entry']) * pos['lot'] * ls
        pnl = raw - trade_cost(pos['lot'])
        acc['equity'] += pnl
        rec = {**pos, 'exit': cp, 'exit_ts': ts15[-1],
               'pnl': pnl, 'status': 'EndOfData'}
        acc['trades'].append(rec)
        all_trades.append(rec)
        acc['open_pos'] = None

    reg_acc(acc, ts15[-1], total_withdrawn, acc_num,
            "ACTIVE/END", acc_logs)

    total = n_confirmed + n_rejected
    if total > 0:
        print(f"\n  تایید: {n_confirmed} | رد: {n_rejected} | "
              f"نرخ: {n_confirmed/total*100:.1f}%")

    return dict(
        all_trades      = all_trades,
        account_logs    = acc_logs,
        eq_curve        = eq_curve,
        eq_ts           = eq_ts_list,
        total_curve     = tot_curve,
        total_withdrawn = total_withdrawn,
        final_equity    = acc['equity'],
        total_accounts  = acc_num,
        n_confirmed     = n_confirmed,
        n_rejected      = n_rejected,
        use_1min        = use_1min,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  آمار
# ═══════════════════════════════════════════════════════════════════════════
def compute_stats(results: dict) -> dict | None:
    if not results['all_trades']:
        return None
    C  = Config
    t  = pd.DataFrame(results['all_trades'])
    t['pnl']          = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60
    al = pd.DataFrame(results['account_logs'])

    tw  = results['total_withdrawn']
    feq = results['final_equity']
    tv  = tw + feq
    tp_ = tv - C.initial_balance
    tr  = tp_ / C.initial_balance * 100
    sd  = t['entry_ts'].min()
    ed  = t['exit_ts'].max()
    td  = max((ed - sd).days, 1)
    ar  = ((tv / C.initial_balance) ** (365.25 / td) - 1) * 100

    wt  = t[t['pnl'] > 0]
    lt  = t[t['pnl'] < 0]
    wr  = len(wt) / len(t) * 100 if len(t) else 0
    aw  = wt['pnl'].mean() if len(wt) else 0
    al_ = lt['pnl'].mean() if len(lt) else 0
    pf  = (wt['pnl'].sum() / abs(lt['pnl'].sum())
           if lt['pnl'].sum() != 0 else np.inf)
    rr  = abs(aw / al_) if al_ != 0 else 0

    max_dd = al['max_dd_pct'].min() if 'max_dd_pct' in al.columns else 0.0
    rc     = pd.Series(results['total_curve']).pct_change().dropna()
    sharpe = rc.mean() / rc.std() * np.sqrt(252*96) if rc.std() > 0 else 0
    neg    = rc[rc < 0]
    sortino= rc.mean() / neg.std() * np.sqrt(252*96) if len(neg) > 1 else 0

    n_target = int((al['reason'] == 'TARGET_HIT').sum())
    n_blown  = int(al['reason'].str.contains(
        'DailyDD|TotalDD|blown', case=False, na=False).sum())
    n_active = int((al['reason'] == 'ACTIVE/END').sum())

    sign = t['pnl'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    cw = cl = mcw = mcl = 0
    for s in sign:
        if s > 0:   cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0: cl += 1; cw = 0; mcl = max(mcl, cl)
        else:       cw = cl = 0

    t['ym'] = t['entry_ts'].dt.to_period('M')
    mg = t.groupby('ym').agg(
        n    = ('pnl', 'count'),
        pnl  = ('pnl', 'sum'),
        wins = ('pnl', lambda x: (x > 0).sum()),
    ).reset_index()
    mg['wr']  = mg['wins'] / mg['n'] * 100
    mg['ret'] = mg['pnl'] / C.initial_balance * 100

    return dict(
        trades=t, acc_logs=al, monthly=mg,
        eq_curve=results['eq_curve'],
        eq_ts=results['eq_ts'],
        total_curve=results['total_curve'],
        total_withdrawn=tw, final_equity=feq,
        total_value=tv, total_profit=tp_,
        total_ret=tr, ann_ret=ar, total_days=td,
        win_r=wr, avg_w=aw, avg_l=al_, pf=pf, rr=rr,
        exp=t['pnl'].mean(), max_dd=max_dd,
        sharpe=sharpe, sortino=sortino, mcw=mcw, mcl=mcl,
        n_accounts=results['total_accounts'],
        n_target=n_target, n_blown=n_blown, n_active=n_active,
        avg_dur=t['duration_min'].mean(),
        n_confirmed=results['n_confirmed'],
        n_rejected=results['n_rejected'],
        use_1min=results['use_1min'],
        avg_monthly=mg['ret'].mean(),
        std_monthly=mg['ret'].std(),
        best_month=mg['ret'].max(),
        worst_month=mg['ret'].min(),
        n_pos_months=int((mg['pnl'] > 0).sum()),
        n_neg_months=int((mg['pnl'] <= 0).sum()),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  گزارش
# ═══════════════════════════════════════════════════════════════════════════
def print_report(s: dict) -> str:
    C   = Config
    W   = 84
    SEP = "═" * W

    def rw(l, v, ok=None):
        lb = f"  {l}"
        vl = str(v)
        m  = "" if ok is None else (" ✅" if ok else " ❌")
        d  = "·" * max(2, W - len(lb) - len(vl) - len(m) - 2)
        return f"{lb} {d} {vl}{m}"

    def box(t):
        i = f"─ {t} "
        return "┌" + i + "─" * (W - len(i) - 1) + "┐"

    bot = "└" + "─" * (W - 1) + "┘"

    dd_ok  = abs(s['max_dd'])  <= 8.0
    pf_ok  = s['pf']           >  1.3
    bl_ok  = s['n_blown']      == 0
    tg_ok  = s['n_target']     >  0
    cg_ok  = s['ann_ret']      >  10.0
    wr_ok  = s['win_r']        >= 50.0
    ws_ok  = s['worst_month']  > -5.0

    passed = all([dd_ok, pf_ok, bl_ok, tg_ok, cg_ok])
    flag   = "✅ PROP PASS" if passed else "⚠️  NEEDS REVIEW"

    nc   = s['n_confirmed']
    nr   = s['n_rejected']
    rate = f"{nc/(nc+nr)*100:.1f}%" if (nc + nr) > 0 else "N/A"
    mode = "15min + 1min" if s['use_1min'] else "15min only"

    lines = [
        "", SEP,
        f"  ▌  CorrArb MTF v6  [{mode}]  —  {flag}  ▐",
        f"  ▌  {s['trades']['entry_ts'].min().date()} → "
        f"{s['trades']['exit_ts'].max().date()}  "
        f"({s['total_days']} روز)  ▐",
        f"  ▌  تایید: {nc} | رد: {nr} | نرخ تایید: {rate}  ▐",
        SEP, "",

        box("نتایج مالی"),
        rw("بالانس هر اکانت",         f"${C.initial_balance:>12,.2f}"),
        rw("کل سود برداشت‌شده",       f"${s['total_withdrawn']:>+12,.2f}"),
        rw("موجودی اکانت فعلی",       f"${s['final_equity']:>12,.2f}"),
        rw("ارزش کل",                 f"${s['total_value']:>12,.2f}"),
        rw("سود خالص کل",             f"${s['total_profit']:>+12,.2f}"),
        rw("بازده کل",                f"{s['total_ret']:>+.2f}%"),
        rw("CAGR", f"{s['ann_ret']:>+.2f}%", ok=cg_ok),
        bot, "",

        box("ریسک"),
        rw("Max DD per Account",
           f"{s['max_dd']:.2f}%",     ok=dd_ok),
        rw("Sharpe",                  f"{s['sharpe']:.2f}"),
        rw("Sortino",                 f"{s['sortino']:.2f}"),
        rw("Profit Factor",
           f"{s['pf']:.2f}",          ok=pf_ok),
        bot, "",

        box("پایداری ماهانه"),
        rw("میانگین ماهانه",          f"{s['avg_monthly']:>+.2f}%"),
        rw("انحراف معیار",            f"{s['std_monthly']:.2f}%"),
        rw("ماه سودده / زیان‌ده",
           f"{s['n_pos_months']} / {s['n_neg_months']}"),
        rw("بهترین ماه",              f"{s['best_month']:>+.2f}%"),
        rw("بدترین ماه",
           f"{s['worst_month']:>+.2f}%", ok=ws_ok),
        bot, "",

        box("پراپ"),
        rw("کل اکانت‌ها",             f"{s['n_accounts']}"),
        rw("✅ Target Hit",
           f"{s['n_target']}",        ok=tg_ok),
        rw("💥 Blown",
           f"{s['n_blown']}",         ok=bl_ok),
        rw("نرخ موفقیت",
           f"{s['n_target']/max(s['n_accounts'],1)*100:.1f}%"),
        bot, "",

        box("معاملات"),
        rw("تعداد کل",                f"{len(s['trades']):,}"),
        rw("Win Rate",
           f"{s['win_r']:.1f}%",      ok=wr_ok),
        rw("Avg Win",                 f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",                f"${s['avg_l']:>+.2f}"),
        rw("RR واقعی",                f"{s['rr']:.2f}"),
        rw("Expectancy",              f"${s['exp']:>+.2f}"),
        rw("Max Cons. Wins",          f"{s['mcw']}"),
        rw("Max Cons. Losses",        f"{s['mcl']}"),
        rw("میانگین مدت",             f"{s['avg_dur']:.0f} min"),
        bot, "",
    ]

    # اکانت‌ها
    lines.append(box("جزئیات اکانت‌ها"))
    lines.append(
        f"  {'#':>4}  {'شروع':>10}  {'پایان':>10}  "
        f"{'PnL':>9}  {'Ret%':>6}  {'#T':>3}  "
        f"{'WR%':>5}  {'MaxDD':>7}  وضعیت"
    )
    lines.append("  " + "─" * (W - 3))
    for _, row in s['acc_logs'].iterrows():
        r   = row['reason']
        ic  = ("💰 WITHDRAW" if r == 'TARGET_HIT' else
               "🔄 ACTIVE"  if r == 'ACTIVE/END'  else
               f"💥 {r[:20]}")
        mdd = row.get('max_dd_pct', 0.0)
        wn  = " ⚠️" if abs(mdd) > 5 else ""
        lines.append(
            f"  {int(row['account']):>4}  "
            f"{str(row['start_ts'])[:10]:>10}  "
            f"{str(row['end_ts'])[:10]:>10}  "
            f"${row['pnl']:>+8.2f}  {row['ret_pct']:>+5.1f}%  "
            f"{row['trades']:>3}  {row['wr']:>4.0f}%  "
            f"{mdd:>+6.2f}%{wn}  {ic}"
        )
    lines += [bot, ""]

    # ماهانه
    lines.append(box("ماهانه"))
    lines.append(
        f"  {'ماه':>7}  {'#T':>3}  {'WR%':>5}  "
        f"{'PnL':>9}  {'Ret%':>7}  نتیجه"
    )
    lines.append("  " + "─" * (W - 3))
    for _, mr in s['monthly'].iterrows():
        ic = "🟢" if mr['ret'] > 0 else "🔴"
        wn = " ⚠️" if mr['ret'] < -4 else ""
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>3}  "
            f"{mr['wr']:>4.1f}%  "
            f"${mr['pnl']:>+8.2f}  {mr['ret']:>+6.2f}%  {ic}{wn}"
        )
    lines += [bot, ""]

    # سالانه
    s['trades']['yr'] = s['trades']['entry_ts'].dt.year
    yg = (s['trades'].groupby('yr')
          .agg(n=('pnl', 'count'), pnl=('pnl', 'sum'),
               wins=('pnl', lambda x: (x > 0).sum()))
          .reset_index())
    yg['wr']  = yg['wins'] / yg['n'] * 100
    yg['ret'] = yg['pnl'] / C.initial_balance * 100

    lines.append(box("سالانه"))
    lines.append(
        f"  {'سال':>5}  {'#T':>5}  {'WR%':>5}  "
        f"{'PnL':>11}  {'Ret%':>7}  نتیجه"
    )
    lines.append("  " + "─" * (W - 3))
    for _, yr in yg.iterrows():
        ic = "🟢" if yr['ret'] > 0 else "🔴"
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>4.1f}%  "
            f"${yr['pnl']:>10.2f}  {yr['ret']:>+6.1f}%  {ic}"
        )
    lines += [bot, ""]

    out = "\n".join(lines)
    print(out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  ذخیره
# ═══════════════════════════════════════════════════════════════════════════
def save_outputs(s: dict, report_txt: str):
    C = Config
    with open("Report_MTF_v6.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    rows = [
        ["CorrArb MTF v6"],
        [f"Mode: {'15min+1min' if s['use_1min'] else '15min only'}"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [], ["=== Summary ==="],
        ["Total Withdrawn",  round(s['total_withdrawn'], 2)],
        ["Final Equity",     round(s['final_equity'], 2)],
        ["CAGR %",           round(s['ann_ret'], 2)],
        ["Profit Factor",    round(s['pf'], 2)],
        ["Win Rate %",       round(s['win_r'], 1)],
        ["Max DD %",         round(s['max_dd'], 2)],
        ["Confirmed",        s['n_confirmed']],
        ["Rejected",         s['n_rejected']],
        [], ["=== Trades ==="],
        ["Acc", "EntryTS", "ExitTS", "Side", "Lot",
         "Entry", "SL", "TP", "Exit", "SL_pips", "TP_pips",
         "PnL", "Status", "DurMin"],
    ]
    for _, tr in s['trades'].iterrows():
        rows.append([
            tr.get('account', ''),
            str(tr['entry_ts'])[:16], str(tr['exit_ts'])[:16],
            'BUY' if tr.get('dir', 0) == 1 else 'SELL',
            tr.get('lot', ''),
            round(float(tr.get('entry', 0)), 5),
            round(float(tr.get('sl',    0)), 5),
            round(float(tr.get('tp',    0)), 5),
            round(float(tr.get('exit',  0)), 5),
            round(float(tr.get('sl_pips', 0)), 1),
            round(float(tr.get('tp_pips', 0)), 1),
            round(float(tr['pnl']), 2),
            tr.get('status', ''),
            round(float(tr.get('duration_min', 0)), 0),
        ])

    pd.DataFrame(rows).to_csv(
        "Report_MTF_v6.csv",
        index=False, header=False, encoding="utf-8-sig"
    )

    wc = [round(tv - ae, 2)
          for tv, ae in zip(s['total_curve'], s['eq_curve'])]
    pd.DataFrame({
        'ts':              s['eq_ts'],
        'account_equity':  s['eq_curve'],
        'total_withdrawn': wc,
        'total_value':     s['total_curve'],
    }).to_csv("eq_MTF_v6.csv", index=False, encoding="utf-8-sig")

    print(f"✅ Report_MTF_v6.txt | Report_MTF_v6.csv | eq_MTF_v6.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 84)
    print("  CorrArb MTF Simulator v6  —  15min + 1min")
    print("═" * 84)
    C = Config
    print(f"  Risk={C.risk_base_pct*100:.1f}%  |  "
          f"Target=+{C.profit_target_pct*100:.0f}%  |  "
          f"DailyDD=-{C.max_daily_loss_pct*100:.0f}%  |  "
          f"TotalDD=-{C.max_total_dd_pct*100:.0f}%")
    print("═" * 84)

    t0 = datetime.now()

    print("\n  ▶ بارگذاری داده...")
    df15, df1, use_1min = load_data()

    print("\n  ▶ محاسبه سیگنال‌ها...")
    signals15 = compute_15min_signals(df15)
    ind_ltf   = compute_ltf_indicators(df1)

    print("\n  ▶ شبیه‌سازی...")
    t1      = datetime.now()
    results = run_backtest(df15, signals15, ind_ltf, use_1min)
    dt      = (datetime.now() - t1).total_seconds()

    print(f"\n  ⏱ {dt:.1f}s | "
          f"معاملات: {len(results['all_trades']):,} | "
          f"اکانت‌ها: {results['total_accounts']}")

    if not results['all_trades']:
        print("\n❌ معامله‌ای انجام نشد.")
        print("  دلایل احتمالی:")
        print("  1. داده 1min موجود نیست → کد به 15min-only تبدیل می‌شود")
        print("  2. شرایط تایید خیلی سخت است")
        print(f"  → confirm_vol_mult={C.confirm_vol_mult}")
        print(f"  → confirm_body_min={C.confirm_body_min}")
    else:
        stats = compute_stats(results)
        if stats:
            report = print_report(stats)
            save_outputs(stats, report)

    print(f"\n  ✅ کل: {(datetime.now()-t0).total_seconds():.1f}s")
