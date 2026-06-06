"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        CorrArb Prop Simulator — نسخه اصلاح‌شده v2                         ║
║                                                                              ║
║  اصلاحات این نسخه:                                                          ║
║  1. Max DD محاسبه per-account (نه کل equity curve)                         ║
║  2. Intra-candle DD check: low/high کندل برای blown                        ║
║  3. Trailing Stop اصلاح‌شده                                                 ║
║  4. Daily DD از بالانس ابتدای روز (نه peak)                                ║
║  5. بررسی blown قبل از ورود به معامله جدید                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
    # ── حساب پراپ ──
    initial_balance     = 5_000.0
    profit_target_pct   = 0.05        # +5% → برداشت
    max_daily_loss_pct  = 0.05        # 5% از بالانس ابتدای روز
    max_total_dd_pct    = 0.10        # 10% از $5,000 → floor = $4,500

    # ── ریسک معامله ──
    risk_per_trade_pct  = 0.010       # 1% ریسک هر معامله

    # ── هزینه‌های معامله ──
    spread_pips         = 1.2
    commission_per_lot  = 7.0
    slippage_pips       = 0.3

    # ── مشخصات بازار ──
    pip                 = 0.0001
    lot_size            = 100_000
    max_lot             = 2.0
    min_lot             = 0.01

    # ── warmup ──
    warmup              = 500

    # ── CorrArb پارامترها ──
    arb_z_fast          = 96
    arb_z_slow          = 480
    arb_z_entry         = 2.0
    arb_z_exit          = 0.4
    arb_z_slow_confirm  = 0.5
    arb_adx_max         = 28
    arb_rsi_long_max    = 45
    arb_rsi_short_min   = 55
    arb_sl_pips         = 22.0
    arb_tp_pips         = 44.0
    arb_hour_start      = 7
    arb_hour_end        = 19
    arb_max_trades_day  = 1
    arb_min_std_pct     = 0.25


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده
# ═══════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))

    if not files_eur:
        raise FileNotFoundError("❌ فایل EURUSD CSV پیدا نشد در data/")
    if not files_gbp:
        raise FileNotFoundError("❌ فایل GBPUSD CSV پیدا نشد در data/")

    def read_pair(paths: list, suffix: str) -> pd.DataFrame:
        frames = []
        for p in paths:
            df = pd.read_csv(
                p, sep=';', header=None,
                names=['ts', 'o', 'h', 'l', 'c', 'v']
            )
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            df = df.set_index('ts')
            df = df[~df.index.duplicated(keep='last')]
            df.columns = [f'{col}_{suffix}' for col in df.columns]
            frames.append(df)
        return pd.concat(frames).sort_index()

    eur = read_pair(files_eur, 'eur')
    gbp = read_pair(files_gbp, 'gbp')
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


# ═══════════════════════════════════════════════════════════════════════════
#  اندیکاتورها
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(c, period=14):
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_adx(h, l, c, period=14):
    up  = h.diff()
    dn  = -l.diff()
    dmp = up.where((up > dn) & (up > 0), 0.0)
    dmn = dn.where((dn > up) & (dn > 0), 0.0)
    tr  = calc_atr(h, l, c, 1)
    s   = tr.rolling(period).sum().replace(0, np.nan)
    dip = 100 * dmp.rolling(period).sum() / s
    din = 100 * dmn.rolling(period).sum() / s
    dx  = (abs(dip - din) / (dip + din).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  سیگنال‌های CorrArb — بدون look-ahead
# ═══════════════════════════════════════════════════════════════════════════
def compute_corrarb_signals(df: pd.DataFrame) -> dict:
    print("  محاسبه سیگنال‌های CorrArb...", end="", flush=True)

    c_e = df['c_eur']
    h_e = df['h_eur']
    l_e = df['l_eur']
    c_g = df['c_gbp']
    C   = Config

    rsi  = calc_rsi(c_e, 14)
    adx  = calc_adx(h_e, l_e, c_e, 14)
    hour = pd.Series(df.index.hour, index=df.index)

    eurgbp   = c_e / c_g
    z_mean_f = eurgbp.rolling(C.arb_z_fast).mean()
    z_std_f  = eurgbp.rolling(C.arb_z_fast).std()
    z_fast   = (eurgbp - z_mean_f) / z_std_f.replace(0, np.nan)

    z_mean_s = eurgbp.rolling(C.arb_z_slow).mean()
    z_std_s  = eurgbp.rolling(C.arb_z_slow).std()
    z_slow   = (eurgbp - z_mean_s) / z_std_s.replace(0, np.nan)

    std_hist = z_std_f.rolling(C.arb_z_slow).mean()
    std_ok   = z_std_f > std_hist * C.arb_min_std_pct

    time_ok  = hour.between(C.arb_hour_start, C.arb_hour_end)
    adx_ok   = adx < C.arb_adx_max

    sig = pd.Series(0, index=df.index)
    sig[
        (z_fast < -C.arb_z_entry) &
        (z_slow < -C.arb_z_slow_confirm) &
        std_ok & adx_ok & time_ok &
        (rsi < C.arb_rsi_long_max)
    ] = 1
    sig[
        (z_fast > C.arb_z_entry) &
        (z_slow > C.arb_z_slow_confirm) &
        std_ok & adx_ok & time_ok &
        (rsi > C.arb_rsi_short_min)
    ] = -1

    # حذف سیگنال تکراری متوالی
    sig = sig.where(sig != sig.shift(), 0)

    print(" ✓")
    print(f"  سیگنال‌ها: {int((sig != 0).sum()):,} | "
          f"Long: {int((sig == 1).sum()):,} | "
          f"Short: {int((sig == -1).sum()):,}")

    return {'sig': sig, 'z_fast': z_fast}


# ═══════════════════════════════════════════════════════════════════════════
#  توابع کمکی
# ═══════════════════════════════════════════════════════════════════════════
def trade_cost(lot: float, C) -> float:
    """هزینه کامل یک معامله (spread رفت+برگشت + کمیسیون)"""
    return (C.spread_pips * 2 * C.pip * lot * C.lot_size) + (C.commission_per_lot * lot)


def lot_calc(equity: float, sl_pips: float, C) -> float:
    if sl_pips <= 0:
        return C.min_lot
    raw = equity * C.risk_per_trade_pct / (sl_pips * C.pip * C.lot_size)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def check_prop_rules(equity: float, day_start_eq: float,
                     prop_floor: float, C) -> tuple[bool, str]:
    """
    بررسی قوانین پراپ — True = blown
    
    Daily DD: از بالانس ابتدای روز (نه peak)
    Total DD: equity < floor ($4,500)
    """
    # Daily DD: ضرر نسبت به بالانس ابتدای روز
    if day_start_eq > 0:
        daily_loss_pct = (equity - day_start_eq) / day_start_eq
        if daily_loss_pct <= -C.max_daily_loss_pct:
            return True, f"DailyDD {daily_loss_pct*100:.2f}%"
    
    # Total DD: equity از floor پایین‌تر رفته
    if equity <= prop_floor:
        dd_pct = (equity - C.initial_balance) / C.initial_balance
        return True, f"TotalDD {dd_pct*100:.2f}% (eq=${equity:.2f})"
    
    return False, ""


def register_account(start_ts, end_ts, final_eq, total_withdrawn,
                     trades, acc_num, reason, logs,
                     per_account_max_dd: float):
    C   = Config
    pnl = final_eq - C.initial_balance
    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    wr   = wins / len(trades) * 100 if trades else 0
    logs.append({
        'account':          acc_num,
        'start_ts':         start_ts,
        'end_ts':           end_ts,
        'initial':          C.initial_balance,
        'final':            round(final_eq, 2),
        'pnl':              round(pnl, 2),
        'ret_pct':          round(pnl / C.initial_balance * 100, 2),
        'trades':           len(trades),
        'wins':             wins,
        'wr':               round(wr, 1),
        'reason':           reason,
        'total_withdrawn':  round(total_withdrawn, 2),
        'max_dd_pct':       round(per_account_max_dd, 4),
    })


# ═══════════════════════════════════════════════════════════════════════════
#  موتور بک‌تست پراپ — اصلاح‌شده
# ═══════════════════════════════════════════════════════════════════════════
def run_prop_backtest(df: pd.DataFrame, signals: dict) -> dict:
    """
    اصلاحات کلیدی:
    
    1. Intra-candle blown check:
       - برای Long: اگر low کندل به prop_floor برسد → blown
       - برای Daily DD: worst-case intra-candle equity بررسی می‌شود
    
    2. Max DD per-account: فقط از $5,000 تا minimum equity در همان اکانت
    
    3. Trailing Stop اصلاح‌شده: فقط sl را بهتر (سمت سود) می‌کند
    
    4. ورود روی open کندل بعدی (pending signal)
    
    5. SL و TP هر دو در یک کندل → SL (conservative)
    """
    C    = Config
    pip  = C.pip
    ls   = C.lot_size

    # آرایه‌های numpy برای سرعت
    open_a  = df['o_eur'].values
    close_a = df['c_eur'].values
    high_a  = df['h_eur'].values
    low_a   = df['l_eur'].values
    sig_a   = signals['sig'].values
    z_a     = signals['z_fast'].values
    ts_a    = df.index

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)   # $4,500
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)  # $5,250

    # ── وضعیت کلی ──
    total_withdrawn  = 0.0
    account_number   = 1
    all_account_logs = []
    all_trades       = []

    global_eq_curve  = []
    global_eq_ts     = []
    global_tot_curve = []

    def reset_account(ts, initial_eq=C.initial_balance):
        """ریست به اکانت جدید"""
        return {
            'equity':        initial_eq,
            'start_ts':      ts,
            'trades':        [],
            'open_pos':      None,
            'blown':         False,
            'blown_reason':  "",
            # per-account DD tracking
            'acc_peak':      initial_eq,   # peak equity این اکانت
            'acc_min_eq':    initial_eq,   # minimum equity این اکانت
            'max_dd_pct':    0.0,          # max drawdown این اکانت (%)
        }

    acc = reset_account(ts_a[C.warmup])

    # ── وضعیت روز ──
    cur_day        = None
    day_start_eq   = acc['equity']
    trades_today   = 0
    pending_signal = 0

    # ── index سیگنال‌ها ──
    sig_bars = {
        i: int(sig_a[i])
        for i in range(C.warmup, len(ts_a) - 1)
        if sig_a[i] != 0
    }

    print(f"\n  شروع شبیه‌سازی... PROP_FLOOR=${PROP_FLOOR:,.0f} | "
          f"هدف=${PROFIT_LEVEL:,.0f}")

    for bar in range(C.warmup, len(ts_a)):
        ts  = ts_a[bar]
        day = ts.date()

        eq = acc['equity']

        # ── ثبت equity curve ──
        global_eq_curve.append(round(eq, 4))
        global_eq_ts.append(ts)
        global_tot_curve.append(round(eq + total_withdrawn, 4))

        # ── به‌روزرسانی per-account DD ──
        if eq > acc['acc_peak']:
            acc['acc_peak'] = eq
        if eq < acc['acc_min_eq']:
            acc['acc_min_eq'] = eq
        # DD از peak این اکانت
        if acc['acc_peak'] > 0:
            dd_now = (eq - acc['acc_peak']) / acc['acc_peak'] * 100
            if dd_now < acc['max_dd_pct']:
                acc['max_dd_pct'] = dd_now

        # ── ریست روزانه ──
        if day != cur_day:
            cur_day    = day
            day_start_eq = eq   # بالانس ابتدای روز (نه peak)
            trades_today = 0

        # ══════════════════════════════════════════════════════════
        #  اگر اکانت blown شده: بستن پوزیشن + ثبت + ریست
        # ══════════════════════════════════════════════════════════
        if acc['blown']:
            if acc['open_pos'] is not None:
                pos = acc['open_pos']
                cp  = close_a[bar]
                raw = pos['dir'] * (cp - pos['entry']) * pos['lot'] * ls
                pnl = raw - trade_cost(pos['lot'], C)
                acc['equity'] += pnl
                rec = {**pos, 'exit': cp, 'exit_ts': ts,
                       'pnl': pnl, 'status': 'blown_close'}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None

            register_account(
                acc['start_ts'], ts, acc['equity'],
                total_withdrawn, acc['trades'],
                account_number, acc['blown_reason'],
                all_account_logs, acc['max_dd_pct']
            )
            print(f"    💥 اکانت #{account_number:>3} | {ts.date()} | "
                  f"${acc['equity']:>8.2f} | {acc['blown_reason']}")

            account_number += 1
            acc = reset_account(ts)
            day_start_eq   = acc['equity']
            trades_today   = 0
            pending_signal = 0
            PROP_FLOOR     = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL   = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ══════════════════════════════════════════════════════════
        #  اجرای سیگنال pending روی open این کندل
        # ══════════════════════════════════════════════════════════
        if (pending_signal != 0
                and acc['open_pos'] is None
                and not acc['blown']
                and trades_today < C.arb_max_trades_day):

            sv  = pending_signal
            slp = C.arb_sl_pips
            tpp = C.arb_tp_pips
            lot = lot_calc(acc['equity'], slp, C)

            # ورود روی open + slippage (در جهت ضرر = بدبینانه)
            ep = open_a[bar] + sv * (C.slippage_pips + C.spread_pips / 2) * pip

            # SL و TP
            sl_price = ep - sv * slp * pip
            tp_price = ep + sv * tpp * pip

            # ── بررسی: آیا SL بلافاصله توسط همین کندل hit می‌شود؟ ──
            # اگر بله، وارد معامله نشو (avoid immediate SL)
            hi = high_a[bar]
            lo = low_a[bar]
            immediate_sl = (
                (sv == 1  and lo <= sl_price) or
                (sv == -1 and hi >= sl_price)
            )
            if not immediate_sl:
                acc['open_pos'] = dict(
                    account   = account_number,
                    dir       = sv,
                    lot       = lot,
                    entry     = ep,
                    sl        = sl_price,
                    tp        = tp_price,
                    entry_ts  = ts,
                    entry_bar = bar,
                )
                trades_today += 1

        pending_signal = 0

        # ══════════════════════════════════════════════════════════
        #  مدیریت پوزیشن باز
        # ══════════════════════════════════════════════════════════
        pos = acc['open_pos']
        if pos is not None:
            hi = high_a[bar]
            lo = low_a[bar]
            cp = close_a[bar]
            d  = pos['dir']
            ep = pos['entry']
            sl = pos['sl']
            tp = pos['tp']

            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            # Z-exit: بازگشت به میانگین
            zn = z_a[bar]
            if not np.isnan(zn) and abs(zn) < C.arb_z_exit:
                hit_tp = True

            # SL و TP هر دو در یک کندل → SL (بدبینانه)
            if hit_sl and hit_tp:
                hit_tp = False

            # ── Intra-candle worst-case equity برای DD check ──
            # بدبینانه‌ترین حالت: SL زده شده
            if not hit_sl:
                # worst case اگر SL زده می‌شد
                worst_raw = d * (sl - ep) * pos['lot'] * ls
                worst_pnl = worst_raw - trade_cost(pos['lot'], C)
                worst_eq  = acc['equity'] + worst_pnl

                # بررسی blown با worst-case equity
                blown, reason = check_prop_rules(
                    worst_eq, day_start_eq, PROP_FLOOR, C
                )
                if blown:
                    # force close روی SL
                    raw = d * (sl - ep) * pos['lot'] * ls
                    pnl = raw - trade_cost(pos['lot'], C)
                    acc['equity'] += pnl
                    rec = {**pos, 'exit': sl, 'exit_ts': ts,
                           'pnl': pnl, 'status': 'blown_SL'}
                    acc['trades'].append(rec)
                    all_trades.append(rec)
                    acc['open_pos'] = None
                    acc['blown']        = True
                    acc['blown_reason'] = reason
                    # به‌روزرسانی DD
                    if acc['equity'] < acc['acc_min_eq']:
                        acc['acc_min_eq'] = acc['equity']
                    if acc['acc_peak'] > 0:
                        dd_now = (acc['equity'] - acc['acc_peak']) / acc['acc_peak'] * 100
                        if dd_now < acc['max_dd_pct']:
                            acc['max_dd_pct'] = dd_now
                    continue

            # ── Trailing Stop ──
            tp_dist = abs(tp - ep)
            if tp_dist > 0:
                progress = d * (cp - ep) / tp_dist
                if progress >= 0.5:
                    # breakeven + ۱۰٪ سود
                    be = ep + d * tp_dist * 0.10
                    if d == 1:
                        # فقط sl را بالا ببر (نه پایین)
                        if be > sl:
                            pos['sl'] = be
                    else:
                        # فقط sl را پایین ببر (نه بالا)
                        if be < sl:
                            pos['sl'] = be
                if progress >= 0.8:
                    lock = ep + d * tp_dist * 0.55
                    if d == 1:
                        if lock > pos['sl']:
                            pos['sl'] = lock
                    else:
                        if lock < pos['sl']:
                            pos['sl'] = lock

            # ── Time Stop: ۳ روز (288 کندل ۱۵ دقیقه‌ای) ──
            bars_held = bar - pos['entry_bar']
            if bars_held >= 288 and not hit_tp and not hit_sl:
                raw = d * (cp - ep) * pos['lot'] * ls
                pnl = raw - trade_cost(pos['lot'], C)
                acc['equity'] += pnl
                rec = {**pos, 'exit': cp, 'exit_ts': ts,
                       'pnl': pnl, 'status': 'TimeStop'}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None

                # بررسی قوانین پس از بستن
                blown, reason = check_prop_rules(
                    acc['equity'], day_start_eq, PROP_FLOOR, C
                )
                acc['blown']        = blown
                acc['blown_reason'] = reason
                # به‌روزرسانی DD
                if acc['equity'] < acc['acc_min_eq']:
                    acc['acc_min_eq'] = acc['equity']
                if acc['acc_peak'] > 0:
                    dd_now = (acc['equity'] - acc['acc_peak']) / acc['acc_peak'] * 100
                    if dd_now < acc['max_dd_pct']:
                        acc['max_dd_pct'] = dd_now
                continue

            # ── بستن روی SL یا TP ──
            if hit_sl or hit_tp:
                exit_px = sl if hit_sl else tp
                status  = 'SL' if hit_sl else 'TP'
                raw = d * (exit_px - ep) * pos['lot'] * ls
                pnl = raw - trade_cost(pos['lot'], C)
                acc['equity'] += pnl
                rec = {**pos, 'exit': exit_px, 'exit_ts': ts,
                       'pnl': pnl, 'status': status}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None

                # به‌روزرسانی DD
                if acc['equity'] < acc['acc_min_eq']:
                    acc['acc_min_eq'] = acc['equity']
                if acc['acc_peak'] > 0:
                    dd_now = (acc['equity'] - acc['acc_peak']) / acc['acc_peak'] * 100
                    if dd_now < acc['max_dd_pct']:
                        acc['max_dd_pct'] = dd_now

                # بررسی قوانین پس از بستن
                blown, reason = check_prop_rules(
                    acc['equity'], day_start_eq, PROP_FLOOR, C
                )
                acc['blown']        = blown
                acc['blown_reason'] = reason

        # ══════════════════════════════════════════════════════════
        #  بررسی هدف برداشت (فقط وقتی پوزیشنی باز نیست)
        # ══════════════════════════════════════════════════════════
        if (acc['equity'] >= PROFIT_LEVEL
                and acc['open_pos'] is None
                and not acc['blown']):
            withdrawn = acc['equity'] - C.initial_balance
            total_withdrawn += withdrawn
            register_account(
                acc['start_ts'], ts, acc['equity'],
                total_withdrawn, acc['trades'],
                account_number, "TARGET_HIT",
                all_account_logs, acc['max_dd_pct']
            )
            print(f"    💰 اکانت #{account_number:>3} | {ts.date()} | "
                  f"برداشت: ${withdrawn:>7.2f} | "
                  f"کل: ${total_withdrawn:>9.2f}")

            account_number += 1
            acc = reset_account(ts)
            day_start_eq   = acc['equity']
            trades_today   = 0
            pending_signal = 0
            PROP_FLOOR     = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL   = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ══════════════════════════════════════════════════════════
        #  ثبت سیگنال برای کندل بعدی
        # ══════════════════════════════════════════════════════════
        if (acc['open_pos'] is None
                and not acc['blown']
                and bar in sig_bars
                and trades_today < C.arb_max_trades_day):
            pending_signal = sig_bars[bar]

    # ── پایان داده: بستن پوزیشن باز ──
    if acc['open_pos'] is not None:
        pos = acc['open_pos']
        cp  = close_a[-1]
        raw = pos['dir'] * (cp - pos['entry']) * pos['lot'] * ls
        pnl = raw - trade_cost(pos['lot'], C)
        acc['equity'] += pnl
        rec = {**pos, 'exit': cp, 'exit_ts': ts_a[-1],
               'pnl': pnl, 'status': 'EndOfData'}
        acc['trades'].append(rec)
        all_trades.append(rec)
        acc['open_pos'] = None

    register_account(
        acc['start_ts'], ts_a[-1], acc['equity'],
        total_withdrawn, acc['trades'],
        account_number, "ACTIVE/END",
        all_account_logs, acc['max_dd_pct']
    )

    return {
        'all_trades':      all_trades,
        'account_logs':    all_account_logs,
        'eq_curve':        global_eq_curve,
        'eq_ts':           global_eq_ts,
        'total_curve':     global_tot_curve,
        'total_withdrawn': total_withdrawn,
        'final_equity':    acc['equity'],
        'total_accounts':  account_number,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  آمار
# ═══════════════════════════════════════════════════════════════════════════
def compute_stats(results: dict) -> dict:
    if not results['all_trades']:
        return None

    C  = Config
    t  = pd.DataFrame(results['all_trades'])
    t['pnl']          = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts']     = pd.to_datetime(t['entry_ts'])
    t['exit_ts']      = pd.to_datetime(t['exit_ts'])
    t['duration_min'] = (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60
    al = pd.DataFrame(results['account_logs'])

    tw   = results['total_withdrawn']
    feq  = results['final_equity']
    tv   = tw + feq
    tp_  = tv - C.initial_balance
    tr   = tp_ / C.initial_balance * 100

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
           if lt['pnl'].sum() != 0 else float('inf'))
    rr  = abs(aw / al_) if al_ != 0 else 0

    # ── Max DD per-account (واقعی) ──
    # بدترین DD در هر اکانت، از ستون max_dd_pct
    if 'max_dd_pct' in al.columns:
        max_dd_per_acc = al['max_dd_pct'].min()  # منفی‌ترین
    else:
        max_dd_per_acc = 0.0

    # ── Sharpe/Sortino روی total curve ──
    rc      = pd.Series(results['total_curve']).pct_change().dropna()
    sharpe  = (rc.mean() / rc.std() * np.sqrt(252 * 96)
               if rc.std() > 0 else 0)
    neg     = rc[rc < 0]
    sortino = (rc.mean() / neg.std() * np.sqrt(252 * 96)
               if len(neg) > 1 else 0)

    n_target = int((al['reason'] == 'TARGET_HIT').sum())
    n_blown  = int(al['reason'].str.contains(
        'DailyDD|TotalDD|blown', case=False, na=False).sum())
    n_active = int((al['reason'] == 'ACTIVE/END').sum())

    # Max consecutive wins/losses
    sign = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cw = cl = mcw = mcl = 0
    for s in sign:
        if s > 0:
            cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0:
            cl += 1; cw = 0; mcl = max(mcl, cl)
        else:
            cw = cl = 0

    # ماهانه
    t['ym'] = t['entry_ts'].dt.to_period('M')
    mg = t.groupby('ym').agg(
        n    =('pnl', 'count'),
        pnl  =('pnl', 'sum'),
        wins =('pnl', lambda x: (x > 0).sum()),
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
        exp=t['pnl'].mean(),
        max_dd=max_dd_per_acc,
        sharpe=sharpe, sortino=sortino,
        mcw=mcw, mcl=mcl,
        n_accounts=results['total_accounts'],
        n_target=n_target, n_blown=n_blown, n_active=n_active,
        avg_dur=t['duration_min'].mean(),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  گزارش
# ═══════════════════════════════════════════════════════════════════════════
def print_report(s: dict) -> str:
    C  = Config
    W  = 78
    SEP = "═" * W

    def rw(lbl, val, ok=None):
        l = f"  {lbl}"
        v = str(val)
        m = "" if ok is None else (" ✅" if ok else " ❌")
        d = "·" * max(2, W - len(l) - len(v) - len(m) - 2)
        return f"{l} {d} {v}{m}"

    def box(title):
        i = f"─ {title} "
        return "┌" + i + "─" * (W - len(i) - 1) + "┐"

    bot = "└" + "─" * (W - 1) + "┘"

    passed = (
        s['total_ret'] > 0
        and s['pf'] > 1.2
        and s['n_blown'] == 0
        and s['n_target'] > 0
        and abs(s['max_dd']) <= 10.0
    )
    flag = "✅ PROP PASS" if passed else "⚠️  NEEDS REVIEW"

    lines = [
        "", SEP,
        f"  ▌  CorrArb Prop Simulator v2  —  {flag}  ▐",
        f"  ▌  {s['trades']['entry_ts'].min().date()} → "
        f"{s['trades']['exit_ts'].max().date()}  ({s['total_days']} روز)  ▐",
        SEP, "",
        box("نتایج مالی تجمیعی"),
        rw("بالانس هر اکانت",          f"${C.initial_balance:>12,.2f}"),
        rw("کل سود برداشت‌شده",        f"${s['total_withdrawn']:>+12,.2f}"),
        rw("موجودی اکانت فعلی",        f"${s['final_equity']:>12,.2f}"),
        rw("ارزش کل (برداشت + اکانت)", f"${s['total_value']:>12,.2f}"),
        rw("سود خالص کل",              f"${s['total_profit']:>+12,.2f}"),
        rw("بازده کل",                 f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه (CAGR)",       f"{s['ann_ret']:>+.2f}%"),
        bot, "",
        box("ریسک"),
        rw("Max DD per Account",
           f"{s['max_dd']:.2f}%",
           ok=(abs(s['max_dd']) <= 10.0)),
        rw("Sharpe",  f"{s['sharpe']:.2f}"),
        rw("Sortino", f"{s['sortino']:.2f}"),
        rw("Profit Factor",
           f"{s['pf']:.2f}",
           ok=(s['pf'] > 1.2)),
        bot, "",
        box("آمار اکانت‌های پراپ"),
        rw("کل اکانت‌ها",          f"{s['n_accounts']}"),
        rw("✅ Target Hit (برداشت)",
           f"{s['n_target']}",     ok=(s['n_target'] > 0)),
        rw("💥 Blown (قانون نقض)",
           f"{s['n_blown']}",      ok=(s['n_blown'] == 0)),
        rw("🔄 فعال / پایان داده", f"{s['n_active']}"),
        rw("نرخ موفقیت اکانت",
           f"{s['n_target'] / max(s['n_accounts'], 1) * 100:.1f}%"),
        bot, "",
        box("معاملات"),
        rw("تعداد کل",               f"{len(s['trades']):,}"),
        rw("Win Rate",               f"{s['win_r']:.1f}%"),
        rw("Avg Win",                f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",               f"${s['avg_l']:>+.2f}"),
        rw("Risk:Reward واقعی",      f"{s['rr']:.2f}"),
        rw("Expectancy",             f"${s['exp']:>+.2f}"),
        rw("Max Cons. Wins",         f"{s['mcw']}"),
        rw("Max Cons. Losses",       f"{s['mcl']}"),
        rw("مدت میانگین معامله",     f"{s['avg_dur']:.0f} min"),
        bot, "",
    ]

    # ── جدول جزئیات اکانت‌ها ──
    lines.append(box("جزئیات هر اکانت"))
    lines.append(
        f"  {'#':>4}  {'شروع':>10}  {'پایان':>10}  "
        f"{'PnL':>9}  {'Ret%':>6}  {'#T':>3}  "
        f"{'WR%':>5}  {'MaxDD%':>7}  وضعیت"
    )
    lines.append("  " + "─" * (W - 3))
    for _, row in s['acc_logs'].iterrows():
        r = row['reason']
        if r == 'TARGET_HIT':
            icon = "💰 WITHDRAW"
        elif any(x in str(r) for x in ['DailyDD', 'TotalDD', 'blown']):
            icon = f"💥 BLOWN ({r[:20]})"
        elif r == 'ACTIVE/END':
            icon = "🔄 ACTIVE"
        else:
            icon = f"⚠️  {r[:18]}"

        mdd = row.get('max_dd_pct', 0.0)
        lines.append(
            f"  {int(row['account']):>4}  "
            f"{str(row['start_ts'])[:10]:>10}  "
            f"{str(row['end_ts'])[:10]:>10}  "
            f"${row['pnl']:>+8.2f}  {row['ret_pct']:>+5.1f}%  "
            f"{row['trades']:>3}  {row['wr']:>4.0f}%  "
            f"{mdd:>+6.2f}%  {icon}"
        )
    lines += [bot, ""]

    # ── ماهانه ──
    lines.append(box("ماهانه"))
    lines.append(f"  {'ماه':>7}  {'#T':>3}  {'WR%':>5}  "
                 f"{'PnL':>9}  {'Ret%':>6}")
    lines.append("  " + "─" * (W - 3))
    for _, mr in s['monthly'].iterrows():
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>3}  "
            f"{mr['wr']:>4.1f}%  "
            f"${mr['pnl']:>+8.2f}  {mr['ret']:>+5.1f}%"
        )
    lines += [bot, ""]

    # ── سالانه ──
    s['trades']['yr'] = s['trades']['entry_ts'].dt.year
    yg = (
        s['trades'].groupby('yr')
        .agg(n=('pnl', 'count'), pnl=('pnl', 'sum'),
             wins=('pnl', lambda x: (x > 0).sum()))
        .reset_index()
    )
    yg['wr']  = yg['wins'] / yg['n'] * 100
    yg['ret'] = yg['pnl'] / C.initial_balance * 100

    lines.append(box("سالانه"))
    lines.append(f"  {'سال':>5}  {'#T':>5}  {'WR%':>5}  "
                 f"{'PnL':>11}  {'Ret%':>7}")
    lines.append("  " + "─" * (W - 3))
    for _, yr in yg.iterrows():
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>4.1f}%  "
            f"${yr['pnl']:>10.2f}  {yr['ret']:>+6.1f}%"
        )
    lines += [bot, ""]

    out = "\n".join(lines)
    print(out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  ذخیره خروجی‌ها
# ═══════════════════════════════════════════════════════════════════════════
def save_outputs(s: dict, report_txt: str):
    C = Config

    with open("Report_CorrArb_Prop.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    # ── Trades CSV ──
    rows = [
        ["CorrArb Prop Simulator v2"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [f"Risk={C.risk_per_trade_pct*100:.1f}%  "
         f"Spread={C.spread_pips}pip  Slip={C.slippage_pips}pip  "
         f"Comm=${C.commission_per_lot}/lot"],
        [],
        ["=== خلاصه ==="],
        ["Total Withdrawn",     round(s['total_withdrawn'], 2)],
        ["Final Equity",        round(s['final_equity'], 2)],
        ["Total Value",         round(s['total_value'], 2)],
        ["Total Return %",      round(s['total_ret'], 2)],
        ["Annual Return %",     round(s['ann_ret'], 2)],
        ["Profit Factor",       round(s['pf'], 2)],
        ["Win Rate %",          round(s['win_r'], 1)],
        ["Max DD per Acc %",    round(s['max_dd'], 2)],
        ["Sharpe",              round(s['sharpe'], 2)],
        ["Accounts Total",      s['n_accounts']],
        ["Accounts Target Hit", s['n_target']],
        ["Accounts Blown",      s['n_blown']],
        [],
        ["=== اکانت‌ها ==="],
        ["#", "Start", "End", "PnL", "Ret%",
         "Trades", "WR%", "MaxDD%", "Reason", "TotalWithdrawn"],
    ]
    for _, row in s['acc_logs'].iterrows():
        rows.append([
            row['account'],
            str(row['start_ts'])[:16],
            str(row['end_ts'])[:16],
            row['pnl'], row['ret_pct'],
            row['trades'], row['wr'],
            row.get('max_dd_pct', 0.0),
            row['reason'],
            row['total_withdrawn'],
        ])

    rows += [
        [],
        ["=== معاملات ==="],
        ["Account", "EntryTS", "ExitTS", "Side",
         "Lot", "Entry", "SL", "TP", "Exit",
         "PnL", "Status", "DurMin"],
    ]
    for _, tr in s['trades'].iterrows():
        rows.append([
            tr.get('account', ''),
            str(tr['entry_ts'])[:16],
            str(tr['exit_ts'])[:16],
            'BUY' if tr.get('dir', 0) == 1 else 'SELL',
            tr.get('lot', ''),
            round(float(tr.get('entry', 0)), 5),
            round(float(tr.get('sl', 0)), 5),
            round(float(tr.get('tp', 0)), 5),
            round(float(tr.get('exit', 0)), 5),
            round(float(tr['pnl']), 2),
            tr.get('status', ''),
            round(float(tr.get('duration_min', 0)), 0),
        ])

    pd.DataFrame(rows).to_csv(
        "Report_CorrArb_Prop.csv", index=False,
        header=False, encoding="utf-8-sig"
    )

    # ── Equity Curve CSV ──
    withdrawn_curve = [
        round(tv - ae, 2)
        for tv, ae in zip(s['total_curve'], s['eq_curve'])
    ]
    eq_df = pd.DataFrame({
        'ts':              s['eq_ts'],
        'account_equity':  s['eq_curve'],
        'total_withdrawn': withdrawn_curve,
        'total_value':     s['total_curve'],
    })
    eq_df['account_dd_pct'] = (
        (eq_df['account_equity'] - eq_df['account_equity'].cummax())
        / eq_df['account_equity'].cummax() * 100
    ).round(4)
    eq_df.to_csv("eq_CorrArb_Prop.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ ذخیره شد:")
    print(f"   📄 Report_CorrArb_Prop.txt")
    print(f"   📊 Report_CorrArb_Prop.csv  ({len(s['trades']):,} معامله)")
    print(f"   📈 eq_CorrArb_Prop.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 78)
    print("  CorrArb Prop Simulator — نسخه اصلاح‌شده v2")
    print("═" * 78)
    C = Config
    print(f"  Risk={C.risk_per_trade_pct*100:.1f}%  |  "
          f"Target=+{C.profit_target_pct*100:.0f}%  |  "
          f"DailyDD=-{C.max_daily_loss_pct*100:.0f}%  |  "
          f"TotalDD=-{C.max_total_dd_pct*100:.0f}%")
    print(f"  SL={C.arb_sl_pips:.0f}pip  |  "
          f"TP={C.arb_tp_pips:.0f}pip  |  "
          f"RR={C.arb_tp_pips/C.arb_sl_pips:.1f}  |  "
          f"Spread={C.spread_pips}pip  |  "
          f"Slip={C.slippage_pips}pip  |  "
          f"Comm=${C.commission_per_lot}/lot")
    print("═" * 78)

    t0 = datetime.now()
    df = load_data()

    signals = compute_corrarb_signals(df)

    print("\n  ▶ شبیه‌سازی پراپ...")
    t1      = datetime.now()
    results = run_prop_backtest(df, signals)
    dt      = (datetime.now() - t1).total_seconds()
    print(f"\n  ⏱ {dt:.1f}s | "
          f"معاملات: {len(results['all_trades']):,} | "
          f"اکانت‌ها: {results['total_accounts']}")

    if not results['all_trades']:
        print("\n❌ هیچ معامله‌ای انجام نشد.")
    else:
        stats = compute_stats(results)
        if stats:
            report = print_report(stats)
            save_outputs(stats, report)
            print(f"\n  ✅ کل: {(datetime.now()-t0).total_seconds():.1f}s")
