"""
CorrArb Prop Simulator — v4 Balanced
هدف: تعادل بین کیفیت و کمیت سیگنال
- DD < 8% per account
- حداقل ۸-۱۲ معامله در ماه
- CAGR هدف: ۱۵-۲۵٪
- بدون تقلب و look-ahead bias
"""

import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — تنظیم‌شده برای تعادل
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05       # +5%
    max_daily_loss_pct = 0.04       # 4% (بافر از 5%)
    max_total_dd_pct   = 0.08       # 8% (بافر از 10%)

    # ── ریسک ──
    risk_base_pct      = 0.008      # 0.8% پایه
    risk_min_pct       = 0.004      # حداقل بعد از ضرر
    risk_max_pct       = 0.010      # حداکثر بعد از برد

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

    # ══════════════════════════════════════════════════════════
    #  استراتژی: ترکیب سه لایه سیگنال
    #
    #  لایه ۱: Z-score نسبت EUR/GBP (اصلی)
    #  لایه ۲: Momentum divergence (تایید)
    #  لایه ۳: Volume/Volatility filter (فیلتر)
    # ══════════════════════════════════════════════════════════

    # Z-score — تنظیم تعادل
    z_fast_period   = 96        # 24h
    z_slow_period   = 384       # 4 روز (کمتر از قبل)
    z_entry         = 1.8       # بین 2.0 و 2.2
    z_exit          = 0.5
    z_slow_confirm  = 0.6       # بین 0.5 و 0.8

    # ADX — بازار رنج (نه ترند قوی)
    adx_max         = 28

    # RSI
    rsi_long_max    = 45
    rsi_short_min   = 55

    # SL/TP
    sl_pips         = 20.0
    tp_pips         = 44.0      # RR = 2.2

    # ساعت معامله — London + NY overlap
    hour_start      = 7
    hour_end        = 18

    # روز معامله — دوشنبه تا پنجشنبه
    trade_days      = [0, 1, 2, 3]   # Mon-Thu

    # حداکثر معامله در روز
    max_trades_day  = 2              # ۲ معامله در روز

    # Volatility filter
    atr_period      = 14
    atr_ma_period   = 96
    atr_max_mult    = 2.5           # حذف extreme volatility
    atr_min_mult    = 0.4           # حذف dead market

    # Correlation
    corr_window     = 48
    corr_min        = 0.65          # کمی کمتر از v3

    # Std filter
    std_min_pct     = 0.20          # کمی کمتر از v3

    # Risk scaling
    consec_loss_n   = 3
    risk_reduce     = 0.65

    # Time stop
    time_stop_bars  = 160           # ~2.5 روز


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده
# ═══════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur:
        raise FileNotFoundError("EURUSD CSV not found in data/")
    if not files_gbp:
        raise FileNotFoundError("GBPUSD CSV not found in data/")

    def read_pair(paths, suffix):
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
    up  = h.diff()
    dn  = -l.diff()
    dmp = up.where((up > dn) & (up > 0), 0.0)
    dmn = dn.where((dn > up) & (dn > 0), 0.0)
    atr1 = calc_atr(h, l, c, 1)
    s    = atr1.rolling(period).sum().replace(0, np.nan)
    dip  = 100 * dmp.rolling(period).sum() / s
    din  = 100 * dmn.rolling(period).sum() / s
    dx   = (abs(dip - din) / (dip + din).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def calc_ema(s, period):
    return s.ewm(span=period, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  سیگنال‌ها — v4 Balanced
# ═══════════════════════════════════════════════════════════════════════════
def compute_signals(df: pd.DataFrame) -> dict:
    """
    سیگنال‌های CorrArb با سه لایه تایید:

    لایه ۱ — Z-score (اصلی):
        ratio = EUR/GBP
        Z_fast > threshold → mean-reversion انتظار داریم

    لایه ۲ — Divergence (تایید):
        EUR پایین‌تر از GBP رفته (Long EUR)
        یا EUR بالاتر از GBP رفته (Short EUR)

    لایه ۳ — فیلترها:
        ADX، RSI، ATR، Correlation، ساعت، روز
    """
    print("  محاسبه سیگنال‌های CorrArb v4...", end="", flush=True)
    C   = Config
    c_e = df['c_eur']
    h_e = df['h_eur']
    l_e = df['l_eur']
    c_g = df['c_gbp']
    h_g = df['h_gbp']
    l_g = df['l_gbp']

    # ── اندیکاتورهای پایه ──
    rsi_eur = calc_rsi(c_e, 14)
    rsi_gbp = calc_rsi(c_g, 14)
    adx_eur = calc_adx(h_e, l_e, c_e, 14)
    atr_eur = calc_atr(h_e, l_e, c_e, C.atr_period)
    atr_ma  = atr_eur.rolling(C.atr_ma_period).mean()

    # ── لایه ۱: Z-score نسبت ──
    ratio    = c_e / c_g
    z_mf     = ratio.rolling(C.z_fast_period).mean()
    z_sf     = ratio.rolling(C.z_fast_period).std()
    z_fast   = (ratio - z_mf) / z_sf.replace(0, np.nan)

    z_ms     = ratio.rolling(C.z_slow_period).mean()
    z_ss     = ratio.rolling(C.z_slow_period).std()
    z_slow   = (ratio - z_ms) / z_ss.replace(0, np.nan)

    # ── لایه ۲: Momentum divergence ──
    # EUR return vs GBP return در ۱۲ ساعت اخیر
    ret_e_12h = c_e.pct_change(48)   # ۴۸ × ۱۵min = ۱۲h
    ret_g_12h = c_g.pct_change(48)
    div_12h   = ret_e_12h - ret_g_12h

    # EUR return vs GBP return در ۲۴ ساعت
    ret_e_24h = c_e.pct_change(96)
    ret_g_24h = c_g.pct_change(96)
    div_24h   = ret_e_24h - ret_g_24h

    # ── Correlation EUR vs GBP ──
    ret_e = c_e.pct_change()
    ret_g = c_g.pct_change()
    corr  = ret_e.rolling(C.corr_window).corr(ret_g)

    # ── فیلترها ──
    std_hist = z_sf.rolling(C.z_slow_period).mean()
    std_ok   = z_sf > std_hist * C.std_min_pct

    # Volatility: نه خیلی کم نه خیلی زیاد
    vol_ok = (
        (atr_eur > atr_ma * C.atr_min_mult) &
        (atr_eur < atr_ma * C.atr_max_mult)
    )

    # زمان: London + NY
    hour     = pd.Series(df.index.hour, index=df.index)
    dow      = pd.Series(df.index.dayofweek, index=df.index)
    time_ok  = (
        hour.between(C.hour_start, C.hour_end) &
        dow.isin(C.trade_days)
    )

    adx_ok  = adx_eur < C.adx_max
    corr_ok = corr > C.corr_min

    # ── سیگنال Long ──
    # EUR ارزان‌تر شده نسبت به GBP → انتظار بازگشت
    long_cond = (
        # لایه ۱: Z اصلی
        (z_fast < -C.z_entry) &
        (z_slow < -C.z_slow_confirm) &
        # لایه ۲: divergence تایید (EUR underperformed)
        (div_12h < -0.0005) &    # EUR حداقل ۰.۰۵٪ از GBP عقب‌تر
        # لایه ۳: فیلترها
        std_ok & vol_ok & time_ok & adx_ok & corr_ok &
        (rsi_eur < C.rsi_long_max) &
        (rsi_eur < rsi_gbp - 5)   # RSI EUR < RSI GBP
    )

    # ── سیگنال Short ──
    short_cond = (
        (z_fast > C.z_entry) &
        (z_slow > C.z_slow_confirm) &
        (div_12h > 0.0005) &
        std_ok & vol_ok & time_ok & adx_ok & corr_ok &
        (rsi_eur > C.rsi_short_min) &
        (rsi_eur > rsi_gbp + 5)
    )

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1

    # حذف تکراری‌های متوالی
    sig = sig.where(sig != sig.shift(), 0)

    n_sig   = int((sig != 0).sum())
    n_long  = int((sig == 1).sum())
    n_short = int((sig == -1).sum())
    print(f" ✓")
    print(f"  سیگنال‌ها: {n_sig:,} | Long: {n_long:,} | Short: {n_short:,}")

    return {
        'sig':    sig,
        'z_fast': z_fast,
        'atr':    atr_eur,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  توابع کمکی پراپ
# ═══════════════════════════════════════════════════════════════════════════
def trade_cost(lot: float) -> float:
    C = Config
    return (C.spread_pips * 2 * C.pip * lot * C.lot_size) + (C.commission_per_lot * lot)


def calc_lot(equity: float, sl_pips: float, consec_loss: int) -> float:
    """
    Position sizing پویا:
    - پایه: risk_base_pct
    - بعد از N ضرر متوالی: کاهش
    """
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
        dd_day = (equity - day_start) / day_start
        if dd_day <= -C.max_daily_loss_pct:
            return True, f"DailyDD {dd_day*100:.2f}%"
    if equity <= prop_floor:
        dd_tot = (equity - C.initial_balance) / C.initial_balance
        return True, f"TotalDD {dd_tot*100:.2f}%"
    return False, ""


def new_acc(ts) -> dict:
    C = Config
    return {
        'equity':      C.initial_balance,
        'start_ts':    ts,
        'trades':      [],
        'open_pos':    None,
        'blown':       False,
        'blown_rsn':   "",
        'peak':        C.initial_balance,
        'min_eq':      C.initial_balance,
        'max_dd_pct':  0.0,
        'consec_loss': 0,
        'consec_win':  0,
    }


def upd_dd(acc: dict):
    eq = acc['equity']
    if eq > acc['peak']:
        acc['peak'] = eq
    if eq < acc['min_eq']:
        acc['min_eq'] = eq
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
    logs.append({
        'account':         num,
        'start_ts':        acc['start_ts'],
        'end_ts':          end_ts,
        'final':           round(acc['equity'], 2),
        'pnl':             round(pnl, 2),
        'ret_pct':         round(pnl / C.initial_balance * 100, 2),
        'trades':          len(acc['trades']),
        'wins':            w,
        'wr':              round(wr, 1),
        'reason':          reason,
        'total_withdrawn': round(tw, 2),
        'max_dd_pct':      round(acc['max_dd_pct'], 4),
    })


# ═══════════════════════════════════════════════════════════════════════════
#  موتور بک‌تست
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, signals: dict) -> dict:
    C    = Config
    pip  = C.pip
    ls   = C.lot_size

    open_a  = df['o_eur'].values
    close_a = df['c_eur'].values
    high_a  = df['h_eur'].values
    low_a   = df['l_eur'].values
    sig_a   = signals['sig'].values
    z_a     = signals['z_fast'].values
    ts_a    = df.index

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    total_withdrawn = 0.0
    acc_num         = 1
    acc_logs        = []
    all_trades      = []
    eq_curve        = []
    eq_ts_list      = []
    tot_curve       = []

    acc          = new_acc(ts_a[C.warmup])
    cur_day      = None
    day_start_eq = C.initial_balance
    trades_today = 0
    pending_sig  = 0

    sig_bars = {
        i: int(sig_a[i])
        for i in range(C.warmup, len(ts_a) - 1)
        if sig_a[i] != 0
    }

    print(f"\n  شروع... PROP_FLOOR=${PROP_FLOOR:,.0f} | "
          f"هدف=${PROFIT_LEVEL:,.0f}")

    for bar in range(C.warmup, len(ts_a)):
        ts  = ts_a[bar]
        day = ts.date()
        eq  = acc['equity']

        eq_curve.append(round(eq, 4))
        eq_ts_list.append(ts)
        tot_curve.append(round(eq + total_withdrawn, 4))
        upd_dd(acc)

        # ── ریست روزانه ──
        if day != cur_day:
            cur_day      = day
            day_start_eq = eq
            trades_today = 0

        # ══════════════════════════════════════════════════════
        #  اکانت blown → ثبت + ریست
        # ══════════════════════════════════════════════════════
        if acc['blown']:
            if acc['open_pos'] is not None:
                pos = acc['open_pos']
                cp  = close_a[bar]
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
            acc_num      += 1
            acc           = new_acc(ts)
            day_start_eq  = acc['equity']
            trades_today  = 0
            pending_sig   = 0
            PROP_FLOOR    = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL  = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ══════════════════════════════════════════════════════
        #  اجرای سیگنال pending روی open این کندل
        # ══════════════════════════════════════════════════════
        if (pending_sig != 0
                and acc['open_pos'] is None
                and not acc['blown']
                and trades_today < C.max_trades_day):

            sv  = pending_sig
            lot = calc_lot(acc['equity'], C.sl_pips, acc['consec_loss'])
            ep  = open_a[bar] + sv * (C.slippage_pips + C.spread_pips/2) * pip
            sl  = ep - sv * C.sl_pips * pip
            tp  = ep + sv * C.tp_pips * pip

            # بررسی immediate SL
            hi = high_a[bar]
            lo = low_a[bar]
            if not ((sv == 1 and lo <= sl) or (sv == -1 and hi >= sl)):
                acc['open_pos'] = dict(
                    account    = acc_num,
                    dir        = sv,
                    lot        = lot,
                    entry      = ep,
                    sl         = sl,
                    tp         = tp,
                    entry_ts   = ts,
                    entry_bar  = bar,
                    initial_sl = sl,
                )
                trades_today += 1

        pending_sig = 0

        # ══════════════════════════════════════════════════════
        #  مدیریت پوزیشن باز
        # ══════════════════════════════════════════════════════
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

            # Z-exit
            zn = z_a[bar]
            if not np.isnan(zn) and abs(zn) < C.z_exit:
                hit_tp = True

            # هر دو → SL
            if hit_sl and hit_tp:
                hit_tp = False

            # ── Intra-candle worst-case blown check ──
            if not hit_sl:
                worst_pnl = (d*(sl - ep)*pos['lot']*ls
                             - trade_cost(pos['lot']))
                blown, rsn = check_prop(
                    acc['equity'] + worst_pnl, day_start_eq, PROP_FLOOR
                )
                if blown:
                    pnl = d*(sl-ep)*pos['lot']*ls - trade_cost(pos['lot'])
                    acc['equity'] += pnl
                    rec = {**pos, 'exit': sl, 'exit_ts': ts,
                           'pnl': pnl, 'status': 'blown_SL'}
                    acc['trades'].append(rec)
                    all_trades.append(rec)
                    acc['open_pos']   = None
                    acc['blown']      = True
                    acc['blown_rsn']  = rsn
                    acc['consec_loss'] += 1
                    acc['consec_win']   = 0
                    upd_dd(acc)
                    continue

            # ── Trailing Stop ──
            tp_dist = abs(tp - ep)
            if tp_dist > 0:
                progress = d * (cp - ep) / tp_dist
                # بعد از ۵۰٪: breakeven
                if progress >= 0.50:
                    be = ep + d * tp_dist * 0.08
                    if d == 1 and be > pos['sl']:
                        pos['sl'] = be
                    elif d == -1 and be < pos['sl']:
                        pos['sl'] = be
                # بعد از ۷۵٪: قفل ۴۵٪
                if progress >= 0.75:
                    lock = ep + d * tp_dist * 0.45
                    if d == 1 and lock > pos['sl']:
                        pos['sl'] = lock
                    elif d == -1 and lock < pos['sl']:
                        pos['sl'] = lock

            # ── Time Stop ──
            if (bar - pos['entry_bar'] >= C.time_stop_bars
                    and not hit_tp and not hit_sl):
                raw = d*(cp-ep)*pos['lot']*ls
                pnl = raw - trade_cost(pos['lot'])
                acc['equity'] += pnl
                st  = 'TP_time' if pnl > 0 else 'SL_time'
                rec = {**pos, 'exit': cp, 'exit_ts': ts,
                       'pnl': pnl, 'status': st}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
                if pnl > 0:
                    acc['consec_win']  += 1
                    acc['consec_loss']  = 0
                else:
                    acc['consec_loss'] += 1
                    acc['consec_win']   = 0
                upd_dd(acc)
                blown, rsn = check_prop(
                    acc['equity'], day_start_eq, PROP_FLOOR
                )
                acc['blown']    = blown
                acc['blown_rsn'] = rsn
                continue

            # ── بستن روی SL / TP ──
            if hit_sl or hit_tp:
                exit_px = sl if hit_sl else tp
                st      = 'SL' if hit_sl else 'TP'
                raw     = d*(exit_px-ep)*pos['lot']*ls
                pnl     = raw - trade_cost(pos['lot'])
                acc['equity'] += pnl
                rec = {**pos, 'exit': exit_px, 'exit_ts': ts,
                       'pnl': pnl, 'status': st}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
                if pnl > 0:
                    acc['consec_win']  += 1
                    acc['consec_loss']  = 0
                else:
                    acc['consec_loss'] += 1
                    acc['consec_win']   = 0
                upd_dd(acc)
                blown, rsn = check_prop(
                    acc['equity'], day_start_eq, PROP_FLOOR
                )
                acc['blown']     = blown
                acc['blown_rsn'] = rsn

        # ══════════════════════════════════════════════════════
        #  هدف برداشت
        # ══════════════════════════════════════════════════════
        if (acc['equity'] >= PROFIT_LEVEL
                and acc['open_pos'] is None
                and not acc['blown']):
            w  = acc['equity'] - C.initial_balance
            total_withdrawn += w
            reg_acc(acc, ts, total_withdrawn, acc_num,
                    "TARGET_HIT", acc_logs)
            print(f"    💰 #{acc_num:>3} | {ts.date()} | "
                  f"برداشت: ${w:>7.2f} | "
                  f"کل: ${total_withdrawn:>9.2f}")
            acc_num      += 1
            acc           = new_acc(ts)
            day_start_eq  = acc['equity']
            trades_today  = 0
            pending_sig   = 0
            PROP_FLOOR    = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL  = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ── ثبت سیگنال بعدی ──
        if (acc['open_pos'] is None
                and not acc['blown']
                and bar in sig_bars
                and trades_today < C.max_trades_day):
            pending_sig = sig_bars[bar]

    # پایان داده
    if acc['open_pos'] is not None:
        pos = acc['open_pos']
        cp  = close_a[-1]
        raw = pos['dir']*(cp-pos['entry'])*pos['lot']*ls
        pnl = raw - trade_cost(pos['lot'])
        acc['equity'] += pnl
        rec = {**pos, 'exit': cp, 'exit_ts': ts_a[-1],
               'pnl': pnl, 'status': 'EndOfData'}
        acc['trades'].append(rec)
        all_trades.append(rec)
        acc['open_pos'] = None

    reg_acc(acc, ts_a[-1], total_withdrawn, acc_num,
            "ACTIVE/END", acc_logs)

    return {
        'all_trades':      all_trades,
        'account_logs':    acc_logs,
        'eq_curve':        eq_curve,
        'eq_ts':           eq_ts_list,
        'total_curve':     tot_curve,
        'total_withdrawn': total_withdrawn,
        'final_equity':    acc['equity'],
        'total_accounts':  acc_num,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  آمار جامع
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

    tw  = results['total_withdrawn']
    feq = results['final_equity']
    tv  = tw + feq
    tp_ = tv - C.initial_balance
    tr  = tp_ / C.initial_balance * 100

    sd = t['entry_ts'].min()
    ed = t['exit_ts'].max()
    td = max((ed - sd).days, 1)
    ar = ((tv / C.initial_balance) ** (365.25 / td) - 1) * 100

    wt  = t[t['pnl'] > 0]
    lt  = t[t['pnl'] < 0]
    wr  = len(wt) / len(t) * 100 if len(t) else 0
    aw  = wt['pnl'].mean() if len(wt) else 0
    al_ = lt['pnl'].mean() if len(lt) else 0
    pf  = (wt['pnl'].sum() / abs(lt['pnl'].sum())
           if lt['pnl'].sum() != 0 else float('inf'))
    rr  = abs(aw / al_) if al_ != 0 else 0

    max_dd = al['max_dd_pct'].min() if 'max_dd_pct' in al.columns else 0.0

    rc      = pd.Series(results['total_curve']).pct_change().dropna()
    sharpe  = rc.mean() / rc.std() * np.sqrt(252*96) if rc.std() > 0 else 0
    neg     = rc[rc < 0]
    sortino = (rc.mean() / neg.std() * np.sqrt(252*96)
               if len(neg) > 1 else 0)

    n_target = int((al['reason'] == 'TARGET_HIT').sum())
    n_blown  = int(al['reason'].str.contains(
        'DailyDD|TotalDD|blown', case=False, na=False).sum())
    n_active = int((al['reason'] == 'ACTIVE/END').sum())

    # consecutive
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
        n   =('pnl', 'count'),
        pnl =('pnl', 'sum'),
        wins=('pnl', lambda x: (x > 0).sum()),
    ).reset_index()
    mg['wr']  = mg['wins'] / mg['n'] * 100
    mg['ret'] = mg['pnl'] / C.initial_balance * 100

    # خلاصه ماهانه
    all_months = pd.period_range(
        start=sd.to_period('M'),
        end=ed.to_period('M'),
        freq='M'
    )
    n_total_months  = len(all_months)
    n_active_months = len(mg)
    n_pos_months    = int((mg['pnl'] > 0).sum())
    n_neg_months    = int((mg['pnl'] <= 0).sum())
    avg_monthly_ret = mg['ret'].mean()
    std_monthly_ret = mg['ret'].std()
    best_month      = mg['ret'].max()
    worst_month     = mg['ret'].min()

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
        max_dd=max_dd, sharpe=sharpe, sortino=sortino,
        mcw=mcw, mcl=mcl,
        n_accounts=results['total_accounts'],
        n_target=n_target, n_blown=n_blown, n_active=n_active,
        avg_dur=t['duration_min'].mean(),
        n_total_months=n_total_months,
        n_active_months=n_active_months,
        n_pos_months=n_pos_months,
        n_neg_months=n_neg_months,
        avg_monthly_ret=avg_monthly_ret,
        std_monthly_ret=std_monthly_ret,
        best_month=best_month,
        worst_month=worst_month,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  گزارش
# ═══════════════════════════════════════════════════════════════════════════
def print_report(s: dict) -> str:
    C   = Config
    W   = 82
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

    # معیارهای pass
    dd_ok      = abs(s['max_dd'])    <= 8.0
    pf_ok      = s['pf']             >  1.3
    blown_ok   = s['n_blown']        == 0
    target_ok  = s['n_target']       >  0
    cagr_ok    = s['ann_ret']        >  10.0
    wr_ok      = s['win_r']          >= 50.0
    worst_ok   = s['worst_month']    > -5.0

    passed = all([dd_ok, pf_ok, blown_ok, target_ok, cagr_ok])
    flag   = "✅ PROP PASS" if passed else "⚠️  NEEDS REVIEW"

    lines = [
        "", SEP,
        f"  ▌  CorrArb Prop Simulator v4  —  {flag}  ▐",
        f"  ▌  {s['trades']['entry_ts'].min().date()} → "
        f"{s['trades']['exit_ts'].max().date()}"
        f"  ({s['total_days']} روز)  ▐",
        SEP, "",

        box("نتایج مالی"),
        rw("بالانس هر اکانت",
           f"${C.initial_balance:>12,.2f}"),
        rw("کل سود برداشت‌شده",
           f"${s['total_withdrawn']:>+12,.2f}"),
        rw("موجودی اکانت فعلی",
           f"${s['final_equity']:>12,.2f}"),
        rw("ارزش کل (برداشت + اکانت)",
           f"${s['total_value']:>12,.2f}"),
        rw("سود خالص کل",
           f"${s['total_profit']:>+12,.2f}"),
        rw("بازده کل",
           f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه (CAGR)",
           f"{s['ann_ret']:>+.2f}%",        ok=cagr_ok),
        bot, "",

        box("ریسک"),
        rw("Max DD per Account",
           f"{s['max_dd']:.2f}%",           ok=dd_ok),
        rw("Sharpe Ratio",
           f"{s['sharpe']:.2f}"),
        rw("Sortino Ratio",
           f"{s['sortino']:.2f}"),
        rw("Profit Factor",
           f"{s['pf']:.2f}",                ok=pf_ok),
        bot, "",

        box("پایداری ماهانه"),
        rw("میانگین بازده ماهانه",
           f"{s['avg_monthly_ret']:>+.2f}%"),
        rw("انحراف معیار ماهانه",
           f"{s['std_monthly_ret']:.2f}%"),
        rw("ماه‌های سودده / زیان‌ده",
           f"{s['n_pos_months']} / {s['n_neg_months']}"),
        rw("بهترین ماه",
           f"{s['best_month']:>+.2f}%"),
        rw("بدترین ماه",
           f"{s['worst_month']:>+.2f}%",    ok=worst_ok),
        bot, "",

        box("آمار پراپ"),
        rw("کل اکانت‌ها",         f"{s['n_accounts']}"),
        rw("✅ Target Hit",
           f"{s['n_target']}",              ok=target_ok),
        rw("💥 Blown",
           f"{s['n_blown']}",               ok=blown_ok),
        rw("🔄 Active/End",       f"{s['n_active']}"),
        rw("نرخ موفقیت",
           f"{s['n_target']/max(s['n_accounts'],1)*100:.1f}%"),
        bot, "",

        box("معاملات"),
        rw("تعداد کل",            f"{len(s['trades']):,}"),
        rw("Win Rate",
           f"{s['win_r']:.1f}%",            ok=wr_ok),
        rw("Avg Win",             f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",            f"${s['avg_l']:>+.2f}"),
        rw("RR واقعی",            f"{s['rr']:.2f}"),
        rw("Expectancy",          f"${s['exp']:>+.2f}"),
        rw("Max Cons. Wins",      f"{s['mcw']}"),
        rw("Max Cons. Losses",    f"{s['mcl']}"),
        rw("میانگین مدت معامله",  f"{s['avg_dur']:.0f} min"),
        bot, "",
    ]

    # ── جزئیات اکانت‌ها ──
    lines.append(box("جزئیات اکانت‌ها"))
    lines.append(
        f"  {'#':>4}  {'شروع':>10}  {'پایان':>10}  "
        f"{'PnL':>9}  {'Ret%':>6}  {'#T':>3}  "
        f"{'WR%':>5}  {'MaxDD':>7}  وضعیت"
    )
    lines.append("  " + "─"*(W-3))
    for _, row in s['acc_logs'].iterrows():
        r    = row['reason']
        icon = ("💰 WITHDRAW" if r == 'TARGET_HIT' else
                "🔄 ACTIVE"  if r == 'ACTIVE/END'  else
                f"💥 {r[:20]}")
        mdd  = row.get('max_dd_pct', 0.0)
        warn = " ⚠️" if abs(mdd) > 5 else ""
        lines.append(
            f"  {int(row['account']):>4}  "
            f"{str(row['start_ts'])[:10]:>10}  "
            f"{str(row['end_ts'])[:10]:>10}  "
            f"${row['pnl']:>+8.2f}  {row['ret_pct']:>+5.1f}%  "
            f"{row['trades']:>3}  {row['wr']:>4.0f}%  "
            f"{mdd:>+6.2f}%{warn}  {icon}"
        )
    lines += [bot, ""]

    # ── ماهانه ──
    lines.append(box("ماهانه"))
    lines.append(
        f"  {'ماه':>7}  {'#T':>3}  {'WR%':>5}  "
        f"{'PnL':>9}  {'Ret%':>7}  نتیجه"
    )
    lines.append("  " + "─"*(W-3))
    for _, mr in s['monthly'].iterrows():
        icon = "🟢" if mr['ret'] > 0 else "🔴"
        warn = " ⚠️" if mr['ret'] < -4 else ""
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>3}  "
            f"{mr['wr']:>4.1f}%  "
            f"${mr['pnl']:>+8.2f}  {mr['ret']:>+6.2f}%  "
            f"{icon}{warn}"
        )
    lines += [bot, ""]

    # ── سالانه ──
    s['trades']['yr'] = s['trades']['entry_ts'].dt.year
    yg = (
        s['trades'].groupby('yr')
        .agg(n=('pnl','count'), pnl=('pnl','sum'),
             wins=('pnl', lambda x: (x>0).sum()))
        .reset_index()
    )
    yg['wr']  = yg['wins'] / yg['n'] * 100
    yg['ret'] = yg['pnl'] / C.initial_balance * 100

    lines.append(box("سالانه"))
    lines.append(
        f"  {'سال':>5}  {'#T':>5}  {'WR%':>5}  "
        f"{'PnL':>11}  {'Ret%':>7}  نتیجه"
    )
    lines.append("  " + "─"*(W-3))
    for _, yr in yg.iterrows():
        icon = "🟢" if yr['ret'] > 0 else "🔴"
        lines.append(
            f"  {int(yr['yr']):>5}  {int(yr['n']):>5}  "
            f"{yr['wr']:>4.1f}%  "
            f"${yr['pnl']:>10.2f}  {yr['ret']:>+6.1f}%  {icon}"
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

    with open("Report_CorrArb_v4.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    rows = [
        ["CorrArb Prop Simulator v4"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [],
        ["=== Summary ==="],
        ["Total Withdrawn",    round(s['total_withdrawn'], 2)],
        ["Final Equity",       round(s['final_equity'], 2)],
        ["Total Value",        round(s['total_value'], 2)],
        ["Total Return %",     round(s['total_ret'], 2)],
        ["CAGR %",             round(s['ann_ret'], 2)],
        ["Profit Factor",      round(s['pf'], 2)],
        ["Win Rate %",         round(s['win_r'], 1)],
        ["Max DD per Acc %",   round(s['max_dd'], 2)],
        ["Monthly Avg %",      round(s['avg_monthly_ret'], 2)],
        ["Monthly Worst %",    round(s['worst_month'], 2)],
        ["Sharpe",             round(s['sharpe'], 2)],
        ["Accounts Total",     s['n_accounts']],
        ["Accounts Hit",       s['n_target']],
        ["Accounts Blown",     s['n_blown']],
        [],
        ["=== Accounts ==="],
        ["#","Start","End","PnL","Ret%","Trades","WR%",
         "MaxDD%","Reason","TotalWithdrawn"],
    ]
    for _, r in s['acc_logs'].iterrows():
        rows.append([
            r['account'], str(r['start_ts'])[:16], str(r['end_ts'])[:16],
            r['pnl'], r['ret_pct'], r['trades'], r['wr'],
            r.get('max_dd_pct', 0), r['reason'], r['total_withdrawn'],
        ])

    rows += [
        [],
        ["=== Trades ==="],
        ["Acc","EntryTS","ExitTS","Side","Lot",
         "Entry","SL","TP","Exit","PnL","Status","DurMin"],
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
            round(float(tr['pnl']), 2),
            tr.get('status', ''),
            round(float(tr.get('duration_min', 0)), 0),
        ])

    pd.DataFrame(rows).to_csv(
        "Report_CorrArb_v4.csv",
        index=False, header=False, encoding="utf-8-sig"
    )

    wc = [round(tv - ae, 2)
          for tv, ae in zip(s['total_curve'], s['eq_curve'])]
    pd.DataFrame({
        'ts':              s['eq_ts'],
        'account_equity':  s['eq_curve'],
        'total_withdrawn': wc,
        'total_value':     s['total_curve'],
    }).to_csv("eq_CorrArb_v4.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ ذخیره شد:")
    print(f"   📄 Report_CorrArb_v4.txt")
    print(f"   📊 Report_CorrArb_v4.csv  ({len(s['trades']):,} معامله)")
    print(f"   📈 eq_CorrArb_v4.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 82)
    print("  CorrArb Prop Simulator v4 — Balanced")
    print("═" * 82)
    C = Config
    print(f"  Risk={C.risk_base_pct*100:.1f}%  |  "
          f"Target=+{C.profit_target_pct*100:.0f}%  |  "
          f"DailyDD=-{C.max_daily_loss_pct*100:.0f}%  |  "
          f"TotalDD=-{C.max_total_dd_pct*100:.0f}%")
    print(f"  SL={C.sl_pips:.0f}pip  |  "
          f"TP={C.tp_pips:.0f}pip  |  "
          f"RR={C.tp_pips/C.sl_pips:.1f}  |  "
          f"Spread={C.spread_pips}pip  |  "
          f"Slip={C.slippage_pips}pip  |  "
          f"Comm=${C.commission_per_lot}/lot")
    print("═" * 82)

    t0 = datetime.now()
    df = load_data()

    signals = compute_signals(df)

    print("\n  ▶ شبیه‌سازی پراپ...")
    t1      = datetime.now()
    results = run_backtest(df, signals)
    dt      = (datetime.now() - t1).total_seconds()
    print(f"\n  ⏱ {dt:.1f}s | "
          f"معاملات: {len(results['all_trades']):,} | "
          f"اکانت‌ها: {results['total_accounts']}")

    if not results['all_trades']:
        print("\n❌ هیچ معامله‌ای انجام نشد.")
    else:
        stats  = compute_stats(results)
        if stats:
            report = print_report(stats)
            save_outputs(stats, report)
            print(f"\n  ✅ کل: {(datetime.now()-t0).total_seconds():.1f}s")
