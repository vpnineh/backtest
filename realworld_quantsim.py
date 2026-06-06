"""
CorrArb Prop Simulator — v3 Professional
اصلاحات:
1. فیلتر کیفیت سیگنال قوی‌تر (Regime Detection)
2. Position Sizing پویا بر اساس volatility
3. Correlation confirmation واقعی بین EUR و GBP
4. Session filter دقیق‌تر
5. Max consecutive loss → کاهش ریسک
6. Volatility regime filter (ATR-based)
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
    profit_target_pct  = 0.05      # +5%
    max_daily_loss_pct = 0.04      # 4% روزانه (بافر امنیتی از 5%)
    max_total_dd_pct   = 0.08      # 8% کل (بافر امنیتی از 10%)

    # ── ریسک پایه ──
    risk_per_trade_pct = 0.008     # 0.8% پایه (کمتر از قبل)
    risk_min_pct       = 0.004     # حداقل ریسک در شرایط بد
    risk_max_pct       = 0.010     # حداکثر ریسک

    # ── هزینه‌ها ──
    spread_pips        = 1.2
    commission_per_lot = 7.0
    slippage_pips      = 0.3

    # ── بازار ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 2.0
    min_lot  = 0.01

    # ── Warmup ──
    warmup = 500

    # ── CorrArb ──
    arb_z_fast         = 96        # 24h
    arb_z_slow         = 480       # 5d
    arb_z_entry        = 2.2       # سخت‌گیرانه‌تر
    arb_z_exit         = 0.5
    arb_z_slow_confirm = 0.8       # تایید قوی‌تر
    arb_adx_max        = 25        # رنج‌تر
    arb_rsi_long_max   = 42        # سخت‌گیرانه‌تر
    arb_rsi_short_min  = 58
    arb_sl_pips        = 20.0
    arb_tp_pips        = 46.0      # RR = 2.3
    arb_hour_start     = 8         # London open
    arb_hour_end       = 17        # قبل از NY close
    arb_max_trades_day = 1
    arb_min_std_pct    = 0.30

    # ── Correlation filter ──
    corr_window        = 48        # پنجره همبستگی (12 ساعت)
    corr_min           = 0.70      # حداقل همبستگی EUR-GBP

    # ── Volatility regime ──
    atr_period         = 14
    atr_slow_period    = 96        # ATR میانگین بلندمدت
    atr_regime_max     = 2.0       # ATR نباید بیشتر از 2x میانگین باشد
    atr_regime_min     = 0.5       # ATR نباید کمتر از 0.5x میانگین باشد

    # ── Risk scaling بر اساس عملکرد ──
    consecutive_loss_reduce = 3    # بعد از N ضرر متوالی → کاهش ریسک
    risk_reduce_factor      = 0.6  # ریسک × 0.6 بعد از N ضرر

    # ── Time stop ──
    time_stop_bars = 192           # 2 روز (192 × 15min)


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده
# ═══════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur:
        raise FileNotFoundError("EURUSD CSV not found")
    if not files_gbp:
        raise FileNotFoundError("GBPUSD CSV not found")

    def read_pair(paths, suffix):
        frames = []
        for p in paths:
            df = pd.read_csv(p, sep=';', header=None,
                             names=['ts','o','h','l','c','v'])
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
        (l - c.shift()).abs()
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
    tr  = calc_atr(h, l, c, 1)
    s   = tr.rolling(period).sum().replace(0, np.nan)
    dip = 100 * dmp.rolling(period).sum() / s
    din = 100 * dmn.rolling(period).sum() / s
    dx  = (abs(dip - din) / (dip + din).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


def calc_rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """همبستگی rolling بین دو سری — causal (بدون look-ahead)"""
    return a.rolling(window).corr(b)


# ═══════════════════════════════════════════════════════════════════════════
#  سیگنال‌های CorrArb — v3
# ═══════════════════════════════════════════════════════════════════════════
def compute_signals(df: pd.DataFrame) -> dict:
    print("  محاسبه سیگنال‌های CorrArb v3...", end="", flush=True)
    C   = Config
    c_e = df['c_eur']
    h_e = df['h_eur']
    l_e = df['l_eur']
    c_g = df['c_gbp']

    # ── اندیکاتورهای پایه ──
    rsi  = calc_rsi(c_e, 14)
    adx  = calc_adx(h_e, l_e, c_e, 14)
    atr  = calc_atr(h_e, l_e, c_e, C.atr_period)
    atr_slow = atr.rolling(C.atr_slow_period).mean()
    hour = pd.Series(df.index.hour, index=df.index)
    dow  = pd.Series(df.index.dayofweek, index=df.index)

    # ── Z-score نسبت EUR/GBP ──
    ratio    = c_e / c_g
    z_mf     = ratio.rolling(C.arb_z_fast).mean()
    z_sf     = ratio.rolling(C.arb_z_fast).std()
    z_fast   = (ratio - z_mf) / z_sf.replace(0, np.nan)

    z_ms     = ratio.rolling(C.arb_z_slow).mean()
    z_ss     = ratio.rolling(C.arb_z_slow).std()
    z_slow   = (ratio - z_ms) / z_ss.replace(0, np.nan)

    # ── Correlation EUR vs GBP return ──
    ret_e  = c_e.pct_change()
    ret_g  = c_g.pct_change()
    corr   = calc_rolling_corr(ret_e, ret_g, C.corr_window)

    # ── فیلترها ──
    std_hist = z_sf.rolling(C.arb_z_slow).mean()
    std_ok   = z_sf > std_hist * C.arb_min_std_pct

    # Volatility regime: ATR نه خیلی زیاد نه خیلی کم
    vol_ok = (
        (atr > atr_slow * C.atr_regime_min) &
        (atr < atr_slow * C.atr_regime_max)
    )

    time_ok = (
        hour.between(C.arb_hour_start, C.arb_hour_end) &
        dow.between(0, 3)   # دوشنبه تا پنجشنبه (جمعه حذف)
    )

    adx_ok  = adx < C.arb_adx_max
    corr_ok = corr > C.corr_min   # همبستگی بالا = arbitrage معنادار

    # ── سیگنال‌ها ──
    sig = pd.Series(0, index=df.index)

    long_cond = (
        (z_fast < -C.arb_z_entry) &
        (z_slow < -C.arb_z_slow_confirm) &
        std_ok & vol_ok & time_ok & adx_ok & corr_ok &
        (rsi < C.arb_rsi_long_max)
    )
    short_cond = (
        (z_fast > C.arb_z_entry) &
        (z_slow > C.arb_z_slow_confirm) &
        std_ok & vol_ok & time_ok & adx_ok & corr_ok &
        (rsi > C.arb_rsi_short_min)
    )

    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    print(" ✓")
    print(f"  سیگنال‌ها: {int((sig != 0).sum()):,} | "
          f"Long: {int((sig == 1).sum()):,} | "
          f"Short: {int((sig == -1).sum()):,}")

    return {
        'sig':    sig,
        'z_fast': z_fast,
        'atr':    atr,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  توابع کمکی
# ═══════════════════════════════════════════════════════════════════════════
def trade_cost(lot: float) -> float:
    C = Config
    return (C.spread_pips * 2 * C.pip * lot * C.lot_size) + (C.commission_per_lot * lot)


def dynamic_lot(equity: float, sl_pips: float,
                consecutive_losses: int) -> float:
    """
    Position sizing پویا:
    - پایه: risk_per_trade_pct
    - بعد از N ضرر متوالی: کاهش ریسک
    """
    C = Config
    base_risk = C.risk_per_trade_pct

    # کاهش ریسک بعد از ضررهای متوالی
    if consecutive_losses >= C.consecutive_loss_reduce:
        base_risk = max(
            base_risk * C.risk_reduce_factor,
            C.risk_min_pct
        )

    if sl_pips <= 0:
        return C.min_lot
    raw = equity * base_risk / (sl_pips * C.pip * C.lot_size)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def check_prop(equity: float, day_start_eq: float,
               prop_floor: float) -> tuple:
    C = Config
    if day_start_eq > 0:
        daily_pct = (equity - day_start_eq) / day_start_eq
        if daily_pct <= -C.max_daily_loss_pct:
            return True, f"DailyDD {daily_pct*100:.2f}%"
    if equity <= prop_floor:
        dd = (equity - C.initial_balance) / C.initial_balance
        return True, f"TotalDD {dd*100:.2f}%"
    return False, ""


def new_account_state(ts):
    C = Config
    return {
        'equity':       C.initial_balance,
        'start_ts':     ts,
        'trades':       [],
        'open_pos':     None,
        'blown':        False,
        'blown_reason': "",
        'peak':         C.initial_balance,
        'min_eq':       C.initial_balance,
        'max_dd_pct':   0.0,
        'consec_loss':  0,   # ضررهای متوالی
        'consec_win':   0,
    }


def update_dd(acc: dict):
    eq = acc['equity']
    if eq > acc['peak']:
        acc['peak'] = eq
    if eq < acc['min_eq']:
        acc['min_eq'] = eq
    if acc['peak'] > 0:
        dd = (eq - acc['peak']) / acc['peak'] * 100
        if dd < acc['max_dd_pct']:
            acc['max_dd_pct'] = dd


def register_account(acc: dict, end_ts, total_withdrawn: float,
                     acc_num: int, reason: str, logs: list):
    C   = Config
    pnl = acc['equity'] - C.initial_balance
    wins = sum(1 for t in acc['trades'] if t.get('pnl', 0) > 0)
    wr   = wins / len(acc['trades']) * 100 if acc['trades'] else 0
    logs.append({
        'account':         acc_num,
        'start_ts':        acc['start_ts'],
        'end_ts':          end_ts,
        'initial':         C.initial_balance,
        'final':           round(acc['equity'], 2),
        'pnl':             round(pnl, 2),
        'ret_pct':         round(pnl / C.initial_balance * 100, 2),
        'trades':          len(acc['trades']),
        'wins':            wins,
        'wr':              round(wr, 1),
        'reason':          reason,
        'total_withdrawn': round(total_withdrawn, 2),
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
    atr_a   = signals['atr'].values
    ts_a    = df.index

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    total_withdrawn  = 0.0
    account_number   = 1
    all_account_logs = []
    all_trades       = []
    eq_curve         = []
    eq_ts            = []
    tot_curve        = []

    acc            = new_account_state(ts_a[C.warmup])
    cur_day        = None
    day_start_eq   = acc['equity']
    trades_today   = 0
    pending_signal = 0

    sig_bars = {
        i: int(sig_a[i])
        for i in range(C.warmup, len(ts_a) - 1)
        if sig_a[i] != 0
    }

    print(f"\n  شروع... PROP_FLOOR=${PROP_FLOOR:,.0f} | هدف=${PROFIT_LEVEL:,.0f}")

    for bar in range(C.warmup, len(ts_a)):
        ts  = ts_a[bar]
        day = ts.date()
        eq  = acc['equity']

        eq_curve.append(round(eq, 4))
        eq_ts.append(ts)
        tot_curve.append(round(eq + total_withdrawn, 4))
        update_dd(acc)

        # ── ریست روزانه ──
        if day != cur_day:
            cur_day      = day
            day_start_eq = eq
            trades_today = 0

        # ══════════════════════════════════════════════════════
        #  اکانت blown
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

            register_account(acc, ts, total_withdrawn,
                             account_number, acc['blown_reason'],
                             all_account_logs)
            print(f"    💥 #{account_number:>3} | {ts.date()} | "
                  f"${acc['equity']:>8.2f} | {acc['blown_reason']}")

            account_number += 1
            acc            = new_account_state(ts)
            day_start_eq   = acc['equity']
            trades_today   = 0
            pending_signal = 0
            PROP_FLOOR     = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL   = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ══════════════════════════════════════════════════════
        #  اجرای pending signal
        # ══════════════════════════════════════════════════════
        if (pending_signal != 0
                and acc['open_pos'] is None
                and not acc['blown']
                and trades_today < C.arb_max_trades_day):

            sv  = pending_signal
            lot = dynamic_lot(acc['equity'], C.arb_sl_pips,
                              acc['consec_loss'])
            ep  = open_a[bar] + sv * (C.slippage_pips + C.spread_pips/2) * pip
            sl  = ep - sv * C.arb_sl_pips * pip
            tp  = ep + sv * C.arb_tp_pips * pip

            # بررسی immediate SL
            hi = high_a[bar]
            lo = low_a[bar]
            imm_sl = (sv == 1 and lo <= sl) or (sv == -1 and hi >= sl)

            if not imm_sl:
                acc['open_pos'] = dict(
                    account   = account_number,
                    dir       = sv,
                    lot       = lot,
                    entry     = ep,
                    sl        = sl,
                    tp        = tp,
                    entry_ts  = ts,
                    entry_bar = bar,
                    initial_sl = sl,   # برای trailing stop
                )
                trades_today += 1

        pending_signal = 0

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
            if not np.isnan(zn) and abs(zn) < C.arb_z_exit:
                hit_tp = True

            # هر دو در یک کندل → SL
            if hit_sl and hit_tp:
                hit_tp = False

            # ── Intra-candle worst-case DD check ──
            if not hit_sl:
                worst_pnl = (d * (sl - ep) * pos['lot'] * ls
                             - trade_cost(pos['lot']))
                worst_eq  = acc['equity'] + worst_pnl
                blown, reason = check_prop(worst_eq, day_start_eq,
                                           PROP_FLOOR)
                if blown:
                    raw = d * (sl - ep) * pos['lot'] * ls
                    pnl = raw - trade_cost(pos['lot'])
                    acc['equity'] += pnl
                    rec = {**pos, 'exit': sl, 'exit_ts': ts,
                           'pnl': pnl, 'status': 'blown_SL'}
                    acc['trades'].append(rec)
                    all_trades.append(rec)
                    acc['open_pos']      = None
                    acc['blown']         = True
                    acc['blown_reason']  = reason
                    acc['consec_loss']  += 1
                    acc['consec_win']    = 0
                    update_dd(acc)
                    continue

            # ── Trailing Stop (اصلاح‌شده) ──
            tp_dist = abs(tp - ep)
            if tp_dist > 0:
                progress = d * (cp - ep) / tp_dist
                # مرحله ۱: بعد از ۵۰٪ پیشرفت → breakeven
                if progress >= 0.50:
                    be = ep + d * tp_dist * 0.08
                    if d == 1 and be > pos['sl']:
                        pos['sl'] = be
                    elif d == -1 and be < pos['sl']:
                        pos['sl'] = be
                # مرحله ۲: بعد از ۷۵٪ → قفل ۴۵٪ سود
                if progress >= 0.75:
                    lock = ep + d * tp_dist * 0.45
                    if d == 1 and lock > pos['sl']:
                        pos['sl'] = lock
                    elif d == -1 and lock < pos['sl']:
                        pos['sl'] = lock

            # ── Time Stop ──
            bars_held = bar - pos['entry_bar']
            if bars_held >= C.time_stop_bars and not hit_tp and not hit_sl:
                raw = d * (cp - ep) * pos['lot'] * ls
                pnl = raw - trade_cost(pos['lot'])
                acc['equity'] += pnl
                status = 'TP_time' if pnl > 0 else 'SL_time'
                rec = {**pos, 'exit': cp, 'exit_ts': ts,
                       'pnl': pnl, 'status': status}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None

                if pnl > 0:
                    acc['consec_win']  += 1
                    acc['consec_loss']  = 0
                else:
                    acc['consec_loss'] += 1
                    acc['consec_win']   = 0

                update_dd(acc)
                blown, reason = check_prop(acc['equity'],
                                           day_start_eq, PROP_FLOOR)
                acc['blown']        = blown
                acc['blown_reason'] = reason
                continue

            # ── بستن روی SL یا TP ──
            if hit_sl or hit_tp:
                exit_px = sl if hit_sl else tp
                status  = 'SL' if hit_sl else 'TP'
                raw = d * (exit_px - ep) * pos['lot'] * ls
                pnl = raw - trade_cost(pos['lot'])
                acc['equity'] += pnl
                rec = {**pos, 'exit': exit_px, 'exit_ts': ts,
                       'pnl': pnl, 'status': status}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None

                if pnl > 0:
                    acc['consec_win']  += 1
                    acc['consec_loss']  = 0
                else:
                    acc['consec_loss'] += 1
                    acc['consec_win']   = 0

                update_dd(acc)
                blown, reason = check_prop(acc['equity'],
                                           day_start_eq, PROP_FLOOR)
                acc['blown']        = blown
                acc['blown_reason'] = reason

        # ══════════════════════════════════════════════════════
        #  بررسی هدف برداشت
        # ══════════════════════════════════════════════════════
        if (acc['equity'] >= PROFIT_LEVEL
                and acc['open_pos'] is None
                and not acc['blown']):
            withdrawn = acc['equity'] - C.initial_balance
            total_withdrawn += withdrawn
            register_account(acc, ts, total_withdrawn,
                             account_number, "TARGET_HIT",
                             all_account_logs)
            print(f"    💰 #{account_number:>3} | {ts.date()} | "
                  f"برداشت: ${withdrawn:>7.2f} | "
                  f"کل: ${total_withdrawn:>9.2f}")
            account_number += 1
            acc            = new_account_state(ts)
            day_start_eq   = acc['equity']
            trades_today   = 0
            pending_signal = 0
            PROP_FLOOR     = C.initial_balance * (1 - C.max_total_dd_pct)
            PROFIT_LEVEL   = C.initial_balance * (1 + C.profit_target_pct)
            continue

        # ══════════════════════════════════════════════════════
        #  ثبت سیگنال بعدی
        # ══════════════════════════════════════════════════════
        if (acc['open_pos'] is None
                and not acc['blown']
                and bar in sig_bars
                and trades_today < C.arb_max_trades_day):
            pending_signal = sig_bars[bar]

    # پایان داده
    if acc['open_pos'] is not None:
        pos = acc['open_pos']
        cp  = close_a[-1]
        raw = pos['dir'] * (cp - pos['entry']) * pos['lot'] * ls
        pnl = raw - trade_cost(pos['lot'])
        acc['equity'] += pnl
        rec = {**pos, 'exit': cp, 'exit_ts': ts_a[-1],
               'pnl': pnl, 'status': 'EndOfData'}
        acc['trades'].append(rec)
        all_trades.append(rec)
        acc['open_pos'] = None

    register_account(acc, ts_a[-1], total_withdrawn,
                     account_number, "ACTIVE/END", all_account_logs)

    return {
        'all_trades':      all_trades,
        'account_logs':    all_account_logs,
        'eq_curve':        eq_curve,
        'eq_ts':           eq_ts,
        'total_curve':     tot_curve,
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
           if lt['pnl'].sum() != 0 else float('inf'))
    rr  = abs(aw / al_) if al_ != 0 else 0

    # Max DD per-account
    max_dd = al['max_dd_pct'].min() if 'max_dd_pct' in al.columns else 0.0

    rc      = pd.Series(results['total_curve']).pct_change().dropna()
    sharpe  = (rc.mean() / rc.std() * np.sqrt(252*96)
               if rc.std() > 0 else 0)
    neg     = rc[rc < 0]
    sortino = (rc.mean() / neg.std() * np.sqrt(252*96)
               if len(neg) > 1 else 0)

    n_target = int((al['reason'] == 'TARGET_HIT').sum())
    n_blown  = int(al['reason'].str.contains(
        'DailyDD|TotalDD|blown', case=False, na=False).sum())
    n_active = int((al['reason'] == 'ACTIVE/END').sum())

    sign = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cw = cl = mcw = mcl = 0
    for s in sign:
        if s > 0:   cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0: cl += 1; cw = 0; mcl = max(mcl, cl)
        else:       cw = cl = 0

    t['ym'] = t['entry_ts'].dt.to_period('M')
    mg = t.groupby('ym').agg(
        n   =('pnl','count'),
        pnl =('pnl','sum'),
        wins=('pnl', lambda x: (x>0).sum()),
    ).reset_index()
    mg['wr']  = mg['wins'] / mg['n'] * 100
    mg['ret'] = mg['pnl'] / C.initial_balance * 100

    # آمار ماهانه برای گزارش پایداری
    monthly_pos  = int((mg['pnl'] > 0).sum())
    monthly_neg  = int((mg['pnl'] < 0).sum())
    monthly_avg  = mg['ret'].mean()
    monthly_std  = mg['ret'].std()
    monthly_best = mg['ret'].max()
    monthly_worst= mg['ret'].min()

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
        monthly_pos=monthly_pos, monthly_neg=monthly_neg,
        monthly_avg=monthly_avg, monthly_std=monthly_std,
        monthly_best=monthly_best, monthly_worst=monthly_worst,
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

    def box(t):
        i = f"─ {t} "
        return "┌" + i + "─" * (W - len(i) - 1) + "┐"

    bot = "└" + "─" * (W - 1) + "┘"

    dd_ok     = abs(s['max_dd']) <= 8.0
    pf_ok     = s['pf'] > 1.3
    blown_ok  = s['n_blown'] == 0
    target_ok = s['n_target'] > 0
    monthly_ok= s['monthly_worst'] > -5.0

    passed = dd_ok and pf_ok and blown_ok and target_ok and monthly_ok
    flag   = "✅ PROP PASS" if passed else "⚠️  NEEDS REVIEW"

    lines = [
        "", SEP,
        f"  ▌  CorrArb Prop Simulator v3  —  {flag}  ▐",
        f"  ▌  {s['trades']['entry_ts'].min().date()} → "
        f"{s['trades']['exit_ts'].max().date()}"
        f"  ({s['total_days']} روز | "
        f"{s['total_days']//365} سال)  ▐",
        SEP, "",

        box("نتایج مالی"),
        rw("بالانس هر اکانت",           f"${C.initial_balance:>12,.2f}"),
        rw("کل سود برداشت‌شده",         f"${s['total_withdrawn']:>+12,.2f}"),
        rw("موجودی اکانت فعلی",         f"${s['final_equity']:>12,.2f}"),
        rw("ارزش کل (برداشت + اکانت)",  f"${s['total_value']:>12,.2f}"),
        rw("سود خالص کل",               f"${s['total_profit']:>+12,.2f}"),
        rw("بازده کل",                  f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه (CAGR)",        f"{s['ann_ret']:>+.2f}%"),
        bot, "",

        box("ریسک — مهم‌ترین بخش"),
        rw("Max DD per Account (باید < 8%)", f"{s['max_dd']:.2f}%",
           ok=dd_ok),
        rw("بدترین ماه",
           f"{s['monthly_worst']:.2f}%",    ok=monthly_ok),
        rw("Sharpe Ratio",                  f"{s['sharpe']:.2f}"),
        rw("Sortino Ratio",                 f"{s['sortino']:.2f}"),
        rw("Profit Factor",                 f"{s['pf']:.2f}", ok=pf_ok),
        bot, "",

        box("پایداری ماهانه"),
        rw("میانگین بازده ماهانه",       f"{s['monthly_avg']:>+.2f}%"),
        rw("انحراف معیار ماهانه",        f"{s['monthly_std']:.2f}%"),
        rw("ماه‌های سودده",              f"{s['monthly_pos']}"),
        rw("ماه‌های زیان‌ده",            f"{s['monthly_neg']}"),
        rw("بهترین ماه",                 f"{s['monthly_best']:>+.2f}%"),
        rw("بدترین ماه",                 f"{s['monthly_worst']:>+.2f}%"),
        bot, "",

        box("آمار اکانت‌های پراپ"),
        rw("کل اکانت‌ها",              f"{s['n_accounts']}"),
        rw("✅ Target Hit",
           f"{s['n_target']}",          ok=target_ok),
        rw("💥 Blown",
           f"{s['n_blown']}",           ok=blown_ok),
        rw("🔄 فعال / پایان داده",     f"{s['n_active']}"),
        rw("نرخ موفقیت اکانت",
           f"{s['n_target']/max(s['n_accounts'],1)*100:.1f}%"),
        bot, "",

        box("معاملات"),
        rw("تعداد کل",                  f"{len(s['trades']):,}"),
        rw("Win Rate",                   f"{s['win_r']:.1f}%"),
        rw("Avg Win",                    f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",                   f"${s['avg_l']:>+.2f}"),
        rw("Risk:Reward واقعی",          f"{s['rr']:.2f}"),
        rw("Expectancy",                 f"${s['exp']:>+.2f}"),
        rw("Max Cons. Wins",             f"{s['mcw']}"),
        rw("Max Cons. Losses",           f"{s['mcl']}"),
        rw("مدت میانگین معامله",         f"{s['avg_dur']:.0f} min"),
        bot, "",
    ]

    # جزئیات اکانت‌ها
    lines.append(box("جزئیات هر اکانت"))
    lines.append(
        f"  {'#':>4}  {'شروع':>10}  {'پایان':>10}  "
        f"{'PnL':>9}  {'Ret%':>6}  {'#T':>3}  "
        f"{'WR%':>5}  {'MaxDD':>7}  وضعیت"
    )
    lines.append("  " + "─" * (W - 3))
    for _, row in s['acc_logs'].iterrows():
        r = row['reason']
        icon = ("💰 WITHDRAW"  if r == 'TARGET_HIT' else
                "🔄 ACTIVE"   if r == 'ACTIVE/END'  else
                f"💥 {r[:22]}")
        mdd = row.get('max_dd_pct', 0.0)
        dd_flag = "⚠️" if abs(mdd) > 6 else ""
        lines.append(
            f"  {int(row['account']):>4}  "
            f"{str(row['start_ts'])[:10]:>10}  "
            f"{str(row['end_ts'])[:10]:>10}  "
            f"${row['pnl']:>+8.2f}  {row['ret_pct']:>+5.1f}%  "
            f"{row['trades']:>3}  {row['wr']:>4.0f}%  "
            f"{mdd:>+6.2f}%{dd_flag}  {icon}"
        )
    lines += [bot, ""]

    # ماهانه
    lines.append(box("ماهانه"))
    lines.append(f"  {'ماه':>7}  {'#T':>3}  {'WR%':>5}  "
                 f"{'PnL':>9}  {'Ret%':>7}  وضعیت")
    lines.append("  " + "─" * (W - 3))
    for _, mr in s['monthly'].iterrows():
        flag_m = "🟢" if mr['ret'] > 0 else "🔴"
        lines.append(
            f"  {str(mr['ym']):>7}  {int(mr['n']):>3}  "
            f"{mr['wr']:>4.1f}%  "
            f"${mr['pnl']:>+8.2f}  {mr['ret']:>+6.2f}%  {flag_m}"
        )
    lines += [bot, ""]

    # سالانه
    s['trades']['yr'] = s['trades']['entry_ts'].dt.year
    yg = (s['trades'].groupby('yr')
          .agg(n=('pnl','count'), pnl=('pnl','sum'),
               wins=('pnl', lambda x: (x>0).sum()))
          .reset_index())
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
#  ذخیره
# ═══════════════════════════════════════════════════════════════════════════
def save_outputs(s: dict, report_txt: str):
    C = Config
    with open("Report_CorrArb_v3.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    rows = [
        ["CorrArb Prop Simulator v3"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [], ["=== Summary ==="],
        ["Total Withdrawn",     round(s['total_withdrawn'], 2)],
        ["Final Equity",        round(s['final_equity'], 2)],
        ["Total Value",         round(s['total_value'], 2)],
        ["Total Return %",      round(s['total_ret'], 2)],
        ["CAGR %",              round(s['ann_ret'], 2)],
        ["Profit Factor",       round(s['pf'], 2)],
        ["Win Rate %",          round(s['win_r'], 1)],
        ["Max DD per Acc %",    round(s['max_dd'], 2)],
        ["Monthly Avg %",       round(s['monthly_avg'], 2)],
        ["Monthly Worst %",     round(s['monthly_worst'], 2)],
        ["Sharpe",              round(s['sharpe'], 2)],
        ["Accounts Total",      s['n_accounts']],
        ["Accounts Hit",        s['n_target']],
        ["Accounts Blown",      s['n_blown']],
        [], ["=== Accounts ==="],
        ["#","Start","End","PnL","Ret%","Trades","WR%",
         "MaxDD%","Reason","TotalWithdrawn"],
    ]
    for _, r in s['acc_logs'].iterrows():
        rows.append([
            r['account'], str(r['start_ts'])[:16], str(r['end_ts'])[:16],
            r['pnl'], r['ret_pct'], r['trades'], r['wr'],
            r.get('max_dd_pct', 0), r['reason'], r['total_withdrawn'],
        ])
    rows += [[], ["=== Trades ==="],
             ["Acc","EntryTS","ExitTS","Side","Lot",
              "Entry","SL","TP","Exit","PnL","Status","DurMin"]]
    for _, tr in s['trades'].iterrows():
        rows.append([
            tr.get('account',''),
            str(tr['entry_ts'])[:16], str(tr['exit_ts'])[:16],
            'BUY' if tr.get('dir',0)==1 else 'SELL',
            tr.get('lot',''),
            round(float(tr.get('entry',0)),5),
            round(float(tr.get('sl',0)),5),
            round(float(tr.get('tp',0)),5),
            round(float(tr.get('exit',0)),5),
            round(float(tr['pnl']),2),
            tr.get('status',''),
            round(float(tr.get('duration_min',0)),0),
        ])
    pd.DataFrame(rows).to_csv("Report_CorrArb_v3.csv",
                               index=False, header=False,
                               encoding="utf-8-sig")

    wc = [round(tv-ae,2) for tv,ae in zip(s['total_curve'],s['eq_curve'])]
    eq_df = pd.DataFrame({
        'ts':              s['eq_ts'],
        'account_equity':  s['eq_curve'],
        'total_withdrawn': wc,
        'total_value':     s['total_curve'],
    })
    eq_df.to_csv("eq_CorrArb_v3.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ ذخیره شد: Report_CorrArb_v3.txt | "
          f"Report_CorrArb_v3.csv | eq_CorrArb_v3.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 82)
    print("  CorrArb Prop Simulator v3 — Professional")
    print("═" * 82)
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
