"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CorrArb Prop Trading Simulator — موتور دو-تایم‌فریمی (واقع‌بینانه)   ║
║  • استراتژی: اصلی (CorrArb) روی EUR/GBP و ترید روی EUR/USD                   ║
║  • معماری: تحلیل سیگنال در ۱۵ دقیقه / اجرای معامله و مدیریت در ۱ دقیقه       ║
║  • هدف: حذف کامل توهم پر شدن אורدرها (Phantom Fills) در بک‌تست               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG (دقیقاً پارامترهای اورجینال خودتان)
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── حساب پراپ ──
    initial_balance      = 5_000.0    
    profit_target_pct    = 0.05       
    max_daily_loss_pct   = 0.05       
    max_total_dd_pct     = 0.10       

    # ── ریسک معامله ──
    risk_per_trade_pct   = 0.010      # 1.0%

    # ── هزینه‌های معامله ──
    spread_pips          = 1.0        # EUR/USD spread
    commission_per_lot   = 6.0        

    # ── مشخصات بازار ──
    pip                  = 0.0001
    lot_size             = 100_000
    max_lot              = 2.0

    # ── اندیکاتورها ──
    warmup               = 500        # کندل 15m

    # ── CorrArb پارامترهای اصلی ──
    arb_z_fast           = 96         
    arb_z_slow           = 480        
    arb_z_entry          = 2.0        
    arb_z_exit           = 0.3        
    arb_z_slow_confirm   = 0.5        
    arb_adx_max          = 30         
    arb_rsi_long_max     = 46         
    arb_rsi_short_min    = 54         
    arb_sl_pips          = 20.0       
    arb_tp_pips          = 44.0       
    arb_hour_start       = 7          
    arb_hour_end         = 19         


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده (نگه داشتن دیتای 1 دقیقه)
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))

    if not files_eur or not files_gbp:
        raise FileNotFoundError("❌ فایل‌های CSV پیدا نشدند.")

    def read_pair(paths: list, suffix: str) -> pd.DataFrame:
        frames = []
        for p in paths:
            df = pd.read_csv(p, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
            df = df.set_index('ts')
            df = df[~df.index.duplicated(keep='last')]
            df.columns = [f'{col}_{suffix}' for col in df.columns]
            frames.append(df)
        return pd.concat(frames).sort_index()

    eur = read_pair(files_eur, 'eur')
    gbp = read_pair(files_gbp, 'gbp')
    df_1m = eur.join(gbp, how='inner').dropna()

    # حذف آخر هفته‌ها
    df_1m = df_1m[df_1m.index.weekday < 5]
    print(f"✅ {len(df_1m):,} کندل ۱ دقیقه‌ای بارگذاری شد.")
    return df_1m

# (اندیکاتورها دقیقاً مثل قبل)
def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rsi(c: pd.Series, period: int = 14):
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14):
    up = h.diff(); dn = -l.diff()
    dmp = up.where((up > dn) & (up > 0), 0.0)
    dmn = dn.where((dn > up) & (dn > 0), 0.0)
    tr = calc_atr(h, l, c, 1)
    s = tr.rolling(period).sum().replace(0, np.nan)
    dip = 100 * dmp.rolling(period).sum() / s
    din = 100 * dmn.rolling(period).sum() / s
    dx = (abs(dip - din) / (dip + din).replace(0, np.nan)) * 100
    return dx.rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  تولید سیگنال در 15m و مپ کردن به 1m
# ═══════════════════════════════════════════════════════════════════════════
def compute_dual_tf_signals(df_1m: pd.DataFrame) -> dict:
    print("  تبدیل به 15m و استخراج سیگنال‌های اصلی...", end="", flush=True)
    C = Config
    
    # 1. ساخت کندل‌های 15 دقیقه برای تحلیل (بدون Lookahead)
    df_15m = pd.DataFrame({
        'c_eur': df_1m['c_eur'].resample('15min').last(),
        'h_eur': df_1m['h_eur'].resample('15min').max(),
        'l_eur': df_1m['l_eur'].resample('15min').min(),
        'c_gbp': df_1m['c_gbp'].resample('15min').last(),
    }).dropna()

    c_e = df_15m['c_eur']; h_e = df_15m['h_eur']; l_e = df_15m['l_eur']; c_g = df_15m['c_gbp']
    
    rsi  = calc_rsi(c_e, 14)
    adx  = calc_adx(h_e, l_e, c_e, 14)
    hour = pd.Series(df_15m.index.hour, index=df_15m.index)

    eurgbp    = c_e / c_g
    z_mean_f  = eurgbp.rolling(C.arb_z_fast).mean()
    z_std_f   = eurgbp.rolling(C.arb_z_fast).std()
    z_fast    = (eurgbp - z_mean_f) / z_std_f.replace(0, np.nan)

    z_mean_s  = eurgbp.rolling(C.arb_z_slow).mean()
    z_std_s   = eurgbp.rolling(C.arb_z_slow).std()
    z_slow    = (eurgbp - z_mean_s) / z_std_s.replace(0, np.nan)

    std_ok    = z_std_f > z_std_f.rolling(C.arb_z_slow).mean() * 0.2
    time_ok   = hour.between(C.arb_hour_start, C.arb_hour_end)
    adx_ok    = adx < C.arb_adx_max

    sig_15m = pd.Series(0, index=df_15m.index)
    sig_15m[(z_fast < -C.arb_z_entry) & (z_slow < -C.arb_z_slow_confirm) & std_ok & adx_ok & time_ok & (rsi < C.arb_rsi_long_max)] = 1
    sig_15m[(z_fast > C.arb_z_entry) & (z_slow > C.arb_z_slow_confirm) & std_ok & adx_ok & time_ok & (rsi > C.arb_rsi_short_min)] = -1
    sig_15m = sig_15m.where(sig_15m != sig_15m.shift(), 0)

    # 2. مپ کردن سیگنال 15m به چارت 1m
    # با shift(1) مطمئن می‌شویم سیگنالِ ساعت 10:00 تا 10:14، دقیقاً در 10:15 روی چارت 1 دقیقه ظاهر می‌شود.
    sig_shifted = sig_15m.shift(1)
    z_shifted = z_fast.shift(1)

    df_1m['sig'] = sig_shifted.reindex(df_1m.index).fillna(0)
    # Z-score را به سمت جلو پُر می‌کنیم تا هر ۱ دقیقه بتوانیم شرط خروج را چک کنیم
    df_1m['z_fast'] = z_shifted.reindex(df_1m.index).ffill()

    print(" ✓")
    print(f"  سیگنال‌های ورود 15m (آماده اجرا در 1m): {int((df_1m['sig'] != 0).sum()):,}")
    
    return {'sig': df_1m['sig'].values, 'z_fast': df_1m['z_fast'].values}


# ═══════════════════════════════════════════════════════════════════════════
#  اجرای پراپ در تایم فریم 1 دقیقه
# ═══════════════════════════════════════════════════════════════════════════
def run_prop_backtest_1m(df_1m: pd.DataFrame, signals: dict) -> dict:
    C    = Config
    pip  = C.pip
    ls   = C.lot_size
    sp   = C.spread_pips
    comm = C.commission_per_lot

    # داده‌های 1 دقیقه برای اجرای دقیق
    close_a = df_1m['c_eur'].values
    high_a  = df_1m['h_eur'].values
    low_a   = df_1m['l_eur'].values
    sig_a   = signals['sig']
    z_a     = signals['z_fast']
    ts_a    = df_1m.index

    total_withdrawn  = 0.0
    account_number   = 1
    all_account_logs = []
    all_trades       = []
    global_eq_curve  = []
    global_eq_ts     = []

    equity      = C.initial_balance
    max_eq      = equity
    open_pos    = None
    cur_day     = None
    day_start_eq = equity

    # گرم‌آپ (باید بر اساس 1 دقیقه محاسبه شود. 500 کندل 15m = 7500 کندل 1m)
    warmup_1m = C.warmup * 15
    acc_start_ts  = ts_a[warmup_1m] if len(ts_a) > warmup_1m else ts_a[0]
    acc_trades    = []
    acc_blown     = False
    PROP_FLOOR    = C.initial_balance * (1 - C.max_total_dd_pct)

    def lot_calc(eq: float, sl_pips: float) -> float:
        raw = eq * C.risk_per_trade_pct / (sl_pips * pip * ls)
        return round(float(np.clip(raw, 0.01, C.max_lot)), 2)

    def close_position(pos: dict, exit_price: float, exit_ts, reason: str) -> float:
        raw  = pos['dir'] * (exit_price - pos['entry']) * pos['lot'] * ls
        cost = sp * pip * pos['lot'] * ls + comm * pos['lot']
        pnl  = raw - cost
        rec  = {**pos, 'exit': exit_price, 'exit_ts': exit_ts, 'pnl': pnl, 'status': reason}
        return pnl, rec

    sig_indices = set(i for i in np.where(sig_a != 0)[0] if i >= warmup_1m)
    print(f"\n  شروع شبیه‌سازی پراپ 1 دقیقه‌ای... (کد اورجینال)")

    for bar in range(warmup_1m, len(ts_a)):
        ts  = ts_a[bar]
        day = ts.date()

        global_eq_curve.append(round(equity, 4))
        global_eq_ts.append(ts)

        if day != cur_day:
            cur_day = day
            day_start_eq = equity 

        # هندل کردن اکانت Blown
        if acc_blown:
            if open_pos is not None:
                pnl, rec = close_position(open_pos, close_a[bar], ts, 'blown_close')
                equity += pnl
                acc_trades.append(rec)
                all_trades.append(rec)
                open_pos = None
            
            _log_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, "BLOWN", all_account_logs)
            
            equity = C.initial_balance
            max_eq = equity
            day_start_eq = equity
            account_number += 1
            acc_start_ts = ts
            acc_trades = []
            acc_blown = False
            PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)
            continue

        # ── مدیریت پوزیشن در تایم فریم 1 دقیقه ──
        if open_pos is not None:
            hi = high_a[bar]
            lo = low_a[bar]
            cp = close_a[bar]
            d  = open_pos['dir']
            ep = open_pos['entry']
            sl = open_pos['sl']
            tp = open_pos['tp']

            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)

            # اگر در 1 دقیقه هم تارگت بخورد هم استاپ، بدبینانه استاپ را لحاظ می‌کنیم!
            if hit_sl and hit_tp:
                hit_tp = False

            # خروج بر اساس Z-Score
            zn = z_a[bar]
            hit_z = not np.isnan(zn) and abs(zn) < C.arb_z_exit

            # Trailing Stop (از کد اصلی)
            tp_dist = abs(tp - ep)
            if tp_dist > 0:
                progress = d * (cp - ep) / tp_dist
                if progress > 0.5:
                    breakeven = ep + d * tp_dist * 0.1
                    open_pos['sl'] = max(sl, breakeven) if d == 1 else min(sl, breakeven)
                if progress > 0.8:
                    lock = ep + d * tp_dist * 0.5
                    open_pos['sl'] = max(open_pos['sl'], lock) if d == 1 else min(open_pos['sl'], lock)

            # Time Stop (در کد اصلی 384 کندل 15m بود که می‌شود 5760 کندل 1m)
            bars_held_1m = bar - open_pos['entry_bar']
            hit_time = bars_held_1m >= 5760 

            if hit_sl or hit_tp or hit_z or hit_time:
                # خروج با قیمت دقیق برخورد
                if hit_sl: exit_px = open_pos['sl']
                elif hit_tp: exit_px = tp
                else: exit_px = cp # Z-Exit یا TimeStop با قیمت کلوز همون دقیقه بسته میشن
                
                reason = 'SL' if hit_sl else ('TP' if hit_tp else ('Z-Exit' if hit_z else 'TimeStop'))
                pnl, rec = close_position(open_pos, exit_px, ts, reason)
                equity += pnl
                max_eq = max(max_eq, equity)
                acc_trades.append(rec)
                all_trades.append(rec)
                open_pos = None
                
                if equity <= PROP_FLOOR or ((equity - day_start_eq) / day_start_eq) <= -C.max_daily_loss_pct:
                    acc_blown = True

        # بررسی هدف برداشت
        if (equity - C.initial_balance) / C.initial_balance >= C.profit_target_pct and open_pos is None:
            withdrawn = equity - C.initial_balance
            total_withdrawn += withdrawn
            _log_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, "TARGET_HIT", all_account_logs)
            print(f"    💰 اکانت #{account_number:>3} | {ts.date()} | برداشت: ${withdrawn:>7.2f} | کل برداشت: ${total_withdrawn:>9.2f}")
            equity = C.initial_balance
            max_eq = equity
            day_start_eq = equity
            account_number += 1
            acc_start_ts = ts
            acc_trades = []
            PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)
            continue

        # ورود به پوزیشن
        if open_pos is None and not acc_blown and bar in sig_indices:
            sv = int(sig_a[bar])
            lot = lot_calc(equity, C.arb_sl_pips)
            ep2 = close_a[bar] + sv * sp * pip / 2  # لحاظ کردن اسپرد در قیمت ورود
            
            open_pos = {
                'account': account_number,
                'dir': sv,
                'lot': lot,
                'entry': ep2,
                'sl': ep2 - sv * C.arb_sl_pips * pip,
                'tp': ep2 + sv * C.arb_tp_pips * pip,
                'entry_ts': ts,
                'entry_bar': bar,
            }

    if open_pos is not None:
        pnl, rec = close_position(open_pos, close_a[-1], ts_a[-1], 'EndOfData')
        equity += pnl
        acc_trades.append(rec)
        all_trades.append(rec)

    _log_account(acc_start_ts, ts_a[-1], equity, total_withdrawn, acc_trades, account_number, "ACTIVE/END", all_account_logs)

    return {'all_trades': all_trades, 'acc_logs': all_account_logs, 'eq_curve': global_eq_curve, 'withdrawn': total_withdrawn, 'final_eq': equity, 'total_acc': account_number}

def _log_account(sts, ets, eq, wd, trds, acc, rsn, logs):
    C = Config
    logs.append({
        'Account': acc, 'Start': str(sts)[:10], 'End': str(ets)[:10],
        'PnL($)': round(eq - C.initial_balance, 2),
        'Ret%': round((eq - C.initial_balance) / C.initial_balance * 100, 2),
        'Trades': len(trds), 
        'Win%': round(sum(1 for t in trds if t['pnl'] > 0) / len(trds) * 100 if trds else 0, 1),
        'Status': rsn
    })

if __name__ == "__main__":
    print("═" * 76)
    print("  CorrArb Prop Simulator — موتور دو-تایم‌فریمی (حذف توهمِ پر شدن)")
    print("═" * 76)
    df_1m = load_data()
    signals = compute_dual_tf_signals(df_1m)
    results = run_prop_backtest_1m(df_1m, signals)
    
    print("\n" + "═" * 76)
    print("  خلاصه نتایج:")
    print(f"  مجموع برداشت خالص: ${results['withdrawn']:,.2f}")
    print(f"  تعداد کل معاملات: {len(results['all_trades'])}")
    
    logs_df = pd.DataFrame(results['acc_logs'])
    print("\n  تاریخچه اکانت‌ها:")
    print(logs_df.to_string(index=False))
