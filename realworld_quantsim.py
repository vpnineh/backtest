"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Z-EV Prop Trading Simulator — موتور معاملاتی مبتنی بر امید ریاضی     ║
║  • استراتژی: Single-Leg Mean Reversion با فیلتر Cross-Asset                  ║
║  • کاهش هزینه: تحلیل روی یورو و پوند، اما معامله فقط روی یورو (نصف شدن هزینه)║
║  • شبیه‌سازی قطعی: بررسی حد سود و ضرر فقط روی Close کندل ۱۵ دقیقه‌ای         ║
║  • قوانین پراپ: Max DD 10% / Daily DD 5% / Target 5%                         ║
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
    initial_balance      = 5_000.0    # بالانس هر اکانت جدید
    profit_target_pct    = 0.05       # 5% سود → برداشت + اکانت جدید
    max_daily_loss_pct   = 0.05       # 5% از بالانس ابتدای روز
    max_total_dd_pct     = 0.10       # 10% از بالانس اولیه ($5,000 ثابت)

    # ── ریسک معامله (کاهش یافته برای بقا در پراپ) ──
    risk_per_trade_pct   = 0.005      # 0.5% ریسک دلاری برای هر معامله

    # ── هزینه‌های معامله ──
    spread_eur_pips      = 1.0        # EUR/USD spread
    commission_per_lot   = 6.0        # کمیسیون هر لات

    # ── مشخصات بازار ──
    pip                  = 0.0001
    lot_size             = 100_000
    max_lot              = 3.0

    # ── اندیکاتورها ──
    warmup               = 500        # کندل warmup

    # ── پارامترهای استراتژی Z-EV ──
    z_period             = 192        # میانگین متحرک (حدود ۲ روز کاری)
    z_entry              = 2.2        # انحراف معیار برای ورود (Extreme)
    z_exit               = 0.0        # بازگشت کامل به میانگین (خروج)
    adx_period           = 14
    adx_max              = 25         # فقط در بازارهای رنج (جلوگیری از ترندهای کشنده)
    sl_multiplier        = 1.5        # ضریب ATR برای استاپ لاس داینامیک
    tp_multiplier        = 2.5        # ضریب ATR برای تارگت اولیه
    hour_start           = 2          # شروع سشن لندن/فرانکفورت
    hour_end             = 18         # پایان سشن نیویورک


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده
# ═══════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))

    if not files_eur: raise FileNotFoundError("❌ فایل EURUSD پیدا نشد.")
    if not files_gbp: raise FileNotFoundError("❌ فایل GBPUSD پیدا نشد.")

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
    raw = eur.join(gbp, how='inner').dropna()

    df = pd.DataFrame({
        'o_eur': raw['o_eur'].resample('15min').first(),
        'h_eur': raw['h_eur'].resample('15min').max(),
        'l_eur': raw['l_eur'].resample('15min').min(),
        'c_eur': raw['c_eur'].resample('15min').last(),
        'o_gbp': raw['o_gbp'].resample('15min').first(),
        'c_gbp': raw['c_gbp'].resample('15min').last(),
    }).dropna()

    df = df[df.index.weekday < 5]
    print(f"✅ {len(df):,} کندل پردازش شد.")
    return df


def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    up = h.diff(); dn = -l.diff()
    dmp = up.where((up > dn) & (up > 0), 0.0)
    dmn = dn.where((dn > up) & (dn > 0), 0.0)
    tr = calc_atr(h, l, c, 1)
    s = tr.rolling(period).sum().replace(0, np.nan)
    dip = 100 * dmp.rolling(period).sum() / s
    din = 100 * dmn.rolling(period).sum() / s
    return (abs(dip - din) / (dip + din).replace(0, np.nan) * 100).rolling(period).mean()


# ═══════════════════════════════════════════════════════════════════════════
#  تولید سیگنال (امید ریاضی مثبت)
# ═══════════════════════════════════════════════════════════════════════════
def compute_signals(df: pd.DataFrame) -> dict:
    print("  استخراج لبه‌های آماری و تولید سیگنال...", end="", flush=True)
    C = Config
    c_e = df['c_eur']; h_e = df['h_eur']; l_e = df['l_eur']
    
    # 1. Z-Score یورو
    rolling_mean = c_e.rolling(C.z_period).mean()
    rolling_std  = c_e.rolling(C.z_period).std()
    z_eur = (c_e - rolling_mean) / rolling_std.replace(0, np.nan)
    
    # 2. فیلتر همبستگی (Cross-Asset): آیا پوند هم در حال استراحت/واگرایی است؟
    c_g = df['c_gbp']
    z_gbp = (c_g - c_g.rolling(C.z_period).mean()) / c_g.rolling(C.z_period).std().replace(0, np.nan)
    corr_ok = (z_eur * z_gbp) > 0 # هر دو باید در یک جهت کلی باشن تا تله نباشه

    # 3. فیلتر رنج بازار
    adx = calc_adx(h_e, l_e, c_e, C.adx_period)
    atr = calc_atr(h_e, l_e, c_e, C.adx_period)

    hour = pd.Series(df.index.hour, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end)
    adx_ok = adx < C.adx_max

    sig = pd.Series(0, index=df.index)
    
    # Long: یورو خیلی ریخته (Z < -2.2) در بازار رنج -> خرید یورو
    sig[(z_eur < -C.z_entry) & corr_ok & adx_ok & time_ok] = 1
    
    # Short: یورو خیلی رفته بالا (Z > 2.2) در بازار رنج -> فروش یورو
    sig[(z_eur > C.z_entry) & corr_ok & adx_ok & time_ok] = -1

    sig = sig.where(sig != sig.shift(), 0)
    
    # محاسبه فاصله استاپ لاس داینامیک بر اساس نوسان (ATR)
    sl_dist = atr * C.sl_multiplier / C.pip
    tp_dist = atr * C.tp_multiplier / C.pip

    print(" ✓")
    print(f"  کل سیگنال‌ها: {int((sig != 0).sum()):,} | Long: {int((sig==1).sum()):,} | Short: {int((sig==-1).sum()):,}")
    
    return {'sig': sig, 'z_eur': z_eur, 'sl_dist': sl_dist, 'tp_dist': tp_dist}


# ═══════════════════════════════════════════════════════════════════════════
#  موتور اجرای واقع‌بینانه پراپ (Close-Based)
# ═══════════════════════════════════════════════════════════════════════════
def run_prop_backtest(df: pd.DataFrame, signals: dict) -> dict:
    C = Config
    close_e = df['c_eur'].values
    sig_a   = signals['sig'].values
    z_a     = signals['z_eur'].values
    sl_d    = signals['sl_dist'].values
    tp_d    = signals['tp_dist'].values
    ts_a    = df.index

    total_withdrawn = 0.0
    account_number  = 1
    all_account_logs = []
    all_trades      = []
    global_eq_curve = []
    global_eq_ts    = []
    
    equity = C.initial_balance
    max_eq = equity
    open_pos = None
    cur_day = None
    day_start_eq = equity
    acc_start_ts = ts_a[C.warmup] if len(ts_a) > C.warmup else ts_a[0]
    acc_trades = []
    acc_blown = False
    
    PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)

    print(f"\n  ▶ اجرای موتور Z-EV... ریسک: {C.risk_per_trade_pct*100}% در هر ترید")

    for bar in range(C.warmup, len(ts_a)):
        ts = ts_a[bar]
        day = ts.date()
        
        global_eq_curve.append(round(equity, 4))
        global_eq_ts.append(ts)

        if day != cur_day:
            cur_day = day
            day_start_eq = equity 

        # وضعیت Blown
        if acc_blown:
            if open_pos is not None:
                equity += open_pos['stop_pnl'] # خروج با استاپ
                acc_trades.append({**open_pos, 'exit_ts': ts, 'pnl': open_pos['stop_pnl'], 'status': 'Blown_Force_Close'})
                open_pos = None
            
            _log_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, "BLOWN", all_account_logs)
            
            equity = C.initial_balance
            day_start_eq = equity
            account_number += 1
            acc_start_ts = ts
            acc_trades = []
            acc_blown = False
            continue

        # مدیریت پوزیشن باز
        if open_pos is not None:
            cp = close_e[bar]
            raw_pnl = open_pos['dir'] * (cp - open_pos['ep']) * open_pos['lot'] * C.lot_size
            net_pnl = raw_pnl - open_pos['cost']
            
            hit_sl = net_pnl <= open_pos['stop_pnl']
            hit_tp = net_pnl >= open_pos['target_pnl']
            
            # خروج استراتژیک: بازگشت کامل به میانگین (Z-Score = 0)
            hit_z = (open_pos['dir'] == 1 and z_a[bar] >= C.z_exit) or (open_pos['dir'] == -1 and z_a[bar] <= -C.z_exit)
            
            # تایم استاپ: ۲ روز کاری (۱۹۲ کندل ۱۵ دقیقه‌ای)
            hit_time = (bar - open_pos['entry_bar']) >= 192

            if hit_sl or hit_tp or hit_z or hit_time:
                status = 'SL' if hit_sl else ('TP' if hit_tp else ('Z-Mean_Exit' if hit_z else 'TimeStop'))
                final_pnl = open_pos['stop_pnl'] if hit_sl else (open_pos['target_pnl'] if hit_tp else net_pnl)
                
                equity += final_pnl
                acc_trades.append({**open_pos, 'exit_ts': ts, 'pnl': final_pnl, 'status': status})
                all_trades.append(acc_trades[-1])
                open_pos = None
                
                # چک کردن قوانین پراپ
                if equity <= PROP_FLOOR or ((equity - day_start_eq) / day_start_eq) <= -C.max_daily_loss_pct:
                    acc_blown = True

        # بررسی هدف برداشت
        if (equity - C.initial_balance) / C.initial_balance >= C.profit_target_pct and open_pos is None:
            withdrawn = equity - C.initial_balance
            total_withdrawn += withdrawn
            _log_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, "TARGET_HIT", all_account_logs)
            equity = C.initial_balance
            day_start_eq = equity
            account_number += 1
            acc_start_ts = ts
            acc_trades = []
            continue

        # ورود به پوزیشن
        if open_pos is None and not acc_blown and bar in set(np.where(sig_a != 0)[0]):
            direction = int(sig_a[bar])
            sl_pips = sl_d[bar]
            tp_pips = tp_d[bar]
            
            if sl_pips > 0 and not np.isnan(sl_pips):
                risk_money = equity * C.risk_per_trade_pct
                raw_lot = risk_money / (sl_pips * C.pip * C.lot_size)
                lot = round(float(np.clip(raw_lot, 0.01, C.max_lot)), 2)
                
                ep = close_e[bar] + direction * C.spread_eur_pips * C.pip / 2
                cost = (C.spread_eur_pips * C.pip * lot * C.lot_size) + (C.commission_per_lot * lot)
                target_money = risk_money * (tp_pips / sl_pips)

                open_pos = {
                    'account': account_number,
                    'dir': direction,
                    'lot': lot,
                    'ep': ep,
                    'cost': cost,
                    'stop_pnl': -risk_money,
                    'target_pnl': target_money,
                    'entry_ts': ts,
                    'entry_bar': bar
                }

    if open_pos is not None:
        equity += net_pnl
        all_trades.append({**open_pos, 'exit_ts': ts_a[-1], 'pnl': net_pnl, 'status': 'EndOfData'})
    
    _log_account(acc_start_ts, ts_a[-1], equity, total_withdrawn, acc_trades, account_number, "ACTIVE/END", all_account_logs)
    
    return {'all_trades': all_trades, 'acc_logs': all_account_logs, 'eq_curve': global_eq_curve, 'withdrawn': total_withdrawn, 'final_eq': equity, 'total_acc': account_number}

def _log_account(sts, ets, eq, wd, trds, acc, rsn, logs):
    logs.append({
        'Account': acc, 'Start': str(sts)[:10], 'End': str(ets)[:10],
        'PnL($)': round(eq - Config.initial_balance, 2),
        'Trades': len(trds), 'Win%': round(sum(1 for t in trds if t['pnl'] > 0) / len(trds) * 100 if trds else 0, 1),
        'Status': rsn
    })

# ═══════════════════════════════════════════════════════════════════════════
#  اجرا
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_data()
    signals = compute_signals(df)
    results = run_prop_backtest(df, signals)
    
    print("\n" + "═" * 60)
    print("  نتایج واقع‌بینانه (Single-Leg EV Engine)")
    print("═" * 60)
    print(f"  مجموع برداشت خالص: ${results['withdrawn']:,.2f}")
    print(f"  تعداد کل معاملات: {len(results['all_trades'])}")
    
    logs_df = pd.DataFrame(results['acc_logs'])
    print("\n  تاریخچه اکانت‌ها:")
    print(logs_df.to_string(index=False))
