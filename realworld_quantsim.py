"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         London Breakout Prop Simulator — موتور شکست نوسان (Fat-Tail)         ║
║  • استراتژی: Volatility Breakout روی باکس آسیا (فقط EUR/USD)                 ║
║  • معماری: ضدِ توهم (تحلیل ساختار در 15m / اجرا و مدیریت در 1m)              ║
║  • هدف: شکار ترندهای بزرگ با Risk:Reward بالا برای غلبه بر هزینه‌های پراپ    ║
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
    initial_balance      = 5_000.0    
    profit_target_pct    = 0.05       
    max_daily_loss_pct   = 0.05       
    max_total_dd_pct     = 0.10       

    risk_per_trade_pct   = 0.005      # 0.5% ریسک (چون وین‌ریت پایین‌تر اما ریوارد بالاست)
    spread_pips          = 1.0        
    commission_per_lot   = 6.0        

    pip                  = 0.0001
    lot_size             = 100_000
    max_lot              = 3.0

    # ── پارامترهای شکست لندن ──
    asian_start_hour     = 0          # شروع رنج آسیا
    asian_end_hour       = 7          # پایان رنج آسیا
    trade_start_hour     = 8          # شروع سشن لندن
    trade_end_hour       = 16         # پایان زمان مجاز ورود
    close_all_hour       = 22         # بستن اجباری تمام پوزیشن‌های باز در انتهای روز
    
    # ── مدیریت ریسک و ریوارد ──
    atr_period           = 14
    buffer_pips          = 2.0        # فیلتر شکست (قیمت باید ۲ پیپ از باکس رد بشه تا فیک‌اوت نشیم)
    sl_atr_multiplier    = 1.0        # حد ضرر = 1 برابر ATR 
    tp_atr_multiplier    = 4.0        # تارگت = 4 برابر ATR (Risk:Reward = 1:4)


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده (فقط EURUSD 1m نیاز است)
# ═══════════════════════════════════════════════════════════════════════════
def load_data():
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    if not files_eur: raise FileNotFoundError("❌ فایل EURUSD پیدا نشد.")

    frames = []
    for p in files_eur:
        df = pd.read_csv(p, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
        df = df.set_index('ts')
        df = df[~df.index.duplicated(keep='last')]
        frames.append(df)
    
    df_1m = pd.concat(frames).sort_index()
    df_1m = df_1m[df_1m.index.weekday < 5]
    print(f"✅ {len(df_1m):,} کندل ۱ دقیقه‌ای بارگذاری شد.")
    return df_1m

def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ═══════════════════════════════════════════════════════════════════════════
#  محاسبه سطوح شکست (روزانه) و انتقال به 1 دقیقه
# ═══════════════════════════════════════════════════════════════════════════
def compute_breakout_levels(df_1m: pd.DataFrame) -> pd.DataFrame:
    print("  تشخیص باکس‌های آسیا و استخراج لبه‌های معاملاتی...", end="", flush=True)
    C = Config
    
    # 1. ساخت کندل‌های 15m برای محاسبه ATR پایدار
    df_15m = pd.DataFrame({
        'h': df_1m['h'].resample('15min').max(),
        'l': df_1m['l'].resample('15min').min(),
        'c': df_1m['c'].resample('15min').last(),
    }).dropna()
    atr_15m = calc_atr(df_15m['h'], df_15m['l'], df_15m['c'], C.atr_period)
    
    df_1m['atr'] = atr_15m.reindex(df_1m.index).ffill()
    df_1m['hour'] = df_1m.index.hour
    df_1m['date'] = df_1m.index.date
    
    # 2. پیدا کردن سقف و کف باکس آسیا برای هر روز
    asian_mask = (df_1m['hour'] >= C.asian_start_hour) & (df_1m['hour'] <= C.asian_end_hour)
    asian_data = df_1m[asian_mask]
    
    daily_asian_high = asian_data.groupby('date')['h'].max()
    daily_asian_low = asian_data.groupby('date')['l'].min()
    
    df_1m['asian_high'] = df_1m['date'].map(daily_asian_high)
    df_1m['asian_low'] = df_1m['date'].map(daily_asian_low)
    
    print(" ✓")
    return df_1m

# ═══════════════════════════════════════════════════════════════════════════
#  اجرای پراپ در تایم فریم 1 دقیقه
# ═══════════════════════════════════════════════════════════════════════════
def run_prop_backtest_1m(df: pd.DataFrame) -> dict:
    C    = Config
    pip  = C.pip
    ls   = C.lot_size
    sp   = C.spread_pips
    comm = C.commission_per_lot

    close_a = df['c'].values
    high_a  = df['h'].values
    low_a   = df['l'].values
    atr_a   = df['atr'].values
    ah_a    = df['asian_high'].values
    al_a    = df['asian_low'].values
    hour_a  = df['hour'].values
    ts_a    = df.index

    total_withdrawn  = 0.0
    account_number   = 1
    all_account_logs = []
    all_trades       = []
    
    equity      = C.initial_balance
    max_eq      = equity
    open_pos    = None
    cur_day     = None
    day_start_eq = equity
    traded_today = False

    acc_start_ts  = ts_a[0]
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

    print(f"\n  شروع شبیه‌سازی 1 دقیقه‌ای (London Breakout)...")

    for bar in range(1000, len(ts_a)): # پرش از روزهای اول برای تکمیل ATR
        ts  = ts_a[bar]
        day = ts.date()

        # ریست روزانه
        if day != cur_day:
            cur_day = day
            day_start_eq = equity 
            traded_today = False

        # اکانت Blown
        if acc_blown:
            if open_pos is not None:
                pnl, rec = close_position(open_pos, close_a[bar], ts, 'blown_close')
                equity += pnl
                acc_trades.append(rec); all_trades.append(rec)
                open_pos = None
            
            _log_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, "BLOWN", all_account_logs)
            
            equity = C.initial_balance; max_eq = equity; day_start_eq = equity
            account_number += 1; acc_start_ts = ts; acc_trades = []; acc_blown = False
            continue

        # ── مدیریت پوزیشن ──
        if open_pos is not None:
            hi = high_a[bar]; lo = low_a[bar]; cp = close_a[bar]
            d  = open_pos['dir']; ep = open_pos['entry']
            sl = open_pos['sl']; tp = open_pos['tp']

            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)
            if hit_sl and hit_tp: hit_tp = False # بدبینانه

            # End of Day Exit (بستن قبل از رول‌اور برای جلوگیری از سواپ و گپ)
            hit_eod = hour_a[bar] >= C.close_all_hour

            if hit_sl or hit_tp or hit_eod:
                if hit_sl: exit_px = sl
                elif hit_tp: exit_px = tp
                else: exit_px = cp
                
                reason = 'SL' if hit_sl else ('TP' if hit_tp else 'End_Of_Day')
                pnl, rec = close_position(open_pos, exit_px, ts, reason)
                equity += pnl; max_eq = max(max_eq, equity)
                acc_trades.append(rec); all_trades.append(rec)
                open_pos = None
                
                if equity <= PROP_FLOOR or ((equity - day_start_eq) / day_start_eq) <= -C.max_daily_loss_pct:
                    acc_blown = True

        # بررسی هدف برداشت
        if (equity - C.initial_balance) / C.initial_balance >= C.profit_target_pct and open_pos is None:
            total_withdrawn += (equity - C.initial_balance)
            _log_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, "TARGET_HIT", all_account_logs)
            equity = C.initial_balance; day_start_eq = equity; account_number += 1
            acc_start_ts = ts; acc_trades = []
            continue

        # ── ورود به پوزیشن ──
        # فقط یک ترید در روز (جلوگیری از انتقام‌گیری از بازار)، فقط در سشن لندن/نیویورک
        if open_pos is None and not acc_blown and not traded_today:
            hr = hour_a[bar]
            if C.trade_start_hour <= hr < C.trade_end_hour:
                curr_c = close_a[bar]
                a_high = ah_a[bar]
                a_low  = al_a[bar]
                curr_atr = atr_a[bar]
                
                if np.isnan(curr_atr) or np.isnan(a_high): continue

                buffer = C.buffer_pips * pip
                sl_dist = curr_atr * C.sl_atr_multiplier
                tp_dist = curr_atr * C.tp_atr_multiplier

                # شرط Long: شکست سقف باکس آسیا
                if curr_c > (a_high + buffer):
                    lot = lot_calc(equity, sl_dist / pip)
                    ep = curr_c + (sp * pip / 2)
                    open_pos = {'account': account_number, 'dir': 1, 'lot': lot, 'entry': ep,
                                'sl': ep - sl_dist, 'tp': ep + tp_dist, 'entry_ts': ts, 'entry_bar': bar}
                    traded_today = True

                # شرط Short: شکست کف باکس آسیا
                elif curr_c < (a_low - buffer):
                    lot = lot_calc(equity, sl_dist / pip)
                    ep = curr_c - (sp * pip / 2)
                    open_pos = {'account': account_number, 'dir': -1, 'lot': lot, 'entry': ep,
                                'sl': ep + sl_dist, 'tp': ep - tp_dist, 'entry_ts': ts, 'entry_bar': bar}
                    traded_today = True

    if open_pos is not None:
        pnl, rec = close_position(open_pos, close_a[-1], ts_a[-1], 'EndOfData')
        equity += pnl; acc_trades.append(rec); all_trades.append(rec)

    _log_account(acc_start_ts, ts_a[-1], equity, total_withdrawn, acc_trades, account_number, "ACTIVE/END", all_account_logs)
    return {'all_trades': all_trades, 'acc_logs': all_account_logs, 'withdrawn': total_withdrawn}

def _log_account(sts, ets, eq, wd, trds, acc, rsn, logs):
    C = Config
    logs.append({
        'Account': acc, 'Start': str(sts)[:10], 'End': str(ets)[:10],
        'PnL($)': round(eq - C.initial_balance, 2),
        'Trades': len(trds), 
        'Win%': round(sum(1 for t in trds if t['pnl'] > 0) / len(trds) * 100 if trds else 0, 1),
        'Status': rsn
    })

if __name__ == "__main__":
    print("═" * 76)
    print("  Prop Simulator — London Volatility Breakout (واقع‌بینانه)")
    print("═" * 76)
    df = load_data()
    df = compute_breakout_levels(df)
    results = run_prop_backtest_1m(df)
    
    print("\n" + "═" * 76)
    print(f"  مجموع برداشت خالص: ${results['withdrawn']:,.2f}")
    print(f"  تعداد کل معاملات: {len(results['all_trades'])}")
    
    logs_df = pd.DataFrame(results['acc_logs'])
    print("\n  تاریخچه اکانت‌ها:")
    print(logs_df.to_string(index=False))
