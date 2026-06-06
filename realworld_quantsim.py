"""
CorrArb Prop Simulator — v6 Prop Master
هدف: رعایت سخت‌گیرانه قوانین پراپ + فیلترهای پیشرفته آماری
داده: 2020 تا 2025 (فریم 15 دقیقه)
"""

import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — تنظیمات منطبق بر قوانین پراپ و استراتژی آماری
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ فرم (Strict Rules) ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05       # +5%
    max_daily_loss_pct = 0.05       # -5% (نسبت به شروع روز)
    max_total_dd_pct   = 0.10       # -10% (نسبت به بالانس اولیه)

    # ── ریسک و مدیریت سرمایه ──
    risk_base_pct      = 0.01       # ریسک پایه ۱٪
    risk_min_pct       = 0.005      # کاهش ریسک در ضرر متوالی
    consec_loss_n      = 2          # تعداد ضرر برای کاهش ریسک
    risk_reduce        = 0.5
    
    # ── هزینه‌های بروکر ──
    spread_pips        = 1.2
    commission_per_lot = 7.0        # دلار
    slippage_pips      = 0.3

    # ── مشخصات دارایی ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 5.0
    min_lot  = 0.01
    warmup   = 500

    # ── استراتژی آماری (Log Ratio) ──
    z_fast_period      = 96         # 24 ساعت
    z_entry            = 2.1        # مرز ورود
    z_exit             = 0.1        # مرز خروج (بازگشت به میانگین)
    z_stop_margin      = 4.0        # کات‌لاس داینامیک در صورت تغییر رژیم بازار
    min_net_profit_usd = 10.0       # فیلتر حداقل سود خالص (جلوگیری از درجا زدن)

    # ── فیلترها ──
    corr_period        = 96         # تایم فریم محاسبه همبستگی (24 ساعته)
    corr_min           = 0.80       # حداقل همبستگی برای ورود مجاز
    hour_start         = 2          # سشن لندن و نیویورک
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4] # دوشنبه تا جمعه
    max_trades_day     = 3

    # خروج‌های اضطراری 
    sl_pips            = 40.0
    tp_pips            = 80.0
    time_stop_bars     = 96         # حداکثر ۱ روز ماندن در پوزیشن
    
    # فیلتر نوسان (ATR)
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده
# ═══════════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    if not files_eur or not files_gbp:
        raise FileNotFoundError("CSV files not found in data/ directory.")

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
        'c_eur': raw['c_eur'].resample('15min').last(),
        'h_eur': raw['h_eur'].resample('15min').max(),
        'l_eur': raw['l_eur'].resample('15min').min(),
        
        'o_gbp': raw['o_gbp'].resample('15min').first(),
        'c_gbp': raw['c_gbp'].resample('15min').last(),
        'h_gbp': raw['h_gbp'].resample('15min').max(),
        'l_gbp': raw['l_gbp'].resample('15min').min(),
    }).dropna()

    # ایجاد دیتای کندلی سنتتیک EURGBP
    df['c_eg'] = df['c_eur'] / df['c_gbp']
    df['o_eg'] = df['o_eur'] / df['o_gbp']
    
    df = df[df.index.weekday < 5]
    print(f"✅ {len(df):,} Candles | {df.index[0].date()} → {df.index[-1].date()}")
    return df

def calc_atr(h, l, c, period=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ═══════════════════════════════════════════════════════════════════════════
#  سیگنال‌ها — با فیلتر همبستگی غلتان
# ═══════════════════════════════════════════════════════════════════════════
def compute_signals(df: pd.DataFrame) -> dict:
    print("  Computing Statistical Signals...", end="", flush=True)
    C = Config
    
    # ── Log Ratio ──
    log_ratio = np.log(df['c_eg'])
    z_mean    = log_ratio.rolling(C.z_fast_period).mean()
    z_std     = log_ratio.rolling(C.z_fast_period).std()
    z_score   = (log_ratio - z_mean) / z_std.replace(0, np.nan)

    # ── Correlation Guard (جدید) ──
    ret_eur = df['c_eur'].pct_change()
    ret_gbp = df['c_gbp'].pct_change()
    corr_series = ret_eur.rolling(C.corr_period).corr(ret_gbp)
    corr_ok = corr_series > C.corr_min

    # ── فیلتر Volatility ──
    atr_eg = calc_atr(df['h_eur']/df['l_gbp'], df['l_eur']/df['h_gbp'], df['c_eg'], C.atr_period)
    atr_ma = atr_eg.rolling(C.atr_ma_period).mean()
    vol_ok = (atr_eg > atr_ma * C.atr_min_mult) & (atr_eg < atr_ma * C.atr_max_mult)

    # ── فیلتر زمان ──
    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # ── شروط ورود ──
    long_cond  = (z_score < -C.z_entry) & vol_ok & time_ok & corr_ok
    short_cond = (z_score > C.z_entry) & vol_ok & time_ok & corr_ok

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0)

    print(f" ✓\n  Signals Generated: {int((sig != 0).sum()):,} | L: {int((sig == 1).sum())} | S: {int((sig == -1).sum())}")

    return {
        'sig':     sig,
        'z_score': z_score
    }

# ═══════════════════════════════════════════════════════════════════════════
#  توابع محاسبات مالی صحیح
# ═══════════════════════════════════════════════════════════════════════════
def calc_dynamic_pnl(dir_trade, entry_px, exit_px, lot_size, gbp_usd_rate):
    C = Config
    gross_gbp = dir_trade * (exit_px - entry_px) * lot_size * C.lot_size
    gross_usd = gross_gbp * gbp_usd_rate
    net_usd = gross_usd - (C.commission_per_lot * lot_size)
    return net_usd

def calc_lot(equity: float, sl_pips: float, consec_loss: int) -> float:
    C = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
    est_pip_value_usd = 10 * 1.25  
    raw = equity * risk / (sl_pips * est_pip_value_usd)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)

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
        'consec_loss': 0
    }

# ═══════════════════════════════════════════════════════════════════════════
#  موتور شبیه‌سازی (Strict Rules Engine)
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, signals: dict) -> dict:
    C = Config
    pip = C.pip

    open_eg  = df['o_eg'].values
    close_eg = df['c_eg'].values
    close_g  = df['c_gbp'].values 
    sig_a    = signals['sig'].values
    z_a      = signals['z_score'].values
    ts_a     = df.index

    # مرزهای ثابت بر اساس بالانس اولیه
    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    total_withdrawn = 0.0
    acc_num = 1
    acc_logs, all_trades, eq_curve, eq_ts_list, tot_curve = [], [], [], [], []

    acc = new_acc(ts_a[C.warmup])
    day_start_eq = C.initial_balance
    trades_today = 0
    pending_sig = 0

    print(f"\n  ▶ Running Strict Prop Simulator...")
    print(f"    Target: +{C.profit_target_pct*100}% | Daily DD: -{C.max_daily_loss_pct*100}% | Total DD: -{C.max_total_dd_pct*100}%")

    for bar in range(C.warmup, len(ts_a)):
        ts = ts_a[bar]
        eq = acc['equity']

        eq_curve.append(round(eq, 4))
        eq_ts_list.append(ts)
        tot_curve.append(round(eq + total_withdrawn, 4))

        if eq > acc['peak']: acc['peak'] = eq
        if eq < acc['min_eq']: acc['min_eq'] = eq
        if acc['peak'] > 0:
            dd = (eq - acc['peak']) / acc['peak'] * 100
            if dd < acc['max_dd_pct']: acc['max_dd_pct'] = dd

        # ── 리ست روزانه Daily DD (شروع روز جدید) ──
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            trades_today = 0

        # بررسی حساب سوخته (پاک‌سازی و ریست اکانت)
        if acc['blown']:
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': acc['blown_rsn'], 'pnl': eq - C.initial_balance})
            print(f"    💥 #{acc_num:>3} | {ts.date()} | Eq: ${eq:>8.2f} | {acc['blown_rsn']}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            trades_today, pending_sig = 0, 0
            continue

        # اجرای سیگنال
        if pending_sig != 0 and acc['open_pos'] is None and trades_today < C.max_trades_day:
            sv = pending_sig
            lot = calc_lot(acc['equity'], C.sl_pips, acc['consec_loss'])
            
            spread_cost_px = (C.slippage_pips + C.spread_pips/2) * pip
            ep = open_eg[bar] + (sv * spread_cost_px)
            
            sl = ep - sv * C.sl_pips * pip
            tp = ep + sv * C.tp_pips * pip

            acc['open_pos'] = {
                'dir': sv, 'lot': lot, 'entry': ep, 'sl': sl, 'tp': tp,
                'entry_ts': ts, 'entry_bar': bar
            }
            trades_today += 1
        pending_sig = 0

        # ── مدیریت پوزیشن و فیلترهای خروج ──
        pos = acc['open_pos']
        if pos is not None:
            cp = close_eg[bar]
            cg = close_g[bar]
            d  = pos['dir']
            ep = pos['entry']
            zn = z_a[bar]
            
            # محاسبه سود و زیان لحظه‌ای کندل
            current_pnl = calc_dynamic_pnl(d, ep, cp, pos['lot'], cg)
            current_eq = acc['equity'] + current_pnl
            
            # 1. تست دراداون پراپ (Daily Limit & Total Limit)
            daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)
            blown_daily = current_eq <= daily_limit
            blown_total = current_eq <= PROP_FLOOR

            if blown_daily or blown_total:
                acc['blown'] = True
                acc['blown_rsn'] = "DailyDD" if blown_daily else "TotalDD"
                # ثبت معامله در نقطه سوختن اکانت
                rec = {**pos, 'exit': cp, 'exit_ts': ts, 'pnl': current_pnl, 'status': "BLOWN"}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
                continue

            # 2. تست Z-Stop (کات لاس آماری داینامیک)
            hit_z_stop = False
            if not np.isnan(zn):
                if d == 1 and zn <= -C.z_stop_margin: hit_z_stop = True
                if d == -1 and zn >= C.z_stop_margin: hit_z_stop = True

            # 3. تست Z-Exit با فیلتر Commission Guard
            hit_z_exit = False
            if not np.isnan(zn):
                if d == 1 and zn >= -C.z_exit: hit_z_exit = True
                if d == -1 and zn <= C.z_exit: hit_z_exit = True
            
            # اگر Z-Exit فعال شد اما سود خالص کافی نبود، خروج لغو می‌شود
            if hit_z_exit and current_pnl < C.min_net_profit_usd:
                hit_z_exit = False

            # 4. تست‌های کلاسیک (TimeStop, SL, TP)
            hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
            hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])
            time_stop = (bar - pos['entry_bar']) >= C.time_stop_bars

            # خروج نهایی در صورت فعال شدن هر یک از تریگرها
            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                exit_px = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                st = 'SL' if hit_sl else ('TP' if hit_tp else ('Z-Stop' if hit_z_stop else ('TimeStop' if time_stop else 'Z-Exit')))
                
                final_pnl = calc_dynamic_pnl(d, ep, exit_px, pos['lot'], cg)
                acc['equity'] += final_pnl
                
                rec = {**pos, 'exit': exit_px, 'exit_ts': ts, 'pnl': final_pnl, 'status': st}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
                
                if final_pnl > 0: acc['consec_loss'] = 0
                else: acc['consec_loss'] += 1

        # ── برداشت سود و دریافت اکانت Fresh ──
        if acc['equity'] >= PROFIT_LEVEL and acc['open_pos'] is None and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': "TARGET_HIT", 'pnl': acc['equity'] - C.initial_balance})
            print(f"    💰 #{acc_num:>3} | {ts.date()} | Target Hit: ${w:>7.2f} | Total Bank: ${total_withdrawn:>9.2f}")
            acc_num += 1
            # ریست کامل به اکانت جدید طبق قوانین
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            trades_today, pending_sig = 0, 0
            continue

        # ثبت سیگنال بعدی
        if acc['open_pos'] is None and not acc['blown'] and trades_today < C.max_trades_day:
            if sig_a[bar] != 0:
                pending_sig = sig_a[bar]

    return {
        'all_trades': all_trades, 'account_logs': acc_logs,
        'eq_curve': eq_curve, 'total_withdrawn': total_withdrawn,
        'final_equity': acc['equity'], 'total_accounts': acc_num
    }

# ═══════════════════════════════════════════════════════════════════════════
#  توابع گزارش‌گیری
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results: dict):
    trades = results['all_trades']
    if not trades:
        print("\n❌ No trades executed.")
        return
        
    df_t = pd.DataFrame(trades)
    wins = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    
    wr = len(wins) / len(df_t) * 100
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf')
    
    # تفکیک دلایل خروج
    exit_reasons = df_t['status'].value_counts()
    
    print("\n" + "═" * 65)
    print(" ▌  CorrArb Prop Simulator v6 (Prop Master) — Report  ▐")
    print("═" * 65)
    print(f" Total Trades:    {len(df_t):,}")
    print(f" Win Rate:        {wr:.2f}%")
    print(f" Profit Factor:   {pf:.2f}")
    print(f" Total Banked:    ${results['total_withdrawn']:,.2f}")
    print(f" Active Equity:   ${results['final_equity']:,.2f}")
    print(f" Avg Win:         ${wins['pnl'].mean():.2f}")
    print(f" Avg Loss:        ${losses['pnl'].mean():.2f}")
    print("-" * 65)
    print(" خروج‌ها بر اساس نوع:")
    for status, count in exit_reasons.items():
        print(f"   {status:<10}: {count} معامله")
    print("═" * 65)

if __name__ == "__main__":
    t0 = datetime.now()
    df = load_data()
    signals = compute_signals(df)
    results = run_backtest(df, signals)
    print_report(results)
    print(f"  ✅ Executed in: {(datetime.now()-t0).total_seconds():.2f}s")
