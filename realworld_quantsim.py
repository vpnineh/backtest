"""
CorrArb Prop Simulator — v5 Pro-Grade
هدف: شبیه‌سازی واقعی و ریاضیاتی آربیتراژ آماری روی پراپ
اصلاحات: محاسبه دقیق ارزش پیپ متقاطع، خروج داینامیک، محاسبه واقعی Daily DD
"""

import pandas as pd
import numpy as np
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — تنظیمات بهینه‌شده
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05       # +5%
    max_daily_loss_pct = 0.045      # 4.5% (بافر ایمن از 5%)
    max_total_dd_pct   = 0.09       # 9.0% (بافر ایمن از 10%)

    # ── ریسک ──
    risk_base_pct      = 0.01       # 1.0% پایه (چون خروج داینامیک است، ضرر واقعی کمتر است)
    risk_min_pct       = 0.005      # حداقل بعد از ضرر
    
    # ── هزینه‌ها ──
    spread_pips        = 1.2
    commission_per_lot = 7.0        # دلار
    slippage_pips      = 0.3

    # ── بازار ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 5.0
    min_lot  = 0.01
    warmup   = 500

    # ── استراتژی آماری (Log Ratio) ──
    z_fast_period   = 96        # 24 ساعت
    z_entry         = 2.1       # ورود در انحراف بالا
    z_exit          = 0.1       # خروج هنگام بازگشت به میانگین (داینامیک)
    
    # خروج‌های اضطراری (Emergency) - برای محاسبه لات سایز و محافظت فاجعه
    sl_pips         = 40.0
    tp_pips         = 80.0

    # ── فیلترها ──
    hour_start      = 2         # پوشش سشن لندن و نیویورک
    hour_end        = 19
    trade_days      = [0, 1, 2, 3, 4] # دوشنبه تا جمعه
    max_trades_day  = 3

    # Volatility Filter
    atr_period      = 14
    atr_ma_period   = 96
    atr_max_mult    = 3.0       # حذف خبرهای بسیار سنگین
    atr_min_mult    = 0.5       # حذف بازارهای کاملا مرده

    consec_loss_n   = 2
    risk_reduce     = 0.5
    time_stop_bars  = 96        # حداکثر ۱ روز نگهداری برای جلوگیری از سواپ و مارجین بیهوده


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
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ═══════════════════════════════════════════════════════════════════════════
#  سیگنال‌ها — Log Ratio Cointegration
# ═══════════════════════════════════════════════════════════════════════════
def compute_signals(df: pd.DataFrame) -> dict:
    print("  Computing Statistical Signals...", end="", flush=True)
    C = Config
    
    # ── Log Ratio (بهبود ویژگی‌های توزیع نرمال) ──
    log_ratio = np.log(df['c_eg'])
    z_mean    = log_ratio.rolling(C.z_fast_period).mean()
    z_std     = log_ratio.rolling(C.z_fast_period).std()
    z_score   = (log_ratio - z_mean) / z_std.replace(0, np.nan)

    # ── فیلتر Volatility ──
    atr_eg = calc_atr(df['h_eur']/df['l_gbp'], df['l_eur']/df['h_gbp'], df['c_eg'], C.atr_period)
    atr_ma = atr_eg.rolling(C.atr_ma_period).mean()
    vol_ok = (atr_eg > atr_ma * C.atr_min_mult) & (atr_eg < atr_ma * C.atr_max_mult)

    # ── فیلتر زمان ──
    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # ── شروط ورود ──
    long_cond  = (z_score < -C.z_entry) & vol_ok & time_ok
    short_cond = (z_score > C.z_entry) & vol_ok & time_ok

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1

    # جلوگیری از صدور سیگنال‌های متوالی در یک جهت
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
    """ محاسبه سود و زیان به دلار با در نظر گرفتن نرخ لحظه‌ای متقاطع """
    C = Config
    # سود ناخالص به پوند (انگار داریم EURGBP معامله میکنیم)
    gross_gbp = dir_trade * (exit_px - entry_px) * lot_size * C.lot_size
    # تبدیل به دلار
    gross_usd = gross_gbp * gbp_usd_rate
    # کسر کمیسیون دلاری
    net_usd = gross_usd - (C.commission_per_lot * lot_size)
    return net_usd

def calc_lot(equity: float, sl_pips: float, consec_loss: int) -> float:
    C = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
    # ارزش هر پیپ حدودی برای محاسبه حجم (فرض نرخ GBPUSD=1.25)
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
#  موتور شبیه‌سازی (Backtest Engine)
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, signals: dict) -> dict:
    C = Config
    pip = C.pip

    # تبدیل داده‌ها به آرایه‌های Numpy برای سرعت بالا
    open_eg  = df['o_eg'].values
    close_eg = df['c_eg'].values
    close_g  = df['c_gbp'].values # برای محاسبه ارزش پیپ داینامیک
    sig_a    = signals['sig'].values
    z_a      = signals['z_score'].values
    ts_a     = df.index

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    total_withdrawn = 0.0
    acc_num = 1
    acc_logs, all_trades, eq_curve, eq_ts_list, tot_curve = [], [], [], [], []

    acc = new_acc(ts_a[C.warmup])
    day_start_eq = C.initial_balance
    trades_today = 0
    pending_sig = 0

    print(f"\n  ▶ Running Prop Simulator... Floor=${PROP_FLOOR:,.0f} | Target=${PROFIT_LEVEL:,.0f}")

    for bar in range(C.warmup, len(ts_a)):
        ts = ts_a[bar]
        eq = acc['equity']

        eq_curve.append(round(eq, 4))
        eq_ts_list.append(ts)
        tot_curve.append(round(eq + total_withdrawn, 4))

        # آپدیت Drawdown
        if eq > acc['peak']: acc['peak'] = eq
        if eq < acc['min_eq']: acc['min_eq'] = eq
        if acc['peak'] > 0:
            dd = (eq - acc['peak']) / acc['peak'] * 100
            if dd < acc['max_dd_pct']: acc['max_dd_pct'] = dd

        # ── 리ست روزانه Daily DD (نصف شب سرور) ──
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            trades_today = 0

        # بررسی حساب سوخته
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
            
            # احتساب اسپرد و اسلیپیج در قیمت ورود
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

        # ── مدیریت پوزیشن ──
        pos = acc['open_pos']
        if pos is not None:
            cp = close_eg[bar]
            cg = close_g[bar]
            d  = pos['dir']
            ep = pos['entry']
            
            # خروج داینامیک Z-Score (بازگشت به میانگین)
            zn = z_a[bar]
            hit_z = (not np.isnan(zn)) and ( (d == 1 and zn >= -C.z_exit) or (d == -1 and zn <= C.z_exit) )
            
            # خروج‌های اضطراری 
            hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
            hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])
            time_stop = (bar - pos['entry_bar']) >= C.time_stop_bars

            # محاسبه Drawdown درون‌کندلی
            current_pnl = calc_dynamic_pnl(d, ep, cp, pos['lot'], cg)
            current_eq = acc['equity'] + current_pnl
            
            daily_dd_pct = (current_eq - day_start_eq) / day_start_eq
            total_dd_pct = (current_eq - C.initial_balance) / C.initial_balance
            
            blown_daily = daily_dd_pct <= -C.max_daily_loss_pct
            blown_total = total_dd_pct <= -C.max_total_dd_pct

            if blown_daily or blown_total:
                hit_sl, st = True, "BLOWN_DD"
                acc['blown'] = True
                acc['blown_rsn'] = "DailyDD" if blown_daily else "TotalDD"

            if hit_z or hit_sl or hit_tp or time_stop:
                exit_px = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                st = st if 'st' in locals() else ('SL' if hit_sl else ('TP' if hit_tp else ('TimeStop' if time_stop else 'Z-Exit')))
                
                final_pnl = calc_dynamic_pnl(d, ep, exit_px, pos['lot'], cg)
                acc['equity'] += final_pnl
                
                rec = {**pos, 'exit': exit_px, 'exit_ts': ts, 'pnl': final_pnl, 'status': st}
                acc['trades'].append(rec)
                all_trades.append(rec)
                acc['open_pos'] = None
                
                if final_pnl > 0: acc['consec_loss'] = 0
                else: acc['consec_loss'] += 1

        # ── برداشت سود ──
        if acc['equity'] >= PROFIT_LEVEL and acc['open_pos'] is None and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': "TARGET_HIT", 'pnl': acc['equity'] - C.initial_balance})
            print(f"    💰 #{acc_num:>3} | {ts.date()} | Withdrawn: ${w:>7.2f} | Total: ${total_withdrawn:>9.2f}")
            acc_num += 1
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
#  توابع گزارش‌گیری ساده‌شده
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
    
    print("\n" + "═" * 60)
    print(" ▌  CorrArb Prop Simulator v5 — Performance Report  ▐")
    print("═" * 60)
    print(f" Total Trades:    {len(df_t):,}")
    print(f" Win Rate:        {wr:.2f}%")
    print(f" Profit Factor:   {pf:.2f}")
    print(f" Total Withdrawn: ${results['total_withdrawn']:,.2f}")
    print(f" Final Equity:    ${results['final_equity']:,.2f}")
    print(f" Avg Win:         ${wins['pnl'].mean():.2f}")
    print(f" Avg Loss:        ${losses['pnl'].mean():.2f}")
    print("═" * 60)

if __name__ == "__main__":
    t0 = datetime.now()
    df = load_data()
    signals = compute_signals(df)
    results = run_backtest(df, signals)
    print_report(results)
    print(f"  ✅ Executed in: {(datetime.now()-t0).total_seconds():.2f}s")
