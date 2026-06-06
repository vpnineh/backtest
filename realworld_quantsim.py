"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         CorrArb Prop Trading Simulator — نسخه آربیتراژ واقعی (Delta-Neutral) ║
║  • استراتژی: Statistical Arbitrage (Pairs Trading)                           ║
║  • خنثی‌سازی دلار: همزمان ورود به EUR/USD و GBP/USD در جهت مخالف             ║
║  • شبیه‌سازی واقعی پراپ: محاسبه سود/ضرر بر اساس Basket Net PnL در Close      ║
║  • قوانین پراپ: Max DD 10% / Daily DD 5%                                     ║
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

    # ── ریسک معامله ──
    risk_per_trade_pct   = 0.010      # 1.0% ریسک دلاری برای کل سبد

    # ── هزینه‌های معامله ──
    spread_eur_pips      = 1.0        # EUR/USD spread
    spread_gbp_pips      = 1.5        # GBP/USD spread (واقع‌بینانه)
    commission_per_lot   = 6.0        # کمیسیون هر لات

    # ── مشخصات بازار ──
    pip                  = 0.0001
    lot_size             = 100_000
    max_lot              = 2.0

    # ── اندیکاتورها ──
    warmup               = 500        # کندل warmup اولیه (یکبار)

    # ── CorrArb پارامترها ──
    arb_z_fast           = 96         # rolling window کوتاه (24 ساعت)
    arb_z_slow           = 480        # rolling window بلند (5 روز)
    arb_z_entry          = 2.0        # Z-score ورود
    arb_z_exit           = 0.3        # Z-score خروج
    arb_z_slow_confirm   = 0.5        # تایید Z کند
    arb_adx_max          = 30         # ADX حداکثر
    arb_rsi_long_max     = 46         # RSI برای ورود
    arb_rsi_short_min    = 54         # RSI برای ورود
    arb_sl_pips          = 20.0       # SL مرجع برای محاسبه حجم
    arb_tp_pips          = 44.0       # TP مرجع برای محاسبه حجم (RR = 2.2)
    arb_hour_start       = 7          # ساعت شروع
    arb_hour_end         = 19         # ساعت پایان


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
    print(f"✅ {len(df):,} کندل | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  اندیکاتورها
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(c: pd.Series, period: int = 14) -> pd.Series:
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_adx(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
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
#  محاسبه سیگنال‌های CorrArb
# ═══════════════════════════════════════════════════════════════════════════
def compute_corrarb_signals(df: pd.DataFrame) -> dict:
    print("  محاسبه اندیکاتورها و سیگنال‌های CorrArb...", end="", flush=True)

    c_e = df['c_eur']
    h_e = df['h_eur']
    l_e = df['l_eur']
    c_g = df['c_gbp']
    C   = Config

    rsi  = calc_rsi(c_e, 14)
    adx  = calc_adx(h_e, l_e, c_e, 14)
    hour = pd.Series(df.index.hour, index=df.index)

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

    sig = sig.where(sig != sig.shift(), 0)

    print(" ✓")
    print(f"  سیگنال‌های ورود: {int((sig != 0).sum()):,} | Long: {int((sig == 1).sum()):,} | Short: {int((sig == -1).sum()):,}")

    return {
        'sig':    sig,
        'z_fast': z_fast,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  موتور بک‌تست پراپ — هجینگ سبد و مدیریت PnL واقعی
# ═══════════════════════════════════════════════════════════════════════════
def run_prop_backtest(df: pd.DataFrame, signals: dict) -> dict:
    C    = Config
    pip  = C.pip
    ls   = C.lot_size
    comm = C.commission_per_lot

    close_e = df['c_eur'].values
    close_g = df['c_gbp'].values
    sig_a   = signals['sig'].values
    z_a     = signals['z_fast'].values
    ts_a    = df.index

    total_withdrawn    = 0.0
    account_number     = 1
    all_account_logs   = []
    all_trades         = []

    global_eq_curve    = []
    global_eq_ts       = []
    global_total_curve = []

    equity      = C.initial_balance
    max_eq      = equity
    open_pos    = None
    cur_day     = None
    day_start_eq = equity

    acc_start_ts  = ts_a[C.warmup] if len(ts_a) > C.warmup else ts_a[0]
    acc_trades    = []
    acc_blown     = False
    acc_blown_reason = ""

    PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)

    def lot_calc(eq: float, sl_pips: float) -> float:
        if sl_pips <= 0: return 0.01
        risk_money = (eq * C.risk_per_trade_pct) / 2
        raw = risk_money / (sl_pips * pip * ls)
        return round(float(np.clip(raw, 0.01, C.max_lot)), 2)

    sig_indices = set(i for i in np.where(sig_a != 0)[0] if i >= C.warmup)

    print(f"\n  شروع شبیه‌سازی پراپ واقعی (Delta-Neutral)... (PROP_FLOOR=${PROP_FLOOR:,.0f})")

    for bar in range(C.warmup, len(ts_a)):
        ts  = ts_a[bar]
        day = ts.date()

        global_eq_curve.append(round(equity, 4))
        global_eq_ts.append(ts)
        global_total_curve.append(round(equity + total_withdrawn, 4))

        if day != cur_day:
            cur_day      = day
            day_start_eq = equity 

        if acc_blown:
            if open_pos is not None:
                rec = {**open_pos, 'exit_ts': ts, 'pnl': open_pos['stop_pnl'], 'status': 'blown_close'}
                equity += open_pos['stop_pnl']
                acc_trades.append(rec)
                all_trades.append(rec)
                open_pos = None
            
            _log_and_reset_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, acc_blown_reason, all_account_logs)
            
            equity = C.initial_balance
            max_eq = equity
            day_start_eq = equity
            account_number += 1
            acc_start_ts = ts
            acc_trades = []
            acc_blown = False
            acc_blown_reason = ""
            PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)
            continue

        if open_pos is not None:
            cp_e = close_e[bar]
            cp_g = close_g[bar]
            
            raw_e = open_pos['dir_e'] * (cp_e - open_pos['ep_e']) * open_pos['lot_e'] * ls
            raw_g = open_pos['dir_g'] * (cp_g - open_pos['ep_g']) * open_pos['lot_g'] * ls
            net_pnl = raw_e + raw_g - open_pos['cost']

            hit_sl = net_pnl <= open_pos['stop_pnl']
            hit_tp = net_pnl >= open_pos['target_pnl']
            
            zn = z_a[bar]
            hit_z = not np.isnan(zn) and abs(zn) < C.arb_z_exit
            hit_time = (bar - open_pos['entry_bar']) >= 384

            if hit_sl or hit_tp or hit_z or hit_time:
                reason = 'SL' if hit_sl else ('TP' if hit_tp else ('Z-Exit' if hit_z else 'TimeStop'))
                final_pnl = open_pos['stop_pnl'] if hit_sl else (open_pos['target_pnl'] if hit_tp else net_pnl)
                
                rec = {**open_pos, 'exit_ts': ts, 'pnl': final_pnl, 'status': reason}
                equity += final_pnl
                max_eq = max(max_eq, equity)
                acc_trades.append(rec)
                all_trades.append(rec)
                open_pos = None
                
                acc_blown, acc_blown_reason = _check_prop_rules(equity, day_start_eq, PROP_FLOOR, C)

        profit_pct = (equity - C.initial_balance) / C.initial_balance
        if profit_pct >= C.profit_target_pct and open_pos is None:
            withdrawn = equity - C.initial_balance
            total_withdrawn += withdrawn
            _log_and_reset_account(acc_start_ts, ts, equity, total_withdrawn, acc_trades, account_number, "TARGET_HIT", all_account_logs)
            print(f"    💰 اکانت #{account_number:>3} | {ts.date()} | برداشت: ${withdrawn:>7.2f} | کل برداشت: ${total_withdrawn:>9.2f}")
            equity = C.initial_balance
            max_eq = equity
            day_start_eq = equity
            account_number += 1
            acc_start_ts = ts
            acc_trades = []
            PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)
            continue

        if open_pos is None and not acc_blown and bar in sig_indices:
            sv = int(sig_a[bar])
            dir_e = sv
            dir_g = -sv 
            
            lot_e = lot_calc(equity, C.arb_sl_pips)
            lot_g = lot_calc(equity, C.arb_sl_pips)
            
            ep_e = close_e[bar] + dir_e * C.spread_eur_pips * pip / 2
            ep_g = close_g[bar] + dir_g * C.spread_gbp_pips * pip / 2
            
            cost_e = (C.spread_eur_pips * pip * lot_e * ls) + (comm * lot_e)
            cost_g = (C.spread_gbp_pips * pip * lot_g * ls) + (comm * lot_g)
            total_cost = cost_e + cost_g
            
            risk_money = equity * C.risk_per_trade_pct
            reward_money = risk_money * (C.arb_tp_pips / C.arb_sl_pips)

            open_pos = {
                'account': account_number,
                'dir_e': dir_e, 'dir_g': dir_g,
                'lot_e': lot_e, 'lot_g': lot_g,
                'ep_e': ep_e, 'ep_g': ep_g,
                'cost': total_cost,
                'stop_pnl': -risk_money,
                'target_pnl': reward_money,
                'entry_ts': ts,
                'entry_bar': bar
            }

    if open_pos is not None:
        rec = {**open_pos, 'exit_ts': ts_a[-1], 'pnl': net_pnl, 'status': 'EndOfData'}
        equity += net_pnl
        acc_trades.append(rec)
        all_trades.append(rec)

    _log_and_reset_account(acc_start_ts, ts_a[-1], equity, total_withdrawn, acc_trades, account_number, "ACTIVE/END", all_account_logs)

    return {
        'all_trades': all_trades,
        'account_logs': all_account_logs,
        'eq_curve': global_eq_curve,
        'eq_ts': global_eq_ts,
        'total_curve': global_total_curve,
        'total_withdrawn': total_withdrawn,
        'final_equity': equity,
        'total_accounts': account_number,
    }

def _check_prop_rules(equity: float, day_start: float, prop_floor: float, C) -> tuple:
    daily_dd = (equity - day_start) / day_start
    if daily_dd <= -C.max_daily_loss_pct:
        return True, f"DailyDD {daily_dd*100:.2f}% (limit: -{C.max_daily_loss_pct*100:.0f}%)"

    if equity <= prop_floor:
        total_dd = (equity - C.initial_balance) / C.initial_balance
        return True, f"TotalDD {total_dd*100:.2f}% (equity: ${equity:.2f} < floor: ${prop_floor:.2f})"

    return False, ""

def _log_and_reset_account(start_ts, end_ts, final_eq, total_withdrawn, trades, acc_num, reason, logs):
    C = Config
    pnl      = final_eq - C.initial_balance
    ret_pct  = pnl / C.initial_balance * 100
    wins     = sum(1 for t in trades if t.get('pnl', 0) > 0)
    wr       = wins / len(trades) * 100 if trades else 0
    logs.append({
        'account':         acc_num,
        'start_ts':        start_ts,
        'end_ts':          end_ts,
        'initial':         C.initial_balance,
        'final':           round(final_eq, 2),
        'pnl':             round(pnl, 2),
        'ret_pct':         round(ret_pct, 2),
        'trades':          len(trades),
        'wins':            wins,
        'wr':              round(wr, 1),
        'reason':          reason,
        'total_withdrawn': round(total_withdrawn, 2),
    })

# ═══════════════════════════════════════════════════════════════════════════
#  آمار و تحلیل
# ═══════════════════════════════════════════════════════════════════════════
def compute_stats(results: dict) -> dict:
    trades   = results['all_trades']
    acc_logs = results['account_logs']
    eq_curve = results['eq_curve']
    eq_ts    = results['eq_ts']
    total_c  = results['total_curve']
    C        = Config

    if not trades: return None

    t = pd.DataFrame(trades)
    t['pnl']         = pd.to_numeric(t['pnl'], errors='coerce').fillna(0)
    t['entry_ts']    = pd.to_datetime(t['entry_ts'])
    t['exit_ts']     = pd.to_datetime(t['exit_ts'])
    t['duration_min']= (t['exit_ts'] - t['entry_ts']).dt.total_seconds() / 60

    al = pd.DataFrame(acc_logs)

    total_withdrawn = results['total_withdrawn']
    final_equity    = results['final_equity']
    total_value     = total_withdrawn + final_equity
    total_profit    = total_value - C.initial_balance
    total_ret       = total_profit / C.initial_balance * 100

    sd          = t['entry_ts'].min()
    ed          = t['exit_ts'].max()
    total_days  = max((ed - sd).days, 1)
    ann_ret     = ((total_value / C.initial_balance) ** (365.25 / total_days) - 1) * 100

    win_t  = t[t['pnl'] > 0]
    loss_t = t[t['pnl'] < 0]
    win_r  = len(win_t) / len(t) * 100 if len(t) > 0 else 0
    avg_w  = win_t['pnl'].mean() if len(win_t) > 0 else 0
    avg_l  = loss_t['pnl'].mean() if len(loss_t) > 0 else 0
    gw     = win_t['pnl'].sum()
    gl     = abs(loss_t['pnl'].sum())
    pf     = gw / gl if gl > 0 else float('inf')
    exp_v  = t['pnl'].mean()
    rr     = abs(avg_w / avg_l) if avg_l != 0 else 0

    eq_s   = pd.Series(eq_curve)
    max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min()

    ret_s  = pd.Series(total_c).pct_change().dropna()
    sharpe = ret_s.mean() / ret_s.std() * np.sqrt(252 * 96) if ret_s.std() > 0 else 0
    neg    = ret_s[ret_s < 0]
    ds     = neg.std() if len(neg) > 0 else 1e-10
    sortino= ret_s.mean() / ds * np.sqrt(252 * 96)

    n_accounts  = results['total_accounts']
    n_target    = int((al['reason'] == 'TARGET_HIT').sum())
    n_blown     = int(al['reason'].str.contains('DailyDD|TotalDD|blown').sum())
    n_active    = int((al['reason'] == 'ACTIVE/END').sum())

    sign   = t['pnl'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    cw = cl = mcw = mcl = 0
    for s in sign:
        if s > 0:
            cw += 1; cl = 0; mcw = max(mcw, cw)
        elif s < 0:
            cl += 1; cw = 0; mcl = max(mcl, cl)
        else:
            cw = cl = 0

    return dict(
        trades=t, acc_logs=al,
        eq_curve=eq_curve, eq_ts=eq_ts, total_curve=total_c,
        total_withdrawn=total_withdrawn,
        final_equity=final_equity, total_value=total_value,
        total_profit=total_profit, total_ret=total_ret,
        ann_ret=ann_ret, total_days=total_days,
        win_r=win_r, avg_w=avg_w, avg_l=avg_l,
        pf=pf, exp=exp_v, rr=rr,
        max_dd=max_dd, sharpe=sharpe, sortino=sortino,
        mcw=mcw, mcl=mcl,
        n_accounts=n_accounts, n_target=n_target,
        n_blown=n_blown, n_active=n_active,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  گزارش‌دهی
# ═══════════════════════════════════════════════════════════════════════════
def print_full_report(s: dict) -> str:
    C   = Config
    W   = 76
    SEP = "═" * W

    def rw(label, value, ok=None):
        lpart = f"  {label}"
        vpart = str(value)
        mark  = "" if ok is None else (" ✅" if ok else " ❌")
        dots  = "·" * max(2, W - len(lpart) - len(vpart) - len(mark) - 2)
        return f"{lpart} {dots} {vpart}{mark}"

    def box(title):
        inner = f"─ {title} "
        return "┌" + inner + "─" * (W - len(inner) - 1) + "┐"

    bot = "└" + "─" * (W - 1) + "┘"

    prop_ok = (s['total_ret'] > 0 and s['pf'] > 1.3 and abs(s['max_dd']) < 10 and s['n_target'] > s['n_blown'])
    flag = "✅ PROP PASS" if prop_ok else "⚠️  در حال بهینه‌سازی"

    lines = [
        "", SEP,
        f"  ▌  CorrArb Prop Simulator   {flag}  ▐",
        f"  ▌  {s['trades']['entry_ts'].min().date()} → {s['trades']['exit_ts'].max().date()}  ({s['total_days']} روز)  ▐",
        SEP, "",
        box("نتایج مالی تجمیعی"),
        rw("بالانس هر اکانت",       f"${C.initial_balance:>12,.2f}"),
        rw("کل سود برداشت‌شده",     f"${s['total_withdrawn']:>+12,.2f}"),
        rw("موجودی اکانت فعلی",     f"${s['final_equity']:>12,.2f}"),
        rw("ارزش کل (برداشت+اکانت)",f"${s['total_value']:>12,.2f}"),
        rw("سود خالص کل",           f"${s['total_profit']:>+12,.2f}"),
        rw("بازده کل",              f"{s['total_ret']:>+.2f}%"),
        rw("بازده سالانه",          f"{s['ann_ret']:>+.2f}%"),
        bot, "",
        box("ریسک"),
        rw("Max DD (per account)",  f"{s['max_dd']:.2f}%", ok=(abs(s['max_dd']) < 10)),
        rw("Sharpe",                f"{s['sharpe']:.2f}"),
        rw("Sortino",               f"{s['sortino']:.2f}"),
        rw("Profit Factor",         f"{s['pf']:.2f}", ok=(s['pf'] > 1.3)),
        bot, "",
        box("آمار اکانت‌های پراپ"),
        rw("کل اکانت‌ها",           f"{s['n_accounts']}"),
        rw("✅ Target Hit (برداشت)", f"{s['n_target']}", ok=(s['n_target'] > 0)),
        rw("💥 Blown (قانون نقض)",  f"{s['n_blown']}", ok=(s['n_blown'] == 0)),
        rw("🔄 فعال/پایان داده",    f"{s['n_active']}"),
        bot, "",
        box("معاملات"),
        rw("تعداد کل سبدها",        f"{len(s['trades']):,}"),
        rw("Win Rate",               f"{s['win_r']:.1f}%"),
        rw("Avg Win",                f"${s['avg_w']:>+.2f}"),
        rw("Avg Loss",               f"${s['avg_l']:>+.2f}"),
        rw("Risk:Reward",            f"{s['rr']:.2f}"),
        rw("Expectancy",             f"${s['exp']:>+.2f}"),
        rw("Max Cons. Wins",         f"{s['mcw']}"),
        rw("Max Cons. Losses",       f"{s['mcl']}"),
        rw("مدت میانگین باز بودن",   f"{s['trades']['duration_min'].mean():.0f} min"),
        bot, "",
    ]

    lines.append(box("جزئیات اکانت‌ها"))
    lines.append(f"  {'#':>4}  {'شروع':>10}  {'پایان':>10}  {'PnL':>9}  {'Ret%':>6}  {'#T':>3}  {'WR%':>5}  نتیجه")
    lines.append("  " + "─" * (W - 3))
    for _, row in s['acc_logs'].iterrows():
        start_str = str(row['start_ts'])[:10]
        end_str   = str(row['end_ts'])[:10]
        reason    = row['reason']
        if reason == 'TARGET_HIT': icon = "💰 WITHDRAW"
        elif 'DD' in str(reason) or 'blown' in str(reason): icon = f"💥 BLOWN  ({reason[:28]})"
        elif reason == 'ACTIVE/END': icon = "🔄 ACTIVE"
        else: icon = f"⚠️  {reason[:20]}"

        lines.append(f"  {int(row['account']):>4}  {start_str:>10}  {end_str:>10}  ${row['pnl']:>+8.2f}  {row['ret_pct']:>+5.1f}%  {row['trades']:>3}  {row['wr']:>4.0f}%  {icon}")
    lines += [bot, ""]

    out = "\n".join(lines)
    print(out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  ذخیره فایل‌های خروجی
# ═══════════════════════════════════════════════════════════════════════════
def save_outputs(s: dict, report_txt: str):
    with open("Report_CorrArb_Prop.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    rows = [
        ["CorrArb Prop Simulator — گزارش کامل"],
        [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
        [f"Risk={Config.risk_per_trade_pct*100:.1f}%  ProfitTarget={Config.profit_target_pct*100:.0f}%  DailyDD={Config.max_daily_loss_pct*100:.0f}%  TotalDD={Config.max_total_dd_pct*100:.0f}%"],
        [], ["=== خلاصه کلی ==="],
        ["کل سود برداشت‌شده", round(s['total_withdrawn'], 2)],
        ["موجودی اکانت فعلی", round(s['final_equity'], 2)],
        ["ارزش کل", round(s['total_value'], 2)],
        ["بازده کل%", round(s['total_ret'], 2)],
        ["Profit Factor", round(s['pf'], 2)],
        ["Win Rate%", round(s['win_r'], 1)],
        ["Max DD%", round(s['max_dd'], 2)],
        ["تعداد اکانت", s['n_accounts']],
        ["اکانت‌های موفق", s['n_target']],
        ["اکانت‌های Blown", s['n_blown']],
        [], ["=== جزئیات اکانت‌ها ==="],
        ["Account", "StartTS", "EndTS", "Initial", "Final", "PnL", "Ret%", "Trades", "WinRate%", "Reason", "TotalWithdrawn"],
    ]
    for _, row in s['acc_logs'].iterrows():
        rows.append([row['account'], str(row['start_ts'])[:16], str(row['end_ts'])[:16], row['initial'], row['final'], row['pnl'], row['ret_pct'], row['trades'], row['wr'], row['reason'], row['total_withdrawn']])

    rows += [
        [], ["=== همه معاملات (سبد آربیتراژ) ==="],
        ["Account", "EntryTS", "ExitTS", "Signal(EUR)", "Lots(E|G)", "Entry(E|G)", "Cost($)", "Target($)", "Stop($)", "PnL($)", "Status", "DurMin"],
    ]
    for _, t in s['trades'].iterrows():
        lots = f"{t.get('lot_e', 0)}|{t.get('lot_g', 0)}"
        entries = f"{round(float(t.get('ep_e', 0)), 5)}|{round(float(t.get('ep_g', 0)), 5)}"
        dir_lbl = 'LONG(Eur)/SHORT(Gbp)' if t.get('dir_e', 0) == 1 else 'SHORT(Eur)/LONG(Gbp)'
        
        rows.append([
            t.get('account', ''), str(t['entry_ts'])[:16], str(t['exit_ts'])[:16],
            dir_lbl, lots, entries,
            round(float(t.get('cost', 0)), 2),
            round(float(t.get('target_pnl', 0)), 2),
            round(float(t.get('stop_pnl', 0)), 2),
            round(float(t['pnl']), 2),
            t.get('status', ''),
            round(float(t.get('duration_min', 0)), 0),
        ])

    pd.DataFrame(rows).to_csv("Report_CorrArb_Prop.csv", index=False, header=False, encoding="utf-8-sig")

    withdrawn_curve = [round(tv - ae, 2) for tv, ae in zip(s['total_curve'], s['eq_curve'])]
    eq_df = pd.DataFrame({
        'ts':             s['eq_ts'],
        'account_equity': s['eq_curve'],
        'total_withdrawn':withdrawn_curve,
        'total_value':    s['total_curve'],
    })
    eq_df['account_dd'] = ((eq_df['account_equity'] - eq_df['account_equity'].cummax()) / eq_df['account_equity'].cummax() * 100).round(4)
    eq_df.to_csv("eq_CorrArb_Prop.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ فایل‌های خروجی ذخیره شدند: Report_CorrArb_Prop.txt, Report_CorrArb_Prop.csv, eq_CorrArb_Prop.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  اجرای اصلی
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 76)
    print("  CorrArb Prop Simulator — شبیه‌سازی واقعی آربیتراژ آماری")
    print("═" * 76)
    
    t0 = datetime.now()
    df = load_data()
    signals = compute_corrarb_signals(df)
    results = run_prop_backtest(df, signals)
    
    if not results['all_trades']:
        print("\n❌ هیچ معامله‌ای انجام نشد. پارامترها را بررسی کنید.")
    else:
        stats = compute_stats(results)
        if stats:
            report = print_full_report(stats)
            save_outputs(stats, report)
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"\n  ✅ اتمام کامل در {elapsed:.1f} ثانیه")
