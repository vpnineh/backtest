"""
CorrArb Portfolio v9 — Dual-Logic (Mean-Reversion + Trend-Following)
با تشخیص رژیم بازار (ADX + ATR-Ratio)
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
ASSETS = [
    'EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD',
    'USDCAD', 'USDCHF', 'EURGBP', 'AUDNZD',
    'XAUUSD', 'XAGUSD'
]

# نمادهایی که "روندساز" هستند — منطق متفاوت
TREND_ASSETS = {'XAUUSD', 'XAGUSD'}

# ─────────────────────────────────────────────
class Config:
    initial_balance    = 5_000.0
    risk_per_trade_pct = 0.01        # 1% ریسک هر ترید
    profit_target_pct  = 0.08        # 8% سود = پاس پراپ
    max_daily_loss_pct = 0.04        # 4% حد روزانه
    max_total_dd_pct   = 0.08        # 8% دراداون کل
    sl_pips            = 25          # استاپ لاس پایه (پیپ)
    tp_ratio           = 2.5         # نسبت TP:SL
    trail_trigger_pips = 15          # فعال‌شدن تریل
    trail_lock_pips    = 8           # قفل سود تریل
    time_stop_bars     = 48          # تایم‌استاپ (بار ۱۵دقیقه)
    z_entry_min        = 1.8         # حداقل Z برای ورود MR
    z_stop_level       = 3.8         # Z-Stop خروج اضطراری
    ml_threshold       = 0.55        # آستانه ML
    adx_trend_level    = 25          # ADX بالای این = بازار روند دارد
    train_end          = '2022-12-31'
    test_start         = '2023-01-01'
    pip_value          = {           # ارزش هر پیپ به دلار (لات استاندارد)
        'EURUSD': 10, 'GBPUSD': 10, 'AUDUSD': 10, 'NZDUSD': 10,
        'USDCAD':  7.7, 'USDCHF': 10, 'EURGBP': 13, 'AUDNZD': 6.5,
        'XAUUSD': 10, 'XAGUSD': 50
    }
    pip_size           = {           # اندازه یک پیپ
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001,
        'NZDUSD': 0.0001, 'USDCAD': 0.0001, 'USDCHF': 0.0001,
        'EURGBP': 0.0001, 'AUDNZD': 0.0001,
        'XAUUSD': 0.1,    'XAGUSD': 0.01
    }

# ─────────────────────────────────────────────
# بلوک ۱: لود دیتا
# ─────────────────────────────────────────────
def load_all_data():
    all_files = glob.glob('data/*.csv')
    portfolio = {}
    print("\n📂 Loading Data...")
    for sym in ASSETS:
        hits = [f for f in all_files if sym in f.upper()]
        if not hits:
            print(f"  ⚠️  {sym}: file not found")
            continue
        df = pd.read_csv(
            hits[0], sep=r'[;,]', engine='python',
            header=None, names=['ts','o','h','l','c','v']
        )
        df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S', errors='coerce')
        df = (df.dropna()
                .drop_duplicates('ts')
                .set_index('ts')
                .sort_index()[['o','h','l','c']])
        df = (df.resample('15min')
                .agg({'o':'first','h':'max','l':'min','c':'last'})
                .dropna())
        portfolio[sym] = df
        print(f"  ✅ {sym}: {len(df):,} bars  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
    return portfolio

# ─────────────────────────────────────────────
# بلوک ۲: اندیکاتورها
# ─────────────────────────────────────────────
def add_indicators(df):
    """همه اندیکاتورها را اضافه می‌کند"""
    c = df['c']
    h, l = df['h'], df['l']

    # ── Z-Score (برای Mean-Reversion)
    ret = np.log(c).diff()
    df['log_ret']  = ret
    roll_mean      = ret.rolling(96).mean()
    roll_std       = ret.rolling(96).std().replace(0, np.nan)
    df['z_score']  = (ret - roll_mean) / roll_std

    # ── RSI
    delta   = c.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df['rsi'] = 100 - 100 / (1 + gain / loss)

    # ── ATR
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ── ADX (تشخیص رژیم — قلب Dual-Logic)
    up_move   = h.diff()
    down_move = (-l.diff())
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14     = tr.rolling(14).sum().replace(0, np.nan)
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).rolling(14).sum() / atr14
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / atr14
    dx        = (100 * (plus_di - minus_di).abs() /
                 (plus_di + minus_di).replace(0, np.nan))
    df['adx']      = dx.rolling(14).mean()
    df['plus_di']  = plus_di
    df['minus_di'] = minus_di

    # ── EMA برای Trend-Following
    df['ema_fast'] = c.ewm(span=20, adjust=False).mean()
    df['ema_slow'] = c.ewm(span=50, adjust=False).mean()
    df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['atr'].replace(0, np.nan)

    # ── Momentum (برای Trend)
    df['mom_12']   = c.pct_change(12)
    df['mom_48']   = c.pct_change(48)

    # ── Volatility Ratio (ATR نسبی)
    df['vol_ratio'] = df['atr'] / df['atr'].rolling(96).mean().replace(0, np.nan)

    return df

# ─────────────────────────────────────────────
# بلوک ۳: تشخیص رژیم
# ─────────────────────────────────────────────
def get_regime(row, sym):
    """
    برای هر بار تشخیص دهد بازار در چه رژیمی است:
    'trend'    → سیستم Trend-Following فعال
    'ranging'  → سیستم Mean-Reversion فعال
    'avoid'    → ورود ممنوع (نوسان خیلی بالا/پایین)
    """
    adx       = row.get('adx', 0)
    vol_ratio = row.get('vol_ratio', 1.0)

    # نوسان خیلی بالا = ریسک غیرقابل‌کنترل
    if vol_ratio > 2.5:
        return 'avoid'

    # نمادهای روندساز — ADX پایین‌تر هم کافیه
    if sym in TREND_ASSETS:
        if adx > 20:
            return 'trend'
        elif adx < 15:
            return 'ranging'
        else:
            return 'avoid'  # ناحیه خاکستری برای طلا

    # فارکس متقاطع — Z-Score عالی است
    if adx < Config.adx_trend_level:
        return 'ranging'
    else:
        return 'trend'

# ─────────────────────────────────────────────
# بلوک ۴: ML Meta-Labeler (دو مدل جداگانه)
# ─────────────────────────────────────────────
def train_ml_models(df, sym):
    """
    دو مدل ML جداگانه:
    - model_mr  : برای رژیم Ranging (Mean-Reversion)
    - model_tr  : برای رژیم Trending (Trend-Following)
    """
    # فیچرهای MR
    feat_mr = ['z_score', 'rsi', 'vol_ratio', 'atr']
    # فیچرهای Trend
    feat_tr = ['ema_diff', 'mom_12', 'mom_48', 'adx', 'rsi', 'vol_ratio']

    # Label: آیا ۱۲ بار بعد سود می‌دهد؟
    df['label_mr'] = np.where(
        df['log_ret'].shift(-12).rolling(12).sum() > 0, 1, 0
    )
    df['label_tr'] = df['label_mr']  # می‌توان جداگانه تنظیم کرد

    df = df.dropna()
    train = df[df.index <= Config.train_end]

    models = {}
    for tag, feats, label_col in [
        ('mr', feat_mr, 'label_mr'),
        ('tr', feat_tr, 'label_tr')
    ]:
        X = train[feats].fillna(0)
        y = train[label_col]
        if len(X) < 100:
            models[tag] = None
            continue
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=30,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X, y)
        models[tag] = (clf, feats)

    return models, df

# ─────────────────────────────────────────────
# بلوک ۵: تولید سیگنال‌ها
# ─────────────────────────────────────────────
def generate_signals(df, sym, models):
    """
    سیگنال‌های ترکیبی بر اساس رژیم + ML
    """
    test = df[df.index >= Config.test_start].copy()
    signals = []

    for i, (ts, row) in enumerate(test.iterrows()):
        regime = get_regime(row, sym)
        if regime == 'avoid':
            continue

        signal_dir = None    # 'L' یا 'S'
        ml_prob    = 0.5
        reason     = ''

        # ══ رژیم Ranging — Mean-Reversion ══
        if regime == 'ranging':
            z = row['z_score']
            if abs(z) < Config.z_entry_min:
                continue
            # ورود خلاف روند Z
            if z > Config.z_entry_min:
                signal_dir = 'S'   # Z بالا → انتظار برگشت پایین
            else:
                signal_dir = 'L'

            # ML فیلتر
            if models.get('mr') and models['mr'] is not None:
                clf, feats = models['mr']
                x_row = pd.DataFrame([row[feats].fillna(0)])
                ml_prob = clf.predict_proba(x_row)[0][1]
                # برای Short: ml_prob باید پایین باشد
                if signal_dir == 'S' and ml_prob > (1 - Config.ml_threshold):
                    continue
                if signal_dir == 'L' and ml_prob < Config.ml_threshold:
                    continue
            reason = f'MR|z={z:.2f}'

        # ══ رژیم Trending — Trend-Following ══
        elif regime == 'trend':
            ema_diff = row['ema_diff']
            mom      = row['mom_12']
            plus_di  = row['plus_di']
            minus_di = row['minus_di']

            # EMA + DI هم‌راستا باشند
            if ema_diff > 0.3 and plus_di > minus_di and mom > 0:
                signal_dir = 'L'
            elif ema_diff < -0.3 and minus_di > plus_di and mom < 0:
                signal_dir = 'S'
            else:
                continue

            # ML فیلتر
            if models.get('tr') and models['tr'] is not None:
                clf, feats = models['tr']
                x_row = pd.DataFrame([row[feats].fillna(0)])
                ml_prob = clf.predict_proba(x_row)[0][1]
                if signal_dir == 'L' and ml_prob < Config.ml_threshold:
                    continue
                if signal_dir == 'S' and ml_prob > (1 - Config.ml_threshold):
                    continue
            reason = f'TR|ema={ema_diff:.2f}'

        if signal_dir:
            signals.append({
                'ts':     ts,
                'sym':    sym,
                'dir':    signal_dir,
                'regime': regime,
                'reason': reason,
                'price':  row['c'],
                'atr':    row['atr'],
                'z':      row['z_score']
            })

    return pd.DataFrame(signals)

# ─────────────────────────────────────────────
# بلوک ۶: موتور بک‌تست (Per-Asset)
# ─────────────────────────────────────────────
def backtest_asset(sig_df, price_df, sym):
    """بک‌تست روی یک نماد — خروجی: لیست تریدها"""
    if sig_df.empty:
        return []

    pip_sz  = Config.pip_size.get(sym, 0.0001)
    pip_val = Config.pip_value.get(sym, 10)

    trades  = []
    in_trade = False
    entry_price = sl = tp = trail_sl = None
    direction   = None
    entry_ts    = None
    bar_count   = 0
    entry_z     = 0

    # ایندکس قیمت برای lookup سریع
    price_idx = price_df.index

    sig_iter = iter(sig_df.itertuples())
    current_sig = next(sig_iter, None)

    for ts, row in price_df[price_df.index >= Config.test_start].iterrows():
        o_, h_, l_, c_ = row['o'], row['h'], row['l'], row['c']

        # ── اگر در معامله هستیم
        if in_trade:
            bar_count += 1
            pnl_pips = 0
            exit_reason = None
            exit_price  = c_

            if direction == 'L':
                # Z-Stop (بازگشت اضطراری برای MR)
                if row.get('z_score', 0) > Config.z_stop_level:
                    exit_reason = 'Z-Stop'; exit_price = o_

                # تریل
                elif h_ > entry_price + Config.trail_trigger_pips * pip_sz:
                    new_trail = h_ - Config.trail_lock_pips * pip_sz
                    if trail_sl is None or new_trail > trail_sl:
                        trail_sl = new_trail
                    if l_ < trail_sl:
                        exit_reason = 'Trail'; exit_price = trail_sl

                # SL / TP / TimeStop
                elif l_ < sl:
                    exit_reason = 'SL'; exit_price = sl
                elif h_ > tp:
                    exit_reason = 'TP'; exit_price = tp
                elif bar_count >= Config.time_stop_bars:
                    exit_reason = 'TimeStop'; exit_price = c_

                if exit_reason:
                    pnl_pips = (exit_price - entry_price) / pip_sz

            else:  # SHORT
                if row.get('z_score', 0) < -Config.z_stop_level:
                    exit_reason = 'Z-Stop'; exit_price = o_
                elif l_ < entry_price - Config.trail_trigger_pips * pip_sz:
                    new_trail = l_ + Config.trail_lock_pips * pip_sz
                    if trail_sl is None or new_trail < trail_sl:
                        trail_sl = new_trail
                    if h_ > trail_sl:
                        exit_reason = 'Trail'; exit_price = trail_sl
                elif h_ > sl:
                    exit_reason = 'SL'; exit_price = sl
                elif l_ < tp:
                    exit_reason = 'TP'; exit_price = tp
                elif bar_count >= Config.time_stop_bars:
                    exit_reason = 'TimeStop'; exit_price = c_

                if exit_reason:
                    pnl_pips = (entry_price - exit_price) / pip_sz

            if exit_reason:
                # محاسبه سود/زیان دلاری (لات کوچک 0.01)
                lot  = 0.01
                pnl$ = pnl_pips * pip_val * lot
                trades.append({
                    'sym':      sym,
                    'entry_ts': entry_ts,
                    'exit_ts':  ts,
                    'dir':      direction,
                    'pnl_pips': round(pnl_pips, 1),
                    'pnl':      round(pnl$, 2),
                    'bars':     bar_count,
                    'reason':   exit_reason
                })
                in_trade = False
                trail_sl = None

        # ── ورود به معامله
        if not in_trade and current_sig is not None:
            if ts >= current_sig.ts:
                atr   = current_sig.atr
                sl_sz = Config.sl_pips * pip_sz
                # ATR-based SL (سازگار با نوسان)
                sl_atr = 1.5 * atr
                sl_sz  = max(sl_sz, sl_atr)

                direction   = current_sig.dir
                entry_price = o_   # ورود در Open بار بعدی
                entry_ts    = ts

                if direction == 'L':
                    sl = entry_price - sl_sz
                    tp = entry_price + sl_sz * Config.tp_ratio
                else:
                    sl = entry_price + sl_sz
                    tp = entry_price - sl_sz * Config.tp_ratio

                in_trade  = True
                bar_count = 0
                trail_sl  = None

                current_sig = next(sig_iter, None)

    return trades

# ─────────────────────────────────────────────
# بلوک ۷: شبیه‌ساز پراپ (Portfolio-Level)
# ─────────────────────────────────────────────
def prop_simulator(all_trades_df):
    """
    شبیه‌ساز پراپ روی تمام تریدها — مدیریت دراداون روزانه و کلی
    """
    print("\n" + "═"*65)
    print(" ▌  CorrArb Prop Simulator v9 — Dual Logic  ▐")
    print("═"*65)

    if all_trades_df.empty:
        print("  ❌ هیچ ترید فعالی یافت نشد!")
        return

    trades = all_trades_df.sort_values('exit_ts').copy()

    # ── وضعیت پراپ
    balance        = Config.initial_balance
    peak_balance   = balance
    daily_start    = balance
    current_date   = None
    account_num    = 1
    accounts       = []
    banked_total   = 0.0
    blown_count    = 0
    passed_count   = 0

    trade_results  = []

    for _, t in trades.iterrows():
        date = t['exit_ts'].date() if hasattr(t['exit_ts'], 'date') else pd.Timestamp(t['exit_ts']).date()

        # ── ریست روزانه
        if date != current_date:
            daily_start  = balance
            current_date = date

        balance    += t['pnl']
        peak_balance = max(peak_balance, balance)

        dd_total = (peak_balance - balance) / peak_balance
        dd_daily = (daily_start - balance) / daily_start if daily_start > 0 else 0

        # ── بررسی شرایط
        reason_out = None
        if balance <= 0 or dd_total >= Config.max_total_dd_pct:
            reason_out = f'BLOWN(DD={dd_total:.1%})'
        elif dd_daily >= Config.max_daily_loss_pct:
            reason_out = f'BLOWN(Daily={dd_daily:.1%})'

        if reason_out:
            blown_count += 1
            accounts.append({
                'num': account_num, 'result': 'BLOWN',
                'final_bal': balance, 'reason': reason_out,
                'trades': len(trade_results)
            })
            # ریست اکانت
            balance      = Config.initial_balance
            peak_balance = balance
            daily_start  = balance
            account_num += 1
            trade_results = []
            continue

        # ── پاس کردن
        profit_pct = (balance - Config.initial_balance) / Config.initial_balance
        if profit_pct >= Config.profit_target_pct:
            banked       = balance * 0.80   # 80% سود به تریدر
            banked_total += banked
            passed_count += 1
            accounts.append({
                'num':       account_num,
                'result':    'PASSED',
                'final_bal': balance,
                'banked':    round(banked, 2),
                'trades':    len(trade_results)
            })
            balance      = Config.initial_balance
            peak_balance = balance
            daily_start  = balance
            account_num += 1
            trade_results = []
            continue

        trade_results.append(t['pnl'])

    # ── آمار نهایی
    wins  = trades[trades['pnl'] > 0]['pnl']
    loss  = trades[trades['pnl'] < 0]['pnl']
    wr    = len(wins) / len(trades) * 100 if len(trades) else 0
    pf    = wins.sum() / abs(loss.sum()) if len(loss) else 0

    # آمار رژیم
    if 'regime' in trades.columns:
        mr_t = trades[trades['regime'] == 'ranging']
        tr_t = trades[trades['regime'] == 'trend']
    else:
        mr_t = tr_t = pd.DataFrame()

    total_months = max(1, (trades['exit_ts'].max() - trades['exit_ts'].min()).days / 30)

    print(f" Total Trades:          {len(trades):>6,}")
    print(f" Win Rate:              {wr:>6.1f}%")
    print(f" Profit Factor:         {pf:>6.2f}")
    print(f" Avg Win:               ${wins.mean():>7.2f}" if len(wins) else "")
    print(f" Avg Loss:              ${loss.mean():>7.2f}" if len(loss) else "")
    print(f"{'─'*65}")

    # آمار رژیم
    if len(mr_t) > 0:
        mr_wr = len(mr_t[mr_t['pnl']>0]) / len(mr_t) * 100
        mr_pf_num = mr_t[mr_t['pnl']>0]['pnl'].sum()
        mr_pf_den = abs(mr_t[mr_t['pnl']<0]['pnl'].sum())
        mr_pf = mr_pf_num / mr_pf_den if mr_pf_den > 0 else 0
        print(f" MR Trades: {len(mr_t):>5} | WR:{mr_wr:.0f}% | PF:{mr_pf:.2f}")
    if len(tr_t) > 0:
        tr_wr = len(tr_t[tr_t['pnl']>0]) / len(tr_t) * 100
        tr_pf_num = tr_t[tr_t['pnl']>0]['pnl'].sum()
        tr_pf_den = abs(tr_t[tr_t['pnl']<0]['pnl'].sum())
        tr_pf = tr_pf_num / tr_pf_den if tr_pf_den > 0 else 0
        print(f" Trend Trades:{len(tr_t):>4} | WR:{tr_wr:.0f}% | PF:{tr_pf:.2f}")

    # آمار به تفکیک نماد
    print(f"{'─'*65}")
    print(" آمار نمادها:")
    for sym in ASSETS:
        sym_t = trades[trades['sym'] == sym]
        if len(sym_t) == 0:
            continue
        sym_wr = len(sym_t[sym_t['pnl']>0]) / len(sym_t) * 100
        sym_pnl = sym_t['pnl'].sum()
        sym_pf_n = sym_t[sym_t['pnl']>0]['pnl'].sum()
        sym_pf_d = abs(sym_t[sym_t['pnl']<0]['pnl'].sum())
        sym_pf = sym_pf_n / sym_pf_d if sym_pf_d > 0 else 0
        flag = "🔥" if sym in TREND_ASSETS else "💱"
        print(f"  {flag} {sym:<8} T:{len(sym_t):>4} WR:{sym_wr:.0f}% "
              f"PF:{sym_pf:.2f} PnL:${sym_pnl:>7.1f}")

    print(f"{'─'*65}")
    print(f" Accounts Passed:       {passed_count:>4}")
    print(f" Accounts Blown:        {blown_count:>4}")
    rate = passed_count/(passed_count+blown_count)*100 if (passed_count+blown_count) > 0 else 0
    print(f" Pass Rate:             {rate:.1f}%")
    print(f" Total Banked:          ${banked_total:>9,.2f}")
    print(f" Monthly Avg:           ${banked_total/total_months:.2f}/ماه")

    # جدول اکانت‌ها
    print(f"\n{'─'*65}")
    print(" جزئیات اکانت‌ها:")
    for acc in accounts[-15:]:   # آخرین ۱۵ اکانت
        icon = '💰' if acc['result'] == 'PASSED' else '💥'
        banked_str = f"| ${acc.get('banked',0):>7.1f}" if acc['result']=='PASSED' else ''
        print(f"  {icon} #{acc['num']:>3} | {acc['result']:<7} "
              f"| Bal:${acc['final_bal']:>7.1f} {banked_str}"
              f"| T:{acc['trades']}")

    print("═"*65)

    # خروجی exit‌ها
    reason_counts = trades['reason'].value_counts()
    print("\n خروج‌ها:")
    for r, cnt in reason_counts.items():
        pct = cnt/len(trades)*100
        print(f"   {r:<15}: {cnt:>4} ({pct:.1f}%)")
    print("═"*65)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 CorrArb v9 — Dual-Logic Portfolio System")
    print("═"*65)

    # ۱. لود دیتا
    portfolio = load_all_data()
    if not portfolio:
        print("❌ هیچ داده‌ای یافت نشد!")
        exit()

    # ۲. اندیکاتور + ML + سیگنال (per asset)
    all_signals = []
    all_trades  = []

    for sym, df in portfolio.items():
        print(f"\n⚙️  Processing {sym}...")

        # اندیکاتورها
        df = add_indicators(df)

        # آموزش ML
        models, df = train_ml_models(df, sym)
        print(f"   ML trained | train-bars: {len(df[df.index<=Config.train_end]):,}")

        # سیگنال‌ها
        sigs = generate_signals(df, sym, models)
        if not sigs.empty:
            all_signals.append(sigs)
            print(f"   Signals: {len(sigs)} | "
                  f"MR: {(sigs['regime']=='ranging').sum()} | "
                  f"Trend: {(sigs['regime']=='trend').sum()}")

            # بک‌تست
            trades = backtest_asset(sigs, df, sym)
            if trades:
                t_df = pd.DataFrame(trades)
                t_df['regime'] = sigs.set_index('ts').reindex(
                    t_df['entry_ts']
                )['regime'].values if 'regime' in sigs.columns else 'unknown'
                all_trades.append(t_df)
                print(f"   Trades: {len(trades)}")
        else:
            print(f"   ⚠️ No signals generated")

    # ۳. ترکیب و شبیه‌سازی پراپ
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined['exit_ts'] = pd.to_datetime(combined['exit_ts'])
        prop_simulator(combined)
    else:
        print("❌ هیچ ترید فعالی یافت نشد!")
