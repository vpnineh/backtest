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

TREND_ASSETS = {'XAUUSD', 'XAGUSD'}

# ─────────────────────────────────────────────
class Config:
    initial_balance    = 5_000.0
    risk_per_trade_pct = 0.01
    profit_target_pct  = 0.08
    max_daily_loss_pct = 0.04
    max_total_dd_pct   = 0.08
    sl_pips            = 25
    tp_ratio           = 2.5
    trail_trigger_pips = 15
    trail_lock_pips    = 8
    time_stop_bars     = 48
    z_entry_min        = 1.8
    z_stop_level       = 3.8
    ml_threshold       = 0.55
    adx_trend_level    = 25
    train_end          = '2022-12-31'
    test_start         = '2023-01-01'
    pip_value = {
        'EURUSD': 10,   'GBPUSD': 10,  'AUDUSD': 10,  'NZDUSD': 10,
        'USDCAD':  7.7, 'USDCHF': 10,  'EURGBP': 13,  'AUDNZD': 6.5,
        'XAUUSD': 10,   'XAGUSD': 50
    }
    pip_size = {
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
        try:
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
        except Exception as e:
            print(f"  ❌ {sym}: {e}")
    return portfolio

# ─────────────────────────────────────────────
# بلوک ۲: اندیکاتورها
# ─────────────────────────────────────────────
def add_indicators(df):
    c = df['c'].copy()
    h = df['h'].copy()
    l = df['l'].copy()

    # Z-Score
    ret = np.log(c).diff()
    df['log_ret'] = ret
    roll_mean = ret.rolling(96).mean()
    roll_std  = ret.rolling(96).std().replace(0, np.nan)
    df['z_score'] = (ret - roll_mean) / roll_std

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df['rsi'] = 100 - 100 / (1 + gain / loss)

    # ATR
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # ADX
    up_move   = h.diff()
    down_move = (-l.diff())
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14     = tr.rolling(14).sum().replace(0, np.nan)
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).rolling(14).sum() / atr14
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / atr14
    dx        = (100 * (plus_di - minus_di).abs() /
                 (plus_di + minus_di).replace(0, np.nan))
    df['adx']      = dx.rolling(14).mean()
    df['plus_di']  = plus_di
    df['minus_di'] = minus_di

    # EMA
    df['ema_fast'] = c.ewm(span=20, adjust=False).mean()
    df['ema_slow'] = c.ewm(span=50, adjust=False).mean()
    df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['atr'].replace(0, np.nan)

    # Momentum
    df['mom_12'] = c.pct_change(12)
    df['mom_48'] = c.pct_change(48)

    # Volatility Ratio
    df['vol_ratio'] = df['atr'] / df['atr'].rolling(96).mean().replace(0, np.nan)

    return df

# ─────────────────────────────────────────────
# بلوک ۳: تشخیص رژیم
# ─────────────────────────────────────────────
def get_regime(adx_val, vol_ratio_val, sym):
    """
    'trend'   → Trend-Following
    'ranging' → Mean-Reversion
    'avoid'   → ورود ممنوع
    """
    if pd.isna(adx_val) or pd.isna(vol_ratio_val):
        return 'avoid'

    if vol_ratio_val > 2.5:
        return 'avoid'

    if sym in TREND_ASSETS:
        if adx_val > 20:
            return 'trend'
        elif adx_val < 15:
            return 'ranging'
        else:
            return 'avoid'

    # فارکس
    if adx_val < Config.adx_trend_level:
        return 'ranging'
    else:
        return 'trend'

# ─────────────────────────────────────────────
# بلوک ۴: ML Meta-Labeler
# ─────────────────────────────────────────────
def train_ml_models(df, sym):
    feat_mr = ['z_score', 'rsi', 'vol_ratio', 'atr']
    feat_tr = ['ema_diff', 'mom_12', 'mom_48', 'adx', 'rsi', 'vol_ratio']

    df = df.copy()
    future_ret = df['log_ret'].shift(-1).rolling(12).sum()
    df['label'] = np.where(future_ret > 0, 1, 0)
    df = df.dropna()

    train = df[df.index <= Config.train_end]
    models = {}

    for tag, feats in [('mr', feat_mr), ('tr', feat_tr)]:
        valid_feats = [f for f in feats if f in train.columns]
        X = train[valid_feats].fillna(0)
        y = train['label']
        if len(X) < 200:
            print(f"   ⚠️ {sym}/{tag}: کم‌داده ({len(X)})")
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
        models[tag] = (clf, valid_feats)

    return models, df

# ─────────────────────────────────────────────
# بلوک ۵: تولید سیگنال‌ها
# ─────────────────────────────────────────────
def generate_signals(df, sym, models):
    test = df[df.index >= Config.test_start].copy()
    signals = []

    for ts, row in test.iterrows():
        adx_val       = row.get('adx', np.nan)
        vol_ratio_val = row.get('vol_ratio', np.nan)
        regime        = get_regime(adx_val, vol_ratio_val, sym)

        if regime == 'avoid':
            continue

        signal_dir = None
        ml_prob    = 0.5
        reason     = ''

        # ══ Mean-Reversion ══
        if regime == 'ranging':
            z = row.get('z_score', 0)
            if pd.isna(z) or abs(z) < Config.z_entry_min:
                continue
            signal_dir = 'S' if z > Config.z_entry_min else 'L'

            if models.get('mr') is not None:
                clf, feats = models['mr']
                x_vals = [row.get(f, 0) for f in feats]
                x_row  = pd.DataFrame([x_vals], columns=feats).fillna(0)
                ml_prob = clf.predict_proba(x_row)[0][1]
                if signal_dir == 'S' and ml_prob > (1 - Config.ml_threshold):
                    continue
                if signal_dir == 'L' and ml_prob < Config.ml_threshold:
                    continue
            reason = f'MR|z={z:.2f}'

        # ══ Trend-Following ══
        elif regime == 'trend':
            ema_diff = row.get('ema_diff', 0)
            mom      = row.get('mom_12', 0)
            plus_di  = row.get('plus_di', 0)
            minus_di = row.get('minus_di', 0)

            if pd.isna(ema_diff) or pd.isna(mom):
                continue

            if ema_diff > 0.3 and plus_di > minus_di and mom > 0:
                signal_dir = 'L'
            elif ema_diff < -0.3 and minus_di > plus_di and mom < 0:
                signal_dir = 'S'
            else:
                continue

            if models.get('tr') is not None:
                clf, feats = models['tr']
                x_vals = [row.get(f, 0) for f in feats]
                x_row  = pd.DataFrame([x_vals], columns=feats).fillna(0)
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
                'atr':    row.get('atr', 0),
                'z':      row.get('z_score', 0)
            })

    return pd.DataFrame(signals)

# ─────────────────────────────────────────────
# بلوک ۶: موتور بک‌تست
# ─────────────────────────────────────────────
def backtest_asset(sig_df, price_df, sym):
    if sig_df.empty:
        return []

    pip_sz  = Config.pip_size.get(sym, 0.0001)
    pip_val = Config.pip_value.get(sym, 10)
    lot     = 0.01

    trades   = []
    in_trade = False
    entry_price = sl = tp = trail_sl = None
    direction   = None
    entry_ts    = None
    bar_count   = 0
    entry_regime = ''

    # سیگنال‌ها را به dict تبدیل برای دسترسی سریع
    sig_list = list(sig_df.itertuples())
    sig_idx  = 0
    n_sigs   = len(sig_list)

    price_test = price_df[price_df.index >= Config.test_start]

    for ts, row in price_test.iterrows():
        o_ = row['o']
        h_ = row['h']
        l_ = row['l']
        c_ = row['c']

        # ── مدیریت معامله باز
        if in_trade:
            bar_count  += 1
            exit_reason = None
            exit_price  = c_

            if direction == 'L':
                # Z-Stop
                z_now = row.get('z_score', 0)
                if not pd.isna(z_now) and z_now > Config.z_stop_level:
                    exit_reason = 'Z-Stop'
                    exit_price  = o_
                elif h_ >= entry_price + Config.trail_trigger_pips * pip_sz:
                    new_trail = h_ - Config.trail_lock_pips * pip_sz
                    trail_sl  = max(trail_sl, new_trail) if trail_sl else new_trail
                    if l_ <= trail_sl:
                        exit_reason = 'Trail'
                        exit_price  = trail_sl
                elif l_ <= sl:
                    exit_reason = 'SL'
                    exit_price  = sl
                elif h_ >= tp:
                    exit_reason = 'TP'
                    exit_price  = tp
                elif bar_count >= Config.time_stop_bars:
                    exit_reason = 'TimeStop'
                    exit_price  = c_

                if exit_reason:
                    pnl_pips = (exit_price - entry_price) / pip_sz

            else:  # SHORT
                z_now = row.get('z_score', 0)
                if not pd.isna(z_now) and z_now < -Config.z_stop_level:
                    exit_reason = 'Z-Stop'
                    exit_price  = o_
                elif l_ <= entry_price - Config.trail_trigger_pips * pip_sz:
                    new_trail = l_ + Config.trail_lock_pips * pip_sz
                    trail_sl  = min(trail_sl, new_trail) if trail_sl else new_trail
                    if h_ >= trail_sl:
                        exit_reason = 'Trail'
                        exit_price  = trail_sl
                elif h_ >= sl:
                    exit_reason = 'SL'
                    exit_price  = sl
                elif l_ <= tp:
                    exit_reason = 'TP'
                    exit_price  = tp
                elif bar_count >= Config.time_stop_bars:
                    exit_reason = 'TimeStop'
                    exit_price  = c_

                if exit_reason:
                    pnl_pips = (entry_price - exit_price) / pip_sz

            if exit_reason:
                pnl_dollar = pnl_pips * pip_val * lot
                trades.append({
                    'sym':      sym,
                    'entry_ts': entry_ts,
                    'exit_ts':  ts,
                    'dir':      direction,
                    'regime':   entry_regime,
                    'pnl_pips': round(pnl_pips, 1),
                    'pnl':      round(pnl_dollar, 2),
                    'bars':     bar_count,
                    'reason':   exit_reason
                })
                in_trade = False
                trail_sl = None

        # ── ورود به معامله
        if not in_trade and sig_idx < n_sigs:
            sig = sig_list[sig_idx]
            if ts >= sig.ts:
                sig_idx += 1

                atr_val = sig.atr if sig.atr and not pd.isna(sig.atr) else pip_sz * 20
                sl_fixed = Config.sl_pips * pip_sz
                sl_atr   = 1.5 * atr_val
                sl_sz    = max(sl_fixed, sl_atr)

                direction    = sig.dir
                entry_price  = o_
                entry_ts     = ts
                entry_regime = sig.regime
                bar_count    = 0
                trail_sl     = None

                if direction == 'L':
                    sl = entry_price - sl_sz
                    tp = entry_price + sl_sz * Config.tp_ratio
                else:
                    sl = entry_price + sl_sz
                    tp = entry_price - sl_sz * Config.tp_ratio

                in_trade = True

    return trades

# ─────────────────────────────────────────────
# بلوک ۷: شبیه‌ساز پراپ
# ─────────────────────────────────────────────
def prop_simulator(all_trades_df):
    print("\n" + "═"*65)
    print(" ▌  CorrArb Prop Simulator v9 — Dual Logic  ▐")
    print("═"*65)

    if all_trades_df.empty:
        print("  ❌ هیچ ترید فعالی یافت نشد!")
        return

    trades = all_trades_df.sort_values('exit_ts').reset_index(drop=True)

    balance      = Config.initial_balance
    peak_balance = balance
    daily_start  = balance
    current_date = None
    account_num  = 1
    accounts     = []
    banked_total = 0.0
    blown_count  = 0
    passed_count = 0
    acc_trades   = 0

    for _, t in trades.iterrows():
        exit_dt = pd.Timestamp(t['exit_ts'])
        date    = exit_dt.date()

        if date != current_date:
            daily_start  = balance
            current_date = date

        balance      += t['pnl']
        acc_trades   += 1
        peak_balance  = max(peak_balance, balance)

        dd_total = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        dd_daily = (daily_start - balance) / daily_start  if daily_start  > 0 else 0

        reason_out = None
        if balance <= 0 or dd_total >= Config.max_total_dd_pct:
            reason_out = f'BLOWN_DD_{dd_total:.1%}'
        elif dd_daily >= Config.max_daily_loss_pct:
            reason_out = f'BLOWN_DAILY_{dd_daily:.1%}'

        if reason_out:
            blown_count += 1
            accounts.append({
                'num':    account_num, 'result': 'BLOWN',
                'final':  round(balance, 2), 'reason': reason_out,
                'trades': acc_trades,        'banked': 0
            })
            balance      = Config.initial_balance
            peak_balance = balance
            daily_start  = balance
            account_num += 1
            acc_trades   = 0
            continue

        profit_pct = (balance - Config.initial_balance) / Config.initial_balance
        if profit_pct >= Config.profit_target_pct:
            banked        = (balance - Config.initial_balance) * 0.80
            banked_total += banked
            passed_count += 1
            accounts.append({
                'num':    account_num, 'result': 'PASSED',
                'final':  round(balance, 2), 'reason': f'+{profit_pct:.1%}',
                'trades': acc_trades,        'banked': round(banked, 2)
            })
            balance      = Config.initial_balance
            peak_balance = balance
            daily_start  = balance
            account_num += 1
            acc_trades   = 0
            continue

    # ── آمار کلی
    wins = trades[trades['pnl'] > 0]['pnl']
    loss = trades[trades['pnl'] < 0]['pnl']
    wr   = len(wins) / len(trades) * 100 if len(trades) else 0
    pf   = wins.sum() / abs(loss.sum()) if len(loss) > 0 and loss.sum() != 0 else 0

    date_range_days = max(1, (trades['exit_ts'].max() - trades['exit_ts'].min()).days)
    total_months    = date_range_days / 30.0

    print(f" Total Trades:          {len(trades):>6,}")
    print(f" Win Rate:              {wr:>6.1f}%")
    print(f" Profit Factor:         {pf:>6.2f}")
    if len(wins) > 0:
        print(f" Avg Win:               ${wins.mean():>7.2f}")
    if len(loss) > 0:
        print(f" Avg Loss:              ${loss.mean():>7.2f}")
    print(f"{'─'*65}")

    # ── آمار رژیم
    for reg in ['ranging', 'trend']:
        sub = trades[trades['regime'] == reg]
        if len(sub) == 0:
            continue
        sub_wr  = len(sub[sub['pnl'] > 0]) / len(sub) * 100
        sub_win = sub[sub['pnl'] > 0]['pnl'].sum()
        sub_los = abs(sub[sub['pnl'] < 0]['pnl'].sum())
        sub_pf  = sub_win / sub_los if sub_los > 0 else 0
        label   = "MR (Ranging)" if reg == 'ranging' else "Trend-Follow"
        print(f" {label}: T={len(sub):>4} | WR={sub_wr:.0f}% | PF={sub_pf:.2f}")

    # ── آمار نمادها
    print(f"{'─'*65}")
    print(" آمار نمادها:")
    for sym in ASSETS:
        sub = trades[trades['sym'] == sym]
        if len(sub) == 0:
            continue
        s_wr  = len(sub[sub['pnl'] > 0]) / len(sub) * 100
        s_pnl = sub['pnl'].sum()
        s_win = sub[sub['pnl'] > 0]['pnl'].sum()
        s_los = abs(sub[sub['pnl'] < 0]['pnl'].sum())
        s_pf  = s_win / s_los if s_los > 0 else 0
        flag  = "🔥" if sym in TREND_ASSETS else "💱"
        print(f"  {flag} {sym:<8} T={len(sub):>4} | WR={s_wr:.0f}% | "
              f"PF={s_pf:.2f} | PnL=${s_pnl:>7.1f}")

    # ── آمار پراپ
    total_acc = passed_count + blown_count
    pass_rate = passed_count / total_acc * 100 if total_acc > 0 else 0
    print(f"{'─'*65}")
    print(f" Accounts Passed:       {passed_count:>4}")
    print(f" Accounts Blown:        {blown_count:>4}")
    print(f" Pass Rate:             {pass_rate:.1f}%")
    print(f" Total Banked:          ${banked_total:>9,.2f}")
    print(f" Monthly Avg:           ${banked_total/total_months:.2f}/ماه")
    print(f" Active Balance:        ${balance:>9,.2f}")

    # ── جدول اکانت‌ها
    print(f"\n{'─'*65}")
    print(" جزئیات اکانت‌ها (آخرین ۲۰):")
    for acc in accounts[-20:]:
        icon   = '💰' if acc['result'] == 'PASSED' else '💥'
        banked = f"| Banked:${acc['banked']:>7.1f}" if acc['result'] == 'PASSED' else ''
        print(f"  {icon} #{acc['num']:>3} | {acc['result']:<6} "
              f"| Bal:${acc['final']:>7.1f} {banked} "
              f"| T:{acc['trades']:>3} | {acc['reason']}")

    # ── خروجی‌ها
    print(f"\n{'─'*65}")
    print(" خروج‌ها:")
    for r, cnt in trades['reason'].value_counts().items():
        pct = cnt / len(trades) * 100
        print(f"   {r:<15}: {cnt:>4} ({pct:.1f}%)")
    print("═"*65)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 CorrArb v9 — Dual-Logic Portfolio System")
    print("═"*65)

    portfolio = load_all_data()
    if not portfolio:
        print("❌ هیچ داده‌ای یافت نشد!")
        exit()

    all_trades = []

    for sym, df in portfolio.items():
        print(f"\n⚙️  Processing {sym}...")

        df = add_indicators(df)
        models, df = train_ml_models(df, sym)

        train_size = len(df[df.index <= Config.train_end])
        print(f"   ML trained | train-bars: {train_size:,}")

        sigs = generate_signals(df, sym, models)

        if sigs.empty:
            print(f"   ⚠️ No signals for {sym}")
            continue

        mr_cnt = (sigs['regime'] == 'ranging').sum()
        tr_cnt = (sigs['regime'] == 'trend').sum()
        print(f"   Signals: {len(sigs)} | MR:{mr_cnt} | Trend:{tr_cnt}")

        trades = backtest_asset(sigs, df, sym)
        if trades:
            t_df = pd.DataFrame(trades)
            all_trades.append(t_df)
            print(f"   Trades executed: {len(trades)}")
        else:
            print(f"   ⚠️ No trades executed for {sym}")

    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        combined['exit_ts'] = pd.to_datetime(combined['exit_ts'])
        combined['entry_ts'] = pd.to_datetime(combined['entry_ts'])
        prop_simulator(combined)
    else:
        print("\n❌ هیچ ترید فعالی یافت نشد!")
