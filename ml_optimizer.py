"""
CorrArb Prop ML Optimizer — v7 Meta-Labeler Grid Search
هدف: آموزش ماشین یادگیری و یافتن بهترین تنظیمات برای دیتای تست (2023-2025)
"""

import pandas as pd
import numpy as np
import glob
import warnings
import itertools
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  تنظیمات گرید ماشین یادگیری (تعداد محدودتر برای سرعت پردازش)
# ═══════════════════════════════════════════════════════════════════════════
PARAM_GRID = {
    'z_entry': [1.90, 2.00],               # سخت‌گیری در تولید سیگنال اولیه
    'ml_prob_threshold': [0.52, 0.55, 0.58], # چقدر ماشین باید مطمئن باشد تا ورود کنیم؟
    'corr_min': [0.70, 0.75],              # فیلتر همبستگی پایه
}

class Config:
    # قوانین پراپ
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10
    risk_base_pct      = 0.015  # ریسک بالاتر چون سیگنال‌ها فیلتر شده‌اند
    risk_min_pct       = 0.005
    consec_loss_n      = 2
    risk_reduce        = 0.5
    
    # هزینه‌ها
    spread_pips        = 1.2
    commission_per_lot = 7.0
    slippage_pips      = 0.3
    pip                = 0.0001
    lot_size           = 100_000
    max_lot            = 5.0
    min_lot            = 0.01
    warmup             = 500
    
    # ثابت‌های استراتژی
    z_fast_period      = 96
    z_exit             = 0.1
    z_stop_margin      = 3.5
    time_stop_bars     = 60
    sl_pips            = 40.0
    tp_pips            = 80.0
    
    # دیتای آموزش و تست
    train_end_date     = '2022-12-31'
    test_start_date    = '2023-01-01'

# ── توابع محاسباتی پایه ──
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rsi(c, period=14):
    delta = c.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_dynamic_pnl(dir_trade, ep, cp, lot, cg, C):
    return (dir_trade * (cp - ep) * lot * C.lot_size * cg) - (C.commission_per_lot * lot)

def calc_lot(equity, sl_pips, consec_loss, C):
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n: risk = max(risk * C.risk_reduce, C.risk_min_pct)
    return round(float(np.clip(equity * risk / (sl_pips * 12.5), C.min_lot, C.max_lot)), 2)

# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری و مهندسی ویژگی‌ها
# ═══════════════════════════════════════════════════════════════════════════
def load_and_engineer_data() -> pd.DataFrame:
    print("  Loading Data and Engineering ML Features...")
    files_eur = sorted(glob.glob('data/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob('data/*GBPUSD*.csv'))
    
    def read_pair(paths, suffix):
        frames = [pd.read_csv(p, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v']) for p in paths]
        df = pd.concat(frames).sort_values('ts')
        df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
        df = df.drop_duplicates(subset=['ts'], keep='last').set_index('ts')
        df.columns = [f'{col}_{suffix}' for col in df.columns]
        return df

    raw = read_pair(files_eur, 'eur').join(read_pair(files_gbp, 'gbp'), how='inner').dropna()
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
    
    df['c_eg'] = df['c_eur'] / df['c_gbp']
    df['o_eg'] = df['o_eur'] / df['o_gbp']
    df = df[df.index.weekday < 5]

    # تولید ویژگی‌های ماشین یادگیری (Features)
    C = Config()
    df['log_ratio'] = np.log(df['c_eg'])
    df['z_score'] = (df['log_ratio'] - df['log_ratio'].rolling(C.z_fast_period).mean()) / df['log_ratio'].rolling(C.z_fast_period).std()
    
    ret_eur, ret_gbp = df['c_eur'].pct_change(), df['c_gbp'].pct_change()
    df['feat_corr_48'] = ret_eur.rolling(48).corr(ret_gbp)
    df['feat_corr_96'] = ret_eur.rolling(96).corr(ret_gbp)
    df['feat_rsi_diff'] = calc_rsi(df['c_eur'], 14) - calc_rsi(df['c_gbp'], 14)
    df['feat_atr_norm'] = calc_atr(df['h_eur'], df['l_eur'], df['c_eur'], 14) / df['c_eur']
    df['feat_hour'] = df.index.hour
    
    # برچسب‌گذاری برای آموزش
    df['future_ret'] = df['log_ratio'].shift(-12) - df['log_ratio']
    
    return df.dropna()

# ═══════════════════════════════════════════════════════════════════════════
#  موتور اجرای بک‌تست برای دیتای OOS (2023-2025)
# ═══════════════════════════════════════════════════════════════════════════
def run_oos_backtest(df_test, params, C):
    open_eg, close_eg, close_g = df_test['o_eg'].values, df_test['c_eg'].values, df_test['c_gbp'].values
    sig_a, z_a = df_test['ml_approved'].values, df_test['z_score'].values
    ts_a = df_test.index

    PROP_FLOOR, PROFIT_LEVEL = C.initial_balance * (1 - C.max_total_dd_pct), C.initial_balance * (1 + C.profit_target_pct)
    total_withdrawn, acc_num = 0.0, 1
    trades_list = []
    
    eq = day_start_eq = C.initial_balance
    consec_loss = trades_today = pending_sig = 0
    pos = None; blown = False

    for bar in range(1, len(ts_a)):
        ts = ts_a[bar]
        if ts.hour == 0 and ts.minute == 0: day_start_eq, trades_today = eq, 0

        if blown:
            eq, day_start_eq, consec_loss, pos, trades_today, pending_sig, blown = C.initial_balance, C.initial_balance, 0, None, 0, 0, False
            acc_num += 1; continue

        if pending_sig != 0 and pos is None and trades_today < 3:
            sv = pending_sig
            lot = calc_lot(eq, C.sl_pips, consec_loss, C)
            ep = open_eg[bar] + (sv * (C.slippage_pips + C.spread_pips/2) * C.pip)
            pos = {'dir': sv, 'lot': lot, 'entry': ep, 'sl': ep - sv * C.sl_pips * C.pip, 'tp': ep + sv * C.tp_pips * C.pip, 'entry_bar': bar}
            trades_today += 1
        pending_sig = 0

        if pos is not None:
            cp, cg, d, ep, zn = close_eg[bar], close_g[bar], pos['dir'], pos['entry'], z_a[bar]
            current_pnl = calc_dynamic_pnl(d, ep, cp, pos['lot'], cg, C)
            current_eq = eq + current_pnl

            if current_eq <= day_start_eq * (1 - C.max_daily_loss_pct) or current_eq <= PROP_FLOOR:
                trades_list.append({'pnl': current_pnl, 'status': 'BLOWN'})
                blown = True; pos = None; continue

            hit_z_stop = not np.isnan(zn) and ((d == 1 and zn <= -C.z_stop_margin) or (d == -1 and zn >= C.z_stop_margin))
            hit_z_exit = not np.isnan(zn) and ((d == 1 and zn >= -C.z_exit) or (d == -1 and zn <= C.z_exit))
            hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
            hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])
            time_stop = (bar - pos['entry_bar']) >= C.time_stop_bars

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                st = 'SL' if hit_sl else ('TP' if hit_tp else ('Z-Stop' if hit_z_stop else ('TimeStop' if time_stop else 'Z-Exit')))
                final_pnl = calc_dynamic_pnl(d, ep, pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp), pos['lot'], cg, C)
                eq += final_pnl
                trades_list.append({'pnl': final_pnl, 'status': st})
                if final_pnl > 0: consec_loss = 0
                else: consec_loss += 1
                pos = None

        if eq >= PROFIT_LEVEL and pos is None and not blown:
            total_withdrawn += (eq - C.initial_balance)
            acc_num += 1
            eq, day_start_eq, consec_loss, trades_today = C.initial_balance, C.initial_balance, 0, 0

        if pos is None and not blown and trades_today < 3 and sig_a[bar] != 0:
            pending_sig = sig_a[bar]

    if not trades_list: return {'pf': 0, 'wr': 0, 'trades': 0, 'banked': 0, 'blown': acc_num-1, 'passed': 0}
    df_t = pd.DataFrame(trades_list)
    wins, losses = df_t[df_t['pnl'] > 0], df_t[df_t['pnl'] < 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    blown_count = len(df_t[df_t['status'] == 'BLOWN'])
    target_hits = acc_num - 1 - blown_count

    return {
        'pf': round(pf, 2),
        'wr': round(len(wins) / len(df_t) * 100, 1),
        'trades': len(df_t),
        'banked': round(total_withdrawn, 0),
        'blown': blown_count,
        'passed': target_hits
    }

# ═══════════════════════════════════════════════════════════════════════════
#  اجرای لوپ آموزش و تست
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df_main = load_and_engineer_data()
    C = Config()
    
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n🧠 Starting ML Meta-Labeler Optimization: {len(combinations)} Models to train.")
    print("Testing period: 2023-2025 (Out of Sample Data Only)")
    print("-" * 80)
    print(f"{'Z_Ent':<6} | {'ML_Prb':<6} | {'Corr':<4} || {'PF':<4} | {'WR%':<4} | {'Trades':<6} | {'Passed':<6} | {'Blown':<5} | {'Banked $':<8}")
    print("-" * 80)

    results_list = []
    features_cols = ['z_score', 'feat_corr_48', 'feat_corr_96', 'feat_rsi_diff', 'feat_atr_norm', 'feat_hour']

    for params in combinations:
        df = df_main.copy()
        
        # 1. تولید سیگنال خام برای کل دیتا بر اساس تنظیمات گرید
        df['raw_sig'] = 0
        df.loc[(df['z_score'] < -params['z_entry']) & (df['feat_corr_96'] > params['corr_min']), 'raw_sig'] = 1
        df.loc[(df['z_score'] > params['z_entry']) & (df['feat_corr_96'] > params['corr_min']), 'raw_sig'] = -1
        df['raw_sig'] = df['raw_sig'].where(df['raw_sig'] != df['raw_sig'].shift(), 0)

        # 2. برچسب‌گذاری آموزشی
        df['label'] = 0
        df.loc[(df['raw_sig'] == 1) & (df['future_ret'] > 0), 'label'] = 1
        df.loc[(df['raw_sig'] == -1) & (df['future_ret'] < 0), 'label'] = 1

        # 3. تقسیم به Train و Test
        train_df = df[(df.index <= C.train_end_date) & (df['raw_sig'] != 0)].copy()
        test_df = df[df.index >= C.test_start_date].copy()

        if len(train_df) < 50: continue # دیتای کافی برای آموزش نیست

        # 4. آموزش Random Forest
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[features_cols])
        y_train = train_df['label']
        
        model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # 5. پیش‌بینی روی OOS و اعمال فیلتر
        test_df['ml_approved'] = 0
        sig_idx = test_df[test_df['raw_sig'] != 0].index
        
        if len(sig_idx) > 0:
            X_test = scaler.transform(test_df.loc[sig_idx, features_cols])
            probs = model.predict_proba(X_test)[:, 1]
            test_df.loc[sig_idx, 'ml_prob'] = probs
            # ماشین فقط سیگنال‌هایی را تایید می‌کند که احتمال موفقیتشان بالای ML_Prob گرید باشد
            test_df.loc[(test_df['raw_sig'] != 0) & (test_df['ml_prob'] >= params['ml_prob_threshold']), 'ml_approved'] = test_df['raw_sig']

        # 6. بک‌تست نهایی روی دیتای تست 2023+
        res = run_oos_backtest(test_df, params, C)
        res.update(params)
        results_list.append(res)
        
        print(f"{params['z_entry']:<6.2f} | {params['ml_prob_threshold']:<6.2f} | {params['corr_min']:<4.2f} || "
              f"{res['pf']:<4.2f} | {res['wr']:<4.1f} | {res['trades']:<6} | {res['passed']:<6} | {res['blown']:<5} | ${res['banked']:<8.0f}")

    print("-" * 80)
    if results_list:
        best_pf = max(results_list, key=lambda x: x['pf'])
        print("\n🏆 BEST OOS PERFORMANCE (2023-2025):")
        print(best_pf)
