"""
CorrArb Portfolio Master — v8 (ML + Multi-Asset)
هدف: مدیریت همزمان چندین جفت‌ارز آربیتراژی، کنترل ریسک متمرکز و اعمال قوانین سخت‌گیرانه پراپ
"""

import os
import zipfile
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  تنظیمات مرکزی پورتفولیو
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ فرم ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05       # 5% تارگت
    max_daily_loss_pct = 0.05       # 5% افت روزانه مجاز
    max_total_dd_pct   = 0.10       # 10% افت کل مجاز

    # ── مدیریت سرمایه پورتفولیو ──
    risk_per_trade     = 0.005      # ریسک هر معامله 0.5% (به دلیل تعدد معاملات پورتفولیو)
    max_concurrent     = 4          # حداکثر ۴ پوزیشن همزمان در کل پورتفولیو
    
    # ── هزینه‌ها و بازار ──
    spread_pips        = 1.2
    commission_per_lot = 7.0
    slippage_pips      = 0.3
    pip                = 0.0001
    lot_size           = 100_000
    
    # ── تنظیمات استراتژی ──
    z_fast_period      = 96
    z_entry            = 1.90       # ورود با تایید ماشین
    z_exit             = 0.1
    z_stop_margin      = 3.5
    time_stop_bars     = 60
    sl_pips            = 40.0
    tp_pips            = 80.0
    corr_min           = 0.70
    
    # ── ماشین یادگیری ──
    train_end_date     = '2022-12-31'
    test_start_date    = '2023-01-01'
    ml_prob_threshold  = 0.55

# جفت‌ارزهای متقاطع که سیستم از روی فایل‌های پایه شما می‌سازد
PORTFOLIO_PAIRS = {
    'EURGBP': ('EURUSD', 'GBPUSD'),
    'AUDNZD': ('AUDUSD', 'NZDUSD'),
    'XAUXAG': ('XAUUSD', 'XAGUSD')  # طلا به نقره
}

# ═══════════════════════════════════════════════════════════════════════════
#  اتوماسیون پوشه و پردازش داده‌ها
# ═══════════════════════════════════════════════════════════════════════════
def prepare_data_directory(data_path="data"):
    print("📁 Scanning and extracting data folder...")
    zip_files = glob.glob(os.path.join(data_path, "*.zip"))
    for z in zip_files:
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            os.remove(z) # پاک کردن فایل زیپ برای خلوت شدن فضا
        except Exception as e:
            pass
    return glob.glob(os.path.join(data_path, "*.csv"))

def load_symbol(symbol, all_csvs):
    files = sorted([f for f in all_csvs if symbol in f.upper()])
    if not files:
        return None
    
    frames = []
    for f in files:
        try:
            # HistData format
            df = pd.read_csv(f, sep=r'[;,]', engine='python', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            frames.append(df)
        except: pass
        
    if not frames: return None
    
    df = pd.concat(frames)
    df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S', errors='coerce')
    df = df.dropna(subset=['ts']).drop_duplicates('ts', keep='last').set_index('ts').sort_index()
    
    # Resample to 15m immediately to save RAM
    resampled = pd.DataFrame({
        'o': df['o'].resample('15min').first(),
        'h': df['h'].resample('15min').max(),
        'l': df['l'].resample('15min').min(),
        'c': df['c'].resample('15min').last(),
    }).dropna()
    
    return resampled

# ═══════════════════════════════════════════════════════════════════════════
#  ویژگی‌ها و ماشین یادگیری
# ═══════════════════════════════════════════════════════════════════════════
def process_pair(pair_name, base_sym, quote_sym, all_csvs):
    print(f"\n⚙️ Processing {pair_name} ({base_sym}/{quote_sym})...")
    df_base = load_symbol(base_sym, all_csvs)
    df_quote = load_symbol(quote_sym, all_csvs)
    
    if df_base is None or df_quote is None:
        print(f"⚠️ Skipped {pair_name}: Missing data.")
        return None
        
    raw = df_base.join(df_quote, how='inner', lsuffix='_base', rsuffix='_quote').dropna()
    
    # ساخت جفت متقاطع
    raw['c_cross'] = raw['c_base'] / raw['c_quote']
    raw['o_cross'] = raw['o_base'] / raw['o_quote']
    
    # ML Features
    raw['log_ratio'] = np.log(raw['c_cross'])
    raw['z_score'] = (raw['log_ratio'] - raw['log_ratio'].rolling(Config.z_fast_period).mean()) / raw['log_ratio'].rolling(Config.z_fast_period).std()
    
    ret_base, ret_quote = raw['c_base'].pct_change(), raw['c_quote'].pct_change()
    raw['corr'] = ret_base.rolling(96).corr(ret_quote)
    raw['hour'] = raw.index.hour
    
    raw['raw_sig'] = 0
    raw.loc[(raw['z_score'] < -Config.z_entry) & (raw['corr'] > Config.corr_min), 'raw_sig'] = 1
    raw.loc[(raw['z_score'] > Config.z_entry) & (raw['corr'] > Config.corr_min), 'raw_sig'] = -1
    raw['raw_sig'] = raw['raw_sig'].where(raw['raw_sig'] != raw['raw_sig'].shift(), 0)
    
    raw['future_ret'] = raw['log_ratio'].shift(-12) - raw['log_ratio']
    raw['label'] = 0
    raw.loc[(raw['raw_sig'] == 1) & (raw['future_ret'] > 0), 'label'] = 1
    raw.loc[(raw['raw_sig'] == -1) & (raw['future_ret'] < 0), 'label'] = 1
    
    raw = raw.dropna()
    
    # ── Train ML ──
    train_df = raw[(raw.index <= Config.train_end_date) & (raw['raw_sig'] != 0)]
    test_df = raw[raw.index >= Config.test_start_date]
    
    if len(train_df) < 50:
        return None
        
    features = ['z_score', 'corr', 'hour']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, train_df['label'])
    
    # ── Test (OOS) ──
    test_df['ml_approved'] = 0
    sig_idx = test_df[test_df['raw_sig'] != 0].index
    if len(sig_idx) > 0:
        X_test = scaler.transform(test_df.loc[sig_idx, features])
        probs = model.predict_proba(X_test)[:, 1]
        test_df.loc[sig_idx, 'ml_prob'] = probs
        test_df.loc[(test_df['raw_sig'] != 0) & (test_df['ml_prob'] >= Config.ml_prob_threshold), 'ml_approved'] = test_df['raw_sig']
        
    return test_df

# ═══════════════════════════════════════════════════════════════════════════
#  موتور اجرای پورتفولیو (Centralized Risk)
# ═══════════════════════════════════════════════════════════════════════════
def run_portfolio_backtest(portfolio_data):
    print("\n🚀 Running Centralized Portfolio Backtest (2023-2025)...")
    
    # تجمیع تمام سیگنال‌ها و داده‌ها روی یک تایم‌لاین واحد
    master_index = None
    for pair, df in portfolio_data.items():
        if master_index is None:
            master_index = df.index
        else:
            master_index = master_index.union(df.index)
            
    master_index = master_index.sort_values()
    
    # متغیرهای حساب
    C = Config()
    eq = day_start_eq = C.initial_balance
    total_withdrawn = 0.0
    acc_num = 1
    blown_count = 0
    open_positions = []
    trades_log = []
    
    PROP_FLOOR = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)
    
    # حلقه اصلی زمان
    for ts in master_index:
        # ریست روزانه پراپ
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = eq
            
        # بررسی چک کردن دراداون پورتفولیو
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)
        if eq <= daily_limit or eq <= PROP_FLOOR:
            # سوختن اکانت
            for pos in open_positions:
                trades_log.append({'pnl': pos['floating_pnl'], 'status': 'BLOWN', 'pair': pos['pair']})
            blown_count += 1
            acc_num += 1
            eq = day_start_eq = C.initial_balance
            open_positions = []
            continue

        # مدیریت پوزیشن‌های باز
        active_pos = []
        for pos in open_positions:
            pair = pos['pair']
            if ts not in portfolio_data[pair].index:
                active_pos.append(pos)
                continue
                
            row = portfolio_data[pair].loc[ts]
            cp = row['c_cross']
            cg = row['c_quote']
            ep, d, lot = pos['entry'], pos['dir'], pos['lot']
            zn = row['z_score']
            
            # محاسبه سود لحظه‌ای
            gross_gbp = d * (cp - ep) * lot * C.lot_size
            net_usd = (gross_gbp * cg) - (C.commission_per_lot * lot)
            pos['floating_pnl'] = net_usd
            
            # شروط خروج
            hit_sl = (d == 1 and cp <= pos['sl']) or (d == -1 and cp >= pos['sl'])
            hit_tp = (d == 1 and cp >= pos['tp']) or (d == -1 and cp <= pos['tp'])
            hit_z_stop = not np.isnan(zn) and ((d == 1 and zn <= -C.z_stop_margin) or (d == -1 and zn >= C.z_stop_margin))
            hit_z_exit = not np.isnan(zn) and ((d == 1 and zn >= -C.z_exit) or (d == -1 and zn <= C.z_exit))
            
            if hit_sl or hit_tp or hit_z_stop or hit_z_exit:
                st = 'SL' if hit_sl else ('TP' if hit_tp else ('Z-Stop' if hit_z_stop else 'Z-Exit'))
                eq += net_usd
                trades_log.append({'pair': pair, 'pnl': net_usd, 'status': st})
            else:
                active_pos.append(pos)
                
        open_positions = active_pos
        
        # برداشت سود (تارگت)
        if eq >= PROFIT_LEVEL and len(open_positions) == 0:
            total_withdrawn += (eq - C.initial_balance)
            acc_num += 1
            eq = day_start_eq = C.initial_balance
            continue

        # ورود به پوزیشن‌های جدید
        if len(open_positions) < C.max_concurrent:
            for pair, df in portfolio_data.items():
                if ts in df.index and len(open_positions) < C.max_concurrent:
                    row = df.loc[ts]
                    sig = row['ml_approved']
                    
                    # اطمینان از اینکه روی این جفت پوزیشن باز نداریم
                    already_open = any(p['pair'] == pair for p in open_positions)
                    
                    if sig != 0 and not already_open:
                        # محاسبه حجم بر اساس ریسک پورتفولیو
                        raw_lot = (eq * C.risk_per_trade) / (C.sl_pips * 10 * row['c_quote'])
                        lot = round(float(np.clip(raw_lot, 0.01, 5.0)), 2)
                        
                        ep = row['o_cross'] + (sig * (C.slippage_pips + C.spread_pips/2) * C.pip)
                        open_positions.append({
                            'pair': pair, 'dir': sig, 'lot': lot, 'entry': ep,
                            'sl': ep - sig * C.sl_pips * C.pip, 'tp': ep + sig * C.tp_pips * C.pip,
                            'floating_pnl': 0.0
                        })

    # گزارش‌گیری
    df_trades = pd.DataFrame(trades_log)
    if df_trades.empty:
        print("❌ No trades executed.")
        return
        
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    target_hits = acc_num - 1 - blown_count
    
    print("\n" + "═" * 65)
    print(" ▌  Portfolio Master Backtest (2023-2025 OOS)  ▐")
    print("═" * 65)
    print(f" Total Trades:    {len(df_trades):,}")
    print(f" Win Rate:        {len(wins) / len(df_trades) * 100:.1f}%")
    print(f" Profit Factor:   {pf:.2f}")
    print(f" Prop Accounts:   Passed: {target_hits} | Blown: {blown_count}")
    print(f" Total Banked:    ${total_withdrawn:,.2f}")
    print("═" * 65)
    print(" Trade Distribution by Asset:")
    print(df_trades['pair'].value_counts().to_string())

if __name__ == "__main__":
    all_csvs = prepare_data_directory()
    portfolio_data = {}
    
    for pair_name, (base_sym, quote_sym) in PORTFOLIO_PAIRS.items():
        processed_df = process_pair(pair_name, base_sym, quote_sym, all_csvs)
        if processed_df is not None:
            portfolio_data[pair_name] = processed_df
            
    if portfolio_data:
        run_portfolio_backtest(portfolio_data)
    else:
        print("❌ Could not process any pairs. Check your data files.")
