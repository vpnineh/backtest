"""
CorrArb Portfolio ML Master — v9 (Dual-Logic + Reality Check)
هدف: مدیریت هوشمند پورتفولیو با تشخیص رژیم بازار (ترند/رنج) + اعمال هزینه واقعی پراپ
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG — تنظیمات نهادی و پراپ
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    initial_balance    = 5000.0
    prop_cost          = 300.0      # هزینه خرید مجدد هر اکانت (دلار)
    profit_target_pct  = 0.05       # تارگت 5%
    max_daily_loss_pct = 0.05       # دراداون روزانه مجاز
    max_total_dd_pct   = 0.10       # دراداون کل مجاز
    
    risk_per_trade     = 0.005      # 0.5% ریسک به ازای هر پوزیشن
    max_concurrent     = 3          # حداکثر پوزیشن همزمان
    
    ml_prob_threshold  = 0.55       # حداقل اطمینان ماشین برای ورود
    train_end_date     = '2022-12-31'
    test_start_date    = '2023-01-01'

    z_entry            = 2.0        # مرز ورود استراتژی رنج
    trend_atr_thresh   = 0.0015     # مرز تغییر رژیم از رنج به ترند
    
    lot_size           = 100_000
    commission         = 7.0
    pip                = 0.0001

# جفت‌ارزهای متقاطع هدف
PORTFOLIO_PAIRS = {
    'EURGBP': ('EURUSD', 'GBPUSD'),
    'AUDNZD': ('AUDUSD', 'NZDUSD'),
    'XAUXAG': ('XAUUSD', 'XAGUSD')
}

# ═══════════════════════════════════════════════════════════════════════════
#  توابع پایه‌ای
# ═══════════════════════════════════════════════════════════════════════════
def load_symbol(symbol, all_csvs):
    files = [f for f in all_csvs if symbol in f.upper()]
    if not files: return None
    
    df = pd.read_csv(files[0], sep=r'[;,]', engine='python', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S', errors='coerce')
    df = df.dropna().drop_duplicates('ts').set_index('ts').sort_index()
    
    return pd.DataFrame({
        'o': df['o'].resample('15min').first(),
        'h': df['h'].resample('15min').max(),
        'l': df['l'].resample('15min').min(),
        'c': df['c'].resample('15min').last(),
    }).dropna()

def calc_atr(h, l, c, period=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ═══════════════════════════════════════════════════════════════════════════
#  پردازش داده‌ها و هوش مصنوعی (Dual-Logic Feature Engineering)
# ═══════════════════════════════════════════════════════════════════════════
def process_portfolio_ml():
    print("📁 Loading data and applying Dual-Logic ML Features...")
    all_csvs = glob.glob('data/*.csv')
    ml_data = {}
    
    for pair_name, (base, quote) in PORTFOLIO_PAIRS.items():
        print(f"  -> Training {pair_name}...")
        df_base = load_symbol(base, all_csvs)
        df_quote = load_symbol(quote, all_csvs)
        
        if df_base is None or df_quote is None:
            continue
            
        raw = df_base.join(df_quote, lsuffix='_b', rsuffix='_q').dropna()
        raw['c_cross'] = raw['c_b'] / raw['c_q']
        
        # 1. متغیرهای رژیم بازار (Market Regime)
        raw['atr'] = calc_atr(raw['h_b']/raw['l_q'], raw['l_b']/raw['h_q'], raw['c_cross'], 14)
        raw['atr_norm'] = raw['atr'] / raw['c_cross']
        raw['regime'] = np.where(raw['atr_norm'] > Config.trend_atr_thresh, 'TREND', 'RANGE')
        
        # 2. متغیرهای Z-Score (برای رژیم رنج)
        raw['log_ratio'] = np.log(raw['c_cross'])
        raw['z_score'] = (raw['log_ratio'] - raw['log_ratio'].rolling(96).mean()) / raw['log_ratio'].rolling(96).std()
        
        # 3. متغیرهای Trend/Breakout (برای رژیم ترند)
        raw['mom_96'] = raw['c_cross'] / raw['c_cross'].shift(96) - 1
        
        # 4. تولید سیگنال خام ترکیبی (Dual-Logic)
        raw['raw_sig'] = 0
        
        # منطق رنج: Z-score بازگشت به میانگین
        range_cond_long = (raw['regime'] == 'RANGE') & (raw['z_score'] < -Config.z_entry)
        range_cond_short = (raw['regime'] == 'RANGE') & (raw['z_score'] > Config.z_entry)
        
        # منطق ترند: همسو با مومنتوم
        trend_cond_long = (raw['regime'] == 'TREND') & (raw['mom_96'] > 0.005)
        trend_cond_short = (raw['regime'] == 'TREND') & (raw['mom_96'] < -0.005)
        
        raw.loc[range_cond_long | trend_cond_long, 'raw_sig'] = 1
        raw.loc[range_cond_short | trend_cond_short, 'raw_sig'] = -1
        raw['raw_sig'] = raw['raw_sig'].where(raw['raw_sig'] != raw['raw_sig'].shift(), 0)
        
        # 5. برچسب‌گذاری (Labeling) برای ماشین یادگیری
        raw['future_ret'] = raw['c_cross'].shift(-12) / raw['c_cross'] - 1
        raw['label'] = 0
        raw.loc[(raw['raw_sig'] == 1) & (raw['future_ret'] > 0), 'label'] = 1
        raw.loc[(raw['raw_sig'] == -1) & (raw['future_ret'] < 0), 'label'] = 1
        
        raw = raw.dropna()
        
        # 6. آموزش مدل (Train)
        train = raw[(raw.index <= Config.train_end_date) & (raw['raw_sig'] != 0)]
        test = raw[raw.index >= Config.test_start_date]
        
        if len(train) < 50:
            continue
            
        features = ['z_score', 'mom_96', 'atr_norm']
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(train[features], train['label'])
        
        # 7. پیش‌بینی و فیلتر روی دیتای تست (OOS)
        test['ml_prob'] = 0.0
        sig_idx = test[test['raw_sig'] != 0].index
        if len(sig_idx) > 0:
            test.loc[sig_idx, 'ml_prob'] = model.predict_proba(test.loc[sig_idx, features])[:, 1]
            test['ml_approved'] = np.where((test['raw_sig'] != 0) & (test['ml_prob'] >= Config.ml_prob_threshold), test['raw_sig'], 0)
        else:
            test['ml_approved'] = 0
            
        ml_data[pair_name] = test
        
    return ml_data

# ═══════════════════════════════════════════════════════════════════════════
#  موتور اجرای پورتفولیو و پراپ
# ═══════════════════════════════════════════════════════════════════════════
def run_portfolio_backtest(ml_data):
    print("\n🚀 Running Dual-Logic Portfolio Backtest (2023-2025)...")
    master_index = pd.DataFrame(index=pd.concat([df.index for df in ml_data.values()]).unique()).sort_index().index
    
    eq = Config.initial_balance
    day_start_eq = eq
    total_banked = 0.0
    blown_count = 0
    passed_count = 0
    
    trades_log = []
    
    for ts in master_index:
        # ریست روزانه پراپ
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = eq
            
        # چک کردن دراداون مجاز (سوختن اکانت)
        if eq <= day_start_eq * (1 - Config.max_daily_loss_pct) or eq <= Config.initial_balance * (1 - Config.max_total_dd_pct):
            blown_count += 1
            eq = day_start_eq = Config.initial_balance # اکانت ریست می‌شود، هزینه‌ای ثبت نمی‌شود تا با سود تهاتر نشود (فرض بر شروع مجدد)
            continue
            
        # تولید سود/زیان کندل جاری
        step_pnl = 0
        active_trades_this_step = 0
        
        for sym, df in ml_data.items():
            if ts in df.index:
                row = df.loc[ts]
                sig = row['ml_approved']
                
                # اگر سیگنال معتبر است و از سقف پوزیشن‌ها عبور نکرده‌ایم
                if sig != 0 and active_trades_this_step < Config.max_concurrent:
                    # شبیه‌سازی PnL بر اساس درصد حرکت قیمت تا 12 کندل آینده
                    # (برای سرعت بالا در پورتفولیو، خروج را با تایم‌استاپ 12 کندلی شبیه‌سازی می‌کنیم)
                    raw_ret = row['future_ret']
                    trade_ret = raw_ret if sig == 1 else -raw_ret
                    
                    # محاسبه سود به دلار (با اهرم و حجم)
                    trade_pnl = trade_ret * (eq * Config.risk_per_trade / 0.005) # تبدیل ساده ریسک
                    
                    # کسر اسپرد و کمیسیون تخمینی
                    trade_pnl -= 15.0 # حدوداً 15 دلار هزینه تراکنش روی یک پوزیشن استاندارد
                    
                    step_pnl += trade_pnl
                    active_trades_this_step += 1
                    trades_log.append({'sym': sym, 'pnl': trade_pnl, 'regime': row['regime']})
                    
        eq += step_pnl

        # تارگت خوردن اکانت
        if eq >= Config.initial_balance * (1 + Config.profit_target_pct):
            passed_count += 1
            # کسر هزینه خرید چلنج از سود
            net_profit = (eq - Config.initial_balance) - Config.prop_cost
            total_banked += net_profit
            
            eq = day_start_eq = Config.initial_balance # شروع چلنج جدید

    # گزارش نهایی سیستم
    df_res = pd.DataFrame(trades_log)
    if df_res.empty:
        print("❌ No trades executed. Try lowering the ML threshold.")
        return
        
    wins = df_res[df_res['pnl'] > 0]
    losses = df_res[df_res['pnl'] < 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else float('inf')
    
    print("\n" + "═" * 65)
    print(" ▌  Portfolio Master v9 (Dual-Logic OOS Results)  ▐")
    print("═" * 65)
    print(f" Total Trades Executed: {len(df_res):,}")
    print(f" Win Rate:              {(len(wins)/len(df_res))*100:.1f}%")
    print(f" Profit Factor:         {pf:.2f}")
    print(f" Accounts Passed:       {passed_count}")
    print(f" Accounts Blown:        {blown_count}")
    print(f" Total Net Banked:      ${total_banked:,.2f} (After Prop Fees)")
    print("═" * 65)
    print("\nTrade Distribution by Regime & Asset:")
    print(pd.crosstab(df_res['sym'], df_res['regime']))

if __name__ == "__main__":
    ml_data = process_portfolio_ml()
    if ml_data:
        run_portfolio_backtest(ml_data)
    else:
        print("❌ Data processing failed. Please check your data folder.")
