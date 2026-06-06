"""
CorrArb Portfolio ML Master — v8
هدف: اجرای ML Meta-Labeler روی پورتفولیوی کامل (10 جفت‌ارز)
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# لیست نمادهایی که در پوشه data موجود دارید
ASSETS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'EURGBP', 'AUDNZD', 'XAUUSD', 'XAGUSD']

class Config:
    initial_balance = 5000.0
    profit_target_pct = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct = 0.10
    train_end_date = '2022-12-31'
    test_start_date = '2023-01-01'

def load_all_data():
    all_files = glob.glob('data/*.csv')
    portfolio = {}
    for sym in ASSETS:
        files = [f for f in all_files if sym in f.upper()]
        if files:
            df = pd.read_csv(files[0], sep=r'[;,]', engine='python', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['ts'] = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S', errors='coerce')
            df = df.dropna().drop_duplicates('ts').set_index('ts').sort_index()
            portfolio[sym] = df[['o', 'h', 'l', 'c']].resample('15min').agg({'o':'first', 'h':'max', 'l':'min', 'c':'last'}).dropna()
    return portfolio

def train_and_predict(portfolio):
    print("🧠 Training ML Models for Portfolio...")
    ml_data = {}
    for sym, df in portfolio.items():
        # Feature Engineering ساده برای یادگیری
        df['log_ret'] = np.log(df['c']).diff()
        df['z_score'] = (df['log_ret'] - df['log_ret'].rolling(96).mean()) / df['log_ret'].rolling(96).std()
        df['rsi'] = 100 - (100 / (1 + df['c'].diff().clip(lower=0).rolling(14).mean() / (-df['c'].diff().clip(upper=0).rolling(14).mean())))
        df['label'] = np.where(df['log_ret'].shift(-12) > 0, 1, 0)
        df = df.dropna()
        
        train = df[df.index <= Config.train_end_date]
        test = df[df.index >= Config.test_start_date]
        
        # مدل ML
        X_train = train[['z_score', 'rsi']].fillna(0)
        model = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X_train, train['label'])
        
        X_test = test[['z_score', 'rsi']].fillna(0)
        test['ml_prob'] = model.predict_proba(X_test)[:, 1]
        ml_data[sym] = test
    return ml_data

def run_portfolio_backtest(ml_data):
    print("🚀 Running Portfolio Backtest on OOS Data...")
    # منطقِ اجرای پورتفولیو که در مرحله قبل نوشتیم را اینجا قرار دهید
    # این موتور همزمان سیگنال‌های ml_prob > 0.55 را از ۱۰ جفت ارزی می‌گیرد
    # و با تقسیم ریسک بین نمادها، دراداون را مدیریت می‌کند.
    pass

if __name__ == "__main__":
    data = load_all_data()
    ml_data = train_and_predict(data)
    run_portfolio_backtest(ml_data)
