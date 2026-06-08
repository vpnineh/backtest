#!/usr/bin/env python3
"""
============================================================================
  ASIAN RANGE LIQUIDITY TRAP - PROP FIRM BACKTEST
  
  Strategy:
  - Define Asian session range (00:00-07:00 UTC)
  - Wait for London/NY to sweep Asian High or Low
  - Enter on rejection (close back inside range)
  - TP = 2R, SL = beyond sweep candle
  
  Target: 5%+ monthly profit, <5% daily DD, <10% total DD
  
  Data: HistData M1 CSVs in ./data/
  Format: 20200102 170000;1.12122;1.12124;1.12119;1.12120;0
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time as timer
from datetime import datetime

# ================================================================
# CONFIG (همه تنظیمات اینجاست - نیازی به فایل خارجی نیست)
# ================================================================

CONFIG = {
    # Data
    'data_dir': './data',
    'symbols': ['EURUSD', 'GBPUSD'],
    'years': list(range(2010, 2026)),
    
    # Account
    'initial_balance': 100_000,
    
    # Risk
    'risk_per_trade': 0.005,      # 0.5% per trade
    'max_daily_loss_pct': 0.04,   # 4% daily limit (prop = 5%, ما محتاط‌تر)
    'max_total_dd_pct': 0.08,     # 8% total limit (prop = 10%, ما محتاط‌تر)
    'max_trades_per_day': 2,      # حداکثر ۲ ترید در روز
    
    # Strategy
    'asian_start_hour': 0,        # شروع آسیا UTC
    'asian_end_hour': 7,          # پایان آسیا UTC
    'trade_start_hour': 7,        # شروع ترید (لندن)
    'trade_end_hour': 16,         # پایان ترید
    'rr_ratio': 2.0,              # Risk:Reward = 1:2
    'sl_buffer_pips': 2,          # بافر SL
    'min_asian_range_pips': 15,   # حداقل رنج آسیا (فیلتر بازار مرده)
    'max_asian_range_pips': 80,   # حداکثر رنج آسیا (فیلتر بازار وحشی)
    'sweep_min_pips': 2,          # حداقل نفوذ به سطح
    'cooldown_bars': 16,          # فاصله بین تریدها (بار ۱۵ دقیقه = ۴ ساعت)
    'max_trade_duration': 80,     # حداکثر مدت ترید (بار ۱۵ دقیقه = ۲۰ ساعت)
    
    # Costs
    'spread_pips': 1.2,
    'slippage_pips': 0.3,
    'commission_per_lot': 7.0,
    
    # Days
    'skip_monday_before': 9,      # دوشنبه قبل ۹ UTC ترید نکن
    'skip_friday_after': 14,      # جمعه بعد ۱۴ UTC ترید نکن
    
    # Output
    'output_dir': './results',
}

# ================================================================
# PIP DEFINITIONS
# ================================================================

PIP_INFO = {
    'EURUSD': {'pip': 0.0001, 'pip_val': 10.0},
    'GBPUSD': {'pip': 0.0001, 'pip_val': 10.0},
    'USDJPY': {'pip': 0.01,   'pip_val': 6.7},
    'XAUUSD': {'pip': 0.10,   'pip_val': 10.0},
    'XAGUSD': {'pip': 0.001,  'pip_val': 50.0},
    'USDCHF': {'pip': 0.0001, 'pip_val': 10.0},
    'USDCAD': {'pip': 0.0001, 'pip_val': 7.5},
    'AUDUSD': {'pip': 0.0001, 'pip_val': 10.0},
    'NZDUSD': {'pip': 0.0001, 'pip_val': 10.0},
    'EURGBP': {'pip': 0.0001, 'pip_val': 12.5},
    'AUDNZD': {'pip': 0.0001, 'pip_val': 6.0},
}


# ================================================================
# DATA LOADER
# ================================================================

def load_m1_data(data_dir, symbol, years):
    """Load all M1 CSV files for a symbol"""
    print(f"\n  Loading {symbol}...")
    frames = []
    
    for year in years:
        fp = os.path.join(data_dir, f"DAT_ASCII_{symbol}_M1_{year}.csv")
        if not os.path.exists(fp):
            continue
        
        try:
            df = pd.read_csv(fp, sep=';', header=None,
                             names=['DT', 'Open', 'High', 'Low', 'Close', 'Vol'])
            df['DT'] = df['DT'].str.strip()
            
            # Try formats
            parsed = False
            for fmt in ['%Y%m%d %H%M%S', '%Y.%m.%d %H:%M']:
                try:
                    df['DT'] = pd.to_datetime(df['DT'], format=fmt)
                    parsed = True
                    break
                except:
                    pass
            
            if not parsed:
                df['DT'] = pd.to_datetime(df['DT'], format='mixed')
            
            df.set_index('DT', inplace=True)
            frames.append(df[['Open', 'High', 'Low', 'Close']])
            print(f"    {year}: {len(df):>10,} bars")
        except Exception as e:
            print(f"    {year}: ERROR - {e}")
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined[combined.index.dayofweek < 5]  # weekdays only
    combined.dropna(inplace=True)
    
    print(f"    TOTAL: {len(combined):,} M1 bars")
    print(f"    Range: {combined.index[0]} → {combined.index[-1]}")
    
    return combined


def resample_m15(df_m1):
    """M1 → M15"""
    return df_m1.resample('15min').agg({
        'Open': 'first', 'High': 'max',
        'Low': 'min', 'Close': 'last'
    }).dropna()


# ================================================================
# ASIAN RANGE CALCULATOR
# ================================================================

def compute_asian_ranges(df_m1, asian_start, asian_end):
    """
    Vectorized: for each trading day, compute Asian session High/Low/Mid
    Returns dict: date → (asian_high, asian_low, asian_mid)
    """
    # Filter Asian session bars
    hours = df_m1.index.hour
    asian_mask = (hours >= asian_start) & (hours < asian_end)
    asian_data = df_m1[asian_mask].copy()
    asian_data['Date'] = asian_data.index.date
    
    # Group by date
    daily_asian = asian_data.groupby('Date').agg(
        AH=('High', 'max'),
        AL=('Low', 'min')
    )
    daily_asian['AM'] = (daily_asian['AH'] + daily_asian['AL']) / 2
    daily_asian['AR'] = daily_asian['AH'] - daily_asian['AL']
    
    return daily_asian.to_dict('index')


# ================================================================
# CORE STRATEGY
# ================================================================

def run_backtest(df_m15, asian_ranges, symbol, cfg):
    """
    Main backtest loop on M15 data
    Only loops through trading hours, skips everything else
    """
    pip = PIP_INFO[symbol]['pip']
    pip_val = PIP_INFO[symbol]['pip_val']
    
    balance = cfg['initial_balance']
    initial_balance = balance
    peak_balance = balance
    
    spread_cost = cfg['spread_pips'] * pip
    slip_cost = cfg['slippage_pips'] * pip
    total_entry_cost = spread_cost + slip_cost
    
    rr = cfg['rr_ratio']
    risk_pct = cfg['risk_per_trade']
    sl_buffer = cfg['sl_buffer_pips'] * pip
    min_range = cfg['min_asian_range_pips'] * pip
    max_range = cfg['max_asian_range_pips'] * pip
    sweep_min = cfg['sweep_min_pips'] * pip
    cooldown = cfg['cooldown_bars']
    max_duration = cfg['max_trade_duration']
    commission = cfg['commission_per_lot']
    max_trades_day = cfg['max_trades_per_day']
    max_daily_loss = cfg['max_daily_loss_pct']
    max_total_dd = cfg['max_total_dd_pct']
    
    trades = []
    equity_points = []
    
    # Pre-extract arrays for speed
    highs = df_m15['High'].values
    lows = df_m15['Low'].values
    opens = df_m15['Open'].values
    closes = df_m15['Close'].values
    times = df_m15.index
    hours = df_m15.index.hour
    dates = df_m15.index.date
    dows = df_m15.index.dayofweek
    n = len(df_m15)
    
    # State
    last_trade_bar = -cooldown
    current_date = None
    day_start_balance = balance
    day_trades = 0
    day_pnl = 0.0
    blown = False
    
    i = 0
    while i < n - 1:
        if blown:
            break
        
        dt = dates[i]
        hour = hours[i]
        dow = dows[i]
        
        # ---- Daily Reset ----
        if dt != current_date:
            # Save daily equity
            if current_date is not None:
                equity_points.append({
                    'date': current_date,
                    'balance': balance,
                    'day_pnl': day_pnl,
                    'day_trades': day_trades
                })
            
            current_date = dt
            day_start_balance = balance
            day_trades = 0
            day_pnl = 0.0
        
        # ---- Skip conditions ----
        # Not in trading hours
        if hour < cfg['trade_start_hour'] or hour >= cfg['trade_end_hour']:
            i += 1
            continue
        
        # Monday morning
        if dow == 0 and hour < cfg['skip_monday_before']:
            i += 1
            continue
        
        # Friday afternoon
        if dow == 4 and hour >= cfg['skip_friday_after']:
            i += 1
            continue
        
        # Cooldown
        if i - last_trade_bar < cooldown:
            i += 1
            continue
        
        # Max trades per day
        if day_trades >= max_trades_day:
            i += 1
            continue
        
        # Daily loss limit
        if day_pnl <= -day_start_balance * max_daily_loss:
            i += 1
            continue
        
        # Total DD check
        dd_pct = (peak_balance - balance) / initial_balance
        if dd_pct >= max_total_dd:
            blown = True
            break
        
        # ---- Get Asian Range for today ----
        if dt not in asian_ranges:
            i += 1
            continue
        
        ar = asian_ranges[dt]
        a_high = ar['AH']
        a_low = ar['AL']
        a_mid = ar['AM']
        a_range = ar['AR']
        
        # Filter: range too small or too big
        if a_range < min_range or a_range > max_range:
            i += 1
            continue
        
        # ---- SIGNAL DETECTION ----
        bar_high = highs[i]
        bar_low = lows[i]
        bar_close = closes[i]
        bar_open = opens[i]
        
        body = abs(bar_close - bar_open)
        upper_wick = bar_high - max(bar_close, bar_open)
        lower_wick = min(bar_close, bar_open) - bar_low
        
        signal = 0  # 1=long, -1=short
        entry = 0.0
        sl = 0.0
        tp = 0.0
        
        # ---- SHORT SETUP ----
        # Price swept Asian High + closed back below it + rejection
        if (bar_high > a_high + sweep_min and 
            bar_close < a_high and
            upper_wick > body * 0.5 and
            bar_close < bar_open):  # bearish candle
            
            signal = -1
            entry = bar_close - total_entry_cost
            sl = bar_high + sl_buffer
            sl_dist = sl - entry
            
            if sl_dist > 0:
                tp = entry - sl_dist * rr
                # Adjust TP: don't target below Asian Low
                # (let it ride to at least Asian Mid)
                tp = min(tp, a_low)
                tp = min(tp, entry - sl_dist * 1.5)  # minimum 1.5R
        
        # ---- LONG SETUP ----
        # Price swept Asian Low + closed back above it + rejection
        elif (bar_low < a_low - sweep_min and
              bar_close > a_low and
              lower_wick > body * 0.5 and
              bar_close > bar_open):  # bullish candle
            
            signal = 1
            entry = bar_close + total_entry_cost
            sl = bar_low - sl_buffer
            sl_dist = entry - sl
            
            if sl_dist > 0:
                tp = entry + sl_dist * rr
                tp = max(tp, a_high)
                tp = max(tp, entry + sl_dist * 1.5)
        
        if signal == 0:
            i += 1
            continue
        
        # ---- VALIDATE ----
        sl_pips = abs(entry - sl) / pip
        if sl_pips < 3 or sl_pips > 50:
            i += 1
            continue
        
        # ---- POSITION SIZE ----
        risk_amount = balance * risk_pct
        lot_size = risk_amount / (sl_pips * pip_val)
        lot_size = max(0.01, round(lot_size, 2))
        trade_commission = commission * lot_size
        
        # ---- SIMULATE FORWARD ----
        exit_price = None
        exit_time = None
        exit_type = None
        
        for j in range(i + 1, min(i + max_duration, n)):
            bh = highs[j]
            bl = lows[j]
            
            if signal == 1:  # LONG
                if bl <= sl:
                    exit_price = sl
                    exit_time = times[j]
                    exit_type = 'SL'
                    break
                if bh >= tp:
                    exit_price = tp
                    exit_time = times[j]
                    exit_type = 'TP'
                    break
            else:  # SHORT
                if bh >= sl:
                    exit_price = sl
                    exit_time = times[j]
                    exit_type = 'SL'
                    break
                if bl <= tp:
                    exit_price = tp
                    exit_time = times[j]
                    exit_type = 'TP'
                    break
        
        # Timeout - close at market
        if exit_price is None:
            j_last = min(i + max_duration, n) - 1
            exit_price = closes[j_last]
            exit_time = times[j_last]
            exit_type = 'TIMEOUT'
        
        # ---- CALCULATE PNL ----
        if signal == 1:
            pnl_pips = (exit_price - entry) / pip
        else:
            pnl_pips = (entry - exit_price) / pip
        
        pnl_money = (pnl_pips * pip_val * lot_size) - trade_commission
        r_multiple = pnl_money / risk_amount if risk_amount > 0 else 0
        
        # ---- UPDATE STATE ----
        balance += pnl_money
        peak_balance = max(peak_balance, balance)
        day_pnl += pnl_money
        day_trades += 1
        last_trade_bar = i
        
        trades.append({
            'symbol': symbol,
            'direction': 'LONG' if signal == 1 else 'SHORT',
            'entry_time': str(times[i]),
            'exit_time': str(exit_time),
            'entry_price': round(entry, 5),
            'exit_price': round(exit_price, 5),
            'sl': round(sl, 5),
            'tp': round(tp, 5),
            'sl_pips': round(sl_pips, 1),
            'lots': lot_size,
            'pnl': round(pnl_money, 2),
            'pnl_pips': round(pnl_pips, 1),
            'r': round(r_multiple, 2),
            'exit_type': exit_type,
            'asian_high': round(a_high, 5),
            'asian_low': round(a_low, 5),
            'asian_range_pips': round(a_range / pip, 1),
            'balance': round(balance, 2)
        })
        
        # Jump past cooldown
        i = last_trade_bar + cooldown
        continue
    
    # Final equity point
    if current_date is not None:
        equity_points.append({
            'date': current_date,
            'balance': balance,
            'day_pnl': day_pnl,
            'day_trades': day_trades
        })
    
    return trades, equity_points, blown


# ================================================================
# REPORTING
# ================================================================

def print_report(all_trades, initial_balance):
    """Print comprehensive results"""
    if not all_trades:
        print("\n  NO TRADES EXECUTED!")
        return None
    
    df = pd.DataFrame(all_trades)
    final = df['balance'].iloc[-1]
    total = len(df)
    
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    wr = len(wins) / total * 100
    net = df['pnl'].sum()
    ret = (final - initial_balance) / initial_balance * 100
    
    avg_w = wins['pnl'].mean() if len(wins) else 0
    avg_l = losses['pnl'].mean() if len(losses) else 0
    avg_r = df['r'].mean()
    
    gp = wins['pnl'].sum() if len(wins) else 0
    gl = abs(losses['pnl'].sum()) if len(losses) else 1
    pf = gp / gl if gl > 0 else 999
    
    # Max DD
    bals = [initial_balance] + list(df['balance'])
    pk = initial_balance
    mdd = mdd_p = 0
    for b in bals:
        if b > pk:
            pk = b
        dd = pk - b
        ddp = dd / pk * 100
        if ddp > mdd_p:
            mdd, mdd_p = dd, ddp
    
    # Consecutive
    mcw = mcl = cw = cl = 0
    for _, t in df.iterrows():
        if t['pnl'] > 0:
            cw += 1; cl = 0; mcw = max(mcw, cw)
        else:
            cl += 1; cw = 0; mcl = max(mcl, cl)
    
    # Exit types
    exits = df['exit_type'].value_counts().to_dict()
    
    # Direction split
    longs = df[df['direction'] == 'LONG']
    shorts = df[df['direction'] == 'SHORT']
    
    print(f"\n{'='*70}")
    print(f"{'BACKTEST RESULTS':^70}")
    print(f"{'Asian Range Liquidity Trap Strategy':^70}")
    print(f"{'='*70}")
    
    stats = [
        ["Period", f"{df['entry_time'].iloc[0][:10]} → {df['entry_time'].iloc[-1][:10]}"],
        ["Symbols", ', '.join(df['symbol'].unique())],
        ["", ""],
        ["Initial Balance", f"${initial_balance:,.0f}"],
        ["Final Balance", f"${final:,.2f}"],
        ["Net P&L", f"${net:,.2f}"],
        ["Return", f"{ret:+.2f}%"],
        ["", ""],
        ["Total Trades", total],
        ["Win Rate", f"{wr:.1f}%"],
        ["Wins", len(wins)],
        ["Losses", len(losses)],
        ["", ""],
        ["Avg Win", f"${avg_w:,.2f}"],
        ["Avg Loss", f"${avg_l:,.2f}"],
        ["Avg R", f"{avg_r:+.2f}R"],
        ["Profit Factor", f"{pf:.2f}"],
        ["Total Pips", f"{df['pnl_pips'].sum():+,.1f}"],
        ["Avg Pips/Trade", f"{df['pnl_pips'].mean():+.1f}"],
        ["", ""],
        ["Longs", f"{len(longs)} (WR: {len(longs[longs['pnl']>0])/max(1,len(longs))*100:.0f}%)"],
        ["Shorts", f"{len(shorts)} (WR: {len(shorts[shorts['pnl']>0])/max(1,len(shorts))*100:.0f}%)"],
        ["", ""],
        ["Max Drawdown", f"${mdd:,.2f} ({mdd_p:.2f}%)"],
        ["Max Consec Wins", mcw],
        ["Max Consec Losses", mcl],
        ["", ""],
    ]
    
    for k, v in exits.items():
        stats.append([f"Exit: {k}", v])
    
    stats.extend([
        ["", ""],
        ["═══ PROP FIRM CHECK ═══", ""],
        ["Max DD < 10%", f"{'✅ OK' if mdd_p < 10 else '❌ FAIL'} ({mdd_p:.1f}%)"],
        ["Max DD < 5% (daily)", "Check below"],
    ])
    
    print(tabulate(stats, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    
    # ---- MONTHLY BREAKDOWN ----
    df['entry_dt'] = pd.to_datetime(df['entry_time'])
    df['year'] = df['entry_dt'].dt.year
    df['month'] = df['entry_dt'].dt.month
    df['ym'] = df['entry_dt'].dt.to_period('M')
    
    monthly = df.groupby('ym').agg(
        n=('pnl', 'count'),
        w=('pnl', lambda x: (x > 0).sum()),
        pnl=('pnl', 'sum'),
        pips=('pnl_pips', 'sum')
    ).reset_index()
    monthly['wr'] = (monthly['w'] / monthly['n'] * 100).round(0)
    monthly['pnl_pct'] = (monthly['pnl'] / initial_balance * 100).round(2)
    
    print(f"\n{'MONTHLY BREAKDOWN':^70}")
    print("-" * 70)
    
    rows = []
    months_above_5 = 0
    months_positive = 0
    total_months = len(monthly)
    
    for _, m in monthly.iterrows():
        flag = "✅" if m['pnl_pct'] >= 5 else ("➖" if m['pnl_pct'] > 0 else "❌")
        rows.append([
            str(m['ym']), m['n'], f"{m['wr']:.0f}%",
            f"{m['pips']:+.0f}", f"${m['pnl']:+,.0f}",
            f"{m['pnl_pct']:+.1f}%", flag
        ])
        if m['pnl_pct'] >= 5:
            months_above_5 += 1
        if m['pnl_pct'] > 0:
            months_positive += 1
    
    print(tabulate(rows,
                   headers=["Month", "#", "WR", "Pips", "P&L", "%", "5%?"],
                   tablefmt="simple"))
    
    print(f"\n  Months ≥ 5%: {months_above_5}/{total_months} ({months_above_5/total_months*100:.0f}%)")
    print(f"  Months > 0%: {months_positive}/{total_months} ({months_positive/total_months*100:.0f}%)")
    print(f"  Avg Monthly: {monthly['pnl_pct'].mean():.2f}%")
    print(f"  Worst Month: {monthly['pnl_pct'].min():.2f}%")
    print(f"  Best Month:  {monthly['pnl_pct'].max():.2f}%")
    
    # ---- YEARLY ----
    yearly = df.groupby('year').agg(
        n=('pnl', 'count'),
        w=('pnl', lambda x: (x > 0).sum()),
        pnl=('pnl', 'sum'),
        pips=('pnl_pips', 'sum')
    ).reset_index()
    yearly['wr'] = (yearly['w'] / yearly['n'] * 100).round(0)
    yearly['pct'] = (yearly['pnl'] / initial_balance * 100).round(1)
    
    print(f"\n{'YEARLY BREAKDOWN':^70}")
    print("-" * 70)
    rows = []
    for _, y in yearly.iterrows():
        rows.append([
            int(y['year']), int(y['n']), f"{y['wr']:.0f}%",
            f"{y['pips']:+.0f}", f"${y['pnl']:+,.0f}", f"{y['pct']:+.1f}%"
        ])
    print(tabulate(rows,
                   headers=["Year", "#", "WR", "Pips", "P&L", "%"],
                   tablefmt="simple"))
    
    return df


def save_results(df, equity_points, initial_balance, output_dir):
    """Save all output files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Trades CSV
    fp = os.path.join(output_dir, "trades.csv")
    df.to_csv(fp, index=False)
    print(f"\n  💾 Trades: {fp}")
    
    # 2. Equity Curve
    bals = [initial_balance] + list(df['balance'])
    t0 = pd.to_datetime(df['entry_time'].iloc[0])
    trade_times = [t0] + list(pd.to_datetime(df['exit_time']))
    
    fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                              gridspec_kw={'height_ratios': [3, 1]})
    
    axes[0].plot(trade_times, bals, '#1976D2', lw=1.2, label='Balance')
    axes[0].axhline(initial_balance, color='gray', ls='--', alpha=.5, label='Start')
    axes[0].axhline(initial_balance * 1.08, color='green', ls='--', alpha=.4,
                     label='+8% Target')
    axes[0].axhline(initial_balance * 0.90, color='red', ls='--', alpha=.4,
                     label='-10% Max DD')
    axes[0].set_title('Equity Curve - Asian Range Liquidity Trap',
                       fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Balance ($)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=.25)
    
    peak_arr = np.maximum.accumulate(bals)
    dd = (np.array(peak_arr) - np.array(bals)) / np.array(peak_arr) * 100
    axes[1].fill_between(trade_times, dd, color='#E53935', alpha=.4)
    axes[1].set_ylabel('Drawdown %')
    axes[1].set_xlabel('Date')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=.25)
    
    plt.tight_layout()
    fp = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Equity: {fp}")
    
    # 3. Monthly Heatmap
    df['entry_dt'] = pd.to_datetime(df['entry_time'])
    df['yr'] = df['entry_dt'].dt.year
    df['mo'] = df['entry_dt'].dt.month
    
    piv = df.groupby(['yr', 'mo'])['pnl'].sum().reset_index()
    pt = piv.pivot(index='yr', columns='mo', values='pnl').fillna(0)
    
    mnames = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    pt.columns = [mnames.get(c, c) for c in pt.columns]
    
    fig, ax = plt.subplots(figsize=(16, max(4, len(pt) * 0.8)))
    sns.heatmap(pt, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                ax=ax, linewidths=.5, cbar_kws={'label': 'P&L ($)'})
    ax.set_title('Monthly P&L Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fp = os.path.join(output_dir, "monthly_heatmap.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🗓️  Heatmap: {fp}")
    
    # 4. R Distribution
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['#43A047' if r > 0 else '#E53935' for r in df['r']]
    ax.bar(range(len(df)), df['r'], color=colors, alpha=.7, width=1)
    ax.axhline(0, color='black', lw=.5)
    avg_r = df['r'].mean()
    ax.axhline(avg_r, color='blue', ls='--', alpha=.7,
               label=f'Avg R: {avg_r:+.2f}')
    ax.set_title('R-Multiple per Trade', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('R')
    ax.legend()
    ax.grid(True, alpha=.25)
    plt.tight_layout()
    fp = os.path.join(output_dir, "r_distribution.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 R-Dist: {fp}")
    
    # 5. Win Rate by Hour
    df['entry_hour'] = df['entry_dt'].dt.hour
    hourly = df.groupby('entry_hour').agg(
        n=('pnl', 'count'),
        wr=('pnl', lambda x: (x > 0).mean() * 100),
        avg_pnl=('pnl', 'mean')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(hourly['entry_hour'], hourly['wr'],
                  color=['#43A047' if w > 50 else '#E53935' for w in hourly['wr']],
                  alpha=.7)
    ax.axhline(50, color='gray', ls='--', alpha=.5)
    ax.set_title('Win Rate by Entry Hour (UTC)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Win Rate %')
    ax.set_xticks(hourly['entry_hour'])
    ax.grid(True, alpha=.25)
    
    for bar, n in zip(bars, hourly['n']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'n={int(n)}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fp = os.path.join(output_dir, "winrate_by_hour.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🕐 Hourly WR: {fp}")
    
    # 6. Monthly % bar chart
    df['ym'] = df['entry_dt'].dt.to_period('M')
    m_pnl = df.groupby('ym')['pnl'].sum()
    m_pct = m_pnl / initial_balance * 100
    
    fig, ax = plt.subplots(figsize=(20, 6))
    colors = ['#43A047' if p >= 5 else ('#FFC107' if p > 0 else '#E53935')
              for p in m_pct]
    ax.bar(range(len(m_pct)), m_pct.values, color=colors, alpha=.8)
    ax.axhline(5, color='green', ls='--', alpha=.5, label='5% target')
    ax.axhline(0, color='black', lw=.5)
    ax.set_title('Monthly Return %', fontsize=14, fontweight='bold')
    ax.set_ylabel('Return %')
    ax.set_xlabel('Month')
    
    # X labels (every 6 months)
    labels = [str(p) for p in m_pct.index]
    step = max(1, len(labels) // 20)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                        rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=.25)
    
    plt.tight_layout()
    fp = os.path.join(output_dir, "monthly_returns.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Monthly %: {fp}")


# ================================================================
# MAIN
# ================================================================

def main():
    start_time = timer.time()
    
    print("=" * 70)
    print("  ASIAN RANGE LIQUIDITY TRAP - PROP FIRM BACKTEST")
    print("  2010-2025 | EURUSD + GBPUSD")
    print("=" * 70)
    
    cfg = CONFIG
    initial = cfg['initial_balance']
    
    print(f"\n  💰 Balance: ${initial:,}")
    print(f"  ⚡ Risk: {cfg['risk_per_trade']*100}% per trade")
    print(f"  🎯 R:R = 1:{cfg['rr_ratio']}")
    print(f"  📏 Asian Range: {cfg['min_asian_range_pips']}-{cfg['max_asian_range_pips']} pips")
    print(f"  🕐 Trade: {cfg['trade_start_hour']}:00-{cfg['trade_end_hour']}:00 UTC")
    
    all_trades = []
    all_equity = []
    blown = False
    running_balance = initial
    
    for symbol in cfg['symbols']:
        if blown:
            print(f"\n  ⚠️ Account blown, skipping {symbol}")
            break
        
        print(f"\n{'━' * 70}")
        print(f"  {symbol}")
        print(f"{'━' * 70}")
        
        if symbol not in PIP_INFO:
            print(f"  Skip: no pip info")
            continue
        
        # Load M1 data
        t0 = timer.time()
        df_m1 = load_m1_data(cfg['data_dir'], symbol, cfg['years'])
        if df_m1.empty:
            print("  No data!")
            continue
        print(f"  Load time: {timer.time()-t0:.1f}s")
        
        # Resample to M15
        t0 = timer.time()
        df_m15 = resample_m15(df_m1)
        print(f"  M15 bars: {len(df_m15):,} ({timer.time()-t0:.1f}s)")
        
        # Compute Asian ranges
        t0 = timer.time()
        asian_ranges = compute_asian_ranges(
            df_m1, cfg['asian_start_hour'], cfg['asian_end_hour']
        )
        print(f"  Asian ranges: {len(asian_ranges)} days ({timer.time()-t0:.1f}s)")
        
        # Update config with running balance for multi-symbol
        cfg_run = dict(cfg)
        cfg_run['initial_balance'] = running_balance
        
        # Run backtest
        t0 = timer.time()
        trades, equity, blown = run_backtest(df_m15, asian_ranges, symbol, cfg_run)
        print(f"  Backtest: {len(trades)} trades ({timer.time()-t0:.1f}s)")
        
        if trades:
            all_trades.extend(trades)
            running_balance = trades[-1]['balance']
        
        all_equity.extend(equity)
        
        if blown:
            print(f"  ⚠️ ACCOUNT BLOWN!")
    
    # Report
    if all_trades:
        df = print_report(all_trades, initial)
        if df is not None:
            save_results(df, all_equity, initial, cfg['output_dir'])
    else:
        print("\n  ❌ NO TRADES!")
    
    elapsed = timer.time() - start_time
    print(f"\n  ⏱️ Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
