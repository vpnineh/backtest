"""
=============================================================================
  FAST VECTORIZED BACKTEST ENGINE
  Liquidity Sweep + SMC Strategy
  
  بهینه‌سازی‌ها:
  - تمام محاسبات با NumPy vectorized
  - Pre-compute همه سطوح، سویینگ‌ها، FVG و سیگنال‌ها
  - لوپ فقط روی سیگنال‌ها (نه همه کندل‌ها)
  - اجرا: چند ثانیه به جای چند ساعت
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import time as time_module
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')


# ============================
# DATA LOADER (Fast)
# ============================

def load_histdata(filepath: str) -> pd.DataFrame:
    """بارگذاری سریع CSV"""
    try:
        df = pd.read_csv(filepath, sep=';', header=None,
                         names=['DateTime','Open','High','Low','Close','Volume'])
        df['DateTime'] = df['DateTime'].str.strip()
        
        # Try common formats
        for fmt in ['%Y%m%d %H%M%S', '%Y.%m.%d %H:%M', '%m/%d/%Y %H:%M']:
            try:
                df['DateTime'] = pd.to_datetime(df['DateTime'], format=fmt)
                break
            except:
                continue
        else:
            df['DateTime'] = pd.to_datetime(df['DateTime'], 
                                             format='mixed', dayfirst=False)
        
        df.set_index('DateTime', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df[df.index.dayofweek < 5]  # Remove weekends
        return df[['Open','High','Low','Close','Volume']].dropna()
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")
        return pd.DataFrame()


def load_all_data(data_dir: str, symbol: str, years: list) -> pd.DataFrame:
    """بارگذاری همه سال‌ها"""
    frames = []
    for year in years:
        fp = os.path.join(data_dir, f"DAT_ASCII_{symbol}_M1_{year}.csv")
        if os.path.exists(fp):
            print(f"  Loading {year}...", end=" ")
            df = load_histdata(fp)
            if not df.empty:
                print(f"{len(df):,} rows")
                frames.append(df)
            else:
                print("empty")
        else:
            print(f"  {year} not found, skipping")
    
    if frames:
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        print(f"  Total: {len(combined):,} M1 candles")
        return combined
    return pd.DataFrame()


def resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample سریع"""
    rule = f'{minutes}min'
    return df.resample(rule).agg({
        'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
    }).dropna()


# ============================
# VECTORIZED INDICATORS
# ============================

def find_swing_highs_vec(highs: np.ndarray, lookback: int = 10) -> np.ndarray:
    """شناسایی Vectorized سویینگ‌ها - برمی‌گرداند boolean array"""
    n = len(highs)
    is_swing = np.zeros(n, dtype=bool)
    half = lookback // 2
    
    for i in range(half, n - half):
        window = highs[i-half:i+half+1]
        if highs[i] == window.max() and np.sum(window == highs[i]) == 1:
            is_swing[i] = True
    return is_swing


def find_swing_lows_vec(lows: np.ndarray, lookback: int = 10) -> np.ndarray:
    n = len(lows)
    is_swing = np.zeros(n, dtype=bool)
    half = lookback // 2
    
    for i in range(half, n - half):
        window = lows[i-half:i+half+1]
        if lows[i] == window.min() and np.sum(window == lows[i]) == 1:
            is_swing[i] = True
    return is_swing


def compute_daily_levels(df_m1: pd.DataFrame) -> pd.DataFrame:
    """محاسبه PDH/PDL برای هر روز - Vectorized"""
    daily = df_m1.resample('1D').agg({'High':'max','Low':'min'}).dropna()
    daily.columns = ['DayHigh','DayLow']
    
    # Shift by 1 day = Previous Day High/Low
    daily['PDH'] = daily['DayHigh'].shift(1)
    daily['PDL'] = daily['DayLow'].shift(1)
    daily.dropna(inplace=True)
    return daily[['PDH','PDL']]


def detect_fvg_vectorized(open_arr, high_arr, low_arr, close_arr, 
                           min_gap: float) -> Tuple[np.ndarray, np.ndarray,
                                                      np.ndarray, np.ndarray]:
    """
    Vectorized FVG detection
    Returns: bullish_fvg_mask, bearish_fvg_mask, fvg_top, fvg_bottom
    """
    n = len(open_arr)
    bull_fvg = np.zeros(n, dtype=bool)
    bear_fvg = np.zeros(n, dtype=bool)
    fvg_top = np.zeros(n)
    fvg_bot = np.zeros(n)
    
    # Bullish FVG: candle[i] low > candle[i-2] high
    if n > 2:
        gap_bull = low_arr[2:] - high_arr[:-2]
        mask_bull = gap_bull >= min_gap
        bull_fvg[2:] = mask_bull
        fvg_top[2:] = np.where(mask_bull, low_arr[2:], 0)
        fvg_bot[2:] = np.where(mask_bull, high_arr[:-2], 0)
        
        # Bearish FVG: candle[i-2] low > candle[i] high
        gap_bear = low_arr[:-2] - high_arr[2:]
        mask_bear = gap_bear >= min_gap
        bear_fvg[2:] = mask_bear
        # For bearish: top = candle[i-2] low, bottom = candle[i] high
        fvg_top[2:] = np.where(mask_bear, low_arr[:-2], fvg_top[2:])
        fvg_bot[2:] = np.where(mask_bear, high_arr[2:], fvg_bot[2:])
    
    return bull_fvg, bear_fvg, fvg_top, fvg_bot


# ============================
# SIGNAL GENERATOR (Vectorized pre-compute)
# ============================

def generate_signals(df_htf: pd.DataFrame, df_mtf: pd.DataFrame,
                     df_m1: pd.DataFrame, daily_levels: pd.DataFrame,
                     config: dict, pip_size: float) -> pd.DataFrame:
    """
    تولید سیگنال‌ها بصورت Vectorized
    خروجی: DataFrame با ستون‌های signal, direction, entry, sl, tp
    """
    cfg = config['strategy']
    rr = config['risk']['reward_to_risk']
    sl_buffer = cfg['sl_buffer_pips'] * pip_size
    sweep_thresh = cfg['sweep_threshold_pips'] * pip_size
    
    print("  [1/5] Computing swing points...")
    t0 = time_module.time()
    
    # Swing points on HTF
    htf_swing_highs = find_swing_highs_vec(df_htf['High'].values, 
                                            cfg['swing_lookback'])
    htf_swing_lows = find_swing_lows_vec(df_htf['Low'].values, 
                                          cfg['swing_lookback'])
    
    # Get swing prices
    swing_high_prices = df_htf['High'].values.copy()
    swing_high_prices[~htf_swing_highs] = np.nan
    
    swing_low_prices = df_htf['Low'].values.copy()
    swing_low_prices[~htf_swing_lows] = np.nan
    
    print(f"      Swing Highs: {htf_swing_highs.sum()}, "
          f"Swing Lows: {htf_swing_lows.sum()} "
          f"({time_module.time()-t0:.1f}s)")
    
    print("  [2/5] Computing FVGs on MTF...")
    t0 = time_module.time()
    
    bull_fvg, bear_fvg, fvg_top, fvg_bot = detect_fvg_vectorized(
        df_mtf['Open'].values, df_mtf['High'].values,
        df_mtf['Low'].values, df_mtf['Close'].values,
        cfg['fvg_min_size_pips'] * pip_size
    )
    print(f"      Bullish FVGs: {bull_fvg.sum()}, "
          f"Bearish FVGs: {bear_fvg.sum()} ({time_module.time()-t0:.1f}s)")
    
    print("  [3/5] Mapping daily levels to MTF...")
    t0 = time_module.time()
    
    # Map PDH/PDL to MTF bars
    mtf_dates = df_mtf.index.date
    mtf_pdh = np.full(len(df_mtf), np.nan)
    mtf_pdl = np.full(len(df_mtf), np.nan)
    
    daily_dict = {}
    for dt, row in daily_levels.iterrows():
        daily_dict[dt.date()] = (row['PDH'], row['PDL'])
    
    for i, d in enumerate(mtf_dates):
        if d in daily_dict:
            mtf_pdh[i], mtf_pdl[i] = daily_dict[d]
    
    # Forward fill
    mask = np.isnan(mtf_pdh)
    idx = np.where(~mask, np.arange(len(mtf_pdh)), 0)
    np.maximum.accumulate(idx, out=idx)
    mtf_pdh = mtf_pdh[idx]
    mtf_pdl = mtf_pdl[idx]
    
    print(f"      Done ({time_module.time()-t0:.1f}s)")
    
    print("  [4/5] Detecting sweeps & generating signals...")
    t0 = time_module.time()
    
    highs = df_mtf['High'].values
    lows = df_mtf['Low'].values
    opens = df_mtf['Open'].values
    closes = df_mtf['Close'].values
    times = df_mtf.index
    hours = df_mtf.index.hour
    
    # Session filter
    london_start = cfg['london_start']
    london_end = cfg['london_end']
    ny_start = cfg['newyork_start']
    ny_end = cfg['newyork_end']
    
    in_session = ((hours >= london_start) & (hours < london_end)) | \
                 ((hours >= ny_start) & (hours < ny_end))
    
    # ========== SWEEP DETECTION (Vectorized) ==========
    
    # Sweep PDH (bearish): High > PDH + threshold AND Close < PDH
    sweep_pdh = (highs > mtf_pdh + sweep_thresh) & (closes < mtf_pdh)
    
    # Sweep PDL (bullish): Low < PDL - threshold AND Close > PDL
    sweep_pdl = (lows < mtf_pdl - sweep_thresh) & (closes > mtf_pdl)
    
    # Rolling swing high/low for recent levels
    # Use rolling max/min of confirmed swing points
    window = 50
    
    # Reindex HTF swings to MTF timeframe
    htf_sh_series = pd.Series(swing_high_prices, index=df_htf.index).dropna()
    htf_sl_series = pd.Series(swing_low_prices, index=df_htf.index).dropna()
    
    # Map nearest HTF swing to MTF
    recent_swing_high = np.full(len(df_mtf), np.nan)
    recent_swing_low = np.full(len(df_mtf), np.nan)
    
    sh_idx = 0
    sl_idx = 0
    sh_vals = htf_sh_series.values
    sh_times = htf_sh_series.index
    sl_vals = htf_sl_series.values
    sl_times = htf_sl_series.index
    
    for i in range(len(df_mtf)):
        ct = times[i]
        # Update recent swing high
        while sh_idx < len(sh_times) and sh_times[sh_idx] <= ct:
            sh_idx += 1
        if sh_idx > 0:
            # Use last 3 swing highs, take max
            start = max(0, sh_idx - 3)
            recent_swing_high[i] = np.max(sh_vals[start:sh_idx])
        
        while sl_idx < len(sl_times) and sl_times[sl_idx] <= ct:
            sl_idx += 1
        if sl_idx > 0:
            start = max(0, sl_idx - 3)
            recent_swing_low[i] = np.min(sl_vals[start:sl_idx])
    
    # Also sweep swing highs/lows
    valid_sh = ~np.isnan(recent_swing_high)
    valid_sl = ~np.isnan(recent_swing_low)
    
    sweep_sh = valid_sh & (highs > recent_swing_high + sweep_thresh) & \
               (closes < recent_swing_high)
    sweep_sl = valid_sl & (lows < recent_swing_low - sweep_thresh) & \
               (closes > recent_swing_low)
    
    # Combined sweep signals
    any_sweep_high = sweep_pdh | sweep_sh  # Bearish setup
    any_sweep_low = sweep_pdl | sweep_sl   # Bullish setup
    
    # ========== MARKET STRUCTURE (Simple version) ==========
    # Bullish bias: close > rolling 50-period high midpoint
    # Bearish bias: close < rolling 50-period low midpoint
    
    rolling_mid = (pd.Series(highs).rolling(window).max().values + 
                   pd.Series(lows).rolling(window).min().values) / 2
    
    bullish_bias = closes > rolling_mid
    bearish_bias = closes < rolling_mid
    
    # ========== COMBINE INTO SIGNALS ==========
    
    # SHORT signal: sweep high + bearish bias + in session
    short_signal = any_sweep_high & bearish_bias & in_session
    
    # LONG signal: sweep low + bullish bias + in session
    long_signal = any_sweep_low & bullish_bias & in_session
    
    # Remove conflicting signals (both at same bar)
    conflict = short_signal & long_signal
    short_signal = short_signal & ~conflict
    long_signal = long_signal & ~conflict
    
    # ========== COMPUTE ENTRY/SL/TP ==========
    
    n = len(df_mtf)
    signal = np.zeros(n, dtype=int)  # 1=long, -1=short
    entry_prices = np.zeros(n)
    sl_prices = np.zeros(n)
    tp_prices = np.zeros(n)
    
    signal[long_signal] = 1
    signal[short_signal] = -1
    
    # Long trades
    long_mask = signal == 1
    entry_prices[long_mask] = closes[long_mask]
    
    # SL = recent swing low - buffer (or PDL - buffer)
    sl_long = np.where(valid_sl[long_mask], 
                       recent_swing_low[long_mask], 
                       mtf_pdl[long_mask])
    sl_long = np.minimum(sl_long, lows[long_mask]) - sl_buffer
    sl_prices[long_mask] = sl_long
    
    sl_dist_long = entry_prices[long_mask] - sl_prices[long_mask]
    tp_prices[long_mask] = entry_prices[long_mask] + sl_dist_long * rr
    
    # Short trades
    short_mask = signal == -1
    entry_prices[short_mask] = closes[short_mask]
    
    sl_short = np.where(valid_sh[short_mask],
                        recent_swing_high[short_mask],
                        mtf_pdh[short_mask])
    sl_short = np.maximum(sl_short, highs[short_mask]) + sl_buffer
    sl_prices[short_mask] = sl_short
    
    sl_dist_short = sl_prices[short_mask] - entry_prices[short_mask]
    tp_prices[short_mask] = entry_prices[short_mask] - sl_dist_short * rr
    
    # ========== FILTER BAD SIGNALS ==========
    
    # Remove signals with SL too small or too large
    sl_dist = np.abs(entry_prices - sl_prices)
    sl_pips = sl_dist / pip_size
    
    valid_sl_size = (sl_pips >= 3) & (sl_pips <= 80)
    signal = np.where(valid_sl_size, signal, 0)
    
    # ========== MINIMUM SPACING (cooldown) ==========
    
    min_spacing = 4  # bars
    signal_indices = np.where(signal != 0)[0]
    
    if len(signal_indices) > 1:
        filtered = [signal_indices[0]]
        for idx in signal_indices[1:]:
            if idx - filtered[-1] >= min_spacing:
                filtered.append(idx)
        
        clean_signal = np.zeros(n, dtype=int)
        for idx in filtered:
            clean_signal[idx] = signal[idx]
        signal = clean_signal
    
    total_signals = np.sum(signal != 0)
    longs = np.sum(signal == 1)
    shorts = np.sum(signal == -1)
    
    print(f"      Signals: {total_signals} "
          f"(Long: {longs}, Short: {shorts}) "
          f"({time_module.time()-t0:.1f}s)")
    
    # Build result DataFrame
    signals_df = pd.DataFrame({
        'signal': signal,
        'entry': entry_prices,
        'sl': sl_prices,
        'tp': tp_prices
    }, index=df_mtf.index)
    
    return signals_df


# ============================
# FAST TRADE SIMULATOR
# ============================

def simulate_trades(df_mtf: pd.DataFrame, signals_df: pd.DataFrame,
                    config: dict, pip_size: float, 
                    symbol: str) -> Dict:
    """
    شبیه‌سازی سریع معاملات
    فقط روی سیگنال‌ها لوپ می‌زنیم + بررسی خروج در بارهای بعدی
    """
    print("  [5/5] Simulating trades...")
    t0 = time_module.time()
    
    initial_balance = config['account']['initial_balance']
    balance = initial_balance
    peak_balance = initial_balance
    risk_pct = config['risk']['risk_per_trade']
    commission_per_lot = config['execution']['commission_per_lot']
    spread = config['execution']['spread_pips'] * pip_size
    slippage = config['execution']['slippage_pips'] * pip_size
    rr = config['risk']['reward_to_risk']
    max_daily_dd = config['risk']['max_daily_loss']
    prop_max_dd = config['prop_rules']['max_total_drawdown']
    breakeven_1r = config['strategy']['breakeven_at_1r']
    pip_value = 10.0  # per standard lot per pip
    
    # Get signal bars
    sig_mask = signals_df['signal'].values != 0
    sig_indices = np.where(sig_mask)[0]
    
    if len(sig_indices) == 0:
        print("      No signals to simulate")
        return {'trades': [], 'equity_curve': [], 'balance': balance}
    
    highs = df_mtf['High'].values
    lows = df_mtf['Low'].values
    closes = df_mtf['Close'].values
    times = df_mtf.index
    n_bars = len(df_mtf)
    
    trades = []
    equity_curve = []
    daily_pnl = {}
    account_blown = False
    
    # For each signal, simulate forward
    for sig_idx in sig_indices:
        if account_blown:
            break
        
        direction = signals_df['signal'].values[sig_idx]  # 1 or -1
        entry = signals_df['entry'].values[sig_idx]
        sl = signals_df['sl'].values[sig_idx]
        tp = signals_df['tp'].values[sig_idx]
        entry_time = times[sig_idx]
        
        # Apply spread/slippage
        if direction == 1:
            entry += spread/2 + slippage
        else:
            entry -= spread/2 + slippage
        
        # Position sizing
        sl_pips = abs(entry - sl) / pip_size
        if sl_pips <= 0:
            continue
        
        risk_amount = balance * risk_pct
        lot_size = risk_amount / (sl_pips * pip_value)
        lot_size = max(0.01, round(lot_size, 2))
        commission = commission_per_lot * lot_size
        
        # Check daily limit
        day_key = str(entry_time.date())
        if day_key in daily_pnl:
            if daily_pnl[day_key] <= -balance * max_daily_dd:
                continue
        
        # Simulate forward - find exit
        current_sl = sl
        hit_breakeven = False
        exit_price = None
        exit_time = None
        exit_type = None
        
        max_forward = min(sig_idx + 500, n_bars)  # Max 500 bars forward
        
        for j in range(sig_idx + 1, max_forward):
            bar_high = highs[j]
            bar_low = lows[j]
            
            if direction == 1:  # LONG
                # Breakeven check
                if breakeven_1r and not hit_breakeven:
                    one_r = abs(entry - sl)
                    if bar_high >= entry + one_r:
                        current_sl = entry + 2 * pip_size
                        hit_breakeven = True
                
                # SL hit
                if bar_low <= current_sl:
                    exit_price = current_sl
                    exit_time = times[j]
                    if hit_breakeven and abs(current_sl - entry) < 5 * pip_size:
                        exit_type = 'BE'
                    else:
                        exit_type = 'SL'
                    break
                
                # TP hit
                if bar_high >= tp:
                    exit_price = tp
                    exit_time = times[j]
                    exit_type = 'TP'
                    break
            
            else:  # SHORT
                if breakeven_1r and not hit_breakeven:
                    one_r = abs(sl - entry)
                    if bar_low <= entry - one_r:
                        current_sl = entry - 2 * pip_size
                        hit_breakeven = True
                
                if bar_high >= current_sl:
                    exit_price = current_sl
                    exit_time = times[j]
                    if hit_breakeven and abs(current_sl - entry) < 5 * pip_size:
                        exit_type = 'BE'
                    else:
                        exit_type = 'SL'
                    break
                
                if bar_low <= tp:
                    exit_price = tp
                    exit_time = times[j]
                    exit_type = 'TP'
                    break
        
        # If no exit found, close at last bar
        if exit_price is None:
            exit_price = closes[max_forward - 1]
            exit_time = times[max_forward - 1]
            exit_type = 'TIMEOUT'
        
        # Calculate PnL
        if direction == 1:
            pnl_pips = (exit_price - entry) / pip_size
        else:
            pnl_pips = (entry - exit_price) / pip_size
        
        pnl = (pnl_pips * pip_value * lot_size) - commission
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0
        
        balance += pnl
        peak_balance = max(peak_balance, balance)
        
        # Track daily PnL
        if day_key not in daily_pnl:
            daily_pnl[day_key] = 0
        daily_pnl[day_key] += pnl
        
        # Check prop rules
        total_dd_pct = (peak_balance - balance) / initial_balance
        if total_dd_pct >= prop_max_dd:
            account_blown = True
        
        trades.append({
            'symbol': symbol,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': round(entry, 5),
            'exit_price': round(exit_price, 5),
            'sl': round(sl, 5),
            'tp': round(tp, 5),
            'lot_size': lot_size,
            'pnl': round(pnl, 2),
            'pnl_pips': round(pnl_pips, 1),
            'r_multiple': round(r_multiple, 2),
            'exit_type': exit_type,
            'balance': round(balance, 2)
        })
        
        equity_curve.append({
            'time': exit_time,
            'balance': balance
        })
    
    print(f"      Trades executed: {len(trades)} ({time_module.time()-t0:.1f}s)")
    if account_blown:
        print(f"      ⚠️  Account blown!")
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'balance': balance,
        'peak_balance': peak_balance,
        'blown': account_blown
    }


# ============================
# RESULTS & REPORTING
# ============================

def print_results(results: Dict, config: dict):
    """چاپ نتایج کامل"""
    trades = results['trades']
    if not trades:
        print("\n  ❌ No trades executed!")
        return
    
    initial = config['account']['initial_balance']
    final = results['balance']
    
    df_trades = pd.DataFrame(trades)
    
    total = len(df_trades)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] < 0]
    
    win_rate = len(wins) / total * 100
    total_pnl = df_trades['pnl'].sum()
    total_pips = df_trades['pnl_pips'].sum()
    
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
    avg_win_pips = wins['pnl_pips'].mean() if len(wins) > 0 else 0
    avg_loss_pips = losses['pnl_pips'].mean() if len(losses) > 0 else 0
    avg_r = df_trades['r_multiple'].mean()
    
    gross_profit = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return_pct = (final - initial) / initial * 100
    
    # Max drawdown from equity curve
    balances = [initial] + [t['balance'] for t in trades]
    peak = initial
    max_dd = 0
    max_dd_pct = 0
    for b in balances:
        if b > peak:
            peak = b
        dd = peak - b
        dd_pct = dd / peak * 100
        if dd_pct > max_dd_pct:
            max_dd = dd
            max_dd_pct = dd_pct
    
    # Consecutive
    max_cw = max_cl = cw = cl = 0
    for _, t in df_trades.iterrows():
        if t['pnl'] > 0:
            cw += 1; cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0
            max_cl = max(max_cl, cl)
    
    # Exit type breakdown
    exit_counts = df_trades['exit_type'].value_counts()
    
    print(f"\n{'='*70}")
    print(f"{'BACKTEST RESULTS':^70}")
    print(f"{'='*70}")
    
    stats = [
        ["Initial Balance", f"${initial:,.2f}"],
        ["Final Balance", f"${final:,.2f}"],
        ["Net P&L", f"${total_pnl:,.2f}"],
        ["Return", f"{return_pct:+.2f}%"],
        ["", ""],
        ["Total Trades", total],
        ["Win Rate", f"{win_rate:.1f}%"],
        ["Wins / Losses", f"{len(wins)} / {len(losses)}"],
        ["", ""],
        ["Avg Win", f"${avg_win:,.2f} ({avg_win_pips:.1f} pips)"],
        ["Avg Loss", f"${avg_loss:,.2f} ({avg_loss_pips:.1f} pips)"],
        ["Avg R-Multiple", f"{avg_r:.2f}R"],
        ["Profit Factor", f"{profit_factor:.2f}"],
        ["Total Pips", f"{total_pips:,.1f}"],
        ["", ""],
        ["Max Drawdown", f"${max_dd:,.2f} ({max_dd_pct:.2f}%)"],
        ["Max Consec Wins", max_cw],
        ["Max Consec Losses", max_cl],
        ["", ""],
    ]
    
    for exit_type, count in exit_counts.items():
        stats.append([f"Exit: {exit_type}", count])
    
    stats.extend([
        ["", ""],
        ["═══ PROP FIRM CHECK ═══", ""],
        ["Phase 1 Target (8%)", 
         "✅ PASSED" if return_pct >= 8 else f"❌ FAILED ({return_pct:.1f}%)"],
        ["Max DD < 10%", 
         "✅ OK" if max_dd_pct < 10 else f"❌ VIOLATED ({max_dd_pct:.1f}%)"],
        ["Account Status", 
         "❌ BLOWN" if results.get('blown') else "✅ ALIVE"],
    ])
    
    print(tabulate(stats, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    
    # Monthly breakdown
    print(f"\n{'MONTHLY BREAKDOWN':^70}")
    print("-" * 70)
    
    df_trades['month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
    monthly = df_trades.groupby('month').agg(
        trades=('pnl', 'count'),
        wins=('pnl', lambda x: (x > 0).sum()),
        pnl=('pnl', 'sum'),
        pips=('pnl_pips', 'sum')
    ).reset_index()
    
    monthly['wr'] = (monthly['wins'] / monthly['trades'] * 100).round(0)
    
    rows = []
    for _, m in monthly.iterrows():
        rows.append([
            str(m['month']), m['trades'], f"{m['wr']:.0f}%",
            f"{m['pips']:+.0f}", f"${m['pnl']:+,.0f}"
        ])
    
    print(tabulate(rows, 
                   headers=["Month", "Trades", "WR%", "Pips", "P&L"],
                   tablefmt="simple"))
    
    return df_trades


def save_results(df_trades: pd.DataFrame, results: Dict, 
                 config: dict):
    """ذخیره نتایج"""
    output_dir = config['output']['output_directory']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trades CSV
    csv_path = os.path.join(output_dir, "trades.csv")
    df_trades.to_csv(csv_path, index=False)
    print(f"\n  💾 Trades saved: {csv_path}")
    
    # Equity Curve
    initial = config['account']['initial_balance']
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [3, 1]})
    
    balances = [initial] + list(df_trades['balance'])
    times = [df_trades['entry_time'].iloc[0]] + list(df_trades['exit_time'])
    times = pd.to_datetime(times)
    
    axes[0].plot(times, balances, color='#2196F3', linewidth=1.5)
    axes[0].axhline(y=initial, color='gray', ls='--', alpha=0.5, 
                     label='Initial')
    axes[0].axhline(y=initial * 1.08, color='green', ls='--', alpha=0.5,
                     label='Target 8%')
    axes[0].axhline(y=initial * 0.90, color='red', ls='--', alpha=0.5,
                     label='Max DD 10%')
    axes[0].fill_between(times, initial * 0.90, min(balances) * 0.99,
                          alpha=0.1, color='red')
    axes[0].set_title('Equity Curve - Liquidity Sweep + SMC', 
                       fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Balance ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    peak = np.maximum.accumulate(balances)
    dd = (np.array(peak) - np.array(balances)) / np.array(peak) * 100
    axes[1].fill_between(times, dd, color='red', alpha=0.4)
    axes[1].set_ylabel('Drawdown %')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    eq_path = os.path.join(output_dir, "equity_curve.png")
    plt.savefig(eq_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Equity curve: {eq_path}")
    
    # Monthly Heatmap
    df_trades['exit_dt'] = pd.to_datetime(df_trades['exit_time'])
    df_trades['year'] = df_trades['exit_dt'].dt.year
    df_trades['month_num'] = df_trades['exit_dt'].dt.month
    
    pivot = df_trades.groupby(['year', 'month_num'])['pnl'].sum().reset_index()
    pivot_table = pivot.pivot(index='year', columns='month_num', values='pnl')
    pivot_table.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                            'Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot_table.columns)]
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(pivot_table) * 1.2)))
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='RdYlGn',
                center=0, ax=ax, linewidths=1,
                cbar_kws={'label': 'P&L ($)'})
    ax.set_title('Monthly P&L Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    hm_path = os.path.join(output_dir, "monthly_heatmap.png")
    plt.savefig(hm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🗓️  Monthly heatmap: {hm_path}")
    
    # R-Distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#4CAF50' if r > 0 else '#F44336' for r in df_trades['r_multiple']]
    ax.bar(range(len(df_trades)), df_trades['r_multiple'], color=colors, 
           alpha=0.7, width=1.0)
    ax.axhline(y=0, color='black', lw=0.5)
    ax.axhline(y=df_trades['r_multiple'].mean(), color='blue', ls='--',
               label=f"Avg: {df_trades['r_multiple'].mean():.2f}R")
    ax.set_title('R-Multiple per Trade', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('R')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    r_path = os.path.join(output_dir, "r_distribution.png")
    plt.savefig(r_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 R-distribution: {r_path}")


# ============================
# MAIN
# ============================

def main():
    total_start = time_module.time()
    
    print("=" * 70)
    print("  ⚡ FAST VECTORIZED BACKTEST ENGINE")
    print("  Liquidity Sweep + Smart Money Concepts")
    print("=" * 70)
    
    # Load config
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    initial = config['account']['initial_balance']
    print(f"\n  Balance: ${initial:,}")
    print(f"  Risk/Trade: {config['risk']['risk_per_trade']*100}%")
    print(f"  R:R: 1:{config['risk']['reward_to_risk']}")
    
    all_trades = []
    final_balance = initial
    
    for symbol in config['data']['symbols']:
        print(f"\n{'━'*70}")
        print(f"  📊 {symbol}")
        print(f"{'━'*70}")
        
        pip_size = {'EURUSD': 0.0001, 'GBPUSD': 0.0001, 
                    'XAUUSD': 0.10}.get(symbol, 0.0001)
        
        # Load data
        df_m1 = load_all_data(config['data']['directory'], symbol, 
                               config['data']['years'])
        if df_m1.empty:
            continue
        
        # Build timeframes
        print(f"\n  Building timeframes...")
        t0 = time_module.time()
        
        cfg_s = config['strategy']
        df_htf = resample(df_m1, cfg_s['htf_minutes'])
        df_mtf = resample(df_m1, cfg_s['mtf_minutes'])
        
        print(f"    HTF ({cfg_s['htf_minutes']}min): {len(df_htf):,}")
        print(f"    MTF ({cfg_s['mtf_minutes']}min): {len(df_mtf):,}")
        print(f"    ({time_module.time()-t0:.1f}s)")
        
        # Daily levels
        print(f"\n  Computing daily levels...")
        daily_levels = compute_daily_levels(df_m1)
        print(f"    {len(daily_levels)} days")
        
        # Generate signals
        print(f"\n  Generating signals...")
        signals = generate_signals(df_htf, df_mtf, df_m1, daily_levels,
                                    config, pip_size)
        
        # Simulate trades
        results = simulate_trades(df_mtf, signals, config, pip_size, symbol)
        
        if results['trades']:
            all_trades.extend(results['trades'])
            final_balance = results['balance']
    
    # Final report
    if all_trades:
        results_combined = {
            'trades': all_trades,
            'balance': final_balance,
            'blown': any(t.get('blown', False) for t in [results])
        }
        
        df_all = print_results(results_combined, config)
        save_results(df_all, results_combined, config)
    else:
        print("\n  ❌ No trades across all symbols!")
    
    elapsed = time_module.time() - total_start
    print(f"\n  ⏱️  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
