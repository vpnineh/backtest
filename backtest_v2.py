"""
=============================================================================
  BACKTEST ENGINE V2 - OPTIMIZED LIQUIDITY SWEEP + SMC
  
  اصلاحات کلیدی:
  1. Sweep detection واقعی با rejection candle pattern
  2. تایید ساختاری چندلایه (HTF bias + MTF CHoCH + LTF entry)
  3. R:R داینامیک (بر اساس فاصله تا نقدینگی مخالف)
  4. Partial TP (50% at 1.5R, rest at 3R)
  5. فیلتر ATR برای جلوگیری از ورود در بازار کم‌نوسان
  6. فیلتر روز هفته (دوشنبه صبح و جمعه عصر ممنوع)
  7. حذف Breakeven ساده → جایگزین Trailing Stop هوشمند
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import yaml, os, time as timer
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')


# =============================================================
# DATA
# =============================================================

def load_csv(fp):
    try:
        df = pd.read_csv(fp, sep=';', header=None,
                         names=['DT','O','H','L','C','V'])
        df['DT'] = df['DT'].str.strip()
        for fmt in ['%Y%m%d %H%M%S','%Y.%m.%d %H:%M','%m/%d/%Y %H:%M']:
            try:
                df['DT'] = pd.to_datetime(df['DT'], format=fmt)
                break
            except: continue
        else:
            df['DT'] = pd.to_datetime(df['DT'], format='mixed')
        df.set_index('DT', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        df = df[df.index.dayofweek < 5]
        df.columns = ['Open','High','Low','Close','Volume']
        return df.dropna()
    except Exception as e:
        print(f"  ERR {fp}: {e}")
        return pd.DataFrame()


def load_symbol(data_dir, symbol, years):
    frames = []
    for y in years:
        fp = os.path.join(data_dir, f"DAT_ASCII_{symbol}_M1_{y}.csv")
        if os.path.exists(fp):
            print(f"    {y}...", end=" ")
            d = load_csv(fp)
            if not d.empty:
                print(f"{len(d):,}")
                frames.append(d)
            else: print("empty")
        else: print(f"    {y} not found")
    if frames:
        c = pd.concat(frames).sort_index()
        c = c[~c.index.duplicated(keep='first')]
        print(f"    TOTAL: {len(c):,} M1 candles")
        return c
    return pd.DataFrame()


def resample_tf(df, minutes):
    return df.resample(f'{minutes}min').agg({
        'Open':'first','High':'max','Low':'min',
        'Close':'last','Volume':'sum'
    }).dropna()


# =============================================================
# STRUCTURE DETECTION
# =============================================================

def swing_highs(highs, period=10):
    """Rolling window swing high detection"""
    n = len(highs)
    result = np.full(n, np.nan)
    half = period // 2
    for i in range(half, n - half):
        win = highs[i-half:i+half+1]
        if highs[i] >= win.max() and np.sum(win >= highs[i]) <= 2:
            result[i] = highs[i]
    return result


def swing_lows(lows, period=10):
    n = len(lows)
    result = np.full(n, np.nan)
    half = period // 2
    for i in range(half, n - half):
        win = lows[i-half:i+half+1]
        if lows[i] <= win.min() and np.sum(win <= lows[i]) <= 2:
            result[i] = lows[i]
    return result


def compute_atr(high, low, close, period=14):
    """ATR vectorized"""
    h = np.array(high)
    l = np.array(low)
    c = np.array(close)
    tr = np.maximum(h - l, np.maximum(
        np.abs(h - np.roll(c, 1)),
        np.abs(l - np.roll(c, 1))
    ))
    tr[0] = h[0] - l[0]
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


def detect_structure_shift(closes, swing_h, swing_l, lookback=20):
    """
    تشخیص CHoCH/BOS:
    - CHoCH Bullish: بعد از روند نزولی، close بالای آخرین swing high
    - CHoCH Bearish: بعد از روند صعودی، close زیر آخرین swing low
    Returns: array of 1 (bullish shift), -1 (bearish shift), 0 (none)
    """
    n = len(closes)
    shifts = np.zeros(n, dtype=int)
    
    last_sh = np.nan
    last_sl = np.nan
    prev_sh = np.nan
    prev_sl = np.nan
    
    for i in range(n):
        if not np.isnan(swing_h[i]):
            prev_sh = last_sh
            last_sh = swing_h[i]
        if not np.isnan(swing_l[i]):
            prev_sl = last_sl
            last_sl = swing_l[i]
        
        if np.isnan(last_sh) or np.isnan(last_sl):
            continue
        if np.isnan(prev_sh) or np.isnan(prev_sl):
            continue
        
        # Bearish structure (LH + LL) then close > last SH = bullish CHoCH
        if prev_sh > last_sh and prev_sl > last_sl:  # bearish trend
            if closes[i] > last_sh:
                shifts[i] = 1  # bullish CHoCH
        
        # Bullish structure (HH + HL) then close < last SL = bearish CHoCH
        if prev_sh < last_sh and prev_sl < last_sl:  # bullish trend
            if closes[i] < last_sl:
                shifts[i] = -1  # bearish CHoCH
    
    return shifts


# =============================================================
# SIGNAL GENERATION V2
# =============================================================

def generate_signals_v2(df_htf, df_mtf, df_m1, config, pip_size):
    """
    سیگنال‌دهی اصلاح شده با فیلترهای بیشتر
    """
    cfg = config['strategy']
    
    print("  [1/6] Swing points (HTF)...")
    t0 = timer.time()
    sh_htf = swing_highs(df_htf['High'].values, cfg['swing_lookback'])
    sl_htf = swing_lows(df_htf['Low'].values, cfg['swing_lookback'])
    print(f"        SH: {np.sum(~np.isnan(sh_htf))}, "
          f"SL: {np.sum(~np.isnan(sl_htf))} ({timer.time()-t0:.1f}s)")
    
    print("  [2/6] Swing points (MTF)...")
    t0 = timer.time()
    sh_mtf = swing_highs(df_mtf['High'].values, 10)
    sl_mtf = swing_lows(df_mtf['Low'].values, 10)
    print(f"        SH: {np.sum(~np.isnan(sh_mtf))}, "
          f"SL: {np.sum(~np.isnan(sl_mtf))} ({timer.time()-t0:.1f}s)")
    
    print("  [3/6] ATR filter...")
    t0 = timer.time()
    atr_htf = compute_atr(df_htf['High'].values, df_htf['Low'].values,
                           df_htf['Close'].values, 14)
    # Map ATR to MTF
    atr_series = pd.Series(atr_htf, index=df_htf.index)
    atr_mtf = atr_series.reindex(df_mtf.index, method='ffill').values
    
    # ATR percentile filter: only trade when ATR > 30th percentile
    atr_rolling_pct = pd.Series(atr_mtf).rolling(500, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    ).values
    atr_ok = atr_rolling_pct > 0.30
    print(f"        ATR filter passes: {np.sum(atr_ok):,} / "
          f"{len(atr_ok):,} ({timer.time()-t0:.1f}s)")
    
    print("  [4/6] Structure shifts (MTF)...")
    t0 = timer.time()
    struct_shifts = detect_structure_shift(
        df_mtf['Close'].values, sh_mtf, sl_mtf
    )
    print(f"        Bullish CHoCH: {np.sum(struct_shifts==1)}, "
          f"Bearish CHoCH: {np.sum(struct_shifts==-1)} ({timer.time()-t0:.1f}s)")
    
    print("  [5/6] Daily levels + Sweep detection...")
    t0 = timer.time()
    
    # PDH/PDL
    daily = df_m1.resample('1D').agg({'High':'max','Low':'min'}).dropna()
    daily['PDH'] = daily['High'].shift(1)
    daily['PDL'] = daily['Low'].shift(1)
    daily.dropna(inplace=True)
    
    pdh_dict = {d.date(): r['PDH'] for d, r in daily.iterrows()}
    pdl_dict = {d.date(): r['PDL'] for d, r in daily.iterrows()}
    
    n = len(df_mtf)
    mtf_pdh = np.full(n, np.nan)
    mtf_pdl = np.full(n, np.nan)
    
    for i, dt in enumerate(df_mtf.index):
        d = dt.date()
        if d in pdh_dict:
            mtf_pdh[i] = pdh_dict[d]
            mtf_pdl[i] = pdl_dict[d]
    
    # Forward fill
    mask = np.isnan(mtf_pdh)
    idx = np.where(~mask, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    mtf_pdh = mtf_pdh[idx]
    mtf_pdl = mtf_pdl[idx]
    
    highs = df_mtf['High'].values
    lows = df_mtf['Low'].values
    opens = df_mtf['Open'].values
    closes = df_mtf['Close'].values
    hours = df_mtf.index.hour
    dow = df_mtf.index.dayofweek  # 0=Mon, 4=Fri
    
    sweep_thresh = cfg['sweep_threshold_pips'] * pip_size
    
    # ====== IMPROVED SWEEP DETECTION ======
    # Sweep = price goes beyond level + shows rejection
    # Rejection = close back inside + wick > body
    
    body = np.abs(closes - opens)
    upper_wick = highs - np.maximum(closes, opens)
    lower_wick = np.minimum(closes, opens) - lows
    candle_range = highs - lows
    
    # Bearish rejection after sweep high: long upper wick + close < level
    rejection_bear = (upper_wick > body * 0.8) & (upper_wick > candle_range * 0.4)
    
    # Bullish rejection after sweep low: long lower wick + close > level
    rejection_bull = (lower_wick > body * 0.8) & (lower_wick > candle_range * 0.4)
    
    # Sweep PDH with rejection
    sweep_pdh = ((highs > mtf_pdh + sweep_thresh) & 
                 (closes < mtf_pdh) & 
                 rejection_bear)
    
    # Sweep PDL with rejection
    sweep_pdl = ((lows < mtf_pdl - sweep_thresh) & 
                 (closes > mtf_pdl) & 
                 rejection_bull)
    
    # Also sweep recent swing highs/lows (HTF)
    recent_sh = np.full(n, np.nan)
    recent_sl = np.full(n, np.nan)
    recent_sh_2 = np.full(n, np.nan)  # opposite target
    recent_sl_2 = np.full(n, np.nan)
    
    sh_vals = []
    sl_vals = []
    
    # Map HTF swings to MTF timeline
    sh_series = pd.Series(sh_htf, index=df_htf.index).dropna()
    sl_series = pd.Series(sl_htf, index=df_htf.index).dropna()
    
    sh_t = sh_series.index.values
    sh_v = sh_series.values
    sl_t = sl_series.index.values
    sl_v = sl_series.values
    
    si = 0
    li = 0
    for i in range(n):
        ct = df_mtf.index[i]
        while si < len(sh_t) and sh_t[si] <= ct:
            si += 1
        while li < len(sl_t) and sl_t[li] <= ct:
            li += 1
        
        if si >= 2:
            recent_sh[i] = sh_v[si-1]
            recent_sh_2[i] = sh_v[si-2]
        elif si >= 1:
            recent_sh[i] = sh_v[si-1]
        
        if li >= 2:
            recent_sl[i] = sl_v[li-1]
            recent_sl_2[i] = sl_v[li-2]
        elif li >= 1:
            recent_sl[i] = sl_v[li-1]
    
    valid_sh = ~np.isnan(recent_sh)
    valid_sl = ~np.isnan(recent_sl)
    
    sweep_swing_h = (valid_sh & 
                     (highs > recent_sh + sweep_thresh) & 
                     (closes < recent_sh) & 
                     rejection_bear)
    
    sweep_swing_l = (valid_sl & 
                     (lows < recent_sl - sweep_thresh) & 
                     (closes > recent_sl) & 
                     rejection_bull)
    
    any_sweep_bear = sweep_pdh | sweep_swing_h
    any_sweep_bull = sweep_pdl | sweep_swing_l
    
    print(f"        Sweeps - Bearish: {any_sweep_bear.sum()}, "
          f"Bullish: {any_sweep_bull.sum()} ({timer.time()-t0:.1f}s)")
    
    print("  [6/6] Combining signals with filters...")
    t0 = timer.time()
    
    # ====== SESSION FILTER ======
    london_s = cfg['london_start']
    london_e = cfg['london_end']
    ny_s = cfg['newyork_start']
    ny_e = cfg['newyork_end']
    
    in_session = ((hours >= london_s) & (hours < london_e)) | \
                 ((hours >= ny_s) & (hours < ny_e))
    
    # ====== DAY FILTER ======
    # No Monday before 10 UTC, No Friday after 14 UTC
    day_ok = ~((dow == 0) & (hours < 10)) & ~((dow == 4) & (hours >= 14))
    
    # ====== STRUCTURE CONFIRMATION ======
    # Need CHoCH within last 10 bars in the right direction
    
    choch_bull_recent = np.zeros(n, dtype=bool)
    choch_bear_recent = np.zeros(n, dtype=bool)
    
    lookback_choch = 10
    for i in range(lookback_choch, n):
        window = struct_shifts[i-lookback_choch:i+1]
        if np.any(window == 1):
            choch_bull_recent[i] = True
        if np.any(window == -1):
            choch_bear_recent[i] = True
    
    # ====== HTF BIAS ======
    # Simple: price above/below 200-period SMA on HTF, mapped to MTF
    sma_htf = pd.Series(df_htf['Close'].values).rolling(200, min_periods=50).mean().values
    sma_series = pd.Series(sma_htf, index=df_htf.index)
    sma_mtf = sma_series.reindex(df_mtf.index, method='ffill').values
    
    htf_bullish = closes > sma_mtf
    htf_bearish = closes < sma_mtf
    
    # ====== FINAL SIGNALS ======
    
    # SHORT: sweep high + rejection + bearish CHoCH + ATR ok + session + day
    short_signal = (any_sweep_bear & 
                    choch_bear_recent & 
                    atr_ok & 
                    in_session & 
                    day_ok)
    
    # LONG: sweep low + rejection + bullish CHoCH + ATR ok + session + day
    long_signal = (any_sweep_bull & 
                   choch_bull_recent & 
                   atr_ok & 
                   in_session & 
                   day_ok)
    
    # Remove conflicts
    conflict = short_signal & long_signal
    short_signal &= ~conflict
    long_signal &= ~conflict
    
    # ====== COMPUTE ENTRY / SL / TP ======
    
    sl_buffer = cfg['sl_buffer_pips'] * pip_size
    
    signal = np.zeros(n, dtype=int)
    entry_arr = np.zeros(n)
    sl_arr = np.zeros(n)
    tp1_arr = np.zeros(n)  # TP1: partial (1.5R)
    tp2_arr = np.zeros(n)  # TP2: full (target liquidity or 3R)
    
    signal[long_signal] = 1
    signal[short_signal] = -1
    
    for i in np.where(signal != 0)[0]:
        if signal[i] == 1:  # LONG
            entry = closes[i]
            
            # SL: below sweep candle low - buffer
            sl = lows[i] - sl_buffer
            
            # Also check recent swing low
            if valid_sl[i]:
                sl = min(sl, recent_sl[i] - sl_buffer)
            
            sl_dist = entry - sl
            if sl_dist <= 0 or sl_dist / pip_size < 5 or sl_dist / pip_size > 60:
                signal[i] = 0
                continue
            
            # TP1: 1.5R
            tp1 = entry + sl_dist * 1.5
            
            # TP2: target opposite liquidity or 3R
            tp2 = entry + sl_dist * 3.0
            if valid_sh[i]:
                # Target: recent swing high
                target_liq = recent_sh[i]
                if target_liq > tp1:  # Only if it's beyond TP1
                    tp2 = target_liq
            
            # Also PDH as target
            if not np.isnan(mtf_pdh[i]) and mtf_pdh[i] > tp1:
                tp2 = max(tp2, mtf_pdh[i])
            
            # Minimum TP2 = 2R
            tp2 = max(tp2, entry + sl_dist * 2.0)
            
            entry_arr[i] = entry
            sl_arr[i] = sl
            tp1_arr[i] = tp1
            tp2_arr[i] = tp2
            
        else:  # SHORT
            entry = closes[i]
            
            sl = highs[i] + sl_buffer
            if valid_sh[i]:
                sl = max(sl, recent_sh[i] + sl_buffer)
            
            sl_dist = sl - entry
            if sl_dist <= 0 or sl_dist / pip_size < 5 or sl_dist / pip_size > 60:
                signal[i] = 0
                continue
            
            tp1 = entry - sl_dist * 1.5
            
            tp2 = entry - sl_dist * 3.0
            if valid_sl[i]:
                target_liq = recent_sl[i]
                if target_liq < tp1:
                    tp2 = target_liq
            
            if not np.isnan(mtf_pdl[i]) and mtf_pdl[i] < tp1:
                tp2 = min(tp2, mtf_pdl[i])
            
            tp2 = min(tp2, entry - sl_dist * 2.0)
            
            entry_arr[i] = entry
            sl_arr[i] = sl
            tp1_arr[i] = tp1
            tp2_arr[i] = tp2
    
    # ====== COOLDOWN ======
    min_spacing = 6  # bars (= 1.5 hours on 15min)
    sig_idx = np.where(signal != 0)[0]
    
    if len(sig_idx) > 1:
        clean = [sig_idx[0]]
        for idx in sig_idx[1:]:
            if idx - clean[-1] >= min_spacing:
                clean.append(idx)
        
        new_signal = np.zeros(n, dtype=int)
        for idx in clean:
            new_signal[idx] = signal[idx]
        signal = new_signal
    
    # ====== MAX 1 TRADE PER DAY ======
    sig_idx = np.where(signal != 0)[0]
    dates_used = set()
    final_signal = np.zeros(n, dtype=int)
    
    for idx in sig_idx:
        d = df_mtf.index[idx].date()
        if d not in dates_used:
            final_signal[idx] = signal[idx]
            dates_used.add(d)
    
    signal = final_signal
    
    total_sigs = np.sum(signal != 0)
    longs = np.sum(signal == 1)
    shorts = np.sum(signal == -1)
    print(f"        Final signals: {total_sigs} "
          f"(L:{longs} S:{shorts}) ({timer.time()-t0:.1f}s)")
    
    return pd.DataFrame({
        'signal': signal,
        'entry': entry_arr,
        'sl': sl_arr,
        'tp1': tp1_arr,
        'tp2': tp2_arr
    }, index=df_mtf.index)


# =============================================================
# TRADE SIMULATOR V2 (with Partial TP + Smart Trailing)
# =============================================================

def simulate_v2(df_mtf, signals, config, pip_size, symbol):
    """
    شبیه‌سازی با:
    - Partial TP: 50% at TP1 (1.5R), SL → Entry for remaining
    - Runner: 50% rides to TP2 with trailing stop
    """
    print("  Simulating trades V2...")
    t0 = timer.time()
    
    balance = config['account']['initial_balance']
    initial = balance
    peak = balance
    risk_pct = config['risk']['risk_per_trade']
    comm = config['execution']['commission_per_lot']
    spread = config['execution']['spread_pips'] * pip_size
    slip = config['execution']['slippage_pips'] * pip_size
    pip_val = 10.0  # per lot per pip
    max_daily_loss = config['risk']['max_daily_loss']
    prop_max_dd = config['prop_rules']['max_total_drawdown']
    
    sig_idx = np.where(signals['signal'].values != 0)[0]
    
    if len(sig_idx) == 0:
        print("    No signals")
        return {'trades': [], 'balance': balance, 'blown': False}
    
    highs = df_mtf['High'].values
    lows = df_mtf['Low'].values
    closes = df_mtf['Close'].values
    n_bars = len(df_mtf)
    times = df_mtf.index
    
    trades = []
    daily_pnl = {}
    blown = False
    
    for si in sig_idx:
        if blown:
            break
        
        direction = signals['signal'].values[si]
        entry = signals['entry'].values[si]
        sl = signals['sl'].values[si]
        tp1 = signals['tp1'].values[si]
        tp2 = signals['tp2'].values[si]
        entry_time = times[si]
        
        # Daily limit check
        day_key = str(entry_time.date())
        if day_key in daily_pnl and daily_pnl[day_key] <= -balance * max_daily_loss:
            continue
        
        # Spread + slippage
        if direction == 1:
            entry += spread/2 + slip
        else:
            entry -= spread/2 + slip
        
        # Position size
        sl_pips = abs(entry - sl) / pip_size
        if sl_pips <= 0:
            continue
        
        risk_amount = balance * risk_pct
        total_lots = max(0.01, round(risk_amount / (sl_pips * pip_val), 2))
        lot_half = max(0.01, round(total_lots / 2, 2))
        commission = comm * total_lots
        
        # ===== SIMULATE FORWARD =====
        current_sl = sl
        tp1_hit = False
        exit_price = None
        exit_time = None
        exit_type = None
        pnl_part1 = 0  # PnL from first half (TP1)
        trailing_sl = None
        
        max_forward = min(si + 400, n_bars)
        
        for j in range(si + 1, max_forward):
            bh = highs[j]
            bl = lows[j]
            
            if direction == 1:  # LONG
                if not tp1_hit:
                    # Check SL first
                    if bl <= current_sl:
                        exit_price = current_sl
                        exit_time = times[j]
                        exit_type = 'SL'
                        break
                    
                    # Check TP1
                    if bh >= tp1:
                        tp1_hit = True
                        pnl_part1 = ((tp1 - entry) / pip_size) * pip_val * lot_half
                        # Move SL to entry for runner
                        current_sl = entry + 2 * pip_size
                        trailing_sl = current_sl
                        continue
                
                else:
                    # Runner phase - trailing stop
                    # Trail SL to 50% of profit from entry
                    current_profit_price = bh
                    new_trail = entry + (current_profit_price - entry) * 0.5
                    if new_trail > trailing_sl:
                        trailing_sl = new_trail
                    current_sl = trailing_sl
                    
                    if bl <= current_sl:
                        exit_price = current_sl
                        exit_time = times[j]
                        exit_type = 'TRAIL'
                        break
                    
                    if bh >= tp2:
                        exit_price = tp2
                        exit_time = times[j]
                        exit_type = 'TP2'
                        break
            
            else:  # SHORT
                if not tp1_hit:
                    if bh >= current_sl:
                        exit_price = current_sl
                        exit_time = times[j]
                        exit_type = 'SL'
                        break
                    
                    if bl <= tp1:
                        tp1_hit = True
                        pnl_part1 = ((entry - tp1) / pip_size) * pip_val * lot_half
                        current_sl = entry - 2 * pip_size
                        trailing_sl = current_sl
                        continue
                
                else:
                    current_profit_price = bl
                    new_trail = entry - (entry - current_profit_price) * 0.5
                    if new_trail < trailing_sl:
                        trailing_sl = new_trail
                    current_sl = trailing_sl
                    
                    if bh >= current_sl:
                        exit_price = current_sl
                        exit_time = times[j]
                        exit_type = 'TRAIL'
                        break
                    
                    if bl <= tp2:
                        exit_price = tp2
                        exit_time = times[j]
                        exit_type = 'TP2'
                        break
        
        # Timeout
        if exit_price is None:
            exit_price = closes[max_forward - 1]
            exit_time = times[max_forward - 1]
            exit_type = 'TIMEOUT'
        
        # Calculate total PnL
        if tp1_hit:
            # Part 1 already calculated
            # Part 2: runner
            if direction == 1:
                pnl_part2 = ((exit_price - entry) / pip_size) * pip_val * lot_half
            else:
                pnl_part2 = ((entry - exit_price) / pip_size) * pip_val * lot_half
            total_pnl = pnl_part1 + pnl_part2 - commission
        else:
            # Full position hit SL
            if direction == 1:
                total_pnl = ((exit_price - entry) / pip_size) * \
                            pip_val * total_lots - commission
            else:
                total_pnl = ((entry - exit_price) / pip_size) * \
                            pip_val * total_lots - commission
        
        if direction == 1:
            total_pips = (exit_price - entry) / pip_size
        else:
            total_pips = (entry - exit_price) / pip_size
        
        r_mult = total_pnl / risk_amount if risk_amount > 0 else 0
        
        balance += total_pnl
        peak = max(peak, balance)
        
        if day_key not in daily_pnl:
            daily_pnl[day_key] = 0
        daily_pnl[day_key] += total_pnl
        
        # Prop check
        dd_pct = (peak - balance) / initial
        if dd_pct >= prop_max_dd:
            blown = True
        
        trades.append({
            'symbol': symbol,
            'dir': 'LONG' if direction == 1 else 'SHORT',
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry': round(entry, 5),
            'exit': round(exit_price, 5),
            'sl': round(sl, 5),
            'tp1': round(tp1, 5),
            'tp2': round(tp2, 5),
            'lots': total_lots,
            'pnl': round(total_pnl, 2),
            'pips': round(total_pips, 1),
            'r': round(r_mult, 2),
            'type': exit_type,
            'tp1_hit': tp1_hit,
            'balance': round(balance, 2)
        })
    
    print(f"    Executed: {len(trades)} trades ({timer.time()-t0:.1f}s)")
    if blown:
        print(f"    ⚠️ Account blown")
    
    return {'trades': trades, 'balance': balance, 
            'peak': peak, 'blown': blown}


# =============================================================
# REPORTING
# =============================================================

def report(results, config):
    trades = results['trades']
    if not trades:
        print("\n  ❌ No trades!")
        return None
    
    df = pd.DataFrame(trades)
    initial = config['account']['initial_balance']
    final = results['balance']
    
    total = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    
    wr = len(wins)/total*100
    net = df['pnl'].sum()
    ret = (final - initial) / initial * 100
    
    avg_w = wins['pnl'].mean() if len(wins) else 0
    avg_l = losses['pnl'].mean() if len(losses) else 0
    avg_r = df['r'].mean()
    
    gp = wins['pnl'].sum() if len(wins) else 0
    gl = abs(losses['pnl'].sum()) if len(losses) else 1
    pf = gp/gl if gl > 0 else float('inf')
    
    bals = [initial] + list(df['balance'])
    pk = initial
    mdd = mdd_p = 0
    for b in bals:
        if b > pk: pk = b
        dd = pk - b
        ddp = dd/pk*100
        if ddp > mdd_p: mdd, mdd_p = dd, ddp
    
    # Consecutive
    mcw = mcl = cw = cl = 0
    for _, t in df.iterrows():
        if t['pnl'] > 0: cw += 1; cl = 0; mcw = max(mcw, cw)
        else: cl += 1; cw = 0; mcl = max(mcl, cl)
    
    # TP1 hit rate
    tp1_rate = df['tp1_hit'].sum() / total * 100
    
    exits = df['type'].value_counts()
    
    print(f"\n{'='*70}")
    print(f"{'BACKTEST RESULTS V2':^70}")
    print(f"{'='*70}")
    
    s = [
        ["Initial Balance", f"${initial:,.2f}"],
        ["Final Balance", f"${final:,.2f}"],
        ["Net P&L", f"${net:,.2f}"],
        ["Return", f"{ret:+.2f}%"],
        ["",""],
        ["Total Trades", total],
        ["Win Rate", f"{wr:.1f}%"],
        ["Wins / Losses", f"{len(wins)} / {len(losses)}"],
        ["",""],
        ["Avg Win", f"${avg_w:,.2f}"],
        ["Avg Loss", f"${avg_l:,.2f}"],
        ["Avg R", f"{avg_r:.2f}R"],
        ["Profit Factor", f"{pf:.2f}"],
        ["Total Pips", f"{df['pips'].sum():,.1f}"],
        ["",""],
        ["TP1 Hit Rate", f"{tp1_rate:.1f}%"],
        ["Max Drawdown", f"${mdd:,.2f} ({mdd_p:.2f}%)"],
        ["Max Consec Wins", mcw],
        ["Max Consec Losses", mcl],
        ["",""],
    ]
    for et, cnt in exits.items():
        s.append([f"Exit: {et}", cnt])
    
    s.extend([
        ["",""],
        ["═══ PROP FIRM ═══", ""],
        ["Phase 1 (8%)", "✅ PASS" if ret >= 8 else f"❌ FAIL ({ret:.1f}%)"],
        ["Phase 2 (5%)", "✅ PASS" if ret >= 5 else f"❌ FAIL ({ret:.1f}%)"],
        ["Max DD < 10%", "✅ OK" if mdd_p < 10 else f"❌ ({mdd_p:.1f}%)"],
        ["Account", "❌ BLOWN" if results.get('blown') else "✅ ALIVE"],
    ])
    
    print(tabulate(s, headers=["Metric","Value"], tablefmt="fancy_grid"))
    
    # Monthly
    print(f"\n{'MONTHLY':^70}")
    print("-"*70)
    
    df['exit_dt'] = pd.to_datetime(df['exit_time'])
    df['month'] = df['exit_dt'].dt.to_period('M')
    
    monthly = df.groupby('month').agg(
        n=('pnl','count'), w=('pnl',lambda x:(x>0).sum()),
        pnl=('pnl','sum'), pips=('pips','sum')
    ).reset_index()
    monthly['wr'] = (monthly['w']/monthly['n']*100).round(0)
    
    rows = []
    for _, m in monthly.iterrows():
        rows.append([str(m['month']), m['n'], f"{m['wr']:.0f}%",
                     f"{m['pips']:+.0f}", f"${m['pnl']:+,.0f}"])
    print(tabulate(rows, headers=["Month","#","WR","Pips","P&L"],
                   tablefmt="simple"))
    
    # Yearly
    print(f"\n{'YEARLY':^70}")
    df['year'] = df['exit_dt'].dt.year
    yearly = df.groupby('year').agg(
        n=('pnl','count'), w=('pnl',lambda x:(x>0).sum()),
        pnl=('pnl','sum'), pips=('pips','sum')
    ).reset_index()
    yearly['wr'] = (yearly['w']/yearly['n']*100).round(0)
    
    rows = []
    for _, y in yearly.iterrows():
        rows.append([y['year'], y['n'], f"{y['wr']:.0f}%",
                     f"{y['pips']:+.0f}", f"${y['pnl']:+,.0f}"])
    print(tabulate(rows, headers=["Year","#","WR","Pips","P&L"],
                   tablefmt="simple"))
    
    return df


def save_all(df, results, config):
    out = config['output']['output_directory']
    os.makedirs(out, exist_ok=True)
    
    # CSV
    fp = os.path.join(out, "trades_v2.csv")
    df.to_csv(fp, index=False)
    print(f"\n  💾 {fp}")
    
    # Equity
    initial = config['account']['initial_balance']
    bals = [initial] + list(df['balance'])
    times = [df['entry_time'].iloc[0]] + list(df['exit_time'])
    times = pd.to_datetime(times)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                              gridspec_kw={'height_ratios': [3,1]})
    
    axes[0].plot(times, bals, '#2196F3', lw=1.5)
    axes[0].axhline(initial, color='gray', ls='--', alpha=.5, label='Start')
    axes[0].axhline(initial*1.08, color='green', ls='--', alpha=.5, label='+8%')
    axes[0].axhline(initial*0.90, color='red', ls='--', alpha=.5, label='-10%')
    axes[0].set_title('Equity Curve V2 - Liquidity Sweep + SMC', 
                       fontweight='bold', fontsize=14)
    axes[0].set_ylabel('Balance ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=.3)
    
    pk = np.maximum.accumulate(bals)
    dd = (np.array(pk)-np.array(bals))/np.array(pk)*100
    axes[1].fill_between(times, dd, color='red', alpha=.4)
    axes[1].set_ylabel('DD%')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=.3)
    
    plt.tight_layout()
    fp = os.path.join(out, "equity_v2.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 {fp}")
    
    # Heatmap
    df['yr'] = pd.to_datetime(df['exit_time']).dt.year
    df['mo'] = pd.to_datetime(df['exit_time']).dt.month
    piv = df.groupby(['yr','mo'])['pnl'].sum().reset_index()
    pt = piv.pivot(index='yr', columns='mo', values='pnl')
    
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    pt.columns = [month_names.get(c,c) for c in pt.columns]
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(pt)*1.2)))
    sns.heatmap(pt, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                ax=ax, linewidths=1)
    ax.set_title('Monthly P&L Heatmap V2', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fp = os.path.join(out, "heatmap_v2.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🗓️  {fp}")
    
    # R distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#4CAF50' if r > 0 else '#F44336' for r in df['r']]
    ax.bar(range(len(df)), df['r'], color=colors, alpha=.7, width=1)
    ax.axhline(0, color='black', lw=.5)
    ax.axhline(df['r'].mean(), color='blue', ls='--',
               label=f"Avg: {df['r'].mean():.2f}R")
    ax.set_title('R-Multiple Distribution V2', fontweight='bold')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('R')
    ax.legend()
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    fp = os.path.join(out, "r_dist_v2.png")
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 {fp}")


# =============================================================
# MAIN
# =============================================================

def main():
    total_t = timer.time()
    
    print("="*70)
    print("  ⚡ BACKTEST ENGINE V2")
    print("  Liquidity Sweep + SMC (Optimized)")
    print("="*70)
    
    with open("config.yml") as f:
        config = yaml.safe_load(f)
    
    initial = config['account']['initial_balance']
    print(f"\n  💰 ${initial:,} | Risk: {config['risk']['risk_per_trade']*100}%"
          f" | R:R: dynamic (1.5R partial + runner)")
    
    all_trades = []
    final_balance = initial
    blown = False
    
    for symbol in config['data']['symbols']:
        if blown:
            break
        
        print(f"\n{'━'*70}")
        print(f"  📊 {symbol}")
        print(f"{'━'*70}")
        
        pip_size = {'EURUSD':0.0001,'GBPUSD':0.0001,
                    'XAUUSD':0.10,'XAGUSD':0.001}.get(symbol, 0.0001)
        
        df_m1 = load_symbol(config['data']['directory'], symbol,
                             config['data']['years'])
        if df_m1.empty:
            continue
        
        print(f"\n  Building TFs...")
        t0 = timer.time()
        cfg_s = config['strategy']
        df_htf = resample_tf(df_m1, cfg_s['htf_minutes'])
        df_mtf = resample_tf(df_m1, cfg_s['mtf_minutes'])
        print(f"    HTF: {len(df_htf):,} | MTF: {len(df_mtf):,} "
              f"({timer.time()-t0:.1f}s)")
        
        print(f"\n  Signal generation...")
        signals = generate_signals_v2(df_htf, df_mtf, df_m1, config, pip_size)
        
        print(f"\n  Trade simulation...")
        # Update config balance for sequential symbols
        config_copy = config.copy()
        config_copy['account'] = config['account'].copy()
        config_copy['account']['initial_balance'] = final_balance if all_trades else initial
        
        res = simulate_v2(df_mtf, signals, config_copy, pip_size, symbol)
        
        if res['trades']:
            all_trades.extend(res['trades'])
            final_balance = res['balance']
        
        blown = res.get('blown', False)
    
    if all_trades:
        combined = {'trades': all_trades, 'balance': final_balance, 
                    'blown': blown}
        df = report(combined, config)
        if df is not None:
            save_all(df, combined, config)
    else:
        print("\n  ❌ No trades!")
    
    print(f"\n  ⏱️ Runtime: {timer.time()-total_t:.1f}s")
    print("="*70)


if __name__ == "__main__":
    main()
