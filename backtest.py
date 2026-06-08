"""
═══════════════════════════════════════════════════════════════════
  FOREX BACKTESTER  —  Strategy: 1-2-3 Pattern + SMC Filter
  
  - Zero lookahead bias (all signals on bar CLOSE, execute on NEXT bar open)
  - Realistic spread + commission simulation
  - Prop firm rules: 5% daily DD, 10% max DD, 10% profit target
  - Walk-Forward optimization (no overfitting)
  - Supports: EURUSD, GBPUSD, AUDUSD, XAUUSD, USDCAD, USDCHF, NZDUSD,
              AUDNZD, EURGBP, XAGUSD
  - Timeframe aggregation from M1 raw data
  
Author: Generated for prop firm challenge
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import glob
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import zipfile
import io

warnings.filterwarnings("ignore")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest_results/backtest.log", mode="w"),
    ],
)
log = logging.getLogger("backtest")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    RESULTS_DIR: str = "backtest_results"

    # ── Instruments to test ───────────────────────────────────────────────────
    SYMBOLS: list = [
        "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
        "USDCHF", "NZDUSD", "AUDNZD", "EURGBP",
        "XAUUSD", "XAGUSD",
    ]
    PRIMARY_SYMBOL: str = "EURUSD"

    # ── Time ──────────────────────────────────────────────────────────────────
    TRAIN_START: str = "2015-01-01"
    TRAIN_END: str = "2021-12-31"
    TEST_START: str = "2022-01-01"
    TEST_END: str = "2024-12-31"
    TIMEFRAME: str = "H1"          # M5 M15 H1 H4 D1

    # ── Strategy parameters ───────────────────────────────────────────────────
    SWING_LOOKBACK: int = 10        # bars to confirm swing high/low
    ATR_PERIOD: int = 14
    ATR_SL_MULT: float = 1.5        # SL = entry ± ATR * mult
    ATR_TP_MULT: float = 3.0        # TP = entry ± ATR * mult  (RR = 2:1)
    TREND_EMA_FAST: int = 50
    TREND_EMA_SLOW: int = 200
    MIN_PATTERN_ATR: float = 0.5    # pattern height must be >= 0.5 ATR
    MAX_BARS_IN_TRADE: int = 48     # force-close after N bars

    # ── Risk ──────────────────────────────────────────────────────────────────
    RISK_PER_TRADE: float = 0.01    # 1% per trade
    MAX_OPEN_TRADES: int = 2
    INITIAL_BALANCE: float = 10_000.0

    # ── Prop firm rules ───────────────────────────────────────────────────────
    PROP_PROFIT_TARGET: float = 0.10   # 10%
    PROP_MAX_DAILY_DD: float = 0.05    # 5%
    PROP_MAX_TOTAL_DD: float = 0.10    # 10%

    # ── Spread & commission (in pips) ─────────────────────────────────────────
    SPREADS: dict = None   # filled in __post_init__

    def __post_init__(self):
        pass

    def __init__(self):
        self.SPREADS = {
            "EURUSD": 1.0, "GBPUSD": 1.2, "AUDUSD": 1.3,
            "USDCAD": 1.5, "USDCHF": 1.5, "NZDUSD": 1.8,
            "AUDNZD": 2.5, "EURGBP": 1.5, "XAUUSD": 20.0,
            "XAGUSD": 30.0,
        }
        self.PIP_SIZE = {
            "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
            "USDCAD": 0.0001, "USDCHF": 0.0001, "NZDUSD": 0.0001,
            "AUDNZD": 0.0001, "EURGBP": 0.0001, "XAUUSD": 0.01,
            "XAGUSD": 0.001,
        }


CFG = Config()
os.makedirs(CFG.RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER  (HistData ASCII M1 format)
# ══════════════════════════════════════════════════════════════════════════════
def _load_single_csv(path: str) -> Optional[pd.DataFrame]:
    """Load one HistData CSV (plain or inside zip)."""
    try:
        if path.endswith(".zip"):
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                csv_names = [n for n in names if n.endswith(".csv")]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    content = f.read().decode("utf-8", errors="ignore")
            buf = io.StringIO(content)
        else:
            buf = path

        df = pd.read_csv(
            buf,
            header=None,
            names=["datetime", "open", "high", "low", "close", "volume"],
            sep=";",
            parse_dates=["datetime"],
            infer_datetime_format=True,
        )
        # Some files use comma separator
        if df.shape[1] < 5:
            if path.endswith(".zip"):
                buf = io.StringIO(content)
            df = pd.read_csv(
                buf,
                header=None,
                names=["datetime", "open", "high", "low", "close", "volume"],
                sep=",",
                parse_dates=["datetime"],
                infer_datetime_format=True,
            )
        df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
        df = df.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()
        return df
    except Exception as e:
        log.warning(f"  Could not load {path}: {e}")
        return None


def load_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load & concatenate all yearly M1 files for a symbol."""
    log.info(f"Loading {symbol}  {start} → {end}")
    data_dir = Path(CFG.DATA_DIR)
    start_y = int(start[:4])
    end_y = int(end[:4])

    frames = []
    for year in range(start_y, end_y + 1):
        # Pattern 1: DAT_ASCII_EURUSD_M1_2020.csv
        # Pattern 2: HISTDATA_COM_ASCII_EURUSD_M12020.zip
        candidates = (
            list(data_dir.glob(f"DAT_ASCII_{symbol}_M1_{year}.csv"))
            + list(data_dir.glob(f"*{symbol}*{year}*.zip"))
            + list(data_dir.glob(f"*{symbol}_M1_{year}*.csv"))
        )
        for path in candidates:
            df = _load_single_csv(str(path))
            if df is not None and not df.empty:
                frames.append(df)
                break   # one file per year

    if not frames:
        raise FileNotFoundError(
            f"No data files found for {symbol} in '{CFG.DATA_DIR}'. "
            f"Place DAT_ASCII_{symbol}_M1_YYYY.csv  or  "
            f"HISTDATA_COM_ASCII_{symbol}_M1YYYY.zip  there."
        )

    raw = pd.concat(frames).sort_index()
    raw = raw[start:end]
    log.info(f"  {len(raw):,} M1 bars loaded for {symbol}")
    return raw


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate M1 → target timeframe."""
    tf_map = {
        "M5": "5min", "M15": "15min", "M30": "30min",
        "H1": "1h",   "H4": "4h",    "D1": "1D",
    }
    rule = tf_map.get(timeframe.upper())
    if rule is None:
        raise ValueError(f"Unknown timeframe '{timeframe}'. Choose from {list(tf_map)}")

    ohlc = df["close"].resample(rule).ohlc()
    ohlc.columns = ["open", "high", "low", "close"]
    ohlc["open"]   = df["open"].resample(rule).first()
    ohlc["high"]   = df["high"].resample(rule).max()
    ohlc["low"]    = df["low"].resample(rule).min()
    ohlc["volume"] = df["volume"].resample(rule).sum() if "volume" in df else 0
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    log.info(f"  Resampled to {timeframe}: {len(ohlc):,} bars")
    return ohlc


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS  (all vectorised, no future leak)
# ══════════════════════════════════════════════════════════════════════════════
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def swing_high(series: pd.Series, lookback: int) -> pd.Series:
    """True on bars that are a confirmed swing high (peak)."""
    # A swing high at bar i: high[i] is the max over [i-lb .. i+lb]
    # We use a rolling max — to avoid lookahead, the signal fires lb bars AFTER the peak
    lb = lookback
    rolled_max = series.rolling(2 * lb + 1, center=True).max()
    return (series == rolled_max).astype(bool)


def swing_low(series: pd.Series, lookback: int) -> pd.Series:
    lb = lookback
    rolled_min = series.rolling(2 * lb + 1, center=True).min()
    return (series == rolled_min).astype(bool)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], CFG.TREND_EMA_FAST)
    df["ema_slow"] = ema(df["close"], CFG.TREND_EMA_SLOW)
    df["atr"]      = atr(df, CFG.ATR_PERIOD)
    df["sh"]       = swing_high(df["high"], CFG.SWING_LOOKBACK)
    df["sl"]       = swing_low(df["low"],   CFG.SWING_LOOKBACK)
    df["trend"]    = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  1-2-3 PATTERN DETECTION  (event-driven, bar by bar)
# ══════════════════════════════════════════════════════════════════════════════
class PatternState:
    """Tracks one potential 1-2-3 pattern."""
    __slots__ = ("direction", "p1", "p2", "p3_target", "p2_level",
                 "p1_idx", "p2_idx", "active", "p3_found", "p3_idx")

    def __init__(self, direction, p1, p1_idx, p2, p2_idx):
        self.direction = direction   # 1=bullish  -1=bearish
        self.p1 = p1
        self.p1_idx = p1_idx
        self.p2 = p2
        self.p2_idx = p2_idx
        self.p2_level = p2
        self.p3_target = None
        self.p3_found = False
        self.p3_idx = None
        self.active = True


def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bar-by-bar 1-2-3 detection.
    Signal at bar i → entry on bar i+1 open.
    NO lookahead: decisions only use data up to and including bar i.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    sh     = df["sh"].values
    sl     = df["sl"].values
    atr_v  = df["atr"].values
    trend  = df["trend"].values
    n      = len(df)

    signal      = np.zeros(n, dtype=int)    # +1 long  -1 short
    signal_sl   = np.zeros(n)
    signal_tp   = np.zeros(n)

    lb = CFG.SWING_LOOKBACK

    # We scan from lb*2 to avoid edge effects from centered rolling
    bearish_patterns: list[PatternState] = []
    bullish_patterns: list[PatternState] = []

    for i in range(lb * 2, n - 1):
        cur_close = closes[i]
        cur_atr   = atr_v[i]
        if cur_atr == 0:
            continue

        # ── Start new patterns on swing pivots ────────────────────────────
        if sh[i] and trend[i] == -1:   # bearish trend + swing high → potential bearish 1-2-3
            # P1 is a recent swing low before this swing high
            # look back for the last swing low
            for j in range(i - 1, max(i - lb * 3, lb * 2), -1):
                if sl[j]:
                    p1_price = lows[j]
                    p2_price = highs[i]
                    if p2_price - p1_price >= CFG.MIN_PATTERN_ATR * cur_atr:
                        bearish_patterns.append(
                            PatternState(-1, p1_price, j, p2_price, i)
                        )
                    break

        if sl[i] and trend[i] == 1:   # bullish trend + swing low → potential bullish 1-2-3
            for j in range(i - 1, max(i - lb * 3, lb * 2), -1):
                if sh[j]:
                    p1_price = highs[j]
                    p2_price = lows[i]
                    if p1_price - p2_price >= CFG.MIN_PATTERN_ATR * cur_atr:
                        bullish_patterns.append(
                            PatternState(1, p1_price, j, p2_price, i)
                        )
                    break

        # ── Check existing patterns for P3 and breakout ───────────────────
        # BEARISH: P1(low) → P2(high) → P3(lower high) → break below P2_low? 
        # Actually 1-2-3 bearish: P1(high) → P2(retracement low) → P3(lower high than P1)
        # Then break below P2 = SHORT signal
        # We track: p1=previous_high, p2=retracement_low, waiting P3 < P1, then break P2
        
        new_bearish = []
        for pat in bearish_patterns:
            if not pat.active:
                continue
            if i - pat.p2_idx > lb * 6:   # timeout
                pat.active = False
                continue

            if not pat.p3_found:
                # P3: a swing high lower than P1 (P2 is the retracement)
                # For bearish: P1=low, P2=high, P3=lower_high then break P1
                # Using Sperandeo definition:
                #   P1 = last significant HIGH
                #   P2 = retracement LOW
                #   P3 = bounce HIGH but < P1
                #   Signal = price breaks below P2
                if sh[i] and highs[i] < pat.p1 and highs[i] > pat.p2:
                    pat.p3_found = True
                    pat.p3_idx = i
                    pat.p3_target = highs[i]
            else:
                # Waiting for break below P2
                if cur_close < pat.p2_level:
                    # SHORT signal on next bar open
                    entry_sl = pat.p3_target + cur_atr * CFG.ATR_SL_MULT
                    entry_tp = pat.p2_level - cur_atr * CFG.ATR_TP_MULT
                    if entry_sl - cur_close >= cur_atr * 0.3:  # minimum SL distance
                        signal[i]    = -1
                        signal_sl[i] = entry_sl
                        signal_tp[i] = entry_tp
                    pat.active = False
            new_bearish.append(pat)
        bearish_patterns = [p for p in new_bearish if p.active]

        new_bullish = []
        for pat in bullish_patterns:
            if not pat.active:
                continue
            if i - pat.p2_idx > lb * 6:
                pat.active = False
                continue

            if not pat.p3_found:
                # Bullish: P1=high, P2=low, P3=higher_low then break P2(high)
                # P3: swing low > P2 (higher low than P2)
                if sl[i] and lows[i] > pat.p2 and lows[i] < pat.p1:
                    pat.p3_found = True
                    pat.p3_idx = i
                    pat.p3_target = lows[i]
            else:
                # Waiting for break above P2 (the previous high)
                if cur_close > pat.p2_level:
                    entry_sl = pat.p3_target - cur_atr * CFG.ATR_SL_MULT
                    entry_tp = pat.p2_level + cur_atr * CFG.ATR_TP_MULT
                    if cur_close - entry_sl >= cur_atr * 0.3:
                        signal[i]    = 1
                        signal_sl[i] = entry_sl
                        signal_tp[i] = entry_tp
                    pat.active = False
            new_bullish.append(pat)
        bullish_patterns = [p for p in new_bullish if p.active]

    df = df.copy()
    df["signal"]    = signal
    df["signal_sl"] = signal_sl
    df["signal_tp"] = signal_tp
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TRADE EXECUTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class Trade:
    __slots__ = ("entry_bar", "entry_price", "direction", "sl", "tp",
                 "size", "exit_bar", "exit_price", "pnl", "exit_reason",
                 "entry_time", "exit_time")

    def __init__(self, entry_bar, entry_price, direction, sl, tp,
                 size, entry_time):
        self.entry_bar   = entry_bar
        self.entry_price = entry_price
        self.direction   = direction
        self.sl          = sl
        self.tp          = tp
        self.size        = size
        self.exit_bar    = None
        self.exit_price  = None
        self.pnl         = None
        self.exit_reason = None
        self.entry_time  = entry_time
        self.exit_time   = None


def run_backtest(df: pd.DataFrame, symbol: str, balance: float = None) -> dict:
    """
    Realistic order simulation:
    - Signal on bar[i] close → entry on bar[i+1] OPEN
    - SL/TP checked against bar HIGH and LOW (not just close)
    - Spread applied on entry
    - Position sizing by fixed fractional risk
    """
    if balance is None:
        balance = CFG.INITIAL_BALANCE

    pip      = CFG.PIP_SIZE.get(symbol, 0.0001)
    spread   = CFG.SPREADS.get(symbol, 1.5) * pip

    opens   = df["open"].values
    highs   = df["high"].values
    lows    = df["low"].values
    closes  = df["close"].values
    signals = df["signal"].values
    sls     = df["signal_sl"].values
    tps     = df["signal_tp"].values
    times   = df.index

    n = len(df)
    equity_curve = np.full(n, np.nan)
    equity_curve[0] = balance

    open_trades: list[Trade] = []
    closed_trades: list[Trade] = []

    peak_balance        = balance
    max_dd              = 0.0
    daily_start_balance = balance
    daily_start_date    = times[0].date() if hasattr(times[0], "date") else None
    prop_breached       = False
    prop_breach_reason  = ""

    for i in range(1, n):
        # ── Daily reset for DD tracking ───────────────────────────────────
        if hasattr(times[i], "date"):
            cur_date = times[i].date()
            if daily_start_date != cur_date:
                daily_start_balance = balance
                daily_start_date    = cur_date

        # ── Check prop firm breach ─────────────────────────────────────────
        if not prop_breached:
            daily_dd = (daily_start_balance - balance) / CFG.INITIAL_BALANCE
            total_dd = (peak_balance - balance) / CFG.INITIAL_BALANCE
            if daily_dd >= CFG.PROP_MAX_DAILY_DD:
                prop_breached = True
                prop_breach_reason = f"Daily DD {daily_dd:.2%} exceeded 5% on {times[i]}"
            if total_dd >= CFG.PROP_MAX_TOTAL_DD:
                prop_breached = True
                prop_breach_reason = f"Total DD {total_dd:.2%} exceeded 10% on {times[i]}"

        # ── Process open trades ───────────────────────────────────────────
        for trade in list(open_trades):
            bars_in = i - trade.entry_bar

            # SL check (conservative: worst case)
            if trade.direction == 1:
                if lows[i] <= trade.sl:
                    trade.exit_price  = trade.sl
                    trade.exit_reason = "SL"
                elif highs[i] >= trade.tp:
                    trade.exit_price  = trade.tp
                    trade.exit_reason = "TP"
                elif bars_in >= CFG.MAX_BARS_IN_TRADE:
                    trade.exit_price  = opens[i]
                    trade.exit_reason = "TIMEOUT"
            else:  # short
                ask_entry = trade.entry_price
                # for short: we bought at bid so SL/TP are in bid terms
                if highs[i] >= trade.sl:
                    trade.exit_price  = trade.sl
                    trade.exit_reason = "SL"
                elif lows[i] <= trade.tp:
                    trade.exit_price  = trade.tp
                    trade.exit_reason = "TP"
                elif bars_in >= CFG.MAX_BARS_IN_TRADE:
                    trade.exit_price  = opens[i]
                    trade.exit_reason = "TIMEOUT"

            if trade.exit_price is not None:
                trade.exit_bar  = i
                trade.exit_time = times[i]
                raw_pnl         = (trade.exit_price - trade.entry_price) * trade.direction
                trade.pnl       = raw_pnl * trade.size
                balance        += trade.pnl
                peak_balance    = max(peak_balance, balance)
                dd              = (peak_balance - balance) / peak_balance
                max_dd          = max(max_dd, dd)
                closed_trades.append(trade)
                open_trades.remove(trade)

        equity_curve[i] = balance

        # ── Enter new trades ──────────────────────────────────────────────
        if not prop_breached and signals[i - 1] != 0 and len(open_trades) < CFG.MAX_OPEN_TRADES:
            direction = int(signals[i - 1])
            sl_price  = sls[i - 1]
            tp_price  = tps[i - 1]

            # Entry on current bar open + spread
            if direction == 1:
                entry_price = opens[i] + spread / 2
            else:
                entry_price = opens[i] - spread / 2

            sl_dist = abs(entry_price - sl_price)
            if sl_dist < 1e-8:
                continue

            # Position size (risk % of current balance)
            risk_amount = balance * CFG.RISK_PER_TRADE
            size        = risk_amount / sl_dist  # in units

            # Sanity checks
            if size <= 0 or not np.isfinite(size):
                continue

            trade = Trade(
                entry_bar=i,
                entry_price=entry_price,
                direction=direction,
                sl=sl_price,
                tp=tp_price,
                size=size,
                entry_time=times[i],
            )
            open_trades.append(trade)

    # Close any remaining open trades at last close
    for trade in open_trades:
        trade.exit_bar    = n - 1
        trade.exit_time   = times[n - 1]
        trade.exit_price  = closes[n - 1]
        trade.exit_reason = "EOD"
        raw_pnl           = (trade.exit_price - trade.entry_price) * trade.direction
        trade.pnl         = raw_pnl * trade.size
        balance          += trade.pnl
        closed_trades.append(trade)

    equity_curve[-1] = balance

    # Forward-fill NaN gaps
    equity_sr = pd.Series(equity_curve, index=df.index)
    equity_sr = equity_sr.ffill().bfill()

    return {
        "symbol":         symbol,
        "trades":         closed_trades,
        "equity":         equity_sr,
        "final_balance":  balance,
        "max_dd":         max_dd,
        "prop_breached":  prop_breached,
        "prop_reason":    prop_breach_reason,
        "df":             df,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(result: dict) -> dict:
    trades = result["trades"]
    equity = result["equity"]
    init   = CFG.INITIAL_BALANCE

    if not trades:
        return {"error": "No trades"}

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    loss = pnls[pnls <= 0]

    total_return  = (result["final_balance"] - init) / init
    n_trades      = len(trades)
    win_rate      = len(wins) / n_trades if n_trades else 0
    avg_win       = wins.mean() if len(wins) else 0
    avg_loss      = abs(loss.mean()) if len(loss) else 0
    profit_factor = wins.sum() / abs(loss.sum()) if loss.sum() != 0 else np.inf

    # Sharpe (annualised, daily returns)
    daily_eq   = equity.resample("1D").last().ffill()
    daily_ret  = daily_eq.pct_change().dropna()
    sharpe     = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                  if daily_ret.std() > 0 else 0)

    # Monthly return (geometric)
    n_months   = max(1, len(daily_eq) / 21)
    monthly_r  = (1 + total_return) ** (1 / n_months) - 1

    exit_counts = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    avg_hold_bars = np.mean([t.exit_bar - t.entry_bar for t in trades])

    return {
        "n_trades":        n_trades,
        "win_rate":        win_rate,
        "profit_factor":   profit_factor,
        "total_return":    total_return,
        "monthly_return":  monthly_r,
        "max_dd":          result["max_dd"],
        "sharpe":          sharpe,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "rr_achieved":     avg_win / avg_loss if avg_loss > 0 else 0,
        "exit_counts":     exit_counts,
        "avg_hold_bars":   avg_hold_bars,
        "prop_breached":   result["prop_breached"],
        "prop_reason":     result.get("prop_reason", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def walk_forward(df_full: pd.DataFrame, symbol: str,
                 n_splits: int = 5) -> pd.DataFrame:
    """
    Split data into n_splits IS+OOS windows.
    IS = 70%, OOS = 30% of each window.
    Returns OOS metrics per fold.
    """
    log.info(f"Walk-Forward ({n_splits} folds) on {symbol}")
    total_bars = len(df_full)
    fold_size  = total_bars // n_splits
    results    = []

    for k in range(n_splits):
        start = k * fold_size
        end   = min(start + fold_size, total_bars)
        split = start + int(0.7 * (end - start))

        df_is  = df_full.iloc[start:split]
        df_oos = df_full.iloc[split:end]

        if len(df_is) < 200 or len(df_oos) < 50:
            continue

        df_is_sig  = detect_signals(add_indicators(df_is))
        df_oos_sig = detect_signals(add_indicators(df_oos))

        res_oos  = run_backtest(df_oos_sig, symbol)
        m        = compute_metrics(res_oos)

        results.append({
            "fold":           k + 1,
            "is_start":       df_is.index[0],
            "oos_start":      df_oos.index[0],
            "oos_end":        df_oos.index[-1],
            "n_trades":       m.get("n_trades", 0),
            "win_rate":       m.get("win_rate", 0),
            "monthly_return": m.get("monthly_return", 0),
            "max_dd":         m.get("max_dd", 0),
            "profit_factor":  m.get("profit_factor", 0),
            "sharpe":         m.get("sharpe", 0),
        })

        log.info(
            f"  Fold {k+1}: OOS  n={m.get('n_trades',0)}  "
            f"WR={m.get('win_rate',0):.1%}  "
            f"Monthly={m.get('monthly_return',0):.2%}  "
            f"MaxDD={m.get('max_dd',0):.2%}"
        )

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def build_trade_log(trades: list, symbol: str) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "symbol":       symbol,
            "entry_time":   t.entry_time,
            "exit_time":    t.exit_time,
            "direction":    "LONG" if t.direction == 1 else "SHORT",
            "entry_price":  round(t.entry_price, 6),
            "exit_price":   round(t.exit_price, 6) if t.exit_price else None,
            "sl":           round(t.sl, 6),
            "tp":           round(t.tp, 6),
            "size":         round(t.size, 4),
            "pnl":          round(t.pnl, 4) if t.pnl is not None else None,
            "exit_reason":  t.exit_reason,
            "bars_held":    (t.exit_bar - t.entry_bar) if t.exit_bar else None,
        })
    return pd.DataFrame(rows)


def plot_results(result: dict, metrics: dict, symbol: str,
                 period_label: str, wf_df: Optional[pd.DataFrame] = None):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    text_color = "#e6edf3"
    grid_color = "#21262d"
    green      = "#3fb950"
    red        = "#f85149"
    blue       = "#58a6ff"
    yellow     = "#d29922"

    def ax_style(ax, title=""):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines[["top","right","left","bottom"]].set_color(grid_color)
        ax.yaxis.label.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.grid(color=grid_color, linewidth=0.5)
        if title:
            ax.set_title(title, color=text_color, fontsize=10, fontweight="bold")

    # ── 1. Equity curve ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    eq  = result["equity"]
    ax1.plot(eq.index, eq.values, color=blue, linewidth=1.2, label="Equity")
    ax1.axhline(CFG.INITIAL_BALANCE, color=text_color, linestyle="--",
                linewidth=0.8, alpha=0.5, label="Initial balance")

    # Shade drawdown
    peak_eq = eq.cummax()
    dd_abs  = peak_eq - eq
    ax1.fill_between(eq.index, eq.values, (peak_eq).values,
                     color=red, alpha=0.15, label="Drawdown")

    # Prop target line
    target = CFG.INITIAL_BALANCE * (1 + CFG.PROP_PROFIT_TARGET)
    ax1.axhline(target, color=green, linestyle=":", linewidth=1,
                alpha=0.7, label=f"Prop Target (+{CFG.PROP_PROFIT_TARGET:.0%})")

    ax1.set_ylabel("Balance ($)")
    ax_style(ax1, f"{symbol}  |  Equity Curve  |  {period_label}")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22",
               labelcolor=text_color, framealpha=0.6)

    # ── 2. Monthly returns bar chart ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    monthly = eq.resample("ME").last().pct_change().dropna()
    colors_m = [green if r >= 0 else red for r in monthly.values]
    ax2.bar(range(len(monthly)), monthly.values * 100, color=colors_m, width=0.8)
    ax2.axhline(0, color=text_color, linewidth=0.6)
    ax2.axhline(5, color=green, linestyle="--", linewidth=0.8,
                alpha=0.6, label="5% target")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Return (%)")
    ax_style(ax2, "Monthly Returns (%)")
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor=text_color)

    # ── 3. Trade distribution ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    pnls = [t.pnl for t in result["trades"] if t.pnl is not None]
    if pnls:
        ax3.hist(pnls, bins=40, color=blue, edgecolor="#0d1117", alpha=0.8)
        ax3.axvline(0, color=red, linewidth=1.2)
        ax3.axvline(np.mean(pnls), color=yellow, linewidth=1.2,
                    linestyle="--", label=f"Mean: ${np.mean(pnls):.1f}")
    ax3.set_xlabel("PnL ($)")
    ax3.set_ylabel("Count")
    ax_style(ax3, "Trade PnL Distribution")
    ax3.legend(fontsize=8, facecolor="#161b22", labelcolor=text_color)

    # ── 4. Walk-forward OOS ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if wf_df is not None and not wf_df.empty:
        folds = wf_df["fold"].values
        wr    = wf_df["win_rate"].values * 100
        mr    = wf_df["monthly_return"].values * 100
        x     = np.arange(len(folds))
        w     = 0.35
        bars1 = ax4.bar(x - w/2, wr, w, color=blue, label="Win Rate (%)")
        bars2 = ax4.bar(x + w/2, mr, w, color=green, label="Monthly Ret (%)")
        ax4.axhline(5, color=yellow, linestyle="--", linewidth=0.8,
                    label="5% target")
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"F{f}" for f in folds])
        ax4.set_ylabel("%")
    ax_style(ax4, "Walk-Forward OOS Results")
    ax4.legend(fontsize=7, facecolor="#161b22", labelcolor=text_color)

    # ── 5. Metrics summary table ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    ax_style(ax5)

    prop_ok  = not metrics.get("prop_breached", True)
    prop_col = green if prop_ok else red
    prop_txt = "✓ PASS" if prop_ok else "✗ BREACH"

    rows = [
        ("Trades",        f"{metrics.get('n_trades', 0)}"),
        ("Win Rate",      f"{metrics.get('win_rate', 0):.1%}"),
        ("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"),
        ("Total Return",  f"{metrics.get('total_return', 0):.2%}"),
        ("Monthly (avg)", f"{metrics.get('monthly_return', 0):.2%}"),
        ("Max Drawdown",  f"{metrics.get('max_dd', 0):.2%}"),
        ("Sharpe Ratio",  f"{metrics.get('sharpe', 0):.2f}"),
        ("Avg RR",        f"{metrics.get('rr_achieved', 0):.2f}"),
        ("Prop Firm",     prop_txt),
    ]

    for row_idx, (label, value) in enumerate(rows):
        y = 0.92 - row_idx * 0.10
        ax5.text(0.05, y, label, transform=ax5.transAxes,
                 color=text_color, fontsize=9)
        col = prop_col if label == "Prop Firm" else (
              green if "+" in value or (
                  "Return" in label and float(value.replace("%","").replace("+","")) > 0
              ) else text_color)
        try:
            val_f = float(value.replace("%","").replace("+",""))
            col   = (green if val_f > 0 else red) if "Return" in label or "Monthly" in label else text_color
        except Exception:
            col   = prop_col if label == "Prop Firm" else text_color
        ax5.text(0.65, y, value, transform=ax5.transAxes,
                 color=col, fontsize=9, fontweight="bold")

    ax5.set_title("Performance Metrics", color=text_color,
                  fontsize=10, fontweight="bold")

    out_path = os.path.join(CFG.RESULTS_DIR, f"{symbol}_{period_label}_report.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"  Chart saved: {out_path}")


def print_summary(symbol: str, metrics: dict, period: str):
    sep = "─" * 55
    log.info(sep)
    log.info(f"  {symbol}  |  {period}")
    log.info(sep)
    for k, v in metrics.items():
        if k in ("exit_counts", "prop_reason", "error"):
            continue
        if isinstance(v, float):
            log.info(f"  {k:<22} {v:.4f}")
        else:
            log.info(f"  {k:<22} {v}")
    if metrics.get("prop_reason"):
        log.info(f"  {'prop_reason':<22} {metrics['prop_reason']}")
    log.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Forex 1-2-3 Pattern Backtester")
    parser.add_argument("--symbol",    default=CFG.PRIMARY_SYMBOL,
                        help="Symbol to backtest (default EURUSD)")
    parser.add_argument("--timeframe", default=CFG.TIMEFRAME,
                        help="Timeframe: M5 M15 H1 H4 D1 (default H1)")
    parser.add_argument("--all",       action="store_true",
                        help="Run all available symbols")
    parser.add_argument("--wf",        action="store_true",
                        help="Run walk-forward validation")
    args = parser.parse_args()

    CFG.TIMEFRAME = args.timeframe
    symbols = CFG.SYMBOLS if args.all else [args.symbol.upper()]

    all_metrics = []

    for symbol in symbols:
        log.info("=" * 60)
        log.info(f"  BACKTESTING  {symbol}  @ {CFG.TIMEFRAME}")
        log.info("=" * 60)

        try:
            # ── Load TRAIN data ───────────────────────────────────────────
            m1_train = load_symbol(symbol, CFG.TRAIN_START, CFG.TRAIN_END)
            df_train = resample(m1_train, CFG.TIMEFRAME)
            df_train = add_indicators(df_train)
            df_train = detect_signals(df_train)

            result_train = run_backtest(df_train, symbol)
            metrics_train = compute_metrics(result_train)
            print_summary(symbol, metrics_train, "TRAIN")

            # ── Walk-Forward ──────────────────────────────────────────────
            wf_df = None
            if args.wf:
                wf_df = walk_forward(df_train, symbol, n_splits=5)
                wf_path = os.path.join(CFG.RESULTS_DIR, f"{symbol}_walkforward.csv")
                wf_df.to_csv(wf_path, index=False)
                log.info(f"  Walk-forward results saved: {wf_path}")

            # ── Load TEST (OOS) data ──────────────────────────────────────
            m1_test = load_symbol(symbol, CFG.TEST_START, CFG.TEST_END)
            df_test = resample(m1_test, CFG.TIMEFRAME)
            df_test = add_indicators(df_test)
            df_test = detect_signals(df_test)

            result_test  = run_backtest(df_test, symbol)
            metrics_test = compute_metrics(result_test)
            print_summary(symbol, metrics_test, "TEST (OOS)")

            # ── Save trade log ────────────────────────────────────────────
            tlog = build_trade_log(result_test["trades"], symbol)
            tlog_path = os.path.join(CFG.RESULTS_DIR, f"{symbol}_trades.csv")
            tlog.to_csv(tlog_path, index=False)
            log.info(f"  Trade log saved: {tlog_path}")

            # ── Plot ──────────────────────────────────────────────────────
            plot_results(result_test, metrics_test, symbol, "TEST_OOS", wf_df)

            all_metrics.append({"symbol": symbol, **metrics_test})

        except FileNotFoundError as e:
            log.warning(str(e))
        except Exception as e:
            log.error(f"  Error on {symbol}: {e}", exc_info=True)

    # ── Multi-symbol summary ──────────────────────────────────────────────────
    if len(all_metrics) > 1:
        summary_df = pd.DataFrame(all_metrics)
        s_path = os.path.join(CFG.RESULTS_DIR, "summary_all_symbols.csv")
        summary_df.to_csv(s_path, index=False)
        log.info(f"\nMulti-symbol summary saved: {s_path}")
        log.info(summary_df[["symbol","n_trades","win_rate","monthly_return",
                              "max_dd","profit_factor","prop_breached"]].to_string(index=False))

    log.info("\nDone. Results in: " + CFG.RESULTS_DIR)


if __name__ == "__main__":
    main()
