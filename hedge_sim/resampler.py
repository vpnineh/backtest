"""
Resampler
=========
Builds higher timeframe (e.g. M5) candles from raw M1 data using the
standard, lossless OHLC aggregation rule. This is NOT a lossy operation
at the candle level:

    open   = first M1 open in the window
    high   = max of all M1 highs in the window
    low    = min of all M1 lows in the window
    close  = last M1 close in the window
    volume = sum of all M1 volumes in the window

Every M5 high/low is therefore the TRUE high/low reached during that
5-minute window (not an approximation) - so grid/exit triggers checked
against the resampled bar's high/low are exactly as accurate as if
checked minute-by-minute for the purpose of "was this price level
reached". What genuinely cannot be recovered from any OHLC bar (M1 or
M5) is the *exact order* in which intermediate prices were visited -
that requires tick data. The engine's `_bar_path` conservative-ordering
model exists specifically to handle that residual, unavoidable
ambiguity in a realistic, non-optimistic way, whether run on M1 or M5.
"""

from __future__ import annotations
import pandas as pd


def resample_ohlc(df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """
    df must have columns: time, open, high, low, close, volume (time = datetime).
    timeframe uses pandas offset alias syntax, e.g. "5min", "15min", "1h".
    """
    if df.empty:
        return df.copy()

    indexed = df.set_index("time").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    resampled = indexed.resample(timeframe, label="left", closed="left").agg(agg)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled = resampled.reset_index()
    return resampled


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Adds a Wilder-style ATR column (price units, not pips) for optional
    ATR-based dynamic grid distance."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df
