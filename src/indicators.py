"""
Technical indicators computed WITHOUT look-ahead bias.

All functions take a Series/DataFrame and return values
that are valid at bar close time t using data[0..t].

CRITICAL: When used in backtest, always use iloc[:-1] or
shift(1) to ensure we only use CLOSED bars, not current.
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average. pandas ewm is correct."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# ATR - True Average True Range
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    TR = max(H-L, |H-PC|, |L-PC|)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing (same as MT4)
    atr_val = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr_val


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI. Uses RMA (same as MT4/TradingView default).
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.fillna(50.0)  # neutral when undefined


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    series: pd.Series, period: int = 20, std_mult: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    middle = series.rolling(window=period, min_periods=period).mean()
    std    = series.rolling(window=period, min_periods=period).std(ddof=0)
    upper  = middle + std_mult * std
    lower  = middle - std_mult * std
    return upper, middle, lower


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder method).
    Returns ADX series only.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index)
    minus_dm_s = pd.Series(minus_dm, index=df.index)

    # Wilder smoothing
    alpha = 1.0 / period
    atr_s     = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di_s = plus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_di_s= minus_dm_s.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di  = 100.0 * plus_di_s  / atr_s.replace(0, np.nan)
    minus_di = 100.0 * minus_di_s / atr_s.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx_val.fillna(0.0)


# ---------------------------------------------------------------------------
# Candle Pattern Detection
# ---------------------------------------------------------------------------

def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect reversal candle patterns on CLOSED bars.
    Returns DataFrame with boolean columns:
      bullish_engulfing, hammer, morning_star
      bearish_engulfing, shooting_star, evening_star

    All patterns reference completed bars - no look-ahead.
    """
    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]

    body      = (c - o).abs()
    range_hl  = h - l
    upper_wick = h - c.where(c > o, o)   # from body top to high
    lower_wick = c.where(c < o, o) - l   # from body bottom to low

    is_bull = c > o
    is_bear = c < o

    prev_body    = body.shift(1)
    prev_is_bull = is_bull.shift(1)
    prev_is_bear = is_bear.shift(1)
    prev_o = o.shift(1)
    prev_c = c.shift(1)

    patterns = pd.DataFrame(index=df.index)

    # --- Bullish Engulfing ---
    # Current bullish bar body completely engulfs previous bearish body
    patterns["bullish_engulfing"] = (
        is_bull &
        prev_is_bear &
        (o <= prev_c) &
        (c >= prev_o) &
        (body >= prev_body)
    )

    # --- Bearish Engulfing ---
    patterns["bearish_engulfing"] = (
        is_bear &
        prev_is_bull &
        (o >= prev_c) &
        (c <= prev_o) &
        (body >= prev_body)
    )

    # --- Hammer (bullish) ---
    # Small body at top, long lower wick, small upper wick
    patterns["hammer"] = (
        (lower_wick >= 2.0 * body.replace(0, np.nan)) &
        (upper_wick <= 0.3 * range_hl) &
        (body > 0) &
        (range_hl > 0)
    ).fillna(False)

    # --- Shooting Star (bearish) ---
    # Small body at bottom, long upper wick, small lower wick
    patterns["shooting_star"] = (
        (upper_wick >= 2.0 * body.replace(0, np.nan)) &
        (lower_wick <= 0.3 * range_hl) &
        (body > 0) &
        (range_hl > 0)
    ).fillna(False)

    # --- Morning Star (3-bar bullish) ---
    # Bar -2: large bearish
    # Bar -1: small body (doji-like)
    # Bar  0: large bullish closing above midpoint of bar -2
    prev2_body    = body.shift(2)
    prev2_is_bear = is_bear.shift(2)
    prev2_mid     = (prev_o.shift(1) + prev_c.shift(1)).shift(0) / 2

    patterns["morning_star"] = (
        prev2_is_bear &
        (prev_body <= 0.3 * prev2_body) &  # middle bar is small
        is_bull &
        (c > (prev_o.shift(1) + prev_c.shift(1)) / 2) &
        (body >= 0.5 * prev2_body)
    ).fillna(False)

    # --- Evening Star (3-bar bearish) ---
    prev2_is_bull = is_bull.shift(2)

    patterns["evening_star"] = (
        prev2_is_bull &
        (prev_body <= 0.3 * prev2_body) &
        is_bear &
        (c < (prev_o.shift(1) + prev_c.shift(1)) / 2) &
        (body >= 0.5 * prev2_body)
    ).fillna(False)

    return patterns


# ---------------------------------------------------------------------------
# Pre-compute all indicators for a timeframe DataFrame
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Adds all indicator columns to df IN-PLACE (returns new df).
    Uses only data available at bar close. Safe for backtest.
    """
    result = df.copy()

    # EMA 200
    result["ema200"] = ema(result["close"], config.ema_trend_period)

    # ATR(14)
    result["atr14"] = atr(result, config.atr_period)

    # ATR(100)
    result["atr100"] = atr(result, config.atr_filter_period)

    # RSI(14)
    result["rsi14"] = rsi(result["close"], config.rsi_period)

    # Bollinger Bands
    bb_u, bb_m, bb_l = bollinger_bands(
        result["close"], config.bb_period, config.bb_std
    )
    result["bb_upper"]  = bb_u
    result["bb_middle"] = bb_m
    result["bb_lower"]  = bb_l

    # ADX(14)
    result["adx14"] = adx(result, config.adx_period)

    # Candle patterns
    patterns = detect_candle_patterns(result)
    result = pd.concat([result, patterns], axis=1)

    return result
