"""
indicators.py
==============
Purely causal (trailing) technical indicators. Every value at row i is
computed using rows <= i only -- this is standard indicator math, not
lookahead. The engine is responsible for only *acting* on the value
from the previously CLOSED bar (see engine.py), which is what actually
prevents lookahead bias in the trading decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    adx_ = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    return pd.DataFrame({"adx": adx_, "plus_di": plus_di, "minus_di": minus_di})


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    raw_k = 100.0 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k, d


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def weekly_prior_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Prior COMPLETED week's high/low, mapped onto every bar of the
    current week. Uses shift(1) on the weekly resample so the current,
    still-forming week is never used -- this is the anti-lookahead
    guarantee for the 'major weekly structure' filter."""
    d = df.set_index("datetime")
    wk_high = d["high"].resample("W-SUN").max()
    wk_low = d["low"].resample("W-SUN").min()
    wk_high_prior = wk_high.shift(1)
    wk_low_prior = wk_low.shift(1)

    week_period = d.index.to_period("W-SUN")
    high_lookup = wk_high_prior.copy()
    high_lookup.index = high_lookup.index.to_period("W-SUN")
    low_lookup = wk_low_prior.copy()
    low_lookup.index = low_lookup.index.to_period("W-SUN")

    out_high = week_period.map(high_lookup)
    out_low = week_period.map(low_lookup)
    return pd.DataFrame({"week_high_prior": out_high.values, "week_low_prior": out_low.values}, index=df.index)


def daily_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP, resets every day, uses tick-volume as a proxy weight
    (real traded volume is not available in retail FX M1 history). This is
    a documented approximation, not exact institutional VWAP."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, 1.0)
    day = df["datetime"].dt.date.values
    day_s = pd.Series(day, index=df.index)
    cum_pv = (typical * vol).groupby(day_s).cumsum()
    cum_v = vol.groupby(day_s).cumsum()
    return cum_pv / cum_v
