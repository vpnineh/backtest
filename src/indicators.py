from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift(1)

    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )

    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)

    return tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    difference = close.diff()

    gain = difference.clip(lower=0)
    loss = -difference.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + relative_strength))

    return result.fillna(50.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_difference = df["high"].diff()
    low_difference = -df["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (high_difference > low_difference) & (high_difference > 0),
            high_difference,
            0.0,
        ),
        index=df.index,
    )

    minus_dm = pd.Series(
        np.where(
            (low_difference > high_difference) & (low_difference > 0),
            low_difference,
            0.0,
        ),
        index=df.index,
    )

    current_atr = atr(df, period)

    plus_dm_smoothed = plus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    minus_dm_smoothed = minus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = 100 * plus_dm_smoothed / current_atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smoothed / current_atr.replace(0, np.nan)

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    standard_deviations: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(period, min_periods=period).mean()
    standard_deviation = close.rolling(
        period,
        min_periods=period,
    ).std(ddof=0)

    upper = middle + standard_deviations * standard_deviation
    lower = middle - standard_deviations * standard_deviation

    return middle, upper, lower


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    previous_open = df["open"].shift(1)
    previous_close = df["close"].shift(1)

    return (
        (previous_close < previous_open)
        & (df["close"] > df["open"])
        & (df["open"] <= previous_close)
        & (df["close"] >= previous_open)
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    previous_open = df["open"].shift(1)
    previous_close = df["close"].shift(1)

    return (
        (previous_close > previous_open)
        & (df["close"] < df["open"])
        & (df["open"] >= previous_close)
        & (df["close"] <= previous_open)
    )


def hammer(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    return (
        (lower_wick >= 2.0 * body)
        & (upper_wick <= body)
        & (body / candle_range <= 0.40)
    )


def shooting_star(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    return (
        (upper_wick >= 2.0 * body)
        & (lower_wick <= body)
        & (body / candle_range <= 0.40)
    )


def morning_star(df: pd.DataFrame) -> pd.Series:
    first_open = df["open"].shift(2)
    first_close = df["close"].shift(2)

    second_open = df["open"].shift(1)
    second_close = df["close"].shift(1)

    first_body = (first_open - first_close).abs()
    second_body = (second_open - second_close).abs()
    first_midpoint = (first_open + first_close) / 2.0

    return (
        (first_close < first_open)
        & (second_body <= first_body * 0.50)
        & (df["close"] > df["open"])
        & (df["close"] >= first_midpoint)
    )


def evening_star(df: pd.DataFrame) -> pd.Series:
    first_open = df["open"].shift(2)
    first_close = df["close"].shift(2)

    second_open = df["open"].shift(1)
    second_close = df["close"].shift(1)

    first_body = (first_close - first_open).abs()
    second_body = (second_open - second_close).abs()
    first_midpoint = (first_open + first_close) / 2.0

    return (
        (first_close > first_open)
        & (second_body <= first_body * 0.50)
        & (df["close"] < df["open"])
        & (df["close"] <= first_midpoint)
    )
