from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import StrategyConfig
from src.indicators import (
    adx,
    atr,
    bearish_engulfing,
    bollinger_bands,
    bullish_engulfing,
    evening_star,
    hammer,
    morning_star,
    rsi,
    shooting_star,
)


@dataclass
class FeatureSet:
    m15: pd.DataFrame
    h4: pd.DataFrame
    d1: pd.DataFrame
    weekly: pd.DataFrame


def resample_closed_bars(
    m1: pd.DataFrame,
    rule: str,
) -> pd.DataFrame:
    """
    M1 timestamps are treated as bar-open timestamps.

    Example:
    Minutes 10:00 ... 10:14 form an M15 candle whose timestamp is 10:15.
    Therefore that candle is only available at 10:15, never earlier.
    """
    return (
        m1.resample(
            rule,
            label="right",
            closed="left",
            origin="start_day",
        )
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            bar_count=("close", "count"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def build_m15(
    m1: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    frame = resample_closed_bars(m1, "15min")

    # Reject incomplete M15 candles.
    frame = frame[frame["bar_count"] >= 14].copy()

    frame["rsi"] = rsi(frame["close"], config.rsi_period)

    middle, upper, lower = bollinger_bands(
        frame["close"],
        config.bb_period,
        config.bb_std,
    )

    frame["bb_middle"] = middle
    frame["bb_upper"] = upper
    frame["bb_lower"] = lower

    frame["atr"] = atr(frame, config.atr_period)
    frame["atr_long"] = atr(frame, config.atr_long_period)
    frame["adx"] = adx(frame, config.adx_period)

    frame["bullish_pattern"] = (
        bullish_engulfing(frame)
        | hammer(frame)
        | morning_star(frame)
    )

    frame["bearish_pattern"] = (
        bearish_engulfing(frame)
        | shooting_star(frame)
        | evening_star(frame)
    )

    frame["buy_entry_setup"] = (
        (frame["rsi"] < config.buy_rsi)
        & (frame["close"] < frame["bb_lower"])
        & (frame["atr"] < config.atr_filter_multiple * frame["atr_long"])
        & (frame["adx"] < config.maximum_entry_adx)
        & frame["bullish_pattern"]
    )

    frame["sell_entry_setup"] = (
        (frame["rsi"] > config.sell_rsi)
        & (frame["close"] > frame["bb_upper"])
        & (frame["atr"] < config.atr_filter_multiple * frame["atr_long"])
        & (frame["adx"] < config.maximum_entry_adx)
        & frame["bearish_pattern"]
    )

    return frame


def build_h4(
    m1: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    frame = resample_closed_bars(m1, "4h")
    frame = frame[frame["bar_count"] >= 230].copy()

    frame["ema200"] = frame["close"].ewm(
        span=config.h4_ema_period,
        adjust=False,
        min_periods=config.h4_ema_period,
    ).mean()

    frame["atr"] = atr(frame, config.atr_period)
    frame["adx"] = adx(frame, config.adx_period)

    frame["ema_slope"] = frame["ema200"].diff()
    frame["previous_abs_slope"] = frame["ema_slope"].abs().shift(1)

    frame["accelerating"] = (
        (frame["adx"] > config.maximum_entry_adx)
        & (frame["adx"].diff() > 0)
        & (frame["ema_slope"].abs() > frame["previous_abs_slope"])
    )

    overextension = config.h4_overextension_atr * frame["atr"]

    frame["buy_trend_allowed"] = (
        (frame["close"] >= frame["ema200"])
        | (frame["close"] <= frame["ema200"] - overextension)
    ) & ~(
        frame["accelerating"]
        & (frame["close"] < frame["ema200"])
        & (frame["ema_slope"] < 0)
    )

    frame["sell_trend_allowed"] = (
        (frame["close"] <= frame["ema200"])
        | (frame["close"] >= frame["ema200"] + overextension)
    ) & ~(
        frame["accelerating"]
        & (frame["close"] > frame["ema200"])
        & (frame["ema_slope"] > 0)
    )

    return frame


def build_daily(
    m1: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    frame = resample_closed_bars(m1, "1D")
    frame = frame[frame["bar_count"] >= 1000].copy()

    frame["atr"] = atr(frame, config.atr_period)
    frame["atr_long"] = atr(frame, config.atr_long_period)

    frame["daily_volatility_allowed"] = (
        frame["atr"] < frame["atr_long"]
    )

    return frame


def build_previous_week_levels(m1: pd.DataFrame) -> pd.DataFrame:
    """
    A week beginning Monday is labelled with the following Monday,
    because its high/low are only known after the week has finished.
    """
    weekly = (
        m1.resample(
            "W-MON",
            label="left",
            closed="left",
        )
        .agg(
            previous_week_high=("high", "max"),
            previous_week_low=("low", "min"),
            bar_count=("close", "count"),
        )
        .dropna()
    )

    weekly.index = weekly.index + pd.Timedelta(days=7)

    return weekly


def build_features(
    m1: pd.DataFrame,
    config: StrategyConfig,
) -> FeatureSet:
    return FeatureSet(
        m15=build_m15(m1, config),
        h4=build_h4(m1, config),
        d1=build_daily(m1, config),
        weekly=build_previous_week_levels(m1),
    )
