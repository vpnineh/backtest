"""
Strategy signal generator.

Computes entry signals based on multi-timeframe indicator data.
ALL signals are based on CLOSED bars only (shift by 1 bar minimum).

This module is STATELESS - given indicator data, returns signals.
The backtester calls this at each execution bar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    NONE = "NONE"
    BUY  = "BUY"
    SELL = "SELL"


@dataclass
class EntrySignal:
    signal:    SignalType
    reason:    str
    pair:      str
    timestamp: object
    price:     float
    atr14:     float
    adx14:     float
    rsi14:     float
    grid_distance_pips: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bar(df: pd.DataFrame, idx: int) -> Optional[pd.Series]:
    """Safely get bar at position idx from DataFrame."""
    if idx < 0 or idx >= len(df):
        return None
    return df.iloc[idx]


def _price_above_bb_upper(bar: pd.Series) -> bool:
    if pd.isna(bar.get("bb_upper")):
        return False
    return bar["close"] > bar["bb_upper"]


def _price_below_bb_lower(bar: pd.Series) -> bool:
    if pd.isna(bar.get("bb_lower")):
        return False
    return bar["close"] < bar["bb_lower"]


# ---------------------------------------------------------------------------
# Trend Filter (H4 EMA200)
# ---------------------------------------------------------------------------

def check_trend_filter(
    h4_bar: pd.Series,
    signal_type: SignalType,
    config,
) -> bool:
    """
    BUY allowed if:
      - price > EMA200 (with trend)
      - OR price is far below EMA200 (overextended, counter-trend reversal)

    SELL allowed if:
      - price < EMA200
      - OR price is far above EMA200

    Never open grid if trend is accelerating (ADX rising fast).
    """
    if pd.isna(h4_bar.get("ema200")) or pd.isna(h4_bar.get("atr14")):
        return False

    price  = h4_bar["close"]
    ema200 = h4_bar["ema200"]
    atr    = h4_bar["atr14"]

    # Overextension threshold
    overext = config.ema_overextension_atr_mult * atr

    if signal_type == SignalType.BUY:
        above_ema = price >= ema200
        overextended_below = (ema200 - price) >= overext
        return above_ema or overextended_below

    elif signal_type == SignalType.SELL:
        below_ema = price <= ema200
        overextended_above = (price - ema200) >= overext
        return below_ema or overextended_above

    return False


# ---------------------------------------------------------------------------
# Entry filters (M15)
# ---------------------------------------------------------------------------

def check_entry_filters(
    m15_bar: pd.Series,
    signal_type: SignalType,
    config,
) -> tuple:
    """
    Check all M15 entry filters.
    Returns (passed: bool, details: dict)
    """
    details = {}

    # RSI filter
    rsi = m15_bar.get("rsi14", np.nan)
    if pd.isna(rsi):
        return False, {"fail": "rsi_nan"}

    if signal_type == SignalType.BUY:
        rsi_ok = rsi < config.rsi_buy_threshold
    else:
        rsi_ok = rsi > config.rsi_sell_threshold

    details["rsi"] = rsi
    details["rsi_ok"] = rsi_ok

    if not rsi_ok:
        return False, details

    # Bollinger Band filter
    if signal_type == SignalType.BUY:
        bb_ok = _price_below_bb_lower(m15_bar)
    else:
        bb_ok = _price_above_bb_upper(m15_bar)

    details["bb_ok"] = bb_ok

    if not bb_ok:
        return False, details

    # ATR filter: current ATR < 1.3 * ATR(100)
    atr14  = m15_bar.get("atr14",  np.nan)
    atr100 = m15_bar.get("atr100", np.nan)

    if pd.isna(atr14) or pd.isna(atr100) or atr100 == 0:
        return False, {"fail": "atr_nan"}

    atr_ok = atr14 < (config.atr_filter_multiplier * atr100)
    details["atr14"]  = atr14
    details["atr100"] = atr100
    details["atr_ok"] = atr_ok

    if not atr_ok:
        return False, details

    # ADX filter
    adx = m15_bar.get("adx14", np.nan)
    if pd.isna(adx):
        return False, {"fail": "adx_nan"}

    adx_ok = adx < config.adx_max
    details["adx"] = adx
    details["adx_ok"] = adx_ok

    if not adx_ok:
        return False, details

    return True, details


# ---------------------------------------------------------------------------
# Candle pattern check (M15)
# ---------------------------------------------------------------------------

def check_candle_pattern(
    m15_bar: pd.Series,
    signal_type: SignalType,
) -> bool:
    """
    Check if the last closed M15 bar shows a reversal pattern.
    """
    if signal_type == SignalType.BUY:
        patterns = ["bullish_engulfing", "hammer", "morning_star"]
    else:
        patterns = ["bearish_engulfing", "shooting_star", "evening_star"]

    for p in patterns:
        val = m15_bar.get(p, False)
        if val is True or val == 1:
            return True

    return False


# ---------------------------------------------------------------------------
# Emergency exit check
# ---------------------------------------------------------------------------

def check_emergency_exit(
    m15_bar: pd.Series,
    weekly_high: float,
    weekly_low: float,
    config,
) -> bool:
    """
    Emergency exit if:
    - ADX > 35
    - Price breaks weekly high/low
    - ATR expands sharply (atr14 > 1.5 * atr100)

    All three conditions must be true simultaneously.
    """
    adx   = m15_bar.get("adx14",  np.nan)
    atr14 = m15_bar.get("atr14",  np.nan)
    atr100= m15_bar.get("atr100", np.nan)
    price = m15_bar["close"]

    if any(pd.isna(x) for x in [adx, atr14, atr100]):
        return False

    adx_trigger    = adx > config.adx_emergency
    price_break    = (price > weekly_high) or (price < weekly_low)
    atr_expansion  = atr14 > (1.5 * atr100)

    return adx_trigger and price_break and atr_expansion


# ---------------------------------------------------------------------------
# Grid distance calculator
# ---------------------------------------------------------------------------

def calculate_grid_distance(
    atr14_pips: float,
    pair_config,
    config,
) -> float:
    """
    Grid distance = max(min_grid, min(max_grid, 0.8 * ATR14))
    Returns distance in PIPS.
    """
    dynamic = config.atr_grid_multiplier * atr14_pips
    grid = max(pair_config.min_grid_pips, min(pair_config.max_grid_pips, dynamic))
    return grid


# ---------------------------------------------------------------------------
# Master signal generator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """
    Stateless signal evaluator.
    Called by backtester at each M5 bar.
    """

    def __init__(self, config):
        self.config = config

    def evaluate(
        self,
        pair: str,
        current_time: pd.Timestamp,
        current_price: float,
        # Pre-computed indicator DataFrames (all bars up to current_time)
        # We use the LAST CLOSED BAR = iloc[-2] relative to current bar
        m15_indicators: pd.DataFrame,
        h4_indicators: pd.DataFrame,
        pair_config,
    ) -> EntrySignal:
        """
        Returns EntrySignal. Called at M5 bar open.
        Uses only data from CLOSED bars (never current bar).
        """
        no_signal = EntrySignal(
            signal=SignalType.NONE,
            reason="no_signal",
            pair=pair,
            timestamp=current_time,
            price=current_price,
            atr14=0.0,
            adx14=0.0,
            rsi14=0.0,
            grid_distance_pips=pair_config.default_grid_pips,
        )

        # Get last CLOSED M15 bar
        # current_time is start of current M5 bar
        # Last closed M15 bar = last M15 bar whose index < current_time
        m15_closed = m15_indicators[m15_indicators.index < current_time]
        if len(m15_closed) < self.config.bb_period + 5:
            return no_signal

        m15_bar = m15_closed.iloc[-1]  # last closed M15 bar

        # Get last CLOSED H4 bar
        h4_closed = h4_indicators[h4_indicators.index < current_time]
        if len(h4_closed) < self.config.ema_trend_period + 5:
            return no_signal

        h4_bar = h4_closed.iloc[-1]

        # ATR in pips for grid calculation
        atr14_price = m15_bar.get("atr14", 0.0)
        atr14_pips  = atr14_price / pair_config.pip_size if atr14_price > 0 else 0.0

        grid_distance = calculate_grid_distance(atr14_pips, pair_config, self.config)

        adx   = m15_bar.get("adx14", 0.0)
        rsi14 = m15_bar.get("rsi14", 50.0)

        # Try BUY signal
        buy_trend = check_trend_filter(h4_bar, SignalType.BUY, self.config)
        if buy_trend:
            filters_ok, details = check_entry_filters(m15_bar, SignalType.BUY, self.config)
            if filters_ok:
                candle_ok = check_candle_pattern(m15_bar, SignalType.BUY)
                if candle_ok:
                    return EntrySignal(
                        signal=SignalType.BUY,
                        reason="all_filters_passed",
                        pair=pair,
                        timestamp=current_time,
                        price=current_price,
                        atr14=atr14_pips,
                        adx14=adx,
                        rsi14=rsi14,
                        grid_distance_pips=grid_distance,
                    )

        # Try SELL signal
        sell_trend = check_trend_filter(h4_bar, SignalType.SELL, self.config)
        if sell_trend:
            filters_ok, details = check_entry_filters(m15_bar, SignalType.SELL, self.config)
            if filters_ok:
                candle_ok = check_candle_pattern(m15_bar, SignalType.SELL)
                if candle_ok:
                    return EntrySignal(
                        signal=SignalType.SELL,
                        reason="all_filters_passed",
                        pair=pair,
                        timestamp=current_time,
                        price=current_price,
                        atr14=atr14_pips,
                        adx14=adx,
                        rsi14=rsi14,
                        grid_distance_pips=grid_distance,
                    )

        return no_signal
