"""
Central configuration for the Martingale Backtest System.
All parameters are defined here. No magic numbers in code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class PairConfig:
    """Per-pair configuration."""
    name: str
    pip_size: float          # 1 pip in price units
    spread_pips: float       # typical spread in pips
    min_grid_pips: float
    max_grid_pips: float
    default_grid_pips: float
    point_value: float       # USD value per pip per 0.01 lot (mini)


@dataclass
class StrategyConfig:
    """Core strategy parameters."""

    # --- Pairs ---
    pairs: List[str] = field(default_factory=lambda: ["EURGBP", "AUDNZD"])

    pair_configs: Dict[str, PairConfig] = field(default_factory=lambda: {
        "EURGBP": PairConfig(
            name="EURGBP",
            pip_size=0.0001,
            spread_pips=1.5,
            min_grid_pips=25,
            max_grid_pips=45,
            default_grid_pips=30,
            point_value=1.0,   # approximate, will be adjusted
        ),
        "AUDNZD": PairConfig(
            name="AUDNZD",
            pip_size=0.0001,
            spread_pips=2.5,
            min_grid_pips=30,
            max_grid_pips=45,
            default_grid_pips=35,
            point_value=0.65,  # approximate NZD-based
        ),
    })

    # --- Account ---
    initial_capital: float = 10_000.0      # USD
    leverage: float = 100.0
    base_risk_pct: float = 0.002           # 0.20% per initial trade
    max_account_exposure_pct: float = 0.10 # 10%

    # --- Position Scaling (Martingale multipliers) ---
    lot_sequence: List[float] = field(
        default_factory=lambda: [1.00, 1.35, 1.80, 2.40, 3.20, 4.30]
    )
    max_grid_levels: int = 7  # hard limit (sequence has 6, leave room)
    base_lot_size: float = 0.01  # will be scaled by risk

    # --- ATR Grid ---
    atr_grid_multiplier: float = 0.8   # grid = 0.8 * ATR(14)
    atr_period: int = 14

    # --- Entry Filters ---
    rsi_period: int = 14
    rsi_buy_threshold: float = 28.0
    rsi_sell_threshold: float = 72.0

    bb_period: int = 20
    bb_std: float = 2.0

    adx_period: int = 14
    adx_max: float = 25.0                 # entry disabled above this
    adx_emergency: float = 35.0           # emergency exit above this

    atr_filter_period: int = 100          # long-term ATR period
    atr_filter_multiplier: float = 1.3    # current ATR < 1.3 * ATR(100)

    ema_trend_period: int = 200           # H4 EMA

    # --- Basket Exit ---
    basket_tp_pct_min: float = 0.008      # 0.8% of account
    basket_tp_pct_max: float = 0.012      # 1.2% of account
    basket_tp_target: float = 0.010       # use 1.0% as primary target

    # --- Emergency / Protection ---
    basket_max_loss_multiplier: float = 2.0  # 2x expected TP => disable new levels
    spread_max_multiplier: float = 2.0       # disable if spread > 2x average

    # --- Risk Limits ---
    max_baskets_per_direction: int = 1
    max_simultaneous_baskets: int = 2
    max_daily_drawdown_pct: float = 0.03   # 3%
    max_weekly_drawdown_pct: float = 0.08  # 8%

    # --- News Filter (minutes) ---
    news_blackout_minutes: int = 60

    # --- Session Filter ---
    # London: 08:00-16:00 UTC
    # London-NY overlap: 13:00-16:00 UTC
    london_open_utc: int = 8
    london_close_utc: int = 16
    ny_open_utc: int = 13
    # Asian session: 00:00-08:00 UTC (avoided)
    asian_close_utc: int = 8
    # Friday cutoff
    friday_cutoff_utc: int = 12

    # --- Timeframes (in minutes) ---
    tf_trend: int = 240    # H4
    tf_entry: int = 15     # M15
    tf_exec: int = 5       # M5  (data resolution)

    # --- Candle Pattern minimum body ratio ---
    engulfing_min_ratio: float = 1.0   # engulfing body >= prev body
    hammer_wick_ratio: float = 2.0     # lower wick >= 2x body
    doji_max_ratio: float = 0.1        # body <= 10% of range

    # --- Overextension threshold (for counter-trend) ---
    ema_overextension_atr_mult: float = 3.0  # price > 3 ATR from EMA200


# Singleton config instance
CONFIG = StrategyConfig()
