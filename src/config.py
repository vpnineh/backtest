from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SymbolConfig(BaseModel):
    symbol: Literal["EURGBP", "AUDNZD"]

    pip_size: float = 0.0001
    contract_size: float = 100_000.0

    # Synthetic average spread because HistData OHLC normally does not
    # contain historical bid/ask spread.
    average_spread_pips: float
    slippage_pips_per_side: float
    commission_usd_per_lot_per_side: float

    minimum_grid_pips: float
    maximum_grid_pips: float

    # Used for sizing the first position.
    hard_stop_reference_pips: float

    # Quote-currency-to-USD conversion pair.
    conversion_symbol: Literal["GBPUSD", "NZDUSD"]


class StrategyConfig(BaseModel):
    rsi_period: int = 14
    buy_rsi: float = 28.0
    sell_rsi: float = 72.0

    bb_period: int = 20
    bb_std: float = 2.0

    atr_period: int = 14
    atr_long_period: int = 100
    atr_filter_multiple: float = 1.3

    adx_period: int = 14
    maximum_entry_adx: float = 25.0
    emergency_adx: float = 35.0

    h4_ema_period: int = 200
    h4_overextension_atr: float = 2.5

    emergency_atr_multiple: float = 1.5

    use_dynamic_grid: bool = True
    dynamic_grid_atr_multiple: float = 0.8

    lot_multipliers: tuple[float, ...] = (
        1.00,
        1.35,
        1.80,
        2.40,
        3.20,
        4.30,
        5.75,
    )

    max_grid_levels: int = 7

    basket_target_account_pct: float = 0.01
    spread_profit_multiple: float = 1.5

    initial_trade_risk_pct: float = 0.002
    maximum_margin_exposure_pct: float = 0.10

    maximum_basket_loss_pct: float = 0.025
    maximum_daily_drawdown_pct: float = 0.03
    maximum_weekly_drawdown_pct: float = 0.08

    maximum_baskets_per_direction_per_day: int = 1

    leverage: float = 100.0

    london_start_hour_utc: int = 7
    london_end_hour_utc: int = 16
    friday_cutoff_hour_utc: int = 12

    news_block_minutes_before: int = 60
    news_block_minutes_after: int = 60

    require_news_file: bool = True

    @model_validator(mode="after")
    def validate_levels(self):
        if self.max_grid_levels > len(self.lot_multipliers):
            raise ValueError(
                "max_grid_levels cannot exceed lot_multipliers length"
            )

        if self.max_grid_levels > 7:
            raise ValueError("Maximum seven grid levels are allowed.")

        return self


class BacktestConfig(BaseModel):
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")

    symbol: Literal["EURGBP", "AUDNZD"] = "EURGBP"

    start_year: int = 2010
    end_year: int = 2025

    initial_balance: float = 10_000.0

    # HistData commonly describes timestamps as EST.
    # Verify this against the exact download source.
    source_timezone: str = "Etc/GMT+5"
    output_timezone: str = "UTC"

    minimum_lot: float = 0.01
    lot_step: float = 0.01
    maximum_lot: float = 100.0

    save_cache: bool = True

    @model_validator(mode="after")
    def validate_years(self):
        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")

        if self.initial_balance < 10_000:
            raise ValueError(
                "This strategy requires at least 10,000 USD."
            )

        return self


SYMBOL_CONFIGS = {
    "EURGBP": SymbolConfig(
        symbol="EURGBP",
        average_spread_pips=1.2,
        slippage_pips_per_side=0.20,
        commission_usd_per_lot_per_side=3.5,
        minimum_grid_pips=25.0,
        maximum_grid_pips=35.0,
        hard_stop_reference_pips=150.0,
        conversion_symbol="GBPUSD",
    ),
    "AUDNZD": SymbolConfig(
        symbol="AUDNZD",
        average_spread_pips=2.0,
        slippage_pips_per_side=0.30,
        commission_usd_per_lot_per_side=3.5,
        minimum_grid_pips=30.0,
        maximum_grid_pips=40.0,
        hard_stop_reference_pips=180.0,
        conversion_symbol="NZDUSD",
    ),
}
