from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BacktestConfig, StrategyConfig, SymbolConfig
from src.features import FeatureSet
from src.models import Basket, ClosedBasket, EquityPoint, Position
from src.news import NewsFilter


class ConversionRateCursor:
    """
    Provides quote-currency -> USD conversion without accessing future data.

    EURGBP uses GBPUSD.
    AUDNZD uses NZDUSD.
    """

    def __init__(self, conversion_data: pd.DataFrame):
        if conversion_data.empty:
            raise ValueError("Conversion-rate data is empty.")

        data = conversion_data.sort_index()

        self.timestamps = data.index.asi8
        self.open_prices = data["open"].to_numpy(dtype=np.float64)
        self.close_prices = data["close"].to_numpy(dtype=np.float64)

        self.position = -1

    def value_at(
        self,
        timestamp: pd.Timestamp,
        use_close: bool = False,
    ) -> float:
        target = timestamp.value

        while (
            self.position + 1 < len(self.timestamps)
            and self.timestamps[self.position + 1] <= target
        ):
            self.position += 1

        if self.position < 0:
            raise RuntimeError(
                f"No conversion rate available at or before {timestamp}"
            )

        value = (
            self.close_prices[self.position]
            if use_close
            else self.open_prices[self.position]
        )

        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(
                f"Invalid conversion rate at {timestamp}: {value}"
            )

        return float(value)


class ClosedFeatureCursor:
    """
    Makes only feature rows whose close timestamp is <= current time available.
    """

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.sort_index()
        self.timestamps = self.frame.index.asi8
        self.records = self.frame.to_dict("records")

        self.position = -1
        self.current: dict[str, Any] | None = None
        self.current_timestamp: pd.Timestamp | None = None

    def update(self, timestamp: pd.Timestamp) -> bool:
        changed = False
        target = timestamp.value

        while (
            self.position + 1 < len(self.timestamps)
            and self.timestamps[self.position + 1] <= target
        ):
            self.position += 1
            self.current = self.records[self.position]
            self.current_timestamp = self.frame.index[self.position]
            changed = True

        return changed


class MartingaleBacktestEngine:
    """
    Event-driven M1 basket/grid backtester.

    Important assumptions
    ---------------------
    1. HistData OHLC is treated as Bid data.
    2. Ask = Bid + modelled spread.
    3. M15/H4/D1 indicators become available only at bar close.
    4. Entry signals are executed at the first M1 open at or after signal close.
    5. At most one grid level can be added per M1 candle.
    6. If grid addition and basket target are both touched in one M1 candle,
       adverse movement is processed first and profit exit is deferred. This is
       intentionally conservative because OHLC cannot reveal tick ordering.
    """

    def __init__(
        self,
        backtest: BacktestConfig,
        strategy: StrategyConfig,
        symbol: SymbolConfig,
        news_filter: NewsFilter,
    ):
        self.backtest = backtest
        self.strategy = strategy
        self.symbol = symbol
        self.news_filter = news_filter

        self.balance = backtest.initial_balance
        self.basket: Basket | None = None

        self.closed_baskets: list[ClosedBasket] = []
        self.equity_points: list[EquityPoint] = []

        self.next_basket_id = 1

        self.total_spread_cost = 0.0
        self.total_slippage_cost = 0.0
        self.total_commission = 0.0

        self.current_day = None
        self.current_week = None

        self.daily_start_equity = backtest.initial_balance
        self.weekly_start_equity = backtest.initial_balance

        self.daily_trading_disabled = False
        self.weekly_trading_disabled = False

        self.daily_direction_counts = {
            1: 0,
            -1: 0,
        }

        self.maximum_margin_used = 0.0
        self.maximum_margin_pct = 0.0

        self.maximum_open_levels = 0
        self.maximum_floating_loss_current_basket = 0.0

        self.last_processed_signal_timestamp = None

        self.rejected_signals: dict[str, int] = {}

    def _reject(self, reason: str) -> None:
        self.rejected_signals[reason] = (
            self.rejected_signals.get(reason, 0) + 1
        )

    def _round_lot_down(self, lot: float) -> float:
        step = self.backtest.lot_step

        rounded = math.floor((lot + 1e-12) / step) * step
        rounded = min(rounded, self.backtest.maximum_lot)

        if rounded < self.backtest.minimum_lot:
            return 0.0

        return round(rounded, 8)

    def _pip_value_usd_per_lot(
        self,
        conversion_rate: float,
    ) -> float:
        # A pip is denominated in the quote currency.
        return (
            self.symbol.contract_size
            * self.symbol.pip_size
            * conversion_rate
        )

    def _spread_pips(
        self,
        timestamp: pd.Timestamp,
        m15: dict[str, Any] | None,
    ) -> float:
        """
        Synthetic spread model.

        It is deliberately wider outside liquid hours and during elevated ATR.
        This is still not a substitute for historical Bid/Ask data.
        """

        spread = self.symbol.average_spread_pips
        hour = timestamp.hour

        # Rollover and very low liquidity.
        if hour in {21, 22}:
            spread *= 2.5
        elif hour < 6 or hour >= 18:
            spread *= 1.6

        if timestamp.weekday() == 4 and hour >= 16:
            spread *= 1.5

        if m15 is not None:
            current_atr = m15.get("atr")
            long_atr = m15.get("atr_long")

            if (
                current_atr is not None
                and long_atr is not None
                and np.isfinite(current_atr)
                and np.isfinite(long_atr)
                and long_atr > 0
            ):
                volatility_ratio = current_atr / long_atr

                if volatility_ratio > 1.0:
                    spread *= min(1.75, volatility_ratio)

        return float(spread)

    def _entry_execution_price(
        self,
        raw_bid: float,
        side: int,
        spread_pips: float,
    ) -> float:
        spread_price = spread_pips * self.symbol.pip_size
        slippage_price = (
            self.symbol.slippage_pips_per_side
            * self.symbol.pip_size
        )

        if side == 1:
            # Buy at Ask plus adverse slippage.
            return raw_bid + spread_price + slippage_price

        # Sell at Bid minus adverse slippage.
        return raw_bid - slippage_price

    def _exit_execution_price(
        self,
        raw_bid: float,
        side: int,
        spread_pips: float,
    ) -> float:
        spread_price = spread_pips * self.symbol.pip_size
        slippage_price = (
            self.symbol.slippage_pips_per_side
            * self.symbol.pip_size
        )

        if side == 1:
            # Close Buy by selling at Bid.
            return raw_bid - slippage_price

        # Close Sell by buying at Ask.
        return raw_bid + spread_price + slippage_price

    def _position_liquidation_pnl(
        self,
        position: Position,
        raw_bid: float,
        spread_pips: float,
        conversion_rate: float,
        include_exit_commission: bool = True,
    ) -> float:
        exit_price = self._exit_execution_price(
            raw_bid,
            position.side,
            spread_pips,
        )

        price_difference = (
            exit_price - position.entry_price
        ) * position.side

        quote_currency_pnl = (
            price_difference
            * self.symbol.contract_size
            * position.lot
        )

        pnl_usd = quote_currency_pnl * conversion_rate

        if include_exit_commission:
            pnl_usd -= (
                self.symbol.commission_usd_per_lot_per_side
                * position.lot
            )

        return pnl_usd

    def _basket_floating_pnl(
        self,
        raw_bid: float,
        spread_pips: float,
        conversion_rate: float,
    ) -> float:
        if self.basket is None:
            return 0.0

        return sum(
            self._position_liquidation_pnl(
                position,
                raw_bid,
                spread_pips,
                conversion_rate,
                include_exit_commission=True,
            )
            for position in self.basket.positions
        )

    def _equity(
        self,
        raw_bid: float,
        spread_pips: float,
        conversion_rate: float,
    ) -> float:
        return self.balance + self._basket_floating_pnl(
            raw_bid,
            spread_pips,
            conversion_rate,
        )

    def _base_currency_usd_rate(
        self,
        pair_price: float,
        quote_to_usd: float,
    ) -> float:
        return pair_price * quote_to_usd

    def _margin_required(
        self,
        total_lots: float,
        pair_price: float,
        conversion_rate: float,
    ) -> float:
        base_to_usd = self._base_currency_usd_rate(
            pair_price,
            conversion_rate,
        )

        notional_usd = (
            total_lots
            * self.symbol.contract_size
            * base_to_usd
        )

        return notional_usd / self.strategy.leverage

    def _margin_allowed(
        self,
        additional_lot: float,
        pair_price: float,
        conversion_rate: float,
        equity: float,
    ) -> bool:
        current_lots = (
            self.basket.total_lots
            if self.basket is not None
            else 0.0
        )

        margin = self._margin_required(
            current_lots + additional_lot,
            pair_price,
            conversion_rate,
        )

        maximum_margin = (
            equity * self.strategy.maximum_margin_exposure_pct
