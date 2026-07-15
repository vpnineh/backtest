from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    entry_time: pd.Timestamp
    side: int
    lot: float
    entry_price: float
    commission_entry: float
    level: int


@dataclass
class Basket:
    basket_id: int
    side: int
    start_time: pd.Timestamp
    start_balance: float
    grid_distance_pips: float

    positions: list[Position] = field(default_factory=list)

    additions_disabled: bool = False
    total_entry_spread_cost: float = 0.0
    total_entry_slippage_cost: float = 0.0
    total_commission: float = 0.0

    @property
    def levels(self) -> int:
        return len(self.positions)

    @property
    def total_lots(self) -> float:
        return sum(position.lot for position in self.positions)

    @property
    def last_entry_price(self) -> float:
        return self.positions[-1].entry_price


@dataclass
class ClosedBasket:
    basket_id: int
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    levels: int
    total_lots: float
    gross_pnl: float
    net_pnl: float
    exit_reason:
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    entry_time: pd.Timestamp
    side: int
    lot: float

    # Actual executable price, including spread/slippage.
    entry_price: float

    # Raw historical Bid price at which the grid level was triggered.
    reference_price: float

    entry_conversion_rate: float
    commission_entry: float
    estimated_spread_cost: float
    entry_slippage_cost: float
    level: int


@dataclass
class Basket:
    basket_id: int
    side: int
    start_time: pd.Timestamp
    start_balance: float
    initial_lot: float
    grid_distance_pips: float

    positions: list[Position] = field(default_factory=list)

    additions_disabled: bool = False
    additions_disabled_reason: str | None = None

    total_estimated_spread_cost: float = 0.0
    total_slippage_cost: float = 0.0
    total_commission: float = 0.0

    @property
    def levels(self) -> int:
        return len(self.positions)

    @property
    def total_lots(self) -> float:
        return sum(position.lot for position in self.positions)

    @property
    def last_entry_price(self) -> float:
        return self.positions[-1].entry_price

    @property
    def last_reference_price(self) -> float:
        return self.positions[-1].reference_price


@dataclass
class ClosedBasket:
    basket_id: int
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp

    levels: int
    total_lots: float

    gross_pnl: float
    net_pnl: float

    spread_cost: float
    slippage_cost: float
    commission: float

    maximum_floating_loss: float
    duration_minutes: float
    exit_reason: str

    balance_after: float


@dataclass
class EquityPoint:
    datetime: pd.Timestamp
    balance: float
    equity: float
    floating_pnl: float
    open_levels: int
