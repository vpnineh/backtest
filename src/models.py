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
