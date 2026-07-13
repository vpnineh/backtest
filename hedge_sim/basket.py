"""
Basket
======
Holds every open Position for ONE side (BUY or SELL) of a hedge cycle.
Tracks weighted-average price, total lots and aggregate floating P/L.
"""

from __future__ import annotations
from datetime import datetime

from .trade import Position


class Basket:
    def __init__(self, direction: str):
        self.direction = direction  # "BUY" or "SELL"
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []

    # ---- state -----------------------------------------------------
    @property
    def total_lots(self) -> float:
        return sum(p.lot_size for p in self.positions)

    @property
    def levels(self) -> int:
        return len(self.positions)

    @property
    def weighted_avg_price(self) -> float:
        lots = self.total_lots
        if lots == 0:
            return 0.0
        return sum(p.entry_price * p.lot_size for p in self.positions) / lots

    def floating_pnl(self, current_price: float, pip_size: float, contract_size: float,
                      quote_to_account_rate: float = 1.0) -> float:
        return sum(
            p.floating_pnl(current_price, pip_size, contract_size, quote_to_account_rate)
            for p in self.positions
        )

    def is_profitable(self, current_price: float, pip_size: float, contract_size: float,
                       quote_to_account_rate: float = 1.0) -> bool:
        return self.floating_pnl(current_price, pip_size, contract_size, quote_to_account_rate) > 0

    # ---- mutation ----------------------------------------------------
    def add_position(self, entry_price: float, lot_size: float, open_time: datetime,
                      level: int, kind: str, commission: float = 0.0) -> Position:
        pos = Position(
            entry_price=entry_price,
            lot_size=lot_size,
            direction=self.direction,
            open_time=open_time,
            level=level,
            kind=kind,
            commission=commission,
        )
        self.positions.append(pos)
        return pos

    def close_all(self, price: float, time: datetime):
        for p in self.positions:
            p.close(price, time)
            self.closed_positions.append(p)
        self.positions = []

    def accrue_daily_swap(self, swap_long_per_lot: float, swap_short_per_lot: float):
        for p in self.positions:
            p.accrue_swap(swap_long_per_lot, swap_short_per_lot)
