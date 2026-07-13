"""
Trade (Position)
=================
A single open position belonging to one basket (BUY side or SELL side).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    entry_price: float
    lot_size: float
    direction: str            # "BUY" or "SELL"
    open_time: datetime
    level: int                 # 0 = initial hedge position, 1..N = grid level
    kind: str                  # "hedge" | "pyramid" | "martingale"
    commission: float = 0.0    # charged at open (round-turn amortized at close too if desired)
    swap_accrued: float = 0.0
    close_price: float | None = None
    close_time: datetime | None = None
    closed: bool = False

    def floating_pnl(self, current_price: float, pip_size: float, contract_size: float,
                      quote_to_account_rate: float = 1.0) -> float:
        """Floating P/L in account currency (simplified constant conversion rate)."""
        sign = 1 if self.direction == "BUY" else -1
        price_diff = (current_price - self.entry_price) * sign
        pnl = price_diff * self.lot_size * contract_size
        return pnl * quote_to_account_rate - self.commission + self.swap_accrued

    def accrue_swap(self, swap_long_per_lot: float, swap_short_per_lot: float, nights: int = 1):
        rate = swap_long_per_lot if self.direction == "BUY" else swap_short_per_lot
        self.swap_accrued += rate * self.lot_size * nights

    def close(self, price: float, time: datetime):
        self.close_price = price
        self.close_time = time
        self.closed = True
