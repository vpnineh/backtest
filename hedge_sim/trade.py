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
    direction: str            
    open_time: datetime
    level: int                 
    kind: str                  
    commission: float = 0.0    
    swap_accrued: float = 0.0
    close_price: float | None = None
    close_time: datetime | None = None
    closed: bool = False

    def floating_pnl(self, current_bid: float, current_ask: float, pip_size: float, contract_size: float,
                      quote_to_account_rate: float = 1.0) -> float:
        """Realistic Floating P/L: BUY uses BID to close, SELL uses ASK to close."""
        if self.direction == "BUY":
            price_diff = (current_bid - self.entry_price)
        else:
            price_diff = (self.entry_price - current_ask)
            
        pnl = price_diff * self.lot_size * contract_size
        return pnl * quote_to_account_rate - self.commission + self.swap_accrued

    def accrue_swap(self, swap_long_per_lot: float, swap_short_per_lot: float, nights: int = 1):
        rate = swap_long_per_lot if self.direction == "BUY" else swap_short_per_lot
        self.swap_accrued += rate * self.lot_size * nights

    def close(self, price: float, time: datetime):
        self.close_price = price
        self.close_time = time
        self.closed = True
