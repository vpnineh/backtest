"""
PyramidManager
==============
Scales UP the winning side of a hedge cycle. Every time price moves
`grid_distance` in favor of the basket (beyond the last position opened
on that side), a new position is added with lot_size = previous * scale_factor.
"""

from __future__ import annotations
from datetime import datetime

from .basket import Basket
from .configuration import StrategyConfig


class PyramidManager:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def maybe_add(self, basket: Basket, current_price: float, current_time: datetime,
                   grid_distance_price: float, commission_per_lot: float) -> bool:
        """Returns True if a new pyramid position was opened."""
        if basket.levels == 0:
            return False
        if basket.levels - 1 >= self.cfg.max_levels:  # -1 excludes the initial hedge slot
            return False

        last = basket.positions[-1]
        favorable_move = (
            (current_price - last.entry_price) if basket.direction == "BUY"
            else (last.entry_price - current_price)
        )
        if favorable_move >= grid_distance_price:
            new_lot = round(last.lot_size * self.cfg.scale_factor, 2)
            commission = commission_per_lot * new_lot
            basket.add_position(
                entry_price=current_price,
                lot_size=new_lot,
                open_time=current_time,
                level=last.level + 1,
                kind="pyramid",
                commission=commission,
            )
            return True
        return False
