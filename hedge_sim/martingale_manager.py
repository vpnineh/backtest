"""
MartingaleManager
==================
Scales the LOSING side of a hedge cycle ("soft" martingale: same scale
factor as the pyramid side, not an aggressive doubling martingale).
Every time price moves `grid_distance` against the basket (beyond the
last position opened on that side), a new position is added with
lot_size = previous * scale_factor, capped by max_levels.
"""

from __future__ import annotations
from datetime import datetime

from .basket import Basket
from .configuration import StrategyConfig


class MartingaleManager:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def maybe_add(self, basket: Basket, current_price: float, current_time: datetime,
                   grid_distance_price: float, commission_per_lot: float) -> bool:
        if basket.levels == 0:
            return False
        if basket.levels - 1 >= self.cfg.max_levels:
            return False

        last = basket.positions[-1]
        adverse_move = (
            (last.entry_price - current_price) if basket.direction == "BUY"
            else (current_price - last.entry_price)
        )
        if adverse_move >= grid_distance_price:
            new_lot = round(last.lot_size * self.cfg.scale_factor, 2)
            commission = commission_per_lot * new_lot
            basket.add_position(
                entry_price=current_price,
                lot_size=new_lot,
                open_time=current_time,
                level=last.level + 1,
                kind="martingale",
                commission=commission,
            )
            return True
        return False
