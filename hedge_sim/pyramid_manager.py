from __future__ import annotations
from datetime import datetime
from .basket import Basket
from .configuration import StrategyConfig

class PyramidManager:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def maybe_add(self, basket: Basket, bid_price: float, ask_price: float, current_time: datetime,
                   grid_distance_price: float, commission_per_lot: float, slippage_pips: float) -> bool:
        if basket.levels == 0 or (basket.levels - 1) >= self.cfg.max_levels:
            return False

        last = basket.positions[-1]
        slip_price = slippage_pips * self.cfg.pip_size
        
        if basket.direction == "BUY":
            favorable_move = bid_price - last.entry_price
            exec_price = ask_price + slip_price
        else:
            favorable_move = last.entry_price - ask_price
            exec_price = bid_price - slip_price

        if favorable_move >= grid_distance_price:
            new_lot = round(last.lot_size * self.cfg.scale_factor, 2)
            commission = commission_per_lot * new_lot
            basket.add_position(exec_price, new_lot, current_time, last.level + 1, "pyramid", commission)
            return True
        return False
