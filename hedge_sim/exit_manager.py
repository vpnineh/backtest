from __future__ import annotations
from abc import ABC, abstractmethod

from .basket import Basket
from .configuration import ExitConfig, StrategyConfig


class ExitStrategy(ABC):
    name = "base"

    def __init__(self, exit_cfg: ExitConfig, strat_cfg: StrategyConfig):
        self.exit_cfg = exit_cfg
        self.strat_cfg = strat_cfg

    @abstractmethod
    def should_close(self, buy_basket: Basket, sell_basket: Basket, bid_price: float, ask_price: float, quote_rate: float) -> tuple[bool, str]:
        ...

    def _pnl(self, basket: Basket, bid_price: float, ask_price: float, quote_rate: float) -> float:
        return basket.floating_pnl(bid_price, ask_price, self.strat_cfg.pip_size, self.strat_cfg.contract_size, quote_rate)

class ExitModeA(ExitStrategy):
    name = "A"
    def should_close(self, buy_basket, sell_basket, bid_price, ask_price, quote_rate=1.0):
        buy_pnl = self._pnl(buy_basket, bid_price, ask_price, quote_rate)
        sell_pnl = self._pnl(sell_basket, bid_price, ask_price, quote_rate)
        if buy_basket.levels and sell_basket.levels and buy_pnl > 0 and sell_pnl > 0:
            return True, "both_baskets_profitable"
        return False, ""

class ExitModeB(ExitStrategy):
    name = "B"
    def should_close(self, buy_basket, sell_basket, bid_price, ask_price, quote_rate=1.0):
        total = self._pnl(buy_basket, bid_price, ask_price, quote_rate) + \
                self._pnl(sell_basket, bid_price, ask_price, quote_rate)
        if total >= self.exit_cfg.target_profit:
            return True, "combined_target_reached"
        return False, ""

class ExitModeC(ExitStrategy):
    name = "C"
    def should_close(self, buy_basket, sell_basket, bid_price, ask_price, quote_rate=1.0):
        if buy_basket.levels == 0 or sell_basket.levels == 0:
            return False, ""
        gap = abs(buy_basket.weighted_avg_price - sell_basket.weighted_avg_price)
        if gap / self.strat_cfg.pip_size <= self.exit_cfg.convergence_pips:
            return True, "weighted_avg_converged"
        return False, ""

class ExitModeD(ExitStrategy):
    name = "D"
    def should_close(self, buy_basket, sell_basket, bid_price, ask_price, quote_rate=1.0):
        total = self._pnl(buy_basket, bid_price, ask_price, quote_rate) + \
                self._pnl(sell_basket, bid_price, ask_price, quote_rate)
        if buy_basket.levels == 0 and sell_basket.levels == 0:
            return False, ""
        if total >= 0:
            buffer_price = self.exit_cfg.equilibrium_buffer_pips * self.strat_cfg.pip_size
            net_lots = buy_basket.total_lots - sell_basket.total_lots
            if abs(net_lots) < 1e-9 or total >= abs(net_lots) * self.strat_cfg.contract_size * buffer_price:
                return True, "equilibrium_breakeven_reached"
        return False, ""

_MODES = {"A": ExitModeA, "B": ExitModeB, "C": ExitModeC, "D": ExitModeD}

def build_exit_strategy(exit_cfg: ExitConfig, strat_cfg: StrategyConfig) -> ExitStrategy:
    mode = exit_cfg.mode.upper().strip()
    return _MODES[mode](exit_cfg, strat_cfg)
