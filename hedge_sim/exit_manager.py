"""
ExitManager
===========
Interchangeable exit algorithms. Every mode implements `should_close(...)`
and returns (bool, reason:str). No fixed take-profit / stop-loss is used
anywhere in this module - purely research-oriented exit logic as required.

Exit Mode A - Close when BOTH baskets are individually profitable.
Exit Mode B - Close when COMBINED floating profit exceeds a target.
Exit Mode C - Close when the two baskets' weighted-average prices converge
              to within `convergence_pips`.
Exit Mode D - Close using a dynamically calculated mathematical equilibrium:
              the price at which total combined P/L (both baskets) is exactly
              zero (breakeven point), reached or crossed with a small buffer.
"""

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
    def should_close(self, buy_basket: Basket, sell_basket: Basket, current_price: float) -> tuple[bool, str]:
        ...

    def _pnl(self, basket: Basket, price: float, quote_to_account_rate: float) -> float:
        return basket.floating_pnl(price, self.strat_cfg.pip_size, self.strat_cfg.contract_size,
                                    quote_to_account_rate)


class ExitModeA(ExitStrategy):
    """Both baskets individually profitable."""
    name = "A"

    def should_close(self, buy_basket, sell_basket, current_price, quote_to_account_rate=1.0):
        buy_pnl = self._pnl(buy_basket, current_price, quote_to_account_rate)
        sell_pnl = self._pnl(sell_basket, current_price, quote_to_account_rate)
        if buy_basket.levels and sell_basket.levels and buy_pnl > 0 and sell_pnl > 0:
            return True, "both_baskets_profitable"
        return False, ""


class ExitModeB(ExitStrategy):
    """Combined floating profit exceeds target."""
    name = "B"

    def should_close(self, buy_basket, sell_basket, current_price, quote_to_account_rate=1.0):
        total = self._pnl(buy_basket, current_price, quote_to_account_rate) + \
                self._pnl(sell_basket, current_price, quote_to_account_rate)
        if total >= self.exit_cfg.target_profit:
            return True, "combined_target_reached"
        return False, ""


class ExitModeC(ExitStrategy):
    """Weighted-average prices of the two baskets converge.

    BUG FIX: right after a fresh hedge open, avg_buy and avg_sell are only
    `spread` apart - which already satisfies most convergence thresholds,
    causing an instant open/close loop. We require that the basket has
    genuinely grown beyond the initial hedge pair (at least one grid level
    added on either side) before convergence is evaluated, since
    "convergence" is only a meaningful signal once there was real divergence.
    """
    name = "C"

    def should_close(self, buy_basket, sell_basket, current_price, quote_to_account_rate=1.0):
        if buy_basket.levels == 0 or sell_basket.levels == 0:
            return False, ""
        if buy_basket.levels <= 1 and sell_basket.levels <= 1:
            return False, ""  # still just the initial hedge pair - nothing has diverged yet
        gap = abs(buy_basket.weighted_avg_price - sell_basket.weighted_avg_price)
        gap_pips = gap / self.strat_cfg.pip_size
        if gap_pips <= self.exit_cfg.convergence_pips:
            return True, "weighted_avg_converged"
        return False, ""


class ExitModeD(ExitStrategy):
    """Mathematical equilibrium: combined P/L breakeven crossing.

    Solves for the price at which total P/L of both baskets = 0, given
    current lot exposure, then closes once price has reached/crossed
    that equilibrium price (with a small buffer), OR once combined P/L
    is already >= 0 (equilibrium already achieved).
    """
    name = "D"

    def should_close(self, buy_basket, sell_basket, current_price, quote_to_account_rate=1.0):
        total = self._pnl(buy_basket, current_price, quote_to_account_rate) + \
                self._pnl(sell_basket, current_price, quote_to_account_rate)
        if buy_basket.levels == 0 and sell_basket.levels == 0:
            return False, ""
        if total >= 0:
            buffer_price = self.exit_cfg.equilibrium_buffer_pips * self.strat_cfg.pip_size
            # require we've cleared breakeven by at least the buffer in P/L terms
            net_lots = buy_basket.total_lots - sell_basket.total_lots
            if abs(net_lots) < 1e-9 or total >= abs(net_lots) * self.strat_cfg.contract_size * buffer_price:
                return True, "equilibrium_breakeven_reached"
        return False, ""


_MODES = {
    "A": ExitModeA,
    "B": ExitModeB,
    "C": ExitModeC,
    "D": ExitModeD,
}


def build_exit_strategy(exit_cfg: ExitConfig, strat_cfg: StrategyConfig) -> ExitStrategy:
    mode = exit_cfg.mode.upper().strip()
    if mode not in _MODES:
        raise ValueError(f"Unknown exit mode '{mode}'. Valid options: {list(_MODES.keys())}")
    return _MODES[mode](exit_cfg, strat_cfg)
