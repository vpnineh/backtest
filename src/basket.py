"""
Basket (grid) management.

A Basket represents one directional grid (BUY or SELL) on one pair.
It manages multiple Position objects and tracks aggregate P&L.

No look-ahead: all decisions based on current market price passed in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Direction(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class BasketStatus(Enum):
    ACTIVE   = "ACTIVE"
    CLOSED   = "CLOSED"
    EMERGENCY= "EMERGENCY"


@dataclass
class Position:
    """Single position inside a grid basket."""
    level:      int          # 0 = first, 1 = second, etc.
    direction:  Direction
    entry_price: float
    lot_size:   float        # in standard lots
    entry_time: object       # pd.Timestamp
    pip_size:   float
    spread_pips: float

    # Filled at close
    exit_price: Optional[float] = None
    exit_time:  Optional[object] = None
    pnl_pips:   float = 0.0
    pnl_usd:    float = 0.0
    is_open:    bool  = True

    def unrealized_pips(self, current_price: float) -> float:
        """Pips profit/loss at current price (no spread on close assumed)."""
        if self.direction == Direction.BUY:
            return (current_price - self.entry_price) / self.pip_size
        else:
            return (self.entry_price - current_price) / self.pip_size

    def unrealized_usd(self, current_price: float, pip_value_per_lot: float) -> float:
        """Approximate USD P&L."""
        pips = self.unrealized_pips(current_price)
        return pips * pip_value_per_lot * self.lot_size

    def close(self, exit_price: float, exit_time, pip_value_per_lot: float):
        """Mark position as closed."""
        self.exit_price = exit_price
        self.exit_time  = exit_time
        self.is_open    = False
        self.pnl_pips   = self.unrealized_pips(exit_price)
        self.pnl_usd    = self.pnl_pips * pip_value_per_lot * self.lot_size


@dataclass
class Basket:
    """
    A grid basket: collection of positions in same direction.
    """
    basket_id:    int
    pair:         str
    direction:    Direction
    open_time:    object       # pd.Timestamp
    pip_size:     float
    spread_pips:  float
    pip_value_per_lot: float   # USD per pip per full lot

    # Grid parameters
    grid_distance_pips: float  # spacing between levels
    lot_sequence:       List[float]  # multipliers [1.0, 1.35, ...]
    base_lot:           float        # base lot size (scaled by risk)
    max_levels:         int

    # State
    positions:    List[Position] = field(default_factory=list)
    status:       BasketStatus   = BasketStatus.ACTIVE
    next_level:   int            = 0
    close_time:   object         = None
    close_reason: str            = ""

    # Protection flag: floating loss too large, no new levels
    locked:       bool           = False

    def _entry_price_for_level(self, market_price: float, level: int) -> float:
        """
        Level 0: entered at market_price.
        Level N: grid_distance_pips away from level N-1 (adverse direction).
        """
        if level == 0:
            return market_price

        grid_price = self.grid_distance_pips * self.pip_size
        if self.direction == Direction.BUY:
            # Add new BUY lower (price dropped)
            return self.positions[level - 1].entry_price - grid_price
        else:
            # Add new SELL higher (price rose)
            return self.positions[level - 1].entry_price + grid_price

    def _lot_for_level(self, level: int) -> float:
        idx = min(level, len(self.lot_sequence) - 1)
        return round(self.base_lot * self.lot_sequence[idx], 2)

    def try_add_level(
        self,
        current_price: float,
        current_time,
        current_spread_pips: float,
    ) -> Optional[Position]:
        """
        Add next grid level if price has moved grid_distance against us.
        Returns new Position if added, None otherwise.

        NO LOOK-AHEAD: uses only current_price passed in.
        """
        if self.locked:
            return None
        if self.next_level >= self.max_levels:
            return None
        if current_spread_pips > self.spread_pips * 2.0:
            logger.debug("Spread too wide, skipping new level.")
            return None

        level = self.next_level

        # Check if price moved enough to trigger next level
        if level > 0:
            last_pos = self.positions[-1]
            grid_price = self.grid_distance_pips * self.pip_size

            if self.direction == Direction.BUY:
                trigger = last_pos.entry_price - grid_price
                if current_price > trigger:
                    return None   # not yet
            else:
                trigger = last_pos.entry_price + grid_price
                if current_price < trigger:
                    return None

        # Determine entry price (market order simulation)
        if self.direction == Direction.BUY:
            entry = current_price + (current_spread_pips * self.pip_size)  # buy at ask
        else:
            entry = current_price  # sell at bid

        lot = self._lot_for_level(level)

        pos = Position(
            level=level,
            direction=self.direction,
            entry_price=entry,
            lot_size=lot,
            entry_time=current_time,
            pip_size=self.pip_size,
            spread_pips=current_spread_pips,
        )

        self.positions.append(pos)
        self.next_level += 1

        logger.debug(
            f"  [{self.pair}] Basket#{self.basket_id} Level {level}: "
            f"{self.direction.value} {lot:.2f}L @ {entry:.5f}"
        )
        return pos

    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.is_open]

    def unrealized_usd(self, current_price: float) -> float:
        return sum(
            p.unrealized_usd(current_price, self.pip_value_per_lot)
            for p in self.open_positions()
        )

    def realized_usd(self) -> float:
        return sum(p.pnl_usd for p in self.positions if not p.is_open)

    def total_pnl_usd(self, current_price: float) -> float:
        return self.unrealized_usd(current_price) + self.realized_usd()

    def average_entry(self) -> float:
        """Lot-weighted average entry price."""
        open_pos = self.open_positions()
        if not open_pos:
            return 0.0
        total_lots = sum(p.lot_size for p in open_pos)
        if total_lots == 0:
            return 0.0
        return sum(p.entry_price * p.lot_size for p in open_pos) / total_lots

    def total_lots(self) -> float:
        return sum(p.lot_size for p in self.open_positions())

    def close_all(
        self,
        exit_price: float,
        exit_time,
        reason: str,
        current_spread_pips: float,
    ):
        """Close all open positions."""
        for pos in self.open_positions():
            if pos.direction == Direction.BUY:
                actual_exit = exit_price  # sell at bid
            else:
                actual_exit = exit_price + (current_spread_pips * self.pip_size)  # buy back at ask

            pos.close(actual_exit, exit_time, self.pip_value_per_lot)

        self.status     = BasketStatus.CLOSED if reason != "EMERGENCY" else BasketStatus.EMERGENCY
        self.close_time = exit_time
        self.close_reason = reason

    def check_protection(self, current_price: float, expected_tp_usd: float):
        """
        Lock basket if floating loss > 2x expected TP.
        """
        pnl = self.total_pnl_usd(current_price)
        if pnl < -(2.0 * expected_tp_usd):
            if not self.locked:
                logger.debug(
                    f"  Basket#{self.basket_id} LOCKED: "
                    f"loss {pnl:.2f} > 2x TP {expected_tp_usd:.2f}"
                )
            self.locked = True

    def summary(self) -> dict:
        """Return basket statistics."""
        total_pnl = sum(p.pnl_usd for p in self.positions)
        return {
            "basket_id":    self.basket_id,
            "pair":         self.pair,
            "direction":    self.direction.value,
            "open_time":    self.open_time,
            "close_time":   self.close_time,
            "levels_used":  len(self.positions),
            "total_lots":   sum(p.lot_size for p in self.positions),
            "pnl_usd":      total_pnl,
            "status":       self.status.value,
            "close_reason": self.close_reason,
        }
