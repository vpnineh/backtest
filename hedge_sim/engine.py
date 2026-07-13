"""
BacktestEngine
==============
Runs the dual-directional hedged grid strategy over historical OHLC data
(M1 or any resampled timeframe such as M5) with a REALISTIC intrabar
price path model, not just bar-close snapshots.

Why intrabar path matters
--------------------------
A single OHLC bar hides the true order in which price visited its open,
high, low and close. Naively checking triggers only against bar-close
(or against high/low without an order) is optimistic: it can silently
assume the strategy got the best of both worlds inside one candle. This
engine instead builds a conservative, deterministic checkpoint path for
every bar:

    if (open - low) <= (high - open):      path = [open, low, high, close]
    else:                                    path = [open, high, low, close]

i.e. price is assumed to travel first toward whichever extreme is
closer to the open (the standard, conservative convention used by most
serious backtesters when only OHLC - not tick - data is available).
Every grid trigger, pyramid/martingale add, and exit-condition check is
evaluated IN ORDER along this path, so:
  - multiple grid levels within one volatile bar are only added if the
    path actually would have crossed each one, in the correct order;
  - exit conditions can fire mid-bar (not only at bar close), and a new
    cycle can legitimately open and close more than once within the
    same bar during fast markets;
  - which basket is "winning" vs "losing" is re-evaluated at every
    checkpoint, so a reversal mid-bar is handled correctly.

No forced stop loss is implemented anywhere. Worst-case metrics
(max floating DD, worst equity, largest basket, max exposure,
maximum adverse excursion, recovery distance/time) are recorded
instead, exactly as required by the research specification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .basket import Basket
from .configuration import Config
from .exit_manager import build_exit_strategy
from .martingale_manager import MartingaleManager
from .pyramid_manager import PyramidManager


@dataclass
class CycleRecord:
    cycle_id: int
    open_time: datetime
    close_time: datetime | None
    open_price: float
    exit_price: float | None
    direction_movement: str            # "UP" | "DOWN" | "FLAT" - net price movement open->close
    realized_pnl: float | None
    max_levels_buy: int
    max_levels_sell: int
    num_pyramid_positions: int
    num_martingale_positions: int
    total_buy_lots: float
    total_sell_lots: float
    avg_buy_price: float
    avg_sell_price: float
    worst_floating_pnl: float          # == -1 * Maximum Adverse Excursion (MAE)
    worst_floating_time: datetime | None
    worst_price: float | None
    best_floating_pnl: float           # == Maximum Favorable Excursion (MFE)
    best_floating_time: datetime | None
    best_price: float | None
    breakeven_price: float | None      # equilibrium price at the worst-excursion snapshot
    recovery_distance_pips: float | None
    recovery_percentage: float | None
    recovery_time_seconds: float | None
    closed: bool
    exit_reason: str = ""


@dataclass
class SimulationState:
    equity_curve: list = field(default_factory=list)   # (time, equity)
    balance_curve: list = field(default_factory=list)  # (time, balance)
    floating_dd_curve: list = field(default_factory=list)
    margin_curve: list = field(default_factory=list)
    open_lots_curve: list = field(default_factory=list)
    exposure_curve: list = field(default_factory=list)
    basket_size_curve: list = field(default_factory=list)  # max(levels buy, levels sell)
    cycles: list = field(default_factory=list)  # CycleRecord
    trade_log: list = field(default_factory=list)  # every open position, dict form


class BacktestEngine:
    def __init__(self, config: Config, data: pd.DataFrame):
        self.cfg = config
        self.data = data
        self.pyramid_mgr = PyramidManager(config.strategy)
        self.martingale_mgr = MartingaleManager(config.strategy)
        self.exit_strategy = build_exit_strategy(config.exit, config.strategy)

        self.balance = config.account.starting_balance
        self.state = SimulationState()

        self._buy: Basket | None = None
        self._sell: Basket | None = None
        self._cycle_id = 0
        self._cycle_open_time: datetime | None = None
        self._cycle_open_price: float | None = None
        self._cycle_worst_pnl = 0.0
        self._cycle_worst_time: datetime | None = None
        self._cycle_worst_price: float | None = None
        self._cycle_worst_snapshot = None  # (avg_buy, buy_lots, avg_sell, sell_lots)
        self._cycle_best_pnl = 0.0
        self._cycle_best_time: datetime | None = None
        self._cycle_best_price: float | None = None
        self._cycle_recovered = False
        self._last_day = None
        self._peak_equity = self.balance

    # ------------------------------------------------------------------
    @staticmethod
    def _bar_path(o: float, h: float, l: float, c: float) -> list[float]:
        """Conservative, deterministic intrabar checkpoint sequence."""
        if h == l == o == c:
            return [o]
        if (o - l) <= (h - o):
            path = [o, l, h, c]
        else:
            path = [o, h, l, c]
        # collapse consecutive duplicate checkpoints (flat/degenerate bars)
        out = [path[0]]
        for p in path[1:]:
            if p != out[-1]:
                out.append(p)
        return out

    def _grid_distance_price(self, atr_value: float | None) -> float:
        cfg = self.cfg.strategy
        if cfg.grid_mode == "atr" and atr_value is not None and not pd.isna(atr_value):
            return max(atr_value * cfg.atr_multiplier, cfg.pip_size)
        return cfg.grid_distance_pips * cfg.pip_size

    def _open_cycle(self, price: float, time: datetime):
        self._cycle_id += 1
        self._buy = Basket("BUY")
        self._sell = Basket("SELL")
        commission = self.cfg.costs.commission_per_lot * self.cfg.strategy.initial_lot
        spread_price = self.cfg.costs.spread_pips * self.cfg.strategy.pip_size
        slip_price = self.cfg.costs.slippage_pips * self.cfg.strategy.pip_size

        buy_price = price + spread_price / 2 + slip_price
        sell_price = price - spread_price / 2 - slip_price

        self._buy.add_position(buy_price, self.cfg.strategy.initial_lot, time, 0, "hedge", commission)
        self._sell.add_position(sell_price, self.cfg.strategy.initial_lot, time, 0, "hedge", commission)

        self._cycle_open_time = time
        self._cycle_open_price = price
        self._cycle_worst_pnl = 0.0
        self._cycle_worst_time = time
        self._cycle_worst_price = price
        self._cycle_worst_snapshot = (buy_price, self.cfg.strategy.initial_lot,
                                       sell_price, self.cfg.strategy.initial_lot)
        self._cycle_best_pnl = 0.0
        self._cycle_best_time = time
        self._cycle_best_price = price
        self._cycle_recovered = False

    def _pnls(self, price: float) -> tuple[float, float]:
        qtr = self.cfg.account.quote_to_account_rate
        buy_pnl = self._buy.floating_pnl(price, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)
        sell_pnl = self._sell.floating_pnl(price, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)
        return buy_pnl, sell_pnl

    def _margin_used(self, price: float) -> float:
        lots = self._buy.total_lots + self._sell.total_lots
        notional = lots * self.cfg.strategy.contract_size * price
        return notional / max(self.cfg.account.leverage, 1e-9)

    @staticmethod
    def _breakeven_price(avg_buy: float, buy_lots: float, avg_sell: float, sell_lots: float,
                          contract_size: float) -> float | None:
        """Solves buy_pnl(P) + sell_pnl(P) = 0 for P (ignoring commission/swap,
        which are already realized and small relative to price-driven P/L)."""
        net_lots = buy_lots - sell_lots
        if abs(net_lots) < 1e-9:
            return None  # P/L independent of price at this exact lot balance - no single breakeven price
        numerator = avg_buy * buy_lots - avg_sell * sell_lots
        return numerator / net_lots

    def _record_curves(self, time, price):
        buy_pnl, sell_pnl = self._pnls(price)
        equity = self.balance + buy_pnl + sell_pnl
        self._peak_equity = max(self._peak_equity, equity)
        floating_dd = self._peak_equity - equity

        self.state.equity_curve.append((time, equity))
        self.state.balance_curve.append((time, self.balance))
        self.state.floating_dd_curve.append((time, floating_dd))
        self.state.margin_curve.append((time, self._margin_used(price)))
        self.state.open_lots_curve.append((time, self._buy.total_lots + self._sell.total_lots))
        self.state.exposure_curve.append((time, abs(self._buy.total_lots - self._sell.total_lots)))
        self.state.basket_size_curve.append((time, max(self._buy.levels, self._sell.levels)))
        return buy_pnl + sell_pnl

    def _update_worst(self, floating_pnl: float, price: float, time: datetime):
        if floating_pnl < self._cycle_worst_pnl:
            self._cycle_worst_pnl = floating_pnl
            self._cycle_worst_time = time
            self._cycle_worst_price = price
            self._cycle_worst_snapshot = (
                self._buy.weighted_avg_price, self._buy.total_lots,
                self._sell.weighted_avg_price, self._sell.total_lots,
            )
            self._cycle_recovered = False
        elif floating_pnl >= 0 and not self._cycle_recovered and self._cycle_worst_pnl < 0:
            self._cycle_recovered = True

        if floating_pnl > self._cycle_best_pnl:
            self._cycle_best_pnl = floating_pnl
            self._cycle_best_time = time
            self._cycle_best_price = price

    def _close_cycle(self, price: float, time: datetime, reason: str):
        buy_pnl, sell_pnl = self._pnls(price)
        realized = buy_pnl + sell_pnl
        self.balance += realized

        total_buy_lots = self._buy.total_lots
        total_sell_lots = self._sell.total_lots
        avg_buy_price = self._buy.weighted_avg_price
        avg_sell_price = self._sell.weighted_avg_price
        max_level_buy = max((p.level for p in self._buy.positions), default=0)
        max_level_sell = max((p.level for p in self._sell.positions), default=0)
        num_pyramid = sum(1 for p in self._buy.positions if p.kind == "pyramid") + \
                      sum(1 for p in self._sell.positions if p.kind == "pyramid")
        num_martingale = sum(1 for p in self._buy.positions if p.kind == "martingale") + \
                         sum(1 for p in self._sell.positions if p.kind == "martingale")

        move = price - self._cycle_open_price if self._cycle_open_price is not None else 0.0
        pip_threshold = 0.1 * self.cfg.strategy.pip_size
        direction = "UP" if move > pip_threshold else "DOWN" if move < -pip_threshold else "FLAT"

        for basket in (self._buy, self._sell):
            for p in basket.positions:
                self.state.trade_log.append({
                    "cycle_id": self._cycle_id, "direction": p.direction, "kind": p.kind,
                    "level": p.level, "entry_price": p.entry_price, "lot_size": p.lot_size,
                    "open_time": p.open_time, "close_time": time, "close_price": price,
                    "commission": p.commission, "swap": p.swap_accrued,
                })
            basket.close_all(price, time)

        recovery_seconds = None
        if self._cycle_recovered and self._cycle_worst_time:
            recovery_seconds = (time - self._cycle_worst_time).total_seconds()

        breakeven_price = None
        recovery_distance_pips = None
        if self._cycle_worst_snapshot is not None:
            ab, bl, as_, sl = self._cycle_worst_snapshot
            breakeven_price = self._breakeven_price(ab, bl, as_, sl, self.cfg.strategy.contract_size)
            if breakeven_price is not None and self._cycle_worst_price is not None:
                recovery_distance_pips = abs(self._cycle_worst_price - breakeven_price) / self.cfg.strategy.pip_size

        max_adverse_excursion = abs(min(self._cycle_worst_pnl, 0.0))
        if max_adverse_excursion > 0:
            recovered_amount = realized - self._cycle_worst_pnl  # distance climbed back from the trough
            recovery_percentage = max(0.0, min(100.0, 100.0 * recovered_amount / max_adverse_excursion))
        else:
            recovery_percentage = 100.0

        self.state.cycles.append(CycleRecord(
            cycle_id=self._cycle_id,
            open_time=self._cycle_open_time,
            close_time=time,
            open_price=self._cycle_open_price,
            exit_price=price,
            direction_movement=direction,
            realized_pnl=realized,
            max_levels_buy=max_level_buy,
            max_levels_sell=max_level_sell,
            num_pyramid_positions=num_pyramid,
            num_martingale_positions=num_martingale,
            total_buy_lots=total_buy_lots,
            total_sell_lots=total_sell_lots,
            avg_buy_price=avg_buy_price,
            avg_sell_price=avg_sell_price,
            worst_floating_pnl=self._cycle_worst_pnl,
            worst_floating_time=self._cycle_worst_time,
            worst_price=self._cycle_worst_price,
            best_floating_pnl=self._cycle_best_pnl,
            best_floating_time=self._cycle_best_time,
            best_price=self._cycle_best_price,
            breakeven_price=breakeven_price,
            recovery_distance_pips=recovery_distance_pips,
            recovery_percentage=recovery_percentage,
            recovery_time_seconds=recovery_seconds,
            closed=True,
            exit_reason=reason,
        ))

    # ------------------------------------------------------------------
    def _process_checkpoint(self, price: float, time: datetime, grid_dist: float):
        """Runs one intrabar checkpoint: grid triggers (possibly multiple
        levels in sequence), then an exit check. May open/close cycles."""
        commission_per_lot = self.cfg.costs.commission_per_lot

        if self._buy is None:
            self._open_cycle(price, time)

        # allow several grid levels to trigger in sequence at this checkpoint
        # (bounded by max_levels inside the managers - no infinite loop risk)
        while True:
            buy_pnl, sell_pnl = self._pnls(price)
            winning, losing = (self._buy, self._sell) if buy_pnl >= sell_pnl else (self._sell, self._buy)
            added_p = self.pyramid_mgr.maybe_add(winning, price, time, grid_dist, commission_per_lot)
            added_m = self.martingale_mgr.maybe_add(losing, price, time, grid_dist, commission_per_lot)
            if not (added_p or added_m):
                break

        floating_pnl = self._record_curves(time, price)
        self._update_worst(floating_pnl, price, time)

        qtr = self.cfg.account.quote_to_account_rate
        should_close, reason = self.exit_strategy.should_close(self._buy, self._sell, price, qtr)
        if should_close:
            self._close_cycle(price, time, reason)
            self._buy = None
            self._sell = None

    def run(self) -> SimulationState:
        has_atr = "atr" in self.data.columns
        cols = ["time", "open", "high", "low", "close"] + (["atr"] if has_atr else [])
        iterator = self.data[cols].itertuples(index=False, name="Bar")

        for bar in iterator:
            time = bar.time
            o, h, l, c = float(bar.open), float(bar.high), float(bar.low), float(bar.close)
            atr_val = float(bar.atr) if has_atr else None

            day = time.date()
            if self._last_day is not None and day != self._last_day and self._buy is not None:
                self._buy.accrue_daily_swap(self.cfg.costs.swap_long_per_lot, self.cfg.costs.swap_short_per_lot)
                self._sell.accrue_daily_swap(self.cfg.costs.swap_long_per_lot, self.cfg.costs.swap_short_per_lot)
            self._last_day = day

            grid_dist = self._grid_distance_price(atr_val)

            for checkpoint_price in self._bar_path(o, h, l, c):
                self._process_checkpoint(checkpoint_price, time, grid_dist)

        # close any still-open cycle at the last known price (mark-to-market, not a strategy exit)
        if self._buy is not None:
            last_row = self.data.iloc[-1]
            self._close_cycle(float(last_row["close"]), last_row["time"], "end_of_data_mtm")

        return self.state
