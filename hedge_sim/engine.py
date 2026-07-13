"""
BacktestEngine
==============
Runs the dual-directional hedged grid strategy candle-by-candle over the
full historical dataset. A "hedge cycle" starts with a simultaneous
BUY + SELL at initial_lot, dynamically manages both baskets (pyramid on
the winner, soft-martingale on the loser) and ends when the configured
ExitStrategy fires. A brand-new cycle then starts on the next candle.

No forced stop loss is implemented anywhere. Worst-case metrics
(max floating DD, worst equity, largest basket, max exposure, etc.) are
recorded instead, as required by the research specification.
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
    realized_pnl: float | None
    max_levels_buy: int
    max_levels_sell: int
    worst_floating_pnl: float
    worst_floating_time: datetime | None
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
        self._cycle_worst_pnl = 0.0
        self._cycle_worst_time: datetime | None = None
        self._cycle_recovered = False
        self._last_day = None

    # ------------------------------------------------------------------
    def _grid_distance_price(self, row) -> float:
        cfg = self.cfg.strategy
        if cfg.grid_mode == "atr" and "atr" in row and not pd.isna(row["atr"]):
            return max(row["atr"] * cfg.atr_multiplier, cfg.pip_size)
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
        self._cycle_worst_pnl = 0.0
        self._cycle_worst_time = time
        self._cycle_recovered = False

    def _current_equity(self, price: float) -> float:
        qtr = self.cfg.account.quote_to_account_rate
        buy_pnl = self._buy.floating_pnl(price, self.cfg.strategy.pip_size,
                                          self.cfg.strategy.contract_size, qtr) if self._buy else 0.0
        sell_pnl = self._sell.floating_pnl(price, self.cfg.strategy.pip_size,
                                            self.cfg.strategy.contract_size, qtr) if self._sell else 0.0
        return self.balance + buy_pnl + sell_pnl, buy_pnl + sell_pnl

    def _margin_used(self, price: float) -> float:
        lots = (self._buy.total_lots if self._buy else 0) + (self._sell.total_lots if self._sell else 0)
        notional = lots * self.cfg.strategy.contract_size * price
        return notional / max(self.cfg.account.leverage, 1e-9)

    def _close_cycle(self, price: float, time: datetime, reason: str):
        qtr = self.cfg.account.quote_to_account_rate
        buy_pnl = self._buy.floating_pnl(price, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)
        sell_pnl = self._sell.floating_pnl(price, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)
        realized = buy_pnl + sell_pnl
        self.balance += realized

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

        self.state.cycles.append(CycleRecord(
            cycle_id=self._cycle_id,
            open_time=self._cycle_open_time,
            close_time=time,
            realized_pnl=realized,
            max_levels_buy=max((p.level for p in self._buy.closed_positions[-50:]), default=0),
            max_levels_sell=max((p.level for p in self._sell.closed_positions[-50:]), default=0),
            worst_floating_pnl=self._cycle_worst_pnl,
            worst_floating_time=self._cycle_worst_time,
            recovery_time_seconds=recovery_seconds,
            closed=True,
            exit_reason=reason,
        ))

    # ------------------------------------------------------------------
    def run(self) -> SimulationState:
        data = self.data
        peak_equity = self.balance

        for _, row in data.iterrows():
            time = row["time"]
            price = float(row["close"])
            high = float(row["high"])
            low = float(row["low"])

            # daily swap accrual (once per new calendar day)
            day = time.date()
            if self._last_day is not None and day != self._last_day and self._buy is not None:
                self._buy.accrue_daily_swap(self.cfg.costs.swap_long_per_lot, self.cfg.costs.swap_short_per_lot)
                self._sell.accrue_daily_swap(self.cfg.costs.swap_long_per_lot, self.cfg.costs.swap_short_per_lot)
            self._last_day = day

            if self._buy is None:
                self._open_cycle(price, time)

            grid_dist = self._grid_distance_price(row)
            commission_per_lot = self.cfg.costs.commission_per_lot

            qtr = self.cfg.account.quote_to_account_rate
            buy_pnl_now = self._buy.floating_pnl(price, self.cfg.strategy.pip_size,
                                                  self.cfg.strategy.contract_size, qtr)
            sell_pnl_now = self._sell.floating_pnl(price, self.cfg.strategy.pip_size,
                                                    self.cfg.strategy.contract_size, qtr)

            winning, losing = (self._buy, self._sell) if buy_pnl_now >= sell_pnl_now else (self._sell, self._buy)

            # use bar extremes for more realistic intrabar triggering
            extreme_favor = high if winning.direction == "BUY" else low
            extreme_adverse = low if losing.direction == "BUY" else high

            self.pyramid_mgr.maybe_add(winning, extreme_favor, time, grid_dist, commission_per_lot)
            self.martingale_mgr.maybe_add(losing, extreme_adverse, time, grid_dist, commission_per_lot)

            # recompute after possible new positions
            equity, floating_pnl = self._current_equity(price)
            peak_equity = max(peak_equity, equity)
            floating_dd = peak_equity - equity

            if floating_pnl < self._cycle_worst_pnl:
                self._cycle_worst_pnl = floating_pnl
                self._cycle_worst_time = time
                self._cycle_recovered = False
            elif floating_pnl >= 0 and not self._cycle_recovered and self._cycle_worst_pnl < 0:
                self._cycle_recovered = True

            self.state.equity_curve.append((time, equity))
            self.state.balance_curve.append((time, self.balance))
            self.state.floating_dd_curve.append((time, floating_dd))
            self.state.margin_curve.append((time, self._margin_used(price)))
            self.state.open_lots_curve.append((time, self._buy.total_lots + self._sell.total_lots))
            self.state.exposure_curve.append((time, abs(self._buy.total_lots - self._sell.total_lots)))
            self.state.basket_size_curve.append((time, max(self._buy.levels, self._sell.levels)))

            should_close, reason = self.exit_strategy.should_close(self._buy, self._sell, price, qtr)
            if should_close:
                self._close_cycle(price, time, reason)
                self._buy = None
                self._sell = None

        # close any still-open cycle at the last known price (mark-to-market, not a strategy exit)
        if self._buy is not None:
            last_row = data.iloc[-1]
            self._close_cycle(float(last_row["close"]), last_row["time"], "end_of_data_mtm")

        return self.state
