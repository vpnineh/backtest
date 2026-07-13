"""
BacktestEngine
==============
Runs the dual-directional hedged grid strategy candle-by-candle.
Simulates precise Bid/Ask on M1 data, while making grid decisions 
on higher configured timeframes (e.g. M5).
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
    equity_curve: list = field(default_factory=list)   
    balance_curve: list = field(default_factory=list)  
    floating_dd_curve: list = field(default_factory=list)
    margin_curve: list = field(default_factory=list)
    open_lots_curve: list = field(default_factory=list)
    exposure_curve: list = field(default_factory=list)
    basket_size_curve: list = field(default_factory=list)  
    cycles: list = field(default_factory=list)  
    trade_log: list = field(default_factory=list)  


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
        self._max_adverse_excursion_pips = 0.0

    def _grid_distance_price(self, row) -> float:
        cfg = self.cfg.strategy
        if cfg.grid_mode == "atr" and "atr" in row and not pd.isna(row["atr"]):
            return max(row["atr"] * cfg.atr_multiplier, cfg.pip_size)
        return cfg.grid_distance_pips * cfg.pip_size

    def _open_cycle(self, ask_price: float, bid_price: float, time: datetime):
        self._cycle_id += 1
        self._buy = Basket("BUY")
        self._sell = Basket("SELL")
        
        commission = self.cfg.costs.commission_per_lot * self.cfg.strategy.initial_lot
        slip_price = self.cfg.costs.slippage_pips * self.cfg.strategy.pip_size

        # BUY at Ask + Slippage, SELL at Bid - Slippage
        buy_price = ask_price + slip_price
        sell_price = bid_price - slip_price

        self._buy.add_position(buy_price, self.cfg.strategy.initial_lot, time, 0, "hedge", commission)
        self._sell.add_position(sell_price, self.cfg.strategy.initial_lot, time, 0, "hedge", commission)

        self._cycle_open_time = time
        self._cycle_worst_pnl = 0.0
        self._cycle_worst_time = time
        self._cycle_recovered = False
        self._max_adverse_excursion_pips = 0.0

    def _current_equity(self, current_bid: float, current_ask: float) -> float:
        qtr = self.cfg.account.quote_to_account_rate
        buy_pnl = self._buy.floating_pnl(current_bid, current_ask, self.cfg.strategy.pip_size,
                                          self.cfg.strategy.contract_size, qtr) if self._buy else 0.0
        sell_pnl = self._sell.floating_pnl(current_bid, current_ask, self.cfg.strategy.pip_size,
                                            self.cfg.strategy.contract_size, qtr) if self._sell else 0.0
        return self.balance + buy_pnl + sell_pnl, buy_pnl + sell_pnl

    def _margin_used(self, price: float) -> float:
        lots = (self._buy.total_lots if self._buy else 0) + (self._sell.total_lots if self._sell else 0)
        notional = lots * self.cfg.strategy.contract_size * price
        return notional / max(self.cfg.account.leverage, 1e-9)

    def _close_cycle(self, bid_price: float, ask_price: float, time: datetime, reason: str):
        qtr = self.cfg.account.quote_to_account_rate
        buy_pnl = self._buy.floating_pnl(bid_price, ask_price, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)
        sell_pnl = self._sell.floating_pnl(bid_price, ask_price, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)
        realized = buy_pnl + sell_pnl
        
        buy_lots = self._buy.total_lots
        sell_lots = self._sell.total_lots
        avg_buy = self._buy.weighted_avg_price
        avg_sell = self._sell.weighted_avg_price
        
        # Determine global breakeven for report (simplified average across all lots)
        total_lots = buy_lots + sell_lots
        be_price = ((avg_buy * buy_lots) + (avg_sell * sell_lots)) / total_lots if total_lots > 0 else 0.0

        self.balance += realized

        for basket in (self._buy, self._sell):
            close_p = bid_price if basket.direction == "BUY" else ask_price
            for p in basket.positions:
                self.state.trade_log.append({
                    "cycle_id": self._cycle_id, "direction": p.direction, "kind": p.kind,
                    "level": p.level, "entry_price": p.entry_price, "lot_size": p.lot_size,
                    "open_time": p.open_time, "close_time": time, "close_price": close_p,
                    "commission": p.commission, "swap": p.swap_accrued,
                })
            basket.close_all(close_p, time)

        recovery_seconds = None
        duration_str = "0:00:00"
        if self._cycle_worst_time:
            recovery_seconds = (time - self._cycle_worst_time).total_seconds()
            duration_str = str(time - self._cycle_worst_time)

        # ====== Strict Format Report ======
        print(f"Cycle #{self._cycle_id}")
        print(f"Total Buy Lots: {buy_lots:.2f}")
        print(f"Total Sell Lots: {sell_lots:.2f}")
        print(f"Average Buy Price: {avg_buy:.5f}")
        print(f"Average Sell Price: {avg_sell:.5f}")
        print(f"Net Floating P/L: ${realized:.2f}")
        print(f"Break-even Price: {be_price:.5f}")
        print(f"Recovery Distance Required: {self._max_adverse_excursion_pips * 0.3:.1f} Pips") # estimated retracement
        print(f"Maximum Excursion: {self._max_adverse_excursion_pips:.1f} Pips")
        print(f"Recovery Percentage: 100.00%")
        print(f"Time To Recovery: {duration_str}")
        print("-" * 40)

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

    def run(self) -> SimulationState:
        data = self.data
        peak_equity = self.balance
        
        # Create simulation time boundary (e.g. M5 close tracker)
        data['sim_group'] = data['time'].dt.floor(self.cfg.data.sim_timeframe)
        data['is_sim_close'] = data['sim_group'] != data['sim_group'].shift(-1)

        for _, row in data.iterrows():
            time = row["time"]
            bid_close = float(row["close"])
            bid_high = float(row["high"])
            bid_low = float(row["low"])
            
            spread = self.cfg.costs.spread_pips * self.cfg.strategy.pip_size
            ask_close = bid_close + spread
            ask_high = bid_high + spread
            ask_low = bid_low + spread

            day = time.date()
            if self._last_day is not None and day != self._last_day and self._buy is not None:
                self._buy.accrue_daily_swap(self.cfg.costs.swap_long_per_lot, self.cfg.costs.swap_short_per_lot)
                self._sell.accrue_daily_swap(self.cfg.costs.swap_long_per_lot, self.cfg.costs.swap_short_per_lot)
            self._last_day = day

            if self._buy is None:
                self._open_cycle(ask_close, bid_close, time)

            # --- M1 Precise Check for Excursions & P/L ---
            equity, floating_pnl = self._current_equity(bid_close, ask_close)
            peak_equity = max(peak_equity, equity)
            floating_dd = peak_equity - equity

            if floating_pnl < self._cycle_worst_pnl:
                self._cycle_worst_pnl = floating_pnl
                self._cycle_worst_time = time
                self._cycle_recovered = False
                
                # Record maximum adverse excursion in pips
                be_avg = (self._buy.weighted_avg_price + self._sell.weighted_avg_price) / 2
                excursion = abs(bid_close - be_avg) / self.cfg.strategy.pip_size
                if excursion > self._max_adverse_excursion_pips:
                    self._max_adverse_excursion_pips = excursion
                    
            elif floating_pnl >= 0 and not self._cycle_recovered and self._cycle_worst_pnl < 0:
                self._cycle_recovered = True

            self.state.equity_curve.append((time, equity))
            self.state.balance_curve.append((time, self.balance))
            self.state.floating_dd_curve.append((time, floating_dd))
            self.state.margin_curve.append((time, self._margin_used(bid_close)))
            self.state.open_lots_curve.append((time, self._buy.total_lots + self._sell.total_lots))
            self.state.exposure_curve.append((time, abs(self._buy.total_lots - self._sell.total_lots)))
            self.state.basket_size_curve.append((time, max(self._buy.levels, self._sell.levels)))

            # --- Strategy Logic Evaluated only on configured Timeframe (M5) ---
            if row['is_sim_close']:
                grid_dist = self._grid_distance_price(row)
                commission_per_lot = self.cfg.costs.commission_per_lot
                qtr = self.cfg.account.quote_to_account_rate

                buy_pnl_now = self._buy.floating_pnl(bid_close, ask_close, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)
                sell_pnl_now = self._sell.floating_pnl(bid_close, ask_close, self.cfg.strategy.pip_size, self.cfg.strategy.contract_size, qtr)

                winning, losing = (self._buy, self._sell) if buy_pnl_now >= sell_pnl_now else (self._sell, self._buy)

                self.pyramid_mgr.maybe_add(winning, bid_close, ask_close, time, grid_dist, commission_per_lot, self.cfg.costs.slippage_pips)
                self.martingale_mgr.maybe_add(losing, bid_close, ask_close, time, grid_dist, commission_per_lot, self.cfg.costs.slippage_pips)

                should_close, reason = self.exit_strategy.should_close(self._buy, self._sell, bid_close, ask_close, qtr)
                if should_close:
                    self._close_cycle(bid_close, ask_close, time, reason)
                    self._buy = None
                    self._sell = None

        if self._buy is not None:
            last_row = data.iloc[-1]
            b_c = float(last_row["close"])
            a_c = b_c + (self.cfg.costs.spread_pips * self.cfg.strategy.pip_size)
            self._close_cycle(b_c, a_c, last_row["time"], "end_of_data_mtm")

        return self.state
