# src/engine.py
import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Any
from loguru import logger
from src.config import TradingCosts, BacktestSettings

@dataclass
class Trade:
    entry_time: Any
    exit_time: Any = None
    side: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    pnl: float = 0.0

class RealisticBacktestEngine:
    def __init__(self, costs: TradingCosts, settings: BacktestSettings):
        self.costs = costs
        self.settings = settings
        self.balance = settings.initial_balance
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        self.open_trade: Trade | None = None
        
        self.total_spread_cost = 0.0
        self.total_slippage_cost = 0.0
        self.total_commission = 0.0
        self.pip_size = costs.pip_size # Dynamic pip size

    def _calculate_lot_size(self) -> float:
        risk_amount = self.balance * self.settings.risk_per_trade_percent
        # Risk in USD = (SL in pips * pip_value_per_lot) * lot_size
        lot_size = risk_amount / (self.costs.sl_pips * self.costs.pip_value_usd_per_lot)
        return round(lot_size, 2)

    def _apply_entry_costs(self, price: float, side: int) -> float:
        spread_price = self.costs.spread_pips * self.pip_size
        slippage_price = self.costs.slippage_pips * self.pip_size
        
        if side == 1:  # Buy
            exec_price = price + (spread_price / 2) + slippage_price
        else:          # Sell
            exec_price = price - (spread_price / 2) - slippage_price
            
        self.total_spread_cost += (spread_price / 2) * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        self.total_slippage_cost += slippage_price * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        return exec_price

    def _apply_exit_costs(self, price: float, side: int) -> float:
        slippage_price = self.costs.slippage_pips * self.pip_size
        comm_cost = self.costs.commission_per_lot_usd * self.open_trade.lot_size
        
        if side == 1:  # Buy (Exit by Selling)
            exec_price = price - slippage_price
        else:          # Sell (Exit by Buying)
            exec_price = price + slippage_price
            
        self.total_slippage_cost += slippage_price * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        self.total_commission += comm_cost
        return exec_price

    def _close_trade(self, exit_price: float, exit_time: Any):
        trade = self.open_trade
        actual_exit = self._apply_exit_costs(exit_price, trade.side)
        trade.exit_price = actual_exit
        trade.exit_time = exit_time
        
        price_diff = (actual_exit - trade.entry_price) * trade.side
        pip_diff = price_diff / self.pip_size
        gross_pnl = pip_diff * self.costs.pip_value_usd_per_lot * trade.lot_size
        
        trade.pnl = gross_pnl
        self.balance += trade.pnl
        self.trades.append(trade)
        self.open_trade = None

    def run(self, df: pl.DataFrame) -> Dict[str, Any]:
        logger.info("Starting Event-Driven Backtest Engine...")
        data = df.to_dicts()
        
        for row in data:
            current_time = row["datetime"]
            current_open = row["open"]
            current_high = row["high"]
            current_low = row["low"]
            current_close = row["close"]
            signal = row["signal"]
            
            if self.open_trade:
                trade = self.open_trade
                if trade.side == 1: # Buy
                    if current_low <= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)
                    elif current_high >= trade.tp_price:
                        self._close_trade(trade.tp_price, current_time)
                else: # Sell
                    if current_high >= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)
                    elif current_low <= trade.tp_price:
                        self._close_trade(trade.tp_price, current_time)
                        
            if not self.open_trade and signal != 0:
                self.open_trade = Trade(
                    entry_time=current_time,
                    side=signal,
                    lot_size=self._calculate_lot_size()
                )
                actual_entry = self._apply_entry_costs(current_open, signal)
                self.open_trade.entry_price = actual_entry
                
                sl_distance = self.costs.sl_pips * self.pip_size
                tp_distance = self.costs.tp_pips * self.pip_size
                
                if signal == 1:
                    self.open_trade.sl_price = actual_entry - sl_distance
                    self.open_trade.tp_price = actual_entry + tp_distance
                else:
                    self.open_trade.sl_price = actual_entry + sl_distance
                    self.open_trade.tp_price = actual_entry - tp_distance

            floating_pnl = 0.0
            if self.open_trade:
                price_diff = (current_close - self.open_trade.entry_price) * self.open_trade.side
                floating_pnl = (price_diff / self.pip_size) * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
                
            self.equity_curve.append(self.balance + floating_pnl)

        if self.open_trade:
            self._close_trade(data[-1]["close"], data[-1]["datetime"])

        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        logger.info("Generating detailed performance report...")
        if not self.trades:
            return {"error": "No trades executed."}

        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        total_trades = len(pnls)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        peak = self.equity_curve[0]
        max_dd = max_dd_pct = 0.0
        for eq in self.equity_curve:
            if eq > peak: peak = eq
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd: max_dd = dd
            if dd_pct > max_dd_pct: max_dd_pct = dd_pct

        return {
            "symbol": self.settings.symbol,
            "timeframe": self.settings.timeframe,
            "years": f"{self.settings.start_year}-{self.settings.end_year}",
            "initial_balance": self.settings.initial_balance,
            "final_balance": self.balance,
            "total_return_pct": ((self.balance - self.settings.initial_balance) / self.settings.initial_balance) * 100,
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else float('inf'),
            "expectancy": sum(pnls) / total_trades,
            "max_drawdown_usd": max_dd,
            "max_drawdown_pct": max_dd_pct * 100,
            "total_spread_cost": self.total_spread_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "total_commission": self.total_commission,
            "total_hidden_costs": self.total_spread_cost + self.total_slippage_cost + self.total_commission
        }
