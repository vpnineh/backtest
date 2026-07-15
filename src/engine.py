# src/engine.py
import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Any
from loguru import logger
from src.config import TradingCosts, StrategyParams, BacktestSettings

@dataclass
class Trade:
    """Standardized Trade Dataclass."""
    entry_time: Any
    exit_time: Any = None
    side: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    sl_price: float = 0.0
    pnl: float = 0.0

class RealisticBacktestEngine:
    def __init__(self, costs: TradingCosts, params: StrategyParams, settings: BacktestSettings):
        self.costs = costs
        self.params = params
        self.settings = settings
        
        self.balance = settings.initial_balance
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        self.open_trade: Trade | None = None
        
        self.total_spread_cost = 0.0
        self.total_slippage_cost = 0.0
        self.total_commission = 0.0
        self.pip_size = costs.pip_size

    def _calculate_lot_size(self, current_atr: float) -> float:
        """
        🔥 STRESS TEST MODE: Fixed Lot Size to find TRUE expectancy.
        """
        FIXED_LOT_SIZE = 0.1 
        return FIXED_LOT_SIZE

    def _apply_entry_costs(self, price: float, side: int) -> float:
        """🔥 FIX: Correct dollar calculation for spread and slippage."""
        spread_cost_usd = (self.costs.spread_pips / 2) * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        slippage_cost_usd = self.costs.slippage_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        
        self.total_spread_cost += spread_cost_usd
        self.total_slippage_cost += slippage_cost_usd
        
        # Convert dollar cost back to price impact for execution price
        price_impact_pips = (spread_cost_usd + slippage_cost_usd) / (self.costs.pip_value_usd_per_lot * self.open_trade.lot_size)
        price_impact_price = price_impact_pips * self.pip_size
        
        if side == 1:  # Buy (pays spread + slippage)
            return price + price_impact_price
        else:          # Sell (pays spread + slippage)
            return price - price_impact_price

    def _apply_exit_costs(self, price: float, side: int) -> float:
        """🔥 FIX: Correct dollar calculation for exit slippage and commission."""
        slippage_cost_usd = self.costs.slippage_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        comm_cost_usd = self.costs.commission_per_lot_usd * self.open_trade.lot_size
        
        self.total_slippage_cost += slippage_cost_usd
        self.total_commission += comm_cost_usd
        
        # Convert dollar slippage back to price impact
        price_impact_pips = slippage_cost_usd / (self.costs.pip_value_usd_per_lot * self.open_trade.lot_size)
        price_impact_price = price_impact_pips * self.pip_size
        
        if side == 1:  # Buy (Exit by Selling, loses slippage)
            return price - price_impact_price
        else:          # Sell (Exit by Buying, loses slippage)
            return price + price_impact_price

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
        logger.info("Starting Event-Driven Engine with FIXED LOT SIZE & CORRECTED COSTS...")
        data = df.to_dicts()
        
        for row in data:
            current_time = row["datetime"]
            current_open = row["open"]
            current_high = row["high"]
            current_low = row["low"]
            current_close = row["close"]
            current_atr = row["atr"]
            signal = row["signal"]
            
            # 1. Manage Open Trade (Trailing Stop Logic)
            if self.open_trade:
                trade = self.open_trade
                trail_distance = current_atr * self.params.trail_atr_mult
                
                if trade.side == 1: # Buy
                    new_sl = current_high - trail_distance
                    if new_sl > trade.sl_price:
                        trade.sl_price = new_sl
                    if current_low <= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)
                else: # Sell
                    new_sl = current_low + trail_distance
                    if new_sl < trade.sl_price:
                        trade.sl_price = new_sl
                    if current_high >= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)
                        
            # 2. Check for New Entry
            if not self.open_trade and signal != 0:
                lot_size = self._calculate_lot_size(current_atr)
                
                if lot_size > 0:
                    self.open_trade = Trade(
                        entry_time=current_time,
                        side=signal,
                        lot_size=lot_size
                    )
                    
                    actual_entry = self._apply_entry_costs(current_open, signal)
                    self.open_trade.entry_price = actual_entry
                    
                    initial_sl_distance = current_atr * self.params.initial_sl_atr_mult
                    if signal == 1:
                        self.open_trade.sl_price = actual_entry - initial_sl_distance
                    else:
                        self.open_trade.sl_price = actual_entry + initial_sl_distance

            # Track Equity
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
            "avg_win": sum(wins)/len(wins) if wins else 0,
            "avg_loss": sum(losses)/len(losses) if losses else 0,
            "total_spread_cost": self.total_spread_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "total_commission": self.total_commission,
            "total_hidden_costs": self.total_spread_cost + self.total_slippage_cost + self.total_commission
        }
