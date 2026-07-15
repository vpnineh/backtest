# src/engine.py
import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from loguru import logger
from src.config import TradingCosts, StrategyParams, BacktestSettings, MartingaleConfig

@dataclass
class Trade:
    entry_time: Any
    exit_time: Any = None
    side: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    lot_size: float = 0.0
    sl_price: float = 0.0
    pnl: float = 0.0
    martingale_level: int = 0
    bars_held: int = 0

class RealisticBacktestEngine:
    def __init__(self, costs: TradingCosts, params: StrategyParams, settings: BacktestSettings, martingale: MartingaleConfig):
        self.costs = costs
        self.params = params
        self.settings = settings
        self.martingale = martingale

        self.balance = settings.initial_balance
        self.initial_balance = settings.initial_balance
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None

        self.current_martingale_level = 0
        self.is_halted_by_circuit_breaker = False

        self.total_spread_cost = 0.0
        self.total_slippage_cost = 0.0
        self.total_commission = 0.0
        self.pip_size = costs.pip_size

    def _calculate_lot_size(self, current_atr: float) -> float:
        if self.is_halted_by_circuit_breaker:
            return 0.0

        if not self.settings.use_dynamic_position_sizing:
            return self.settings.fixed_lot_size

        base_risk_amount = self.balance * self.settings.base_risk_per_trade_percent

        if self.martingale.enabled and self.current_martingale_level > 0:
            effective_level = min(self.current_martingale_level, self.martingale.max_levels)
            multiplier_factor = self.martingale.multiplier ** effective_level
            risk_amount = base_risk_amount * multiplier_factor
        else:
            risk_amount = base_risk_amount

        # Hard SL for Mean Reversion (3 * ATR)
        sl_pips = (current_atr * 3.0) / self.pip_size if current_atr > 0 else 30.0
        risk_per_lot_usd = sl_pips * self.costs.pip_value_usd_per_lot

        if risk_per_lot_usd <= 0:
            return 0.0

        lot_size = risk_amount / risk_per_lot_usd
        lot_size = max(0.01, round(lot_size, 2))
        return lot_size

    def _apply_entry_costs(self, price: float, side: int) -> float:
        spread_cost_usd = self.costs.spread_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        slippage_entry_usd = self.costs.slippage_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        comm_entry_usd = (self.costs.commission_per_lot_usd / 2) * self.open_trade.lot_size

        self.total_spread_cost += spread_cost_usd
        self.total_slippage_cost += slippage_entry_usd
        self.total_commission += comm_entry_usd

        total_entry_cost_usd = spread_cost_usd + slippage_entry_usd + comm_entry_usd
        price_impact_pips = total_entry_cost_usd / (self.costs.pip_value_usd_per_lot * self.open_trade.lot_size)
        price_impact_price = price_impact_pips * self.pip_size

        if side == 1:
            return price + price_impact_price
        else:
            return price - price_impact_price

    def _apply_exit_costs(self, price: float, side: int) -> float:
        slippage_exit_usd = self.costs.slippage_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        comm_exit_usd = (self.costs.commission_per_lot_usd / 2) * self.open_trade.lot_size

        self.total_slippage_cost += slippage_exit_usd
        self.total_commission += comm_exit_usd

        total_exit_cost_usd = slippage_exit_usd + comm_exit_usd
        price_impact_pips = total_exit_cost_usd / (self.costs.pip_value_usd_per_lot * self.open_trade.lot_size)
        price_impact_price = price_impact_pips * self.pip_size

        if side == 1:
            return price - price_impact_price
        else:
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
        trade.martingale_level = self.current_martingale_level
        self.balance += trade.pnl

        if self.martingale.enabled:
            if trade.pnl > 0:
                if self.martingale.reset_on_win and self.current_martingale_level > 0:
                    logger.info(f"Win detected. Resetting Martingale level from {self.current_martingale_level} to 0.")
                self.current_martingale_level = 0
            elif trade.pnl < 0:
                if self.current_martingale_level < self.martingale.max_levels:
                    self.current_martingale_level += 1
                    logger.warning(f"Loss detected. Martingale level increased to {self.current_martingale_level}.")
                else:
                    logger.warning(f"Max Martingale level ({self.martingale.max_levels}) reached. Capping volume.")

        current_dd = (self.initial_balance - self.balance) / self.initial_balance
        if current_dd >= self.martingale.circuit_breaker_dd_percent:
            self.is_halted_by_circuit_breaker = True
            logger.error(f"🚨 CIRCUIT BREAKER TRIGGERED! Drawdown: {current_dd:.2%}. Trading Halted.")

        self.trades.append(trade)
        self.open_trade = None

    def run(self, df: pl.DataFrame) -> Dict[str, Any]:
        logger.info("Starting Mean Reversion Engine with HARD STOP LOSS & Martingale...")
        data = df.to_dicts()

        for row in data:
            if self.is_halted_by_circuit_breaker:
                self.equity_curve.append(self.balance)
                continue

            current_time = row["datetime"]
            current_open = row["open"]
            current_high = row["high"]
            current_low = row["low"]
            current_close = row["close"]
            current_bb_middle = row["bb_middle"]
            
            # 🔥 FIX: Safe ATR access with fallback
            current_atr = row.get("atr", 0.0)
            if current_atr is None or current_atr <= 0:
                current_atr = 30.0 * self.pip_size  # Fallback to 30 pips
            
            signal = row["signal"]

            if self.open_trade:
                trade = self.open_trade
                trade.bars_held += 1

                # 🔥 FIX 1: Check Hard Stop Loss FIRST
                if trade.side == 1:  # Buy
                    if current_low <= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)
                        continue
                    # 🔥 FIX 2: Check Mean Reversion Target
                    elif current_low <= current_bb_middle:
                        self._close_trade(current_bb_middle, current_time)
                        continue
                else:  # Sell
                    if current_high >= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)
                        continue
                    elif current_high >= current_bb_middle:
                        self._close_trade(current_bb_middle, current_time)
                        continue

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
                    
                    # 🔥 FIX 3: Set Hard Stop Loss on Entry (3 * ATR)
                    sl_distance = current_atr * 3.0
                    if signal == 1:
                        self.open_trade.sl_price = actual_entry - sl_distance
                    else:
                        self.open_trade.sl_price = actual_entry + sl_distance

            floating_pnl = 0.0
            if self.open_trade:
                price_diff = (current_close - self.open_trade.entry_price) * self.open_trade.side
                floating_pnl = (price_diff / self.pip_size) * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size

            self.equity_curve.append(self.balance + floating_pnl)

        if self.open_trade:
            self._close_trade(data[-1]["close"], data[-1]["datetime"])

        return self._generate_report()

    def _generate_report(self) -> Dict[str, Any]:
        logger.info("Generating final performance report...")
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
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        avg_bars_held = sum(t.bars_held for t in self.trades) / total_trades
        max_level_reached = max([t.martingale_level for t in self.trades]) if self.trades else 0

        total_hidden_costs = self.total_spread_cost + self.total_slippage_cost + self.total_commission
        net_pnl = self.balance - self.settings.initial_balance
        estimated_gross_pnl = net_pnl + total_hidden_costs

        return {
            "symbol": self.settings.symbol,
            "timeframe": self.settings.timeframe,
            "years": f"{self.settings.start_year}-{self.settings.end_year}",
            "sizing_mode": "martingale_mean_reversion",
            "initial_balance": self.settings.initial_balance,
            "final_balance": self.balance,
            "total_return_pct": ((self.balance - self.settings.initial_balance) / self.settings.initial_balance) * 100,
            "total_trades": total_trades,
            "win_rate_pct": win_rate * 100,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else float('inf'),
            "expectancy": sum(pnls) / total_trades,
            "max_drawdown_usd": max_dd,
            "max_drawdown_pct": max_dd_pct * 100,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "payoff_ratio": (sum(wins) / len(wins)) / abs(sum(losses) / len(losses)) if losses and wins else 0,
            "avg_bars_held": avg_bars_held,
            "max_martingale_level_reached": max_level_reached,
            "halted_by_circuit_breaker": self.is_halted_by_circuit_breaker,
            "total_spread_cost": self.total_spread_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "total_commission": self.total_commission,
            "total_hidden_costs": total_hidden_costs,
            "net_pnl": net_pnl,
            "estimated_gross_pnl_before_costs": estimated_gross_pnl,
        }
