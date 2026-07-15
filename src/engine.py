# src/engine.py (FINAL PRODUCTION-READY VERSION - v2, look-ahead bias fixed)
import polars as pl
from dataclasses import dataclass
from typing import List, Dict, Any
from loguru import logger
from src.config import TradingCosts, StrategyParams, BacktestSettings

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
        if not self.settings.use_dynamic_position_sizing:
            # STRESS TEST MODE: لات ثابت - برای سنجش expectancy خام
            # سیگنال، مستقل از اثر money management.
            # ⚠️ در این حالت total_return_pct و max_drawdown_pct معیار
            # ریسک واقعی حساب نیستند (چون حجم معامله با رشد/افت بالانس
            # تغییر نمی‌کند). فقط برای مقایسه‌ی کیفیت خام سیگنال مناسب
            # است.
            return self.settings.fixed_lot_size

        # --- Position sizing واقعی بر اساس ریسک ثابت هر معامله ---
        if current_atr is None or current_atr <= 0:
            return 0.0

        risk_amount_usd = self.balance * self.settings.risk_per_trade_percent
        sl_distance_price = current_atr * self.params.initial_sl_atr_mult
        sl_distance_pips = sl_distance_price / self.pip_size

        if sl_distance_pips <= 0:
            return 0.0

        risk_per_lot_usd = sl_distance_pips * self.costs.pip_value_usd_per_lot
        if risk_per_lot_usd <= 0:
            return 0.0

        lot_size = risk_amount_usd / risk_per_lot_usd
        # گرد کردن به step معمول بروکر (0.01) و حداقل منطقی برای جلوگیری
        # از لات صفر روی حساب‌های خیلی کوچک یا استاپ خیلی گشاد
        lot_size = max(0.01, round(lot_size, 2))
        return lot_size

    def _apply_entry_costs(self, price: float, side: int) -> float:
        # Spread is paid on entry (full spread impact)
        spread_cost_usd = self.costs.spread_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        # Slippage on entry
        slippage_entry_usd = self.costs.slippage_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        # Commission (usually charged per side, so half of round-trip here)
        comm_entry_usd = (self.costs.commission_per_lot_usd / 2) * self.open_trade.lot_size

        self.total_spread_cost += spread_cost_usd
        self.total_slippage_cost += slippage_entry_usd
        self.total_commission += comm_entry_usd

        total_entry_cost_usd = spread_cost_usd + slippage_entry_usd + comm_entry_usd
        price_impact_pips = total_entry_cost_usd / (self.costs.pip_value_usd_per_lot * self.open_trade.lot_size)
        price_impact_price = price_impact_pips * self.pip_size

        if side == 1:  # Buy
            return price + price_impact_price
        else:          # Sell
            return price - price_impact_price

    def _apply_exit_costs(self, price: float, side: int) -> float:
        # Slippage on exit
        slippage_exit_usd = self.costs.slippage_pips * self.costs.pip_value_usd_per_lot * self.open_trade.lot_size
        # Commission on exit (second half)
        comm_exit_usd = (self.costs.commission_per_lot_usd / 2) * self.open_trade.lot_size

        self.total_slippage_cost += slippage_exit_usd
        self.total_commission += comm_exit_usd

        total_exit_cost_usd = slippage_exit_usd + comm_exit_usd
        price_impact_pips = total_exit_cost_usd / (self.costs.pip_value_usd_per_lot * self.open_trade.lot_size)
        price_impact_price = price_impact_pips * self.pip_size

        if side == 1:  # Buy (Exit by Selling)
            return price - price_impact_price
        else:          # Sell (Exit by Buying)
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
        mode_label = (
            "DYNAMIC RISK-BASED SIZING (realistic)"
            if self.settings.use_dynamic_position_sizing
            else "FIXED LOT (stress test) - %% metrics are NOT real risk-adjusted"
        )
        logger.info(f"Starting Event-Driven Engine | Sizing mode: {mode_label}")
        data = df.to_dicts()

        for row in data:
            current_time = row["datetime"]
            current_open = row["open"]
            current_high = row["high"]
            current_low = row["low"]
            current_close = row["close"]
            current_atr = row["atr"]
            signal = row["signal"]

            if self.open_trade:
                trade = self.open_trade
                trail_distance = current_atr * self.params.trail_atr_mult

                if trade.side == 1:  # Buy
                    new_sl = current_high - trail_distance
                    if new_sl > trade.sl_price:
                        trade.sl_price = new_sl
                    if current_low <= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)
                else:  # Sell
                    new_sl = current_low + trail_distance
                    if new_sl < trade.sl_price:
                        trade.sl_price = new_sl
                    if current_high >= trade.sl_price:
                        self._close_trade(trade.sl_price, current_time)

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

        return {
            "symbol": self.settings.symbol,
            "timeframe": self.settings.timeframe,
            "years": f"{self.settings.start_year}-{self.settings.end_year}",
            "sizing_mode": (
                "dynamic_risk_based" if self.settings.use_dynamic_position_sizing else "fixed_lot_stress_test"
            ),
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
            "total_spread_cost": self.total_spread_cost,
            "total_slippage_cost": self.total_slippage_cost,
            "total_commission": self.total_commission,
            "total_hidden_costs": self.total_spread_cost + self.total_slippage_cost + self.total_commission
        }
