"""
StatisticsEngine
=================
Turns the raw SimulationState produced by BacktestEngine into the full
set of research statistics requested in the specification.
"""

from __future__ import annotations
import math

import numpy as np
import pandas as pd

from .engine import SimulationState


class StatisticsEngine:
    def __init__(self, state: SimulationState, starting_balance: float):
        self.state = state
        self.starting_balance = starting_balance

    # ------------------------------------------------------------------
    def _equity_series(self) -> pd.Series:
        times, vals = zip(*self.state.equity_curve) if self.state.equity_curve else ([], [])
        return pd.Series(vals, index=pd.to_datetime(times))

    def _cycles_df(self) -> pd.DataFrame:
        if not self.state.cycles:
            return pd.DataFrame()
        return pd.DataFrame([c.__dict__ for c in self.state.cycles])

    def _trades_df(self) -> pd.DataFrame:
        if not self.state.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.state.trade_log)

    # ------------------------------------------------------------------
    def compute(self) -> dict:
        eq = self._equity_series()
        cycles = self._cycles_df()
        trades = self._trades_df()

        net_profit = (eq.iloc[-1] - self.starting_balance) if len(eq) else 0.0

        gross_profit = cycles.loc[cycles["realized_pnl"] > 0, "realized_pnl"].sum() if len(cycles) else 0.0
        gross_loss = cycles.loc[cycles["realized_pnl"] < 0, "realized_pnl"].sum() if len(cycles) else 0.0
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0

        # drawdown series (on equity curve)
        if len(eq):
            running_max = eq.cummax()
            dd = running_max - eq
            max_drawdown = dd.max()
            avg_drawdown = dd[dd > 0].mean() if (dd > 0).any() else 0.0
        else:
            max_drawdown = 0.0
            avg_drawdown = 0.0

        recovery_factor = (net_profit / max_drawdown) if max_drawdown else float("inf") if net_profit > 0 else 0.0

        # Sharpe on daily resampled equity returns (simplification, no risk-free rate)
        sharpe = None
        if len(eq) > 2:
            daily_eq = eq.resample("1D").last().dropna()
            rets = daily_eq.pct_change().dropna()
            if len(rets) > 1 and rets.std() > 0:
                sharpe = (rets.mean() / rets.std()) * math.sqrt(252)

        avg_trade_duration = None
        if len(trades):
            durations = (pd.to_datetime(trades["close_time"]) - pd.to_datetime(trades["open_time"])).dt.total_seconds()
            avg_trade_duration = durations.mean()

        max_basket_size = max((v for _, v in self.state.basket_size_curve), default=0)
        largest_floating_loss = min((v for _, v in self.state.floating_dd_curve), default=0.0) * -1 \
            if self.state.floating_dd_curve else 0.0
        max_floating_dd = max((v for _, v in self.state.floating_dd_curve), default=0.0)
        max_margin_used = max((v for _, v in self.state.margin_curve), default=0.0)
        max_exposure = max((v for _, v in self.state.exposure_curve), default=0.0)
        max_open_lots = max((v for _, v in self.state.open_lots_curve), default=0.0)
        worst_equity = eq.min() if len(eq) else self.starting_balance

        num_pyramid = int((trades["kind"] == "pyramid").sum()) if len(trades) else 0
        num_martingale = int((trades["kind"] == "martingale").sum()) if len(trades) else 0
        num_cycles = len(cycles)
        winning_cycles = int((cycles["realized_pnl"] > 0).sum()) if len(cycles) else 0
        losing_cycles = int((cycles["realized_pnl"] <= 0).sum()) if len(cycles) else 0

        recov_times = cycles["recovery_time_seconds"].dropna() if len(cycles) else pd.Series(dtype=float)
        avg_recovery_time = recov_times.mean() if len(recov_times) else None
        max_recovery_time = recov_times.max() if len(recov_times) else None

        worst_pnls = cycles["worst_floating_pnl"].dropna() if len(cycles) else pd.Series(dtype=float)
        avg_recovery_distance = worst_pnls.mean() if len(worst_pnls) else None

        avg_levels = None
        max_levels_required = None
        if len(cycles):
            levels_used = cycles[["max_levels_buy", "max_levels_sell"]].max(axis=1)
            avg_levels = levels_used.mean()
            max_levels_required = levels_used.max()

        return {
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "recovery_factor": recovery_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "largest_drawdown": max_drawdown,
            "avg_trade_duration_seconds": avg_trade_duration,
            "max_basket_size": max_basket_size,
            "largest_floating_loss": largest_floating_loss,
            "max_floating_drawdown": max_floating_dd,
            "max_margin_used": max_margin_used,
            "largest_exposure": max_exposure,
            "max_open_lots": max_open_lots,
            "worst_equity": worst_equity,
            "final_balance": eq.iloc[-1] if len(eq) else self.starting_balance,
            "num_pyramid_trades": num_pyramid,
            "num_martingale_trades": num_martingale,
            "num_hedge_cycles": num_cycles,
            "winning_cycles": winning_cycles,
            "losing_cycles": losing_cycles,
            "avg_recovery_distance": avg_recovery_distance,
            "avg_recovery_time_seconds": avg_recovery_time,
            "max_recovery_time_seconds": max_recovery_time,
            "avg_levels_required": avg_levels,
            "max_levels_required": max_levels_required,
        }
