"""
ChartEngine
===========
Generates every chart requested in the specification and saves them as
PNG files into the results directory. Uses matplotlib only (headless-safe).
"""

from __future__ import annotations
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .engine import SimulationState


class ChartEngine:
    def __init__(self, state: SimulationState, output_dir: str):
        self.state = state
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _series(self, curve):
        if not curve:
            return pd.Series(dtype=float)
        times, vals = zip(*curve)
        return pd.Series(vals, index=pd.to_datetime(times))

    def _line_chart(self, series: pd.Series, title: str, ylabel: str, filename: str, color="#2563eb"):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(series.index, series.values, color=color, linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=130)
        plt.close(fig)

    def _hist_chart(self, values, title: str, xlabel: str, filename: str, color="#dc2626"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if len(values):
            ax.hist(values, bins=min(40, max(5, len(values) // 3)), color=color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=130)
        plt.close(fig)

    def generate_all(self):
        eq = self._series(self.state.equity_curve)
        bal = self._series(self.state.balance_curve)
        dd = self._series(self.state.floating_dd_curve)
        margin = self._series(self.state.margin_curve)
        lots = self._series(self.state.open_lots_curve)
        exposure = self._series(self.state.exposure_curve)
        basket = self._series(self.state.basket_size_curve)

        self._line_chart(eq, "Equity Curve", "Equity", "equity_curve.png", "#2563eb")
        self._line_chart(bal, "Balance Curve", "Balance", "balance_curve.png", "#16a34a")
        self._line_chart(dd, "Floating Drawdown", "Drawdown", "floating_drawdown.png", "#dc2626")
        self._line_chart(margin, "Margin Usage", "Margin", "margin_usage.png", "#9333ea")
        self._line_chart(lots, "Open Lots", "Lots", "open_lots.png", "#ea580c")
        self._line_chart(exposure, "Net Exposure", "Lots (net)", "exposure.png", "#0891b2")
        self._line_chart(basket, "Basket Size (levels)", "Levels", "basket_size.png", "#65a30d")

        if self.state.cycles:
            recov = [c.recovery_time_seconds / 3600.0 for c in self.state.cycles
                     if c.recovery_time_seconds is not None]
            self._line_chart(pd.Series(recov), "Recovery Duration per Cycle (hours)",
                              "Hours", "recovery_duration.png", "#f59e0b")

            basket_sizes = [max(c.max_levels_buy, c.max_levels_sell) for c in self.state.cycles]
            self._hist_chart(basket_sizes, "Distribution of Basket Sizes", "Levels used",
                              "dist_basket_sizes.png", "#7c3aed")

            recov_dist = [c.worst_floating_pnl for c in self.state.cycles if c.worst_floating_pnl is not None]
            self._hist_chart(recov_dist, "Distribution of Recovery Distance (worst floating P/L)",
                              "Worst floating P/L", "dist_recovery_distance.png", "#dc2626")
