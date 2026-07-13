"""
SimulationReport
================
Writes the final statistics, trade log and cycle log to disk (JSON + CSV)
and prints a human-readable console summary.
"""

from __future__ import annotations
import json
import os

import pandas as pd

from .configuration import Config
from .engine import SimulationState


class SimulationReport:
    def __init__(self, config: Config, state: SimulationState, stats: dict):
        self.config = config
        self.state = state
        self.stats = stats

    def save(self):
        out_dir = self.config.output.results_dir
        os.makedirs(out_dir, exist_ok=True)
        name = self.config.output.report_name

        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, default=str)

        if self.config.output.save_cycle_log and self.state.cycles:
            pd.DataFrame([c.__dict__ for c in self.state.cycles]).to_csv(
                os.path.join(out_dir, f"{name}_cycles.csv"), index=False)

        if self.config.output.save_trade_log and self.state.trade_log:
            pd.DataFrame(self.state.trade_log).to_csv(
                os.path.join(out_dir, f"{name}_trades.csv"), index=False)

    def print_summary(self):
        s = self.stats
        print("\n" + "=" * 60)
        print(f"  HEDGE-GRID STRATEGY RESEARCH REPORT  [{self.config.data.symbol}]")
        print("=" * 60)
        print(f"Exit mode                 : {self.config.exit.mode}")
        print(f"Scale factor               : {self.config.strategy.scale_factor}")
        print(f"Grid distance (pips)        : {self.config.strategy.grid_distance_pips}")
        print(f"Max levels                  : {self.config.strategy.max_levels}")
        print("-" * 60)
        print(f"Net profit                  : {s['net_profit']:.2f}")
        print(f"Gross profit / Gross loss   : {s['gross_profit']:.2f} / {s['gross_loss']:.2f}")
        print(f"Profit factor                : {s['profit_factor']:.3f}")
        print(f"Recovery factor              : {s['recovery_factor']:.3f}")
        print(f"Sharpe ratio                 : {s['sharpe_ratio']}")
        print(f"Max drawdown (equity)        : {s['max_drawdown']:.2f}")
        print(f"Max floating drawdown        : {s['max_floating_drawdown']:.2f}")
        print(f"Worst equity                 : {s['worst_equity']:.2f}")
        print(f"Max margin used               : {s['max_margin_used']:.2f}")
        print(f"Largest exposure (net lots)  : {s['largest_exposure']:.2f}")
        print(f"Max open lots                 : {s['max_open_lots']:.2f}")
        print(f"Max basket size (levels)     : {s['max_basket_size']}")
        print("-" * 60)
        print(f"Hedge cycles (win/lose/total): {s['winning_cycles']} / {s['losing_cycles']} / {s['num_hedge_cycles']}")
        print(f"Pyramid trades / Martingale trades : {s['num_pyramid_trades']} / {s['num_martingale_trades']}")
        print(f"Avg / Max levels required     : {s['avg_levels_required']} / {s['max_levels_required']}")
        print(f"Avg / Max recovery time (h)  : "
              f"{(s['avg_recovery_time_seconds'] or 0) / 3600:.2f} / "
              f"{(s['max_recovery_time_seconds'] or 0) / 3600:.2f}")
        print("=" * 60 + "\n")
