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

    def _format_cycle(self, c) -> str:
        lines = [f"Cycle #{c.cycle_id}"]
        lines.append(f"  Open Date                  : {c.open_time}")
        lines.append(f"  Close Date                  : {c.close_time}")
        lines.append(f"  Direction Movement          : {c.direction_movement}  "
                      f"({c.open_price:.5f} -> {c.exit_price:.5f})")
        lines.append(f"  Total Buy Lots              : {c.total_buy_lots:.2f}")
        lines.append(f"  Total Sell Lots             : {c.total_sell_lots:.2f}")
        lines.append(f"  Average Buy Price           : {c.avg_buy_price:.5f}")
        lines.append(f"  Average Sell Price          : {c.avg_sell_price:.5f}")
        lines.append(f"  Number of Pyramid Positions : {c.num_pyramid_positions}")
        lines.append(f"  Number of Martingale Positions : {c.num_martingale_positions}")
        lines.append(f"  Net Floating P/L (final)    : {c.realized_pnl:.2f}")
        be = f"{c.breakeven_price:.5f}" if c.breakeven_price is not None else "N/A (net-neutral lots)"
        lines.append(f"  Break-even Price            : {be}")
        rd = f"{c.recovery_distance_pips:.1f} pips" if c.recovery_distance_pips is not None else "N/A"
        lines.append(f"  Recovery Distance Required  : {rd}  (from worst point to break-even)")
        lines.append(f"  Maximum Adverse Excursion   : {abs(min(c.worst_floating_pnl, 0.0)):.2f}  "
                      f"(worst floating P/L: {c.worst_floating_pnl:.2f} @ {c.worst_price:.5f})")
        lines.append(f"  Maximum Favorable Excursion : {max(c.best_floating_pnl, 0.0):.2f}  "
                      f"(best floating P/L: {c.best_floating_pnl:.2f} @ {c.best_price:.5f})")
        rp = f"{c.recovery_percentage:.1f}%" if c.recovery_percentage is not None else "N/A"
        lines.append(f"  Recovery Percentage         : {rp}")
        if c.recovery_time_seconds is not None:
            hrs = c.recovery_time_seconds / 3600.0
            lines.append(f"  Time To Recovery            : {hrs:.2f} hours")
        else:
            lines.append(f"  Time To Recovery            : did not fully recover before close")
        lines.append(f"  Exit Price                  : {c.exit_price:.5f}  (reason: {c.exit_reason})")
        lines.append(f"  Max levels (buy/sell)       : {c.max_levels_buy} / {c.max_levels_sell}")
        return "\n".join(lines)

    def print_cycle_reports(self, n: int | None = None):
        cycles = self.state.cycles
        if not cycles:
            print("No hedge cycles were executed.")
            return
        n = n if n is not None else self.config.output.print_last_n_cycles
        selected = cycles[-n:] if n and n > 0 else cycles
        print("\n" + "-" * 60)
        print(f"  PER-CYCLE DETAIL (last {len(selected)} of {len(cycles)} cycles)")
        print("-" * 60)
        for c in selected:
            print(self._format_cycle(c))
            print()

    def save_cycle_reports_txt(self):
        """Writes the Cycle #N formatted report for EVERY cycle to a text file."""
        out_dir = self.config.output.results_dir
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.config.output.report_name}_cycles_detail.txt")
        with open(path, "w", encoding="utf-8") as f:
            for c in self.state.cycles:
                f.write(self._format_cycle(c))
                f.write("\n\n")
        return path

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
