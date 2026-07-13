#!/usr/bin/env python3
"""
Entry point for the Hedge-Grid research backtester.

Usage:
    python main.py --config config.yaml

If config.yaml has `exit.mode: "ALL"`, every exit mode (A, B, C, D) is run
on the EXACT SAME loaded/resampled data and the SAME other parameters, and
a side-by-side comparison table is produced. This is the correct way to
evaluate exit algorithms per the project spec - you compare all of them,
you don't pick one in advance.
"""
from __future__ import annotations
import argparse
import copy
import os
import time as _time

import pandas as pd

from hedge_sim.configuration import Config
from hedge_sim.data_loader import DataLoader
from hedge_sim.resampler import resample_ohlc, add_atr
from hedge_sim.engine import BacktestEngine
from hedge_sim.statistics_engine import StatisticsEngine
from hedge_sim.chart_engine import ChartEngine
from hedge_sim.simulation_report import SimulationReport

ALL_EXIT_MODES = ["A", "B", "C", "D"]


def load_data(config: Config) -> pd.DataFrame:
    print(f"[data] Loading data from '{config.data.path}' ...")
    data = DataLoader(config.data).load()
    print(f"       Loaded {len(data):,} native candles from {data['time'].iloc[0]} to {data['time'].iloc[-1]}")

    if config.data.resample_timeframe:
        before = len(data)
        data = resample_ohlc(data, config.data.resample_timeframe)
        print(f"       Resampled {before:,} -> {len(data):,} candles at '{config.data.resample_timeframe}' "
              f"(open=first, high=max, low=min, close=last, volume=sum - lossless OHLC aggregation)")

    if config.strategy.grid_mode == "atr":
        data = add_atr(data, config.strategy.atr_period)
        print(f"       ATR({config.strategy.atr_period}) computed for dynamic grid distance")

    return data


def run_single_mode(config: Config, data: pd.DataFrame, results_subdir: str | None = None) -> dict:
    """Runs the engine once for whatever exit.mode is set on `config`,
    saves the full report/charts, and returns the stats dict."""
    cfg = copy.deepcopy(config)
    if results_subdir:
        cfg.output.results_dir = os.path.join(config.output.results_dir, results_subdir)

    print(f"[run] Exit Mode {cfg.exit.mode} - running simulation over {len(data):,} candles ...")
    t0 = _time.time()
    engine = BacktestEngine(cfg, data)
    state = engine.run()
    elapsed = _time.time() - t0
    print(f"      finished in {elapsed:.1f}s, {len(state.cycles)} hedge cycles executed.")

    stats = StatisticsEngine(state, cfg.account.starting_balance).compute()
    stats["exit_mode"] = cfg.exit.mode
    stats["elapsed_seconds"] = elapsed

    report = SimulationReport(cfg, state, stats)
    report.save()
    report.print_summary()
    report.print_cycle_reports()
    detail_path = report.save_cycle_reports_txt()
    print(f"      Full per-cycle detail saved to '{detail_path}'")

    if cfg.output.generate_charts:
        ChartEngine(state, cfg.output.results_dir).generate_all()
        print(f"      Charts saved to '{cfg.output.results_dir}/'")

    return stats


def print_comparison_table(all_stats: list[dict]):
    cols = ["exit_mode", "num_hedge_cycles", "winning_cycles", "losing_cycles",
            "net_profit", "profit_factor", "max_drawdown", "max_floating_drawdown",
            "worst_equity", "max_margin_used", "avg_levels_required", "max_levels_required"]
    df = pd.DataFrame(all_stats)[cols]
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print("\n" + "=" * 100)
    print("  EXIT MODE COMPARISON  (same data, same strategy params - only the exit algorithm differs)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    return df


def main():
    parser = argparse.ArgumentParser(description="Hedge-Grid strategy research backtester")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    print(f"[config] Loading configuration from {args.config} ...")
    config = Config.from_yaml(args.config)

    data = load_data(config)

    mode = str(config.exit.mode).upper().strip()

    if mode == "ALL":
        print(f"\nexit.mode = ALL -> running every exit strategy (A, B, C, D) on the SAME data.\n")
        all_stats = []
        for m in ALL_EXIT_MODES:
            cfg_m = copy.deepcopy(config)
            cfg_m.exit.mode = m
            stats = run_single_mode(cfg_m, data, results_subdir=f"mode_{m}")
            all_stats.append(stats)

        comparison_df = print_comparison_table(all_stats)
        out_dir = config.output.results_dir
        os.makedirs(out_dir, exist_ok=True)
        comparison_path = os.path.join(out_dir, "exit_mode_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison table saved to '{comparison_path}'")
        print(f"Per-mode full reports/charts saved under '{out_dir}/mode_A/', 'mode_B/', 'mode_C/', 'mode_D/'")
    else:
        run_single_mode(config, data)


if __name__ == "__main__":
    main()
