#!/usr/bin/env python3
"""
Entry point for the Hedge-Grid research backtester.

Usage:
    python main.py --config config.yaml
"""
from __future__ import annotations
import argparse
import time as _time

from hedge_sim.configuration import Config
from hedge_sim.data_loader import DataLoader
from hedge_sim.engine import BacktestEngine
from hedge_sim.statistics_engine import StatisticsEngine
from hedge_sim.chart_engine import ChartEngine
from hedge_sim.simulation_report import SimulationReport


def main():
    parser = argparse.ArgumentParser(description="Hedge-Grid strategy research backtester")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    print(f"[1/5] Loading configuration from {args.config} ...")
    config = Config.from_yaml(args.config)

    print(f"[2/5] Loading data from '{config.data.path}' ...")
    data = DataLoader(config.data).load()
    print(f"      Loaded {len(data):,} candles from {data['time'].iloc[0]} to {data['time'].iloc[-1]}")

    print("[3/5] Running simulation ...")
    t0 = _time.time()
    engine = BacktestEngine(config, data)
    state = engine.run()
    print(f"      Simulation finished in {_time.time() - t0:.1f}s, {len(state.cycles)} hedge cycles executed.")

    print("[4/5] Computing statistics ...")
    stats = StatisticsEngine(state, config.account.starting_balance).compute()

    print("[5/5] Saving report & charts ...")
    report = SimulationReport(config, state, stats)
    report.save()
    report.print_summary()

    if config.output.generate_charts:
        ChartEngine(state, config.output.results_dir).generate_all()
        print(f"Charts saved to '{config.output.results_dir}/'")

    print(f"Full report saved to '{config.output.results_dir}/{config.output.report_name}.json'")


if __name__ == "__main__":
    main()
