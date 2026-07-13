#!/usr/bin/env python3
"""
Entry point for the Hedge-Grid research backtester.
Added multi-processing scaffolding for optimization/parallel runs.
"""
from __future__ import annotations
import argparse
import time as _time
from concurrent.futures import ProcessPoolExecutor

from hedge_sim.configuration import Config
from hedge_sim.data_loader import DataLoader
from hedge_sim.engine import BacktestEngine
from hedge_sim.statistics_engine import StatisticsEngine
from hedge_sim.chart_engine import ChartEngine
from hedge_sim.simulation_report import SimulationReport


def run_single_backtest(config: Config):
    print(f"[Core] Loading data from '{config.data.path}'...")
    data = DataLoader(config.data).load()
    
    t0 = _time.time()
    engine = BacktestEngine(config, data)
    state = engine.run()
    
    stats = StatisticsEngine(state, config.account.starting_balance).compute()
    report = SimulationReport(config, state, stats)
    report.save()
    
    if config.output.generate_charts:
        ChartEngine(state, config.output.results_dir).generate_all()
        
    return _time.time() - t0, len(state.cycles)


def main():
    parser = argparse.ArgumentParser(description="Hedge-Grid strategy research backtester")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    
    # -------------------------------------------------------------
    # Parallel processing wrapper: 
    # Used for running Multiple configs (Optimization) simultaneously
    # -------------------------------------------------------------
    configs_to_run = [config] # List of configs can be expanded here for Grid Search
    
    print(f"Starting {len(configs_to_run)} simulation(s) using {config.execution.max_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=config.execution.max_workers) as executor:
        futures = [executor.submit(run_single_backtest, cfg) for cfg in configs_to_run]
        
        for future in futures:
            duration, cycles = future.result()
            print(f"Simulation finished in {duration:.1f}s, {cycles} hedge cycles executed.")
            
    print(f"Full report & charts saved to '{config.output.results_dir}/'")


if __name__ == "__main__":
    main()
