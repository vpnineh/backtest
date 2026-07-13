"""
Optimizer / Monte Carlo scaffolding
====================================
Architecture hooks required by the spec: the engine is fully driven by
`Config`, so parameter optimization is just "build N configs, run N
backtests, compare stats". Monte Carlo is "resample the realized
cycle P/Ls and re-derive equity outcomes".

IMPORTANT - what is and isn't safe to parallelize
--------------------------------------------------
A single continuous backtest (BacktestEngine.run over one price series)
is path-dependent: open positions and balance at time T depend on
everything that happened before T. Splitting that single run across
processes/threads would silently corrupt results (a position opened in
"chunk 2" might really have needed context from the end of "chunk 1").
This module therefore NEVER parallelizes a single run.

What IS embarrassingly parallel (independent units of work, safe to run
on separate cores with zero accuracy cost):
  - grid_search / grid_search_parallel: each parameter combination is a
    fully independent backtest.
  - monte_carlo_cycle_resample / monte_carlo_cycle_resample_parallel:
    each Monte Carlo draw independently resamples the SAME already-computed
    list of realized cycle P/Ls - it does not re-run the engine at all.
"""

from __future__ import annotations
import copy
import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import pandas as pd

from .configuration import Config
from .engine import BacktestEngine
from .statistics_engine import StatisticsEngine


def _run_one_combo(args) -> dict:
    base_config, keys, combo, objective = args
    cfg = copy.deepcopy(base_config)
    for key, value in zip(keys, combo):
        section, attr = key.split(".")
        setattr(getattr(cfg, section), attr, value)

    # NOTE: `data` is re-passed via the module-level _DATA_CACHE set by the
    # parent process before submission, to avoid re-pickling large frames
    # per task when running under ProcessPoolExecutor with fork start method.
    data = _DATA_CACHE
    engine = BacktestEngine(cfg, data)
    state = engine.run()
    stats = StatisticsEngine(state, cfg.account.starting_balance).compute()

    row = dict(zip(keys, combo))
    row["score"] = objective(stats)
    row.update(stats)
    return row


_DATA_CACHE = None  # populated in the worker via initializer


def _init_worker(data: pd.DataFrame):
    global _DATA_CACHE
    _DATA_CACHE = data


def grid_search(base_config: Config, data: pd.DataFrame, param_grid: dict,
                 objective: Callable[[dict], float] = lambda stats: stats["net_profit"]) -> pd.DataFrame:
    """Sequential grid search (single core) - simplest, always correct."""
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results = [_run_one_combo_sequential(base_config, keys, combo, objective, data) for combo in combos]
    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


def _run_one_combo_sequential(base_config, keys, combo, objective, data) -> dict:
    cfg = copy.deepcopy(base_config)
    for key, value in zip(keys, combo):
        section, attr = key.split(".")
        setattr(getattr(cfg, section), attr, value)
    engine = BacktestEngine(cfg, data)
    state = engine.run()
    stats = StatisticsEngine(state, cfg.account.starting_balance).compute()
    row = dict(zip(keys, combo))
    row["score"] = objective(stats)
    row.update(stats)
    return row


def grid_search_parallel(base_config: Config, data: pd.DataFrame, param_grid: dict,
                          objective: Callable[[dict], float] = lambda stats: stats["net_profit"],
                          n_workers: int = 0) -> pd.DataFrame:
    """
    Parallel grid search across CPU cores. SAFE because every parameter
    combination is an entirely independent backtest run - no shared state.
    n_workers=0 uses os.cpu_count().
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    workers = n_workers or os.cpu_count() or 1

    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(data,)) as pool:
        tasks = [(base_config, keys, combo, objective) for combo in combos]
        for row in pool.map(_run_one_combo, tasks):
            results.append(row)

    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


def _mc_batch(args) -> list[dict]:
    pnls, batch_size, seed = args
    rng = random.Random(seed)
    outcomes = []
    for _ in range(batch_size):
        sample = [rng.choice(pnls) for _ in pnls]
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in sample:
            running += pnl
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)
        outcomes.append({"total_pnl": running, "max_drawdown": max_dd})
    return outcomes


def monte_carlo_cycle_resample(state, starting_balance: float, n_runs: int = 500,
                                seed: int | None = None) -> pd.DataFrame:
    """Sequential Monte Carlo resample of realized cycle P/Ls (single core)."""
    pnls = [c.realized_pnl for c in state.cycles if c.realized_pnl is not None]
    if not pnls:
        return pd.DataFrame()
    outcomes = _mc_batch((pnls, n_runs, seed))
    df = pd.DataFrame(outcomes)
    df["final_balance"] = starting_balance + df["total_pnl"]
    return df


def monte_carlo_cycle_resample_parallel(state, starting_balance: float, n_runs: int = 5000,
                                         seed: int | None = None, n_workers: int = 0) -> pd.DataFrame:
    """
    Parallel Monte Carlo. SAFE because each draw independently resamples the
    SAME fixed list of already-computed realized cycle P/Ls - no engine
    re-run, no shared mutable state between draws.
    """
    pnls = [c.realized_pnl for c in state.cycles if c.realized_pnl is not None]
    if not pnls:
        return pd.DataFrame()

    workers = n_workers or os.cpu_count() or 1
    per_worker = max(1, n_runs // workers)
    rng = random.Random(seed)
    tasks = [(pnls, per_worker, rng.randint(0, 2**31 - 1)) for _ in range(workers)]

    all_outcomes = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(_mc_batch, tasks):
            all_outcomes.extend(batch)

    df = pd.DataFrame(all_outcomes)
    df["final_balance"] = starting_balance + df["total_pnl"]
    return df
