"""
Optimizer / Monte Carlo scaffolding
====================================
Architecture hooks required by the spec: the engine is fully driven by
`Config`, so parameter optimization is just "build N configs, run N
backtests, compare stats". Monte Carlo is "shuffle/resample the realized
cycle P/Ls (or the price series) and re-derive equity curves".

These are intentionally lightweight - they exist so the framework is
*ready* for optimization/Monte Carlo studies without hardcoding any
particular search algorithm or objective function.
"""

from __future__ import annotations
import copy
import itertools
import random
from typing import Callable, Iterable

import pandas as pd

from .configuration import Config
from .engine import BacktestEngine
from .statistics_engine import StatisticsEngine


def grid_search(base_config: Config, data: pd.DataFrame, param_grid: dict,
                 objective: Callable[[dict], float] = lambda stats: stats["net_profit"]) -> pd.DataFrame:
    """
    param_grid example:
        {
          "strategy.scale_factor": [1.1, 1.2, 1.3],
          "strategy.grid_distance_pips": [10, 15, 20],
          "exit.mode": ["A", "B", "C", "D"],
        }
    Dotted keys are resolved against the Config dataclasses.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    results = []

    for combo in combos:
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
        results.append(row)

    return pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)


def monte_carlo_cycle_resample(state, starting_balance: float, n_runs: int = 500,
                                seed: int | None = None) -> pd.DataFrame:
    """Resamples realized cycle P/Ls (with replacement) to build a
    distribution of possible equity outcomes - a fast proxy Monte Carlo
    that does not require re-running the full candle-by-candle engine."""
    rng = random.Random(seed)
    pnls = [c.realized_pnl for c in state.cycles if c.realized_pnl is not None]
    if not pnls:
        return pd.DataFrame()

    outcomes = []
    for _ in range(n_runs):
        sample = [rng.choice(pnls) for _ in pnls]
        final_balance = starting_balance + sum(sample)
        running = starting_balance
        peak = starting_balance
        max_dd = 0.0
        for pnl in sample:
            running += pnl
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)
        outcomes.append({"final_balance": final_balance, "max_drawdown": max_dd})

    return pd.DataFrame(outcomes)
