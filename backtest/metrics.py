"""
metrics.py
===========
Performance statistics computed strictly from realized trades + the
mark-to-market equity curve produced by engine.py. No forward-looking
adjustments of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import numpy as np
import pandas as pd

from .engine import Trade

BARS_PER_YEAR = {
    "M1": 365 * 24 * 60, "M5": 365 * 24 * 12, "M15": 365 * 24 * 4,
    "M30": 365 * 24 * 2, "H1": 365 * 24, "H4": 365 * 6, "D1": 252,
}


@dataclass
class BacktestStats:
    symbol: str
    timeframe: str
    start_year: int
    end_year: int
    initial_balance: float
    final_balance: float
    net_profit_usd: float
    net_profit_pct: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_usd: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    max_dd_duration_bars: int
    sharpe_annualized: float
    calmar_ratio: float
    recovery_factor: float
    max_consecutive_losses: int
    avg_grid_levels_used: float
    avg_trade_duration_bars: float
    exposure_time_pct: float


def compute_drawdown(equity: np.ndarray):
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd_pct = dd.min() * 100.0 if len(dd) else 0.0
    max_dd_usd = (equity - running_max).min() if len(dd) else 0.0

    # longest duration underwater
    max_dur = 0
    cur_dur = 0
    for val in dd:
        if val < 0:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dd_pct, max_dd_usd, max_dur


def compute_stats(df: pd.DataFrame, trades: List[Trade], cfg) -> BacktestStats:
    equity = df["equity"].values
    initial = cfg.initial_balance
    final = equity[-1] if len(equity) else initial

    net_profit = final - initial
    net_profit_pct = 100.0 * net_profit / initial if initial else 0.0

    pnls = np.array([t.pnl_usd for t in trades])
    total_trades = len(trades)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = 100.0 * len(wins) / total_trades if total_trades else 0.0
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    expectancy = pnls.mean() if total_trades else 0.0

    max_dd_pct, max_dd_usd, max_dd_dur = compute_drawdown(equity)

    # returns per bar for Sharpe (only over bars once trading actually starts)
    rets = np.diff(equity) / np.where(equity[:-1] == 0, 1, equity[:-1])
    bars_per_year = BARS_PER_YEAR.get(cfg.timeframe.upper(), 252)
    if rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    years = max((cfg.end_year - cfg.start_year + 1), 1)
    cagr_like = net_profit_pct / years
    calmar = (cagr_like / abs(max_dd_pct)) if max_dd_pct != 0 else 0.0

    recovery_factor = (net_profit / abs(max_dd_usd)) if max_dd_usd != 0 else 0.0

    # consecutive losses
    max_consec = 0
    cur = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    avg_grid = np.mean([t.grid_levels_used for t in trades]) if trades else 0.0

    durations = []
    for t in trades:
        try:
            durations.append((t.close_time - t.open_time).total_seconds())
        except Exception:
            pass
    avg_duration_bars = 0.0
    if durations and cfg.timeframe.upper() in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
        bar_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}[cfg.timeframe.upper()]
        avg_duration_bars = float(np.mean(durations) / bar_seconds)

    # exposure: fraction of total elapsed time a basket was open (approx,
    # from trade open/close timestamps relative to total backtest span)
    if len(df) > 1 and trades:
        total_span = (df["datetime"].iat[-1] - df["datetime"].iat[0]).total_seconds()
        busy = sum((t.close_time - t.open_time).total_seconds() for t in trades)
        exposure_pct = 100.0 * min(busy / total_span, 1.0) if total_span > 0 else 0.0
    else:
        exposure_pct = 0.0

    return BacktestStats(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        initial_balance=initial,
        final_balance=final,
        net_profit_usd=net_profit,
        net_profit_pct=net_profit_pct,
        total_trades=total_trades,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        expectancy_usd=expectancy,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_usd=max_dd_usd,
        max_dd_duration_bars=max_dd_dur,
        sharpe_annualized=sharpe,
        calmar_ratio=calmar,
        recovery_factor=recovery_factor,
        max_consecutive_losses=max_consec,
        avg_grid_levels_used=avg_grid,
        avg_trade_duration_bars=avg_duration_bars,
        exposure_time_pct=exposure_pct,
    )


def stats_to_dict(stats: BacktestStats) -> dict:
    return asdict(stats)
