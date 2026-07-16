"""
report.py
==========
Writes out the artifacts a person (or a GitHub Actions job) needs to
judge the backtest: a text/JSON summary, a full trade log CSV, and an
equity-curve PNG.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from .engine import Trade
from .metrics import BacktestStats, stats_to_dict


def write_trade_log(trades: List[Trade], out_path: Path):
    rows = []
    for t in trades:
        rows.append({
            "open_time": t.open_time,
            "close_time": t.close_time,
            "direction": t.direction,
            "lots": t.lots,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl_usd": round(t.pnl_usd, 2),
            "reason": t.reason,
            "grid_levels_used": t.grid_levels_used,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_summary(stats: BacktestStats, out_path_json: Path, out_path_txt: Path, cfg, diag: dict = None):
    d = stats_to_dict(stats)
    if diag:
        d["diagnostics"] = diag
    out_path_json.write_text(json.dumps(d, indent=2, default=str))

    lines = [
        "=" * 70,
        f"ADAPTIVE RECOVERY GRID - BACKTEST REPORT",
        "=" * 70,
        f"Symbol:              {stats.symbol}",
        f"Timeframe:            {stats.timeframe}",
        f"Period:               {stats.start_year} - {stats.end_year}",
        "-" * 70,
        f"Initial balance:      {stats.initial_balance:,.2f} USD",
        f"Final balance:        {stats.final_balance:,.2f} USD",
        f"Net profit:           {stats.net_profit_usd:,.2f} USD  ({stats.net_profit_pct:.2f}%)",
        "-" * 70,
        f"Total trades (closes):{stats.total_trades}",
        f"Win rate:             {stats.win_rate_pct:.2f}%",
        f"Profit factor:        {stats.profit_factor:.2f}",
        f"Expectancy / close:   {stats.expectancy_usd:,.2f} USD",
        f"Avg grid levels used: {stats.avg_grid_levels_used:.2f}",
        f"Avg trade duration:   {stats.avg_trade_duration_bars:.1f} bars",
        f"Exposure time:        {stats.exposure_time_pct:.2f}%",
        "-" * 70,
        f"Max drawdown:         {stats.max_drawdown_pct:.2f}%  ({stats.max_drawdown_usd:,.2f} USD)",
        f"Max DD duration:      {stats.max_dd_duration_bars} bars",
        f"Max consecutive losses:{stats.max_consecutive_losses}",
        f"Sharpe (annualized):  {stats.sharpe_annualized:.2f}",
        f"Calmar ratio:         {stats.calmar_ratio:.2f}",
        f"Recovery factor:      {stats.recovery_factor:.2f}",
        "=" * 70,
    ]

    if diag:
        lines += [
            "DIAGNOSTICS (why the engine did/didn't trade -- debug aid, not P&L):",
            "-" * 70,
            f"Regime distribution:  {diag.get('regime_distribution_pct', {})}",
            f"Baskets opened:       {diag.get('baskets_opened', 0)}",
            f"Recovery add-ons:     {diag.get('recovery_additions', 0)}",
            f"Recovery checks:      {diag.get('recovery_checked', 0)} "
            f"(approved: {diag.get('recovery_approved', 0)})",
            f"Recovery reject reasons (count): {diag.get('recovery_rejected_reasons', {})}",
            f"Breakout stops triggered: {diag.get('breakout_stops_triggered', 0)}",
            f"Forced closes (max floating DD): {diag.get('forced_closes_max_dd', 0)}",
            f"Daily loss lock events:  {diag.get('daily_lock_events', 0)}",
            f"Weekly loss lock events: {diag.get('weekly_lock_events', 0)}",
            "Entry filter independent hit-rates (% of valid bars, NOT AND-ed",
            "except the *_ALL_COMBINED rows -- use this to spot the bottleneck):",
        ]
        for k, v in diag.get("filter_hit_rates_pct", {}).items():
            lines.append(f"    {k:<32s} {v:6.3f}%")
        lines.append("=" * 70)

    lines += [
        "NOTES / LIMITATIONS (read before trusting these numbers):",
        " - Costs: spread + commission + slippage are modeled, not scraped from",
        "   a real broker feed (M1 history has no true bid/ask spread ticks).",
        " - P&L is converted to USD using real GBPUSD/NZDUSD history where",
        "   available, else a fixed fallback rate (see console warnings).",
        " - News calendar is NOT integrated (no offline dataset provided);",
        "   the breakout/volatility-spike filters partially substitute for it.",
        " - No parameter fitting/optimization was performed on this data.",
        "   Re-run with different --start-year/--end-year to sanity check",
        "   robustness across regimes.",
        "=" * 70,
    ]
    out_path_txt.write_text("\n".join(lines))
    print("\n".join(lines))


def plot_equity_curve(df: pd.DataFrame, out_path_png: Path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(df["datetime"], df["equity"], label="Equity", linewidth=1.2)
    axes[0].plot(df["datetime"], df["balance"], label="Balance", linewidth=0.8, alpha=0.6)
    axes[0].set_title(f"{cfg.symbol} {cfg.timeframe} | {cfg.start_year}-{cfg.end_year} - Adaptive Recovery Grid")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    running_max = df["equity"].cummax()
    dd = (df["equity"] - running_max) / running_max * 100.0
    axes[1].fill_between(df["datetime"], dd, 0, color="red", alpha=0.4)
    axes[1].set_ylabel("Drawdown %")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path_png, dpi=130)
    plt.close(fig)
