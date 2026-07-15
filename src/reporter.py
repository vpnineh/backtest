"""
Report generator: produces charts and statistics.
"""

from __future__ import annotations

import os
import logging
from typing import List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from .backtester import BacktestResult

logger = logging.getLogger(__name__)

sns.set_theme(style="darkgrid")


def generate_report(result: BacktestResult, output_dir: str = "results"):
    """Generate all charts and save summary CSV."""
    os.makedirs(output_dir, exist_ok=True)

    summary = result.summary()

    # Print summary
    print("\n" + "="*60)
    print("BACKTEST SUMMARY - Professional Martingale Strategy")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k:<30} {v}")
    print("="*60 + "\n")

    # Save summary
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "summary.csv"), index=False
    )

    # Charts
    _plot_equity_curve(result, output_dir)
    _plot_drawdown(result, output_dir)
    _plot_basket_stats(result, output_dir)
    _plot_monthly_returns(result, output_dir)
    _plot_level_distribution(result, output_dir)

    # Trade log CSV
    if result.trade_log:
        trade_df = pd.DataFrame([vars(t) for t in result.trade_log])
        trade_df.to_csv(os.path.join(output_dir, "trade_log.csv"), index=False)

    # Basket log CSV
    if result.basket_log:
        pd.DataFrame(result.basket_log).to_csv(
            os.path.join(output_dir, "basket_log.csv"), index=False
        )

    logger.info(f"Reports saved to: {output_dir}/")


def _plot_equity_curve(result: BacktestResult, output_dir: str):
    fig, ax = plt.subplots(figsize=(14, 6))

    eq = result.equity_curve
    ax.plot(eq.index, eq.values, linewidth=1.5, color="#2196F3", label="Equity")
    ax.axhline(
        result.initial_capital, color="#FF5722", linestyle="--",
        linewidth=1, label=f"Initial: ${result.initial_capital:,.0f}"
    )

    ax.fill_between(
        eq.index, result.initial_capital, eq.values,
        where=(eq.values >= result.initial_capital),
        alpha=0.2, color="green", label="Profit"
    )
    ax.fill_between(
        eq.index, result.initial_capital, eq.values,
        where=(eq.values < result.initial_capital),
        alpha=0.2, color="red", label="Loss"
    )

    ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equity_curve.png"), dpi=150)
    plt.close()


def _plot_drawdown(result: BacktestResult, output_dir: str):
    eq   = result.equity_curve
    peak = eq.cummax()
    dd   = (peak - eq) / peak * 100  # in %

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dd.index, 0, -dd.values, color="#F44336", alpha=0.7)
    ax.plot(dd.index, -dd.values, color="#B71C1C", linewidth=0.8)

    ax.set_title("Drawdown (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drawdown.png"), dpi=150)
    plt.close()


def _plot_basket_stats(result: BacktestResult, output_dir: str):
    if not result.basket_log:
        return

    df = pd.DataFrame(result.basket_log)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # P&L distribution
    ax = axes[0]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in df["pnl_usd"]]
    ax.bar(range(len(df)), df["pnl_usd"].values, color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Basket P&L (USD)")
    ax.set_xlabel("Basket #")
    ax.set_ylabel("P&L (USD)")

    # Win/loss pie
    ax = axes[1]
    wins   = (df["pnl_usd"] > 0).sum()
    losses = (df["pnl_usd"] <= 0).sum()
    ax.pie(
        [wins, losses],
        labels=[f"Win ({wins})", f"Loss ({losses})"],
        colors=["#4CAF50", "#F44336"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Win Rate")

    # Levels used histogram
    ax = axes[2]
    ax.hist(df["levels_used"], bins=range(1, 9), color="#2196F3", edgecolor="white")
    ax.set_title("Grid Levels Used per Basket")
    ax.set_xlabel("Levels")
    ax.set_ylabel("Count")
    ax.set_xticks(range(1, 8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "basket_stats.png"), dpi=150)
    plt.close()


def _plot_monthly_returns(result: BacktestResult, output_dir: str):
    eq = result.equity_curve

    # Monthly returns from equity curve
    monthly = eq.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100

    if len(monthly_ret) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in monthly_ret.values]
    ax.bar(
        range(len(monthly_ret)),
        monthly_ret.values,
        color=colors,
        width=0.8,
    )
    ax.axhline(0, color="black", linewidth=0.8)

    labels = [d.strftime("%Y-%m") for d in monthly_ret.index]
    step   = max(1, len(labels) // 20)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)

    ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_returns.png"), dpi=150)
    plt.close()


def _plot_level_distribution(result: BacktestResult, output_dir: str):
    """Show how often each grid level was reached and its contribution to P&L."""
    if not result.trade_log:
        return

    df = pd.DataFrame([vars(t) for t in result.trade_log])

    level_pnl = df.groupby("level")["pnl_usd"].sum()
    level_cnt = df.groupby("level")["pnl_usd"].count()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in level_pnl.values]
    ax.bar(level_pnl.index, level_pnl.values, color=colors)
    ax.set_title("P&L by Grid Level (USD)")
    ax.set_xlabel("Grid Level")
    ax.set_ylabel("Total P&L (USD)")
    ax.axhline(0, color="black", linewidth=0.8)

    ax = axes[1]
    ax.bar(level_cnt.index, level_cnt.values, color="#2196F3")
    ax.set_title("Trade Count by Grid Level")
    ax.set_xlabel("Grid Level")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "level_analysis.png"), dpi=150)
    plt.close()
