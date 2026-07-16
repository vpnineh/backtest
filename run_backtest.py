#!/usr/bin/env python3
"""
run_backtest.py
================
CLI entry point.

Example:
    python run_backtest.py --symbol EURGBP --timeframe H1 --start-year 2020 --end-year 2024
    python run_backtest.py --config config/default.yaml --symbol AUDNZD --timeframe M15

All strategy parameters live in config/default.yaml. Anything passed on
the CLI overrides the YAML for that single run; nothing is auto-tuned.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import yaml

from backtest.data_loader import load_and_resample
from backtest.engine import BacktestEngine, StrategyConfig, RegimeParams
from backtest.fx_convert import build_conversion_series
from backtest.metrics import compute_stats
from backtest.report import write_trade_log, write_summary, plot_equity_curve

REPO_ROOT = Path(__file__).resolve().parent


def build_config(args) -> StrategyConfig:
    cfg_path = Path(args.config)
    raw = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}

    regime_raw = raw.pop("regime", {})
    regime = RegimeParams(**regime_raw) if regime_raw else RegimeParams()

    valid_keys = {f.name for f in fields(StrategyConfig)}
    raw = {k: v for k, v in raw.items() if k in valid_keys}
    cfg = StrategyConfig(**raw)
    cfg.regime = regime

    # CLI overrides
    if args.symbol:
        cfg.symbol = args.symbol
    if args.timeframe:
        cfg.timeframe = args.timeframe
    if args.start_year:
        cfg.start_year = args.start_year
    if args.end_year:
        cfg.end_year = args.end_year
    if args.initial_balance:
        cfg.initial_balance = args.initial_balance

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Adaptive Recovery Grid backtester (EURGBP / AUDNZD)")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output"))
    parser.add_argument("--symbol", choices=["EURGBP", "AUDNZD"], default=None)
    parser.add_argument("--timeframe", choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"], default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--initial-balance", type=float, default=None)
    args = parser.parse_args()

    cfg = build_config(args)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_backtest] symbol={cfg.symbol} timeframe={cfg.timeframe} "
          f"years={cfg.start_year}-{cfg.end_year}")

    t0 = time.time()
    df = load_and_resample(data_dir, cfg.symbol, cfg.timeframe, cfg.start_year, cfg.end_year)
    print(f"[run_backtest] loaded & resampled {len(df):,} bars in {time.time() - t0:.1f}s")

    conv = build_conversion_series(data_dir, cfg.symbol, cfg.timeframe, cfg.start_year, cfg.end_year)

    t0 = time.time()
    engine = BacktestEngine(df, cfg, conv_rate_at=conv.rate_at)
    result_df = engine.run()
    print(f"[run_backtest] simulated {len(result_df):,} bars, "
          f"{len(engine.trades)} closed trades in {time.time() - t0:.1f}s")

    stats = compute_stats(result_df, engine.trades, cfg)

    tag = f"{cfg.symbol}_{cfg.timeframe}_{cfg.start_year}-{cfg.end_year}"
    write_trade_log(engine.trades, out_dir / f"trades_{tag}.csv")
    write_summary(stats, out_dir / f"summary_{tag}.json", out_dir / f"summary_{tag}.txt", cfg, diag=engine.diag, trades=engine.trades)
    plot_equity_curve(result_df, out_dir / f"equity_{tag}.png", cfg)

    print(f"[run_backtest] artifacts written to {out_dir}/")


if __name__ == "__main__":
    main()
