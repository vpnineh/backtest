"""
Main entry point for the Professional Martingale Strategy Backtest.

Usage:
    python main.py
    python main.py --start 2015-01-01 --end 2023-12-31
    python main.py --pairs EURGBP --years 2018 2019 2020
"""

import argparse
import logging
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from src.data_loader import DataStore
from src.backtester import Backtester
from src.reporter import generate_report


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("backtest.log"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Professional Martingale Strategy Backtester"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing CSV/ZIP data files"
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: all available)"
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: all available)"
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Pairs to test (default: EURGBP AUDNZD)"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Specific years to load (default: all)"
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Initial capital in USD"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    logger.info("="*60)
    logger.info("Professional Martingale Strategy Backtester")
    logger.info("="*60)

    # Override config from args
    if args.pairs:
        CONFIG.pairs = args.pairs
    if args.capital:
        CONFIG.initial_capital = args.capital

    logger.info(f"Pairs:           {CONFIG.pairs}")
    logger.info(f"Initial capital: ${CONFIG.initial_capital:,.2f}")
    logger.info(f"Data directory:  {args.data_dir}")
    logger.info(f"Period:          {args.start or 'all'} → {args.end or 'all'}")

    # Load data
    data_store = DataStore(
        data_dir=args.data_dir,
        pairs=CONFIG.pairs,
        years=args.years,
    )

    try:
        data_store.load()
    except ValueError as e:
        logger.error(f"Data loading failed: {e}")
        logger.error(
            "Make sure data files exist in the data/ directory.\n"
            "Expected filenames:\n"
            "  HISTDATA_COM_ASCII_EURGBP_M1{year}.zip\n"
            "  HISTDATA_COM_ASCII_AUDNZD_M1{year}.zip\n"
            "  or: DAT_ASCII_EURGBP_M1_{year}.csv"
        )
        sys.exit(1)

    # Run backtest
    engine = Backtester(config=CONFIG, data_store=data_store)

    result = engine.run(
        start_date=args.start,
        end_date=args.end,
    )

    # Generate reports
    generate_report(result, output_dir=args.output_dir)

    logger.info("Done. Check results/ directory for reports.")
    return result


if __name__ == "__main__":
    main()
