# main.py
import sys
import json
from pathlib import Path
from loguru import logger
from src.config import TradingCosts, StrategyParams, BacktestSettings
from src.etl import extract_histdata_to_parquet
from src.strategy import TrendFollowingStrategy
from src.engine import RealisticBacktestEngine
import polars as pl

def setup_logger():
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    logger.add("backtest.log", rotation="10 MB", level="DEBUG")

def print_report(report: dict):
    if "error" in report:
        print("\n" + "="*60)
        print("⚠️  BACKTEST FAILED")
        print("="*60)
        print(f"Reason: {report['error']}")
        print("="*60 + "\n")
        return

    print("\n" + "="*60)
    print(f"📊 BACKTEST REPORT: {report['symbol']} {report['timeframe']} ({report['years']})")
    print(f"Sizing Mode:          {report.get('sizing_mode', 'unknown')}")
    print("="*60)
    print(f"Initial Balance:      ${report['initial_balance']:,.2f}")
    print(f"Final Balance:        ${report['final_balance']:,.2f}")
    print(f"Total Return:         {report['total_return_pct']:.2f}%")
    print("-" * 60)
    print(f"Total Trades:         {report['total_trades']}")
    print(f"Win Rate:             {report['win_rate_pct']:.2f}%")
    print(f"Profit Factor:        {report['profit_factor']:.2f}")
    print(f"Expectancy per Trade: ${report['expectancy']:.2f}")
    print(f"Payoff Ratio (W/L):   {report['payoff_ratio']:.2f}")
    print("-" * 60)
    print(f"Average Win:          ${report['avg_win']:,.2f}")
    print(f"Average Loss:         ${report['avg_loss']:,.2f}")
    print("-" * 60)
    print(f"Avg Bars Held (All):     {report['avg_bars_held']:.1f}")
    print(f"Avg Bars Held (Winners): {report['avg_bars_held_winners']:.1f}")
    print(f"Avg Bars Held (Losers):  {report['avg_bars_held_losers']:.1f}")
    print("-" * 60)
    print(f"Max Drawdown ($):     ${report['max_drawdown_usd']:,.2f}")
    print(f"Max Drawdown (%):     {report['max_drawdown_pct']:.2f}%")
    print("-" * 60)
    print("💸 REALISTIC COST BREAKDOWN:")
    print(f"Total Spread Paid:    ${report['total_spread_cost']:,.2f}")
    print(f"Total Slippage Paid:  ${report['total_slippage_cost']:,.2f}")
    print(f"Total Commission:     ${report['total_commission']:,.2f}")
    print(f"TOTAL HIDDEN COSTS:   ${report['total_hidden_costs']:,.2f}")
    print("-" * 60)
    print(f"Net PnL (with costs):        ${report['net_pnl']:,.2f}")
    print(f"Est. Gross PnL (no costs):   ${report['estimated_gross_pnl_before_costs']:,.2f}")
    print("="*60 + "\n")

    if report.get('sizing_mode') == 'fixed_lot_stress_test':
        print("⚠️  NOTE: Sizing mode = FIXED LOT. Total Return %% and Max Drawdown %%")
        print("   are NOT representative of real account risk. Use only to judge")
        print("   raw signal quality (Expectancy, Payoff Ratio, Win Rate).")
        print("   Set use_dynamic_position_sizing=True in config.py for realistic")
        print("   risk-adjusted results.\n")

def main():
    setup_logger()
    logger.info("Initializing Institutional Trend Following System...")

    costs = TradingCosts()
    params = StrategyParams()
    settings = BacktestSettings()

    logger.info(f"Target: {settings.symbol} | Timeframe: {settings.timeframe} | Years: {settings.start_year}-{settings.end_year}")
    logger.info(f"Sizing Mode: {'DYNAMIC (risk-based)' if settings.use_dynamic_position_sizing else 'FIXED LOT (stress test)'}")

    parquet_path = extract_histdata_to_parquet(settings)

    logger.info("Loading Parquet data into memory...")
    df = pl.read_parquet(parquet_path)

    strategy = TrendFollowingStrategy(params)
    df_signals = strategy.generate_signals(df)

    logger.info(f"Signal distribution: {df_signals['signal'].value_counts()}")

    engine = RealisticBacktestEngine(costs, params, settings)
    report = engine.run(df_signals)

    print_report(report)

    with open("report.txt", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    if "error" not in report:
        logger.success("Backtest completed successfully.")
    else:
        logger.warning("Backtest finished with no trades executed.")

if __name__ == "__main__":
    main()
