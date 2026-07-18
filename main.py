# main.py
import logging
import sys
from pathlib import Path
import json
import pandas as pd
from config import BacktestConfig
from engine.backtest_engine import BacktestEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backtest.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    # Configuration
    config = BacktestConfig()
    
    # Test parameters
    symbols = ['EURUSD', 'GBPUSD', 'EURGBP', 'AUDNZD']
    modes = ['AUTO', 'A', 'B', 'C']
    
    start_year = 2010
    end_year = 2025
    
    # Results storage
    all_results = {}
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Run backtests
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'='*60}")
        
        for mode in modes:
            logger.info(f"\nMode: {mode}")
            
            try:
                engine = BacktestEngine(config, mode=mode)
                results = engine.run(symbol, start_year, end_year)
                
                # Store results
                key = f"{symbol}_{mode}"
                all_results[key] = results
                
                # Save trades
                if engine.trades:
                    trades_df = pd.DataFrame(engine.trades)
                    trades_df.to_csv(results_dir / f"trades_{key}.csv", index=False)
                
                # Save equity curve
                if engine.equity_curve:
                    equity_df = pd.DataFrame(engine.equity_curve)
                    equity_df.to_csv(results_dir / f"equity_{key}.csv", index=False)
                
                # Print summary
                print(f"\n{symbol} - Mode {mode} Results:")
                print(f"  Final Balance: ${results['final_balance']:,.2f}")
                print(f"  Net Profit: {results['net_profit_pct']:.2f}%")
                print(f"  Total Trades: {results['total_trades']}")
                print(f"  Win Rate: {results['win_rate']:.2f}%")
                print(f"  Profit Factor: {results['profit_factor']:.2f}")
                print(f"  Max DD: {results['max_drawdown_pct']:.2f}%")
                print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
                print(f"  Avg R: {results['avg_r_multiple']:.2f}")
                
                # Validation checks
                print(f"\n  Validation:")
                print(f"    ✓ PF >= 1.2: {'✓' if results['profit_factor'] >= 1.2 else '✗ FAIL'}")
                print(f"    ✓ DD <= 20%: {'✓' if results['max_drawdown_pct'] <= 20 else '✗ FAIL'}")
                print(f"    ✓ Sharpe >= 0.5: {'✓' if results['sharpe_ratio'] >= 0.5 else '✗ FAIL'}")
                print(f"    ✓ Win Rate 40-70%: {'✓' if 40 <= results['win_rate'] <= 70 else '✗ SUSPICIOUS'}")
                
            except Exception as e:
                logger.error(f"Error testing {symbol} mode {mode}: {e}", exc_info=True)
    
    # Save summary
    with open(results_dir / 'summary.json', 'w') as f:
        # Convert numpy types to python types for JSON serialization
        clean_results = {}
        for k, v in all_results.items():
            clean_results[k] = {key: float(val) if isinstance(val, (int, float)) else val 
                               for key, val in v.items()}
        json.dump(clean_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("Backtest Complete!")
    logger.info(f"Results saved to: {results_dir.absolute()}")
    logger.info(f"{'='*60}")

if __name__ == '__main__':
    main()
