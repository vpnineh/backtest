# main_debug.py
import logging
import sys
from pathlib import Path
import json
import pandas as pd
from config import BacktestConfig
from engine.data_loader import DataLoader, TimeFrameConverter
from engine.regime_detector import RegimeDetector

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log')
    ]
)

logger = logging.getLogger(__name__)

def debug_data_loading():
    """Test data loading step by step"""
    logger.info("="*60)
    logger.info("STEP 1: Testing Data Loading")
    logger.info("="*60)
    
    loader = DataLoader()
    
    try:
        # Test EURUSD
        m1_data = loader.load_symbol('EURUSD', 2010, 2011)
        
        logger.info(f"M1 Data Shape: {m1_data.shape}")
        logger.info(f"M1 Data Columns: {m1_data.columns.tolist()}")
        logger.info(f"M1 First 5 rows:\n{m1_data.head()}")
        logger.info(f"M1 Data types:\n{m1_data.dtypes}")
        logger.info(f"M1 Date range: {m1_data['time'].min()} to {m1_data['time'].max()}")
        
        # Check for NaN
        logger.info(f"M1 NaN count:\n{m1_data.isna().sum()}")
        
        # Convert to H1
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Testing H1 Conversion")
        logger.info("="*60)
        
        h1_data = TimeFrameConverter.resample_to_h1(m1_data)
        
        logger.info(f"H1 Data Shape: {h1_data.shape}")
        logger.info(f"H1 First 5 rows:\n{h1_data.head()}")
        logger.info(f"H1 Date range: {h1_data['time'].min()} to {h1_data['time'].max()}")
        logger.info(f"H1 NaN count:\n{h1_data.isna().sum()}")
        
        # Test indicators
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Testing Indicators")
        logger.info("="*60)
        
        detector = RegimeDetector()
        h1_data = detector.detect(h1_data)
        
        logger.info(f"After indicators Shape: {h1_data.shape}")
        logger.info(f"Indicators columns: {h1_data.columns.tolist()}")
        logger.info(f"Indicator NaN count:\n{h1_data.isna().sum()}")
        
        # Show sample with indicators
        sample = h1_data[['time', 'close', 'ema50', 'ema200', 'atr14', 'rsi14', 'adx14', 'regime']].tail(20)
        logger.info(f"Sample with indicators:\n{sample}")
        
        # Regime distribution
        logger.info(f"\nRegime distribution:\n{h1_data['regime'].value_counts()}")
        
        # Test signal generation
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Testing Signal Generation")
        logger.info("="*60)
        
        from engine.strategies import ModeA_ConservativeTrend, ModeB_BalancedRange, ModeC_AggressiveMomentum
        
        signals_a = 0
        signals_b = 0
        signals_c = 0
        
        for i in range(200, len(h1_data)):
            current = h1_data.iloc[i]
            prev = h1_data.iloc[i-1]
            
            sig_a = ModeA_ConservativeTrend.check_signal(current, prev)
            sig_b = ModeB_BalancedRange.check_signal(current)
            sig_c = ModeC_AggressiveMomentum.check_signal(current, prev)
            
            if sig_a:
                signals_a += 1
                if signals_a <= 3:
                    logger.info(f"Mode A Signal {sig_a} at {current['time']}")
            
            if sig_b:
                signals_b += 1
                if signals_b <= 3:
                    logger.info(f"Mode B Signal {sig_b} at {current['time']}")
            
            if sig_c:
                signals_c += 1
                if signals_c <= 3:
                    logger.info(f"Mode C Signal {sig_c} at {current['time']}")
        
        logger.info(f"\nTotal signals in sample:")
        logger.info(f"  Mode A: {signals_a}")
        logger.info(f"  Mode B: {signals_b}")
        logger.info(f"  Mode C: {signals_c}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during debug: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    success = debug_data_loading()
    
    if success:
        print("\n" + "="*60)
        print("✅ DEBUG COMPLETED - Check debug.log for details")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ DEBUG FAILED - Check debug.log for errors")
        print("="*60)
        sys.exit(1)
