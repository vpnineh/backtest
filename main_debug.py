# main_debug.py - ENHANCED
import logging
import sys
from pathlib import Path
import pandas as pd
from config import BacktestConfig
from engine.data_loader import DataLoader, TimeFrameConverter
from engine.regime_detector import RegimeDetector
from engine.strategies import ModeA_ConservativeTrend, ModeB_BalancedRange, ModeC_AggressiveMomentum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def debug_full():
    """Complete debug with signal counting"""
    
    logger.info("="*60)
    logger.info("LOADING DATA")
    logger.info("="*60)
    
    loader = DataLoader()
    m1_data = loader.load_symbol('EURUSD', 2020, 2021)  # 2 years for quick test
    
    logger.info(f"✓ Loaded {len(m1_data):,} M1 bars")
    
    h1_data = TimeFrameConverter.resample_to_h1(m1_data)
    logger.info(f"✓ Converted to {len(h1_data):,} H1 bars")
    
    logger.info("\n" + "="*60)
    logger.info("CALCULATING INDICATORS")
    logger.info("="*60)
    
    detector = RegimeDetector()
    h1_data = detector.detect(h1_data)
    h1_data = h1_data.dropna().reset_index(drop=True)
    
    logger.info(f"✓ Indicators ready: {len(h1_data):,} bars")
    
    # Regime distribution
    regime_counts = h1_data['regime'].value_counts()
    logger.info(f"\nRegime Distribution:")
    for regime, count in regime_counts.items():
        pct = (count / len(h1_data)) * 100
        logger.info(f"  {regime}: {count} ({pct:.1f}%)")
    
    logger.info("\n" + "="*60)
    logger.info("COUNTING SIGNALS")
    logger.info("="*60)
    
    signals = {
        'A_BUY': [], 'A_SELL': [],
        'B_BUY': [], 'B_SELL': [],
        'C_BUY': [], 'C_SELL': []
    }
    
    for i in range(200, len(h1_data)):
        current = h1_data.iloc[i]
        prev = h1_data.iloc[i-1]
        
        # Mode A
        sig_a = ModeA_ConservativeTrend.check_signal(current, prev)
        if sig_a == 'BUY':
            signals['A_BUY'].append(i)
        elif sig_a == 'SELL':
            signals['A_SELL'].append(i)
        
        # Mode B
        sig_b = ModeB_BalancedRange.check_signal(current)
        if sig_b == 'BUY':
            signals['B_BUY'].append(i)
        elif sig_b == 'SELL':
            signals['B_SELL'].append(i)
        
        # Mode C
        sig_c = ModeC_AggressiveMomentum.check_signal(current, prev)
        if sig_c == 'BUY':
            signals['C_BUY'].append(i)
        elif sig_c == 'SELL':
            signals['C_SELL'].append(i)
    
    logger.info("\nSignal Counts:")
    logger.info(f"  Mode A BUY:  {len(signals['A_BUY'])}")
    logger.info(f"  Mode A SELL: {len(signals['A_SELL'])}")
    logger.info(f"  Mode A Total: {len(signals['A_BUY']) + len(signals['A_SELL'])}")
    logger.info("")
    logger.info(f"  Mode B BUY:  {len(signals['B_BUY'])}")
    logger.info(f"  Mode B SELL: {len(signals['B_SELL'])}")
    logger.info(f"  Mode B Total: {len(signals['B_BUY']) + len(signals['B_SELL'])}")
    logger.info("")
    logger.info(f"  Mode C BUY:  {len(signals['C_BUY'])}")
    logger.info(f"  Mode C SELL: {len(signals['C_SELL'])}")
    logger.info(f"  Mode C Total: {len(signals['C_BUY']) + len(signals['C_SELL'])}")
    
    total_signals = sum(len(v) for v in signals.values())
    logger.info(f"\n✓ TOTAL SIGNALS: {total_signals}")
    
    # Show sample signals
    if signals['A_BUY']:
        logger.info("\nSample Mode A BUY signals:")
        for idx in signals['A_BUY'][:3]:
            row = h1_data.iloc[idx]
            logger.info(f"  {row['time']}: Close={row['close']:.5f}, RSI={row['rsi14']:.1f}, ADX={row['adx14']:.1f}")
    
    if total_signals < 50:
        logger.warning("\n⚠️  WARNING: Less than 50 signals found!")
        logger.warning("Strategy conditions may be too strict.")
        return False
    else:
        logger.info(f"\n✅ SUCCESS: {total_signals} signals found")
        return True

if __name__ == '__main__':
    success = debug_full()
    sys.exit(0 if success else 1)
