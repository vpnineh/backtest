# engine/strategies.py - RELAXED VERSION
import pandas as pd
import numpy as np
from typing import Optional, Literal

class ModeA_ConservativeTrend:
    """Mode A: Simplified Trend Following"""
    
    @staticmethod
    def check_signal(row: pd.Series, prev_row: pd.Series) -> Optional[Literal['BUY', 'SELL']]:
        """Relaxed trend signals"""
        
        # Skip if indicators not ready
        if pd.isna(row['ema50']) or pd.isna(row['ema200']) or pd.isna(row['adx14']):
            return None
        
        # BUY: Simple trend following
        if (
            row['ema50'] > row['ema200'] and  # Uptrend
            row['close'] > row['ema50'] and    # Price above EMA
            row['adx14'] >= 20 and             # Some trend (was 25)
            row['rsi14'] >= 40 and             # Not oversold (was 45-65)
            row['rsi14'] <= 70                 # Not overbought
        ):
            return 'BUY'
        
        # SELL: Simple trend following
        if (
            row['ema50'] < row['ema200'] and   # Downtrend
            row['close'] < row['ema50'] and    # Price below EMA
            row['adx14'] >= 20 and             # Some trend
            row['rsi14'] >= 30 and             # Not oversold
            row['rsi14'] <= 60                 # Not overbought (was 35-55)
        ):
            return 'SELL'
        
        return None

class ModeB_BalancedRange:
    """Mode B: Simplified Range Trading"""
    
    @staticmethod
    def check_signal(row: pd.Series) -> Optional[Literal['BUY', 'SELL']]:
        """Relaxed range signals"""
        
        # Skip if indicators not ready
        if pd.isna(row['bb_lower']) or pd.isna(row['bb_upper']) or pd.isna(row['rsi14']):
            return None
        
        # BUY at lower band (relaxed)
        if (
            row['close'] <= row['bb_lower'] * 1.001 and  # Near lower band (was exact)
            row['rsi14'] <= 35                            # Oversold (was 30)
        ):
            return 'BUY'
        
        # SELL at upper band (relaxed)
        if (
            row['close'] >= row['bb_upper'] * 0.999 and  # Near upper band
            row['rsi14'] >= 65                            # Overbought (was 70)
        ):
            return 'SELL'
        
        return None

class ModeC_AggressiveMomentum:
    """Mode C: Simplified Momentum"""
    
    @staticmethod
    def check_signal(row: pd.Series, prev_row: pd.Series) -> Optional[Literal['BUY', 'SELL']]:
        """Relaxed momentum signals"""
        
        # Skip if indicators not ready
        if pd.isna(row['ema50']) or pd.isna(row['atr14']):
            return None
        
        candle_body = abs(row['close'] - row['open'])
        
        # BUY momentum (relaxed)
        if (
            row['close'] > row['ema50'] and              # Above EMA
            candle_body >= row['atr14'] * 0.5 and        # Strong candle (was 0.8)
            row['close'] > row['open']                    # Bullish candle (removed prev_high condition)
        ):
            return 'BUY'
        
        # SELL momentum (relaxed)
        if (
            row['close'] < row['ema50'] and              # Below EMA
            candle_body >= row['atr14'] * 0.5 and        # Strong candle
            row['close'] < row['open']                    # Bearish candle
        ):
            return 'SELL'
        
        return None
