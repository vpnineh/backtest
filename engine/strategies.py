# engine/strategies.py
import pandas as pd
import numpy as np
from typing import Optional, Literal

class ModeA_ConservativeTrend:
    """Mode A: Conservative Trend Following on H1"""
    
    @staticmethod
    def check_signal(row: pd.Series, prev_row: pd.Series) -> Optional[Literal['BUY', 'SELL']]:
        """Check for trend following signals"""
        
        if row['regime'] != 'TREND':
            return None
        
        # BUY conditions
        if (
            row['ema50'] > row['ema200'] and
            prev_row['low'] <= prev_row['ema50'] and
            row['close'] > row['ema50'] and
            45 <= row['rsi14'] <= 65 and
            row['adx14'] >= 25
        ):
            return 'BUY'
        
        # SELL conditions
        if (
            row['ema50'] < row['ema200'] and
            prev_row['high'] >= prev_row['ema50'] and
            row['close'] < row['ema50'] and
            35 <= row['rsi14'] <= 55 and
            row['adx14'] >= 25
        ):
            return 'SELL'
        
        return None

class ModeB_BalancedRange:
    """Mode B: Range Mean Reversion"""
    
    @staticmethod
    def check_signal(row: pd.Series) -> Optional[Literal['BUY', 'SELL']]:
        """Check for range trading signals"""
        
        if row['regime'] != 'RANGE':
            return None
        
        # BUY at lower band
        if row['close'] <= row['bb_lower'] and row['rsi14'] <= 30:
            return 'BUY'
        
        # SELL at upper band
        if row['close'] >= row['bb_upper'] and row['rsi14'] >= 70:
            return 'SELL'
        
        return None

class ModeC_AggressiveMomentum:
    """Mode C: Momentum Breakout"""
    
    @staticmethod
    def check_signal(row: pd.Series, prev_row: pd.Series) -> Optional[Literal['BUY', 'SELL']]:
        """Check for momentum signals"""
        
        if row['regime'] != 'MOMENTUM':
            return None
        
        candle_body = abs(row['open'] - row['close'])
        
        # BUY momentum
        if (
            row['close'] > row['ema50'] and
            candle_body >= row['atr14'] * 0.8 and
            row['close'] > prev_row['high']
        ):
            return 'BUY'
        
        # SELL momentum
        if (
            row['close'] < row['ema50'] and
            candle_body >= row['atr14'] * 0.8 and
            row['close'] < prev_row['low']
        ):
            return 'SELL'
        
        return None
