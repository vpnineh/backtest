# engine/regime_detector.py
import pandas as pd
import numpy as np
from .indicators import Indicators

class RegimeDetector:
    def __init__(self):
        self.ind = Indicators()
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime for each H1 bar
        Returns: 'TREND', 'RANGE', or 'MOMENTUM'
        """
        # Calculate indicators
        df['ema50'] = self.ind.ema(df['close'], 50)
        df['ema200'] = self.ind.ema(df['close'], 200)
        df['atr14'] = self.ind.atr(df['high'], df['low'], df['close'], 14)
        df['rsi14'] = self.ind.rsi(df['close'], 14)
        df['adx14'] = self.ind.adx(df['high'], df['low'], df['close'], 14)
        
        bb_upper, bb_middle, bb_lower = self.ind.bollinger_bands(df['close'], 20, 2.0)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Calculate BB width
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR average
        df['atr_avg50'] = df['atr14'].rolling(50).mean()
        
        # Candle range
        df['candle_range'] = df['high'] - df['low']
        
        # Detect regimes
        df['regime'] = 'NONE'
        
        # TREND STATE
        trend_cond = (
            (df['adx14'] >= 25) &
            (np.abs(df['ema50'] - df['ema200']) > df['atr14'] * 0.5)
        )
        df.loc[trend_cond, 'regime'] = 'TREND'
        
        # RANGE STATE
        range_cond = (
            (df['adx14'] <= 20) &
            (df['bb_width'] <= 0.04)
        )
        df.loc[range_cond, 'regime'] = 'RANGE'
        
        # MOMENTUM STATE
        momentum_cond = (
            (df['atr14'] > df['atr_avg50'] * 1.3) &
            (df['candle_range'] > df['atr14'] * 1.5)
        )
        df.loc[momentum_cond, 'regime'] = 'MOMENTUM'
        
        return df
