# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

# ==========================================
# 🎯 USER CONFIGURATION (CHANGE THESE)
# ==========================================
SYMBOL = "AUDUSD"       # 🔥 Changed to AUDNZD
TIMEFRAME = "M15"        # Keeping H1 to test the exact same logic
START_YEAR = 2012       # Full available data range
END_YEAR = 2025       
# ==========================================

class TradingCosts(BaseModel):
    """Realistic trading costs calibrated specifically for AUDNZD."""
    spread_pips: float = Field(default=1.8, description="AUDNZD typically has a slightly wider spread than majors")
    slippage_pips: float = Field(default=0.3, description="Average slippage per execution")
    commission_per_lot_usd: float = Field(default=5.0, description="Commission per 1 Lot per side")
    
    # 🔥 CRITICAL FIX FOR CROSS PAIRS:
    # For AUDNZD, 1 pip (0.0001) on 1 standard lot (100,000 units) = 10 NZD.
    # Converted to USD (assuming NZD/USD ~ 0.60), it equals roughly $6.00.
    pip_value_usd_per_lot: float = Field(default=6.0, description="Value of 1 pip for 1 Lot in USD (AUDNZD specific)")
    pip_size: float = Field(default=0.0001, description="0.0001 for 4-digit pairs like AUDNZD")

class StrategyParams(BaseModel):
    """
    Trend Following Strategy Parameters.
    🔥 We use the EXACT SAME parameters as EURGBP to test robustness.
    """
    ema_trend_period: int = 200      
    donchian_period: int = 20        
    
    atr_period: int = 14
    initial_sl_atr_mult: float = 2.0 
    trail_atr_mult: float = 2.5      
    
    # Time filters (UTC) - AUDNZD is also highly active during London/NY overlap
    london_start_hour: int = 7
    london_end_hour: int = 16

class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    risk_per_trade_percent: float = 0.01  
    data_dir: Path = Path("data")
    
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    
    @property
    def parquet_filename(self) -> str:
        # Will generate: AUDNZD_H1_2012_2025.parquet
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
