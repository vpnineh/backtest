# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

# ==========================================
# 🎯 USER CONFIGURATION (CHANGE THESE)
# ==========================================
SYMBOL = "EURGBP"
TIMEFRAME = "M5"      
START_YEAR = 2018     
END_YEAR = 2025       
# ==========================================

class TradingCosts(BaseModel):
    """Realistic trading costs for EURGBP."""
    spread_pips: float = Field(default=1.2, description="Average spread in pips")
    slippage_pips: float = Field(default=0.3, description="Average slippage per execution")
    commission_per_lot_usd: float = Field(default=5.0, description="Commission per 1 Lot per side")
    pip_value_usd_per_lot: float = Field(default=12.5, description="Value of 1 pip for 1 Lot in USD")
    pip_size: float = Field(default=0.0001, description="0.0001 for 4-digit pairs, 0.01 for 5-digit")

class StrategyParams(BaseModel):
    """Mean Reversion Strategy Parameters - OPTIMIZED FOR M5"""
    ema_trend_period: int = 50      
    bb_period: int = 50             
    bb_std_dev: float = 2.0         
    
    rsi_period: int = 14
    rsi_oversold: float = 35.0      
    rsi_overbought: float = 65.0    
    
    sl_pips: float = 8.0            
    tp_pips: float = 12.0           
    
    london_start_hour: int = 7
    london_end_hour: int = 16

class MartingaleConfig(BaseModel):
    """Principled Martingale Configuration."""
    enabled: bool = Field(default=True, description="Enable Martingale on losses")
    multiplier: float = Field(default=1.3, gt=1.0, le=2.0, description="Volume multiplier after a loss (Soft Martingale)")
    max_levels: int = Field(default=3, ge=1, le=5, description="Maximum consecutive loss levels before capping volume")
    reset_on_win: bool = Field(default=True, description="Reset to level 0 after a win")
    circuit_breaker_dd_percent: float = Field(default=0.20, gt=0, le=0.50, description="Max drawdown % to halt trading completely (Safety Net)")

class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    risk_per_trade_percent: float = 0.01  # 1% risk per trade (Base risk)
    data_dir: Path = Path("data")
    
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    
    @property
    def parquet_filename(self) -> str:
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
