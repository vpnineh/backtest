# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

# ==========================================
# 🎯 USER CONFIGURATION (CHANGE THESE)
# ==========================================
SYMBOL = "AUDNZD"
TIMEFRAME = "M15"
START_YEAR = 2012
END_YEAR = 2025
# ==========================================

class TradingCosts(BaseModel):
    spread_pips: float = Field(default=1.8)
    slippage_pips: float = Field(default=0.3)
    commission_per_lot_usd: float = Field(default=5.0)
    pip_value_usd_per_lot: float = Field(default=6.0)
    pip_size: float = Field(default=0.0001)

class StrategyParams(BaseModel):
    bb_period: int = 50
    bb_std_dev: float = 2.5
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    london_start_hour: int = 7
    london_end_hour: int = 16

class MartingaleConfig(BaseModel):
    enabled: bool = Field(default=True)
    multiplier: float = Field(default=1.3, gt=1.0, le=2.0)
    max_levels: int = Field(default=3, ge=1, le=5)
    reset_on_win: bool = Field(default=True)
    circuit_breaker_dd_percent: float = Field(default=0.30, gt=0, le=0.50)

class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    base_risk_per_trade_percent: float = 0.01
    
    # 🔥 CRITICAL FIX: Set to FALSE to disable infinite compounding.
    # This forces the engine to use 'fixed_lot_size' so we can see the 
    # TRUE raw expectancy of the strategy without mathematical illusions.
    use_dynamic_position_sizing: bool = False 
    fixed_lot_size: float = 0.1
    
    data_dir: Path = Path("data")
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    
    @property
    def parquet_filename(self) -> str:
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
