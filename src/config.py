# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

# ==========================================
# 🎯 USER CONFIGURATION (CHANGE THESE)
# ==========================================
SYMBOL = "EURGBP"
# Options: "M1", "M5", "M15", "H1", "H4", "D1"
# Note: The system will automatically read M1 data and resample it if you choose > M1.
TIMEFRAME = "M1"      
START_YEAR = 2012     # Inclusive
END_YEAR = 2025       # Inclusive
# ==========================================

class TradingCosts(BaseModel):
    """Realistic trading costs."""
    spread_pips: float = Field(default=1.2, description="Average spread in pips")
    slippage_pips: float = Field(default=0.3, description="Average slippage per execution")
    commission_per_lot_usd: float = Field(default=5.0, description="Commission per 1 Lot per side")
    pip_value_usd_per_lot: float = Field(default=12.5, description="Value of 1 pip for 1 Lot in USD")
    pip_size: float = Field(default=0.0001, description="0.0001 for 4-digit pairs, 0.01 for 5-digit")

class StrategyParams(BaseModel):
    """Mean Reversion Strategy Parameters."""
    ema_trend_period: int = 200
    bb_period: int = 200
    bb_std_dev: float = 2.5
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    sl_pips: float = 15.0
    tp_pips: float = 10.0 
    
    # Time filters (UTC)
    london_start_hour: int = 7
    london_end_hour: int = 16

class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    risk_per_trade_percent: float = 0.01  # 1% risk
    data_dir: Path = Path("data")
    
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    
    @property
    def parquet_filename(self) -> str:
        # Example: EURGBP_M5_2012_2025.parquet
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
