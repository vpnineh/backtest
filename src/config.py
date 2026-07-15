# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

# ==========================================
# 🎯 USER CONFIGURATION (CHANGE THESE)
# ==========================================
SYMBOL = "EURGBP"
TIMEFRAME = "M5"      # Optimized for M5 timeframe
START_YEAR = 2018     # Shorter, more recent period for faster iteration
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
    # Trend & Volatility Indicators (Scaled down for M5 noise)
    ema_trend_period: int = 50      # 50 * 5min = 250 mins (Half a trading day)
    bb_period: int = 50             # Matches EMA period
    bb_std_dev: float = 2.0         # Slightly tighter bands for M5
    
    # Momentum Indicator
    rsi_period: int = 14
    rsi_oversold: float = 35.0      # Relaxed from 30 to catch more M5 dips
    rsi_overbought: float = 65.0    # Relaxed from 70 to catch more M5 spikes
    
    # Risk Management (Positive Risk/Reward Ratio)
    sl_pips: float = 8.0            # 8 pips SL is realistic for M5 noise
    tp_pips: float = 12.0           # 12 pips TP (Risk 8 to make 12 -> R:R = 1.5)
    
    # Time filters (UTC)
    london_start_hour: int = 7
    london_end_hour: int = 16

class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    risk_per_trade_percent: float = 0.01  # 1% risk per trade
    data_dir: Path = Path("data")
    
    # Dynamic properties based on user config
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    
    @property
    def parquet_filename(self) -> str:
        # Example: EURGBP_M5_2018_2025.parquet
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
