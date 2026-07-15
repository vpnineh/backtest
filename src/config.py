# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

# ==========================================
# 🎯 USER CONFIGURATION (CHANGE THESE)
# ==========================================
SYMBOL = "EURGBP"
TIMEFRAME = "H1"      # 🔥 CRITICAL: Moved to H1 to eliminate spread noise
START_YEAR = 2012     # Let's test the full 10+ years on H1
END_YEAR = 2025       
# ==========================================

class TradingCosts(BaseModel):
    """Realistic trading costs."""
    spread_pips: float = Field(default=1.2)
    slippage_pips: float = Field(default=0.3)
    commission_per_lot_usd: float = Field(default=5.0)
    pip_value_usd_per_lot: float = Field(default=12.5)
    pip_size: float = Field(default=0.0001)

class StrategyParams(BaseModel):
    """Trend Following Strategy Parameters (Donchian Breakout + ATR Trailing)"""
    # Trend Filter
    ema_trend_period: int = 200      # Only trade in direction of long-term trend
    
    # Breakout Logic
    donchian_period: int = 20        # Breakout of 20-period high/low
    
    # Volatility & Risk Management (ATR-based)
    atr_period: int = 14
    initial_sl_atr_mult: float = 2.0 # Initial Stop Loss = 2 * ATR
    trail_atr_mult: float = 2.5      # Trailing Stop distance = 2.5 * ATR
    
    # Time filters (UTC)
    london_start_hour: int = 7
    london_end_hour: int = 16

class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    risk_per_trade_percent: float = 0.01  # Strict 1% risk per trade (No Martingale!)
    data_dir: Path = Path("data")
    
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    
    @property
    def parquet_filename(self) -> str:
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
