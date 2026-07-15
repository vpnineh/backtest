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
    """
    Realistic trading costs for AUDNZD.
    """
    spread_pips: float = Field(default=1.8, description="AUDNZD typical spread")
    slippage_pips: float = Field(default=0.3, description="Average slippage per execution")
    commission_per_lot_usd: float = Field(default=5.0, description="Commission per 1 Lot per side")
    
    # AUDNZD: 1 pip (0.0001) on 1 lot = 10 NZD ≈ $6 USD
    pip_value_usd_per_lot: float = Field(default=6.0, description="Value of 1 pip for 1 Lot in USD (AUDNZD)")
    pip_size: float = Field(default=0.0001, description="0.0001 for 4-digit pairs")


class StrategyParams(BaseModel):
    """
    Mean Reversion Strategy Parameters for Range-Bound Pairs.
    """
    # Bollinger Bands
    bb_period: int = 50
    bb_std_dev: float = 2.5  # انحراف معیار بالاتر برای فیلتر نویز
    
    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    
    # Time filters (UTC)
    london_start_hour: int = 7
    london_end_hour: int = 16


class MartingaleConfig(BaseModel):
    """
    Principled Martingale Configuration.
    """
    enabled: bool = Field(default=True, description="Enable Martingale on losses")
    multiplier: float = Field(default=1.3, gt=1.0, le=2.0, description="Volume multiplier after a loss (Soft Martingale)")
    max_levels: int = Field(default=3, ge=1, le=5, description="Maximum consecutive loss levels")
    reset_on_win: bool = Field(default=True, description="Reset to level 0 after a win")
    circuit_breaker_dd_percent: float = Field(default=0.30, gt=0, le=0.50, description="Max drawdown % to halt trading")


class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    base_risk_per_trade_percent: float = 0.01  # ریسک پایه 1% (قبل از اعمال مارتینگل)
    
    use_dynamic_position_sizing: bool = True
    fixed_lot_size: float = 0.1
    
    data_dir: Path = Path("data")
    
    symbol: str = SYMBOL
    timeframe: str = TIMEFRAME
    start_year: int = START_YEAR
    end_year: int = END_YEAR
    
    @property
    def parquet_filename(self) -> str:
        return f"{self.symbol}_{self.timeframe}_{self.start_year}_{self.end_year}.parquet"
