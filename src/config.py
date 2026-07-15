# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path

class TradingCosts(BaseModel):
    """Realistic trading costs for EURGBP M1."""
    spread_pips: float = Field(default=1.2, description="Average spread in pips (EURGBP is ~1.0-1.5)")
    slippage_pips: float = Field(default=0.3, description="Average slippage per execution in pips")
    commission_per_lot_usd: float = Field(default=5.0, description="Commission per 1 Standard Lot per side ($)")
    pip_value_usd_per_lot: float = Field(default=12.5, description="Value of 1 pip for 1 Lot in USD (EURGBP ~$12.5)")

class StrategyParams(BaseModel):
    """Mean Reversion Strategy Parameters."""
    ema_trend_period: int = 200
    bb_period: int = 200
    bb_std_dev: float = 2.5
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    sl_pips: float = 15.0
    tp_pips: float = 10.0  # Mean reversion usually has lower TP than SL
    
    # Time filters (UTC)
    london_start_hour: int = 7
    london_end_hour: int = 16

class BacktestSettings(BaseModel):
    initial_balance: float = 10000.0
    risk_per_trade_percent: float = 0.01  # 1% risk per trade
    data_dir: Path = Path("data")
