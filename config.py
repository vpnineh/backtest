# config.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class BacktestConfig:
    # Account
    initial_balance: float = 10000.0
    
    # Costs
    spread_pips: Dict[str, float] = None
    commission_per_lot: float = 7.0  # USD round trip
    slippage_pips: float = 0.2
    
    # Risk limits
    daily_loss_limit: float = 0.02  # 2%
    monthly_dd_limit: float = 0.08  # 8%
    max_total_exposure: float = 0.05  # 5%
    
    # Session filter (UTC)
    london_session: tuple = (8, 17)
    newyork_session: tuple = (13, 21)
    no_trade_session: tuple = (22, 7)
    
    # Mode settings
    mode_a_risk: float = 0.005  # 0.5%
    mode_a_max_positions: int = 3
    mode_a_sl_atr: float = 1.5
    mode_a_tp_atr: float = 3.0
    
    mode_b_risk: float = 0.01  # 1%
    mode_b_max_positions: int = 5
    mode_b_sl_atr: float = 2.0
    
    mode_c_risk: float = 0.015  # 1.5%
    mode_c_max_positions: int = 3
    mode_c_sl_atr: float = 1.5
    mode_c_tp_atr: float = 4.0
    
    # Symbols
    symbols: list = None
    
    def __post_init__(self):
        if self.spread_pips is None:
            self.spread_pips = {
                'EURUSD': 1.5, 'GBPUSD': 2.0, 'EURGBP': 1.8,
                'AUDNZD': 3.0, 'AUDUSD': 1.5, 'NZDUSD': 2.0,
                'USDCAD': 2.0, 'USDCHF': 2.0,
                'XAUUSD': 25.0, 'XAGUSD': 3.0
            }
        
        if self.symbols is None:
            self.symbols = [
                'EURUSD', 'GBPUSD', 'EURGBP', 'AUDNZD',
                'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
                'XAUUSD', 'XAGUSD'
            ]
