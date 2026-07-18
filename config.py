# config.py - RELAXED VERSION
from dataclasses import dataclass
from typing import Dict

@dataclass
class BacktestConfig:
    # Account
    initial_balance: float = 10000.0
    
    # Costs (REDUCED for testing)
    spread_pips: Dict[str, float] = None
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.2
    
    # Risk limits
    daily_loss_limit: float = 0.03  # 3% (was 2%)
    monthly_dd_limit: float = 0.12  # 12% (was 8%)
    max_total_exposure: float = 0.08  # 8% (was 5%)
    
    # Session filter (DISABLED for testing)
    london_session: tuple = (0, 24)  # All day
    newyork_session: tuple = (0, 24)  # All day
    no_trade_session: tuple = (25, 25)  # Disabled
    
    # Mode settings (REDUCED SL/TP for more trades)
    mode_a_risk: float = 0.005  # 0.5%
    mode_a_max_positions: int = 5  # was 3
    mode_a_sl_atr: float = 1.2  # was 1.5
    mode_a_tp_atr: float = 2.5  # was 3.0
    
    mode_b_risk: float = 0.01  # 1%
    mode_b_max_positions: int = 5
    mode_b_sl_atr: float = 1.5  # was 2.0
    
    mode_c_risk: float = 0.015  # 1.5%
    mode_c_max_positions: int = 5  # was 3
    mode_c_sl_atr: float = 1.2  # was 1.5
    mode_c_tp_atr: float = 3.0  # was 4.0
    
    # Symbols
    symbols: list = None
    
    def __post_init__(self):
        if self.spread_pips is None:
            self.spread_pips = {
                'EURUSD': 1.0, 'GBPUSD': 1.5, 'EURGBP': 1.5,  # Reduced
                'AUDNZD': 2.0, 'AUDUSD': 1.0, 'NZDUSD': 1.5,
                'USDCAD': 1.5, 'USDCHF': 1.5,
                'XAUUSD': 20.0, 'XAGUSD': 2.5
            }
        
        if self.symbols is None:
            self.symbols = [
                'EURUSD', 'GBPUSD', 'EURGBP', 'AUDNZD',
                'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF',
                'XAUUSD', 'XAGUSD'
            ]
