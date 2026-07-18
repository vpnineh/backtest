# engine/position_manager.py
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

@dataclass
class Position:
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    entry_time: pd.Timestamp
    lot_size: float
    sl: float
    tp: float
    mode: str  # 'A', 'B', or 'C'
    risk_amount: float
    be_moved: bool = False
    trailing_active: bool = False
    
    def calculate_pnl(self, current_price: float, point_value: float) -> float:
        """Calculate current P&L"""
        if self.direction == 'BUY':
            points = current_price - self.entry_price
        else:
            points = self.entry_price - current_price
        
        return points * self.lot_size * point_value
    
    def check_exit(self, current_price: float, current_high: float, current_low: float) -> Optional[str]:
        """Check if position should be closed"""
        if self.direction == 'BUY':
            if current_low <= self.sl:
                return 'SL'
            if current_high >= self.tp:
                return 'TP'
        else:
            if current_high >= self.sl:
                return 'SL'
            if current_low <= self.tp:
                return 'TP'
        
        return None

class PositionManager:
    def __init__(self):
        self.positions: List[Position] = []
    
    def add_position(self, position: Position):
        """Add new position"""
        self.positions.append(position)
    
    def close_position(self, position: Position, exit_price: float, exit_time: pd.Timestamp, exit_reason: str, point_value: float):
        """Close position and return trade result"""
        pnl = position.calculate_pnl(exit_price, point_value)
        
        trade_result = {
            'symbol': position.symbol,
            'mode': position.mode,
            'direction': position.direction,
            'entry_time': position.entry_time,
            'entry_price': position.entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'lot_size': position.lot_size,
            'pnl': pnl,
            'risk_amount': position.risk_amount,
            'r_multiple': pnl / position.risk_amount if position.risk_amount > 0 else 0
        }
        
        self.positions.remove(position)
        return trade_result
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a symbol"""
        return [p for p in self.positions if p.symbol == symbol]
    
    def get_positions_by_mode(self, mode: str) -> List[Position]:
        """Get all positions for a mode"""
        return [p for p in self.positions if p.mode == mode]
    
    def update_positions(self, current_data: dict, point_value: float, atr: float):
        """Update stop loss (BE and trailing)"""
        for pos in self.positions:
            if pos.symbol not in current_data:
                continue
            
            current_price = current_data[pos.symbol]['close']
            r_profit = pos.calculate_pnl(current_price, point_value) / pos.risk_amount
            
            # Move to break-even at +1R
            if not pos.be_moved and r_profit >= 1.0:
                pos.sl = pos.entry_price
                pos.be_moved = True
            
            # Activate trailing at +2R
            if not pos.trailing_active and r_profit >= 2.0:
                pos.trailing_active = True
            
            # Update trailing stop
            if pos.trailing_active:
                trail_distance = atr * 1.0
                
                if pos.direction == 'BUY':
                    new_sl = current_price - trail_distance
                    pos.sl = max(pos.sl, new_sl)
                else:
                    new_sl = current_price + trail_distance
                    pos.sl = min(pos.sl, new_sl)
