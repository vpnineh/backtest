# engine/risk_manager.py
from typing import Optional
import pandas as pd
from config import BacktestConfig

class RiskManager:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.daily_loss = {}
        self.monthly_dd = {}
        self.last_date = None
        self.last_month = None
    
    def calculate_lot_size(self, balance: float, risk_percent: float, sl_points: float, point_value: float) -> float:
        """Calculate position size"""
        risk_money = balance * risk_percent
        lot = risk_money / (sl_points * point_value)
        
        # Round to 0.01
        lot = round(lot, 2)
        
        # Minimum 0.01
        return max(0.01, lot)
    
    def check_daily_limit(self, current_date: pd.Timestamp, current_balance: float, initial_balance: float) -> bool:
        """Check if daily loss limit exceeded"""
        date_key = current_date.date()
        
        # Reset daily counter
        if self.last_date != date_key:
            self.daily_loss[date_key] = 0
            self.last_date = date_key
        
        daily_loss_pct = abs(self.daily_loss.get(date_key, 0)) / initial_balance
        
        return daily_loss_pct < self.config.daily_loss_limit
    
    def update_daily_loss(self, current_date: pd.Timestamp, pnl: float):
        """Update daily loss tracker"""
        date_key = current_date.date()
        if date_key not in self.daily_loss:
            self.daily_loss[date_key] = 0
        
        if pnl < 0:
            self.daily_loss[date_key] += pnl
    
    def check_monthly_limit(self, current_date: pd.Timestamp, current_balance: float, peak_balance: float) -> bool:
        """Check if monthly drawdown limit exceeded"""
        month_key = (current_date.year, current_date.month)
        
        # Reset monthly tracker
        if self.last_month != month_key:
            self.monthly_dd[month_key] = peak_balance
            self.last_month = month_key
        
        month_peak = self.monthly_dd.get(month_key, peak_balance)
        dd_pct = (month_peak - current_balance) / month_peak if month_peak > 0 else 0
        
        return dd_pct < self.config.monthly_dd_limit
    
    def check_total_exposure(self, current_risk: float, new_risk: float, balance: float) -> bool:
        """Check if total exposure limit exceeded"""
        total_risk_pct = (current_risk + new_risk) / balance
        return total_risk_pct <= self.config.max_total_exposure
    
    def check_session(self, current_time: pd.Timestamp) -> bool:
        """Check if current time is in allowed trading session"""
        hour = current_time.hour
        
        # Check if in allowed sessions
        london_ok = self.config.london_session[0] <= hour < self.config.london_session[1]
        ny_ok = self.config.newyork_session[0] <= hour < self.config.newyork_session[1]
        
        # Check if in no-trade session
        no_trade_start, no_trade_end = self.config.no_trade_session
        if no_trade_start > no_trade_end:  # Crosses midnight
            in_no_trade = hour >= no_trade_start or hour < no_trade_end
        else:
            in_no_trade = no_trade_start <= hour < no_trade_end
        
        return (london_ok or ny_ok) and not in_no_trade
