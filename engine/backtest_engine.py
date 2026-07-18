# engine/backtest_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from config import BacktestConfig
from .data_loader import DataLoader, TimeFrameConverter
from .regime_detector import RegimeDetector
from .strategies import ModeA_ConservativeTrend, ModeB_BalancedRange, ModeC_AggressiveMomentum
from .position_manager import PositionManager, Position
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, config: BacktestConfig, mode: str = 'AUTO'):
        self.config = config
        self.mode = mode  # 'A', 'B', 'C', or 'AUTO'
        
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.peak_balance = config.initial_balance
        
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(config)
        
        self.trades = []
        self.equity_curve = []
        
        # Symbol configurations
        self.point_values = {
            'EURUSD': 10, 'GBPUSD': 10, 'EURGBP': 10,
            'AUDNZD': 10, 'AUDUSD': 10, 'NZDUSD': 10,
            'USDCAD': 10, 'USDCHF': 10,
            'XAUUSD': 1, 'XAGUSD': 50
        }
    
    def run(self, symbol: str, start_year: int, end_year: int):
        """Run backtest for a single symbol"""
        logger.info(f"Loading data for {symbol}...")
        
        # Load and prepare data
        loader = DataLoader()
        m1_data = loader.load_symbol(symbol, start_year, end_year)
        
        # Convert to H1
        h1_data = TimeFrameConverter.resample_to_h1(m1_data)
        
        # Detect regimes
        logger.info(f"Detecting market regimes...")
        detector = RegimeDetector()
        h1_data = detector.detect(h1_data)
        
        # Remove NaN rows
        h1_data = h1_data.dropna().reset_index(drop=True)
        
        logger.info(f"Starting backtest simulation...")
        
        # Main simulation loop
        for i in range(100, len(h1_data)):  # Start after warm-up period
            current_row = h1_data.iloc[i]
            prev_row = h1_data.iloc[i-1]
            
            current_time = current_row['time']
            
            # Update existing positions
            self._update_positions(symbol, current_row)
            
            # Check exits
            self._check_exits(symbol, current_row, current_time)
            
            # Check for new entries
            if self.risk_manager.check_session(current_time):
                if self.risk_manager.check_daily_limit(current_time, self.balance, self.config.initial_balance):
                    if self.risk_manager.check_monthly_limit(current_time, self.balance, self.peak_balance):
                        self._check_entries(symbol, current_row, prev_row, current_time)
            
            # Update equity curve
            self._update_equity(current_time)
            
            # Update peak
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
        
        logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        
        return self.get_results()
    
    def _check_entries(self, symbol: str, current_row: pd.Series, prev_row: pd.Series, current_time: pd.Timestamp):
        """Check for new entry signals"""
        
        # Determine active mode
        active_mode = self.mode
        if self.mode == 'AUTO':
            regime = current_row['regime']
            if regime == 'TREND':
                active_mode = 'A'
            elif regime == 'RANGE':
                active_mode = 'B'
            elif regime == 'MOMENTUM':
                active_mode = 'C'
            else:
                return
        
        # Check mode-specific max positions
        mode_positions = len(self.position_manager.get_positions_by_mode(active_mode))
        
        max_positions_map = {
            'A': self.config.mode_a_max_positions,
            'B': self.config.mode_b_max_positions,
            'C': self.config.mode_c_max_positions
        }
        
        if mode_positions >= max_positions_map.get(active_mode, 3):
            return
        
        # Get signal
        signal = None
        
        if active_mode == 'A':
            signal = ModeA_ConservativeTrend.check_signal(current_row, prev_row)
        elif active_mode == 'B':
            signal = ModeB_BalancedRange.check_signal(current_row)
        elif active_mode == 'C':
            signal = ModeC_AggressiveMomentum.check_signal(current_row, prev_row)
        
        if signal is None:
            return
        
        # Calculate position parameters
        atr = current_row['atr14']
        
        if active_mode == 'A':
            sl_distance = atr * self.config.mode_a_sl_atr
            tp_distance = atr * self.config.mode_a_tp_atr
            risk_pct = self.config.mode_a_risk
        elif active_mode == 'B':
            sl_distance = atr * self.config.mode_b_sl_atr
            tp_distance = abs(current_row['close'] - current_row['bb_middle'])
            risk_pct = self.config.mode_b_risk
        else:  # C
            sl_distance = atr * self.config.mode_c_sl_atr
            tp_distance = atr * self.config.mode_c_tp_atr
            risk_pct = self.config.mode_c_risk
        
        # Apply spread
        spread = self.config.spread_pips[symbol] * 0.0001 if 'JPY' not in symbol else self.config.spread_pips[symbol] * 0.01
        
        entry_price = current_row['close'] + spread if signal == 'BUY' else current_row['close'] - spread
        
        # Calculate SL and TP
        if signal == 'BUY':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance
        
        # Calculate lot size
        sl_points = abs(entry_price - sl)
        point_value = self.point_values[symbol]
        lot_size = self.risk_manager.calculate_lot_size(self.balance, risk_pct, sl_points, point_value)
        
        risk_amount = self.balance * risk_pct
        
        # Check total exposure
        current_risk = sum(p.risk_amount for p in self.position_manager.positions)
        
        if not self.risk_manager.check_total_exposure(current_risk, risk_amount, self.balance):
            return
        
        # Open position
        position = Position(
            symbol=symbol,
            direction=signal,
            entry_price=entry_price,
            entry_time=current_time,
            lot_size=lot_size,
            sl=sl,
            tp=tp,
            mode=active_mode,
            risk_amount=risk_amount
        )
        
        self.position_manager.add_position(position)
        
        # Deduct commission
        commission = lot_size * self.config.commission_per_lot
        self.balance -= commission
    
    def _check_exits(self, symbol: str, current_row: pd.Series, current_time: pd.Timestamp):
        """Check and execute exits"""
        positions = self.position_manager.get_positions_by_symbol(symbol)
        
        for pos in positions[:]:  # Copy list to avoid modification during iteration
            exit_reason = pos.check_exit(current_row['close'], current_row['high'], current_row['low'])
            
            if exit_reason:
                # Determine exit price
                if exit_reason == 'SL':
                    exit_price = pos.sl
                else:  # TP
                    exit_price = pos.tp
                
                # Apply slippage
                slippage = self.config.slippage_pips * 0.0001
                if pos.direction == 'BUY':
                    exit_price -= slippage
                else:
                    exit_price += slippage
                
                # Close position
                point_value = self.point_values[symbol]
                trade_result = self.position_manager.close_position(pos, exit_price, current_time, exit_reason, point_value)
                
                # Apply commission
                commission = pos.lot_size * self.config.commission_per_lot
                trade_result['pnl'] -= commission
                
                # Update balance
                self.balance += trade_result['pnl']
                
                # Update risk tracker
                self.risk_manager.update_daily_loss(current_time, trade_result['pnl'])
                
                # Save trade
                self.trades.append(trade_result)
    
    def _update_positions(self, symbol: str, current_row: pd.Series):
        """Update trailing stops and BE"""
        current_data = {symbol: current_row}
        point_value = self.point_values[symbol]
        self.position_manager.update_positions(current_data, point_value, current_row['atr14'])
    
    def _update_equity(self, current_time: pd.Timestamp):
        """Update equity curve"""
        self.equity_curve.append({
            'time': current_time,
            'balance': self.balance,
            'equity': self.equity
        })
    
    def get_results(self) -> Dict:
        """Generate backtest statistics"""
        if not self.trades:
            return {
                'initial_balance': self.config.initial_balance,
                'final_balance': self.balance,
                'net_profit_pct': 0,
                'total_trades': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        net_profit = trades_df['pnl'].sum()
        net_profit_pct = (net_profit / self.config.initial_balance) * 100
        
        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['peak'] = equity_df['balance'].cummax()
        equity_df['dd'] = (equity_df['peak'] - equity_df['balance']) / equity_df['peak']
        max_dd = equity_df['dd'].max() * 100
        
        # R-multiples
        avg_r = trades_df['r_multiple'].mean()
        
        # Sharpe (simplified - monthly returns)
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_pnl = trades_df.groupby('month')['pnl'].sum()
        monthly_returns = monthly_pnl / self.config.initial_balance
        
        sharpe = 0
        if len(monthly_returns) > 1:
            sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12) if monthly_returns.std() > 0 else 0
        
        return {
            'initial_balance': self.config.initial_balance,
            'final_balance': self.balance,
            'net_profit': net_profit,
            'net_profit_pct': net_profit_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown_pct': max_dd,
            'avg_r_multiple': avg_r,
            'sharpe_ratio': sharpe,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
            'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0,
        }
