"""
=============================================================================
  LIQUIDITY SWEEP + SMART MONEY CONCEPTS (SMC) BACKTEST ENGINE
  
  استراتژی:
  1. شناسایی سقف/کف‌های مهم (Previous Day High/Low + Swing Points)
  2. انتظار برای Liquidity Sweep (نفوذ و بازگشت)
  3. تشخیص تغییر ساختار (CHoCH / BOS) در تایم پایین
  4. ورود در FVG (Fair Value Gap) بعد از تغییر ساختار
  5. هدف: نقدینگی مخالف (سمت دیگر)
  
  مدیریت ریسک:
  - ریسک ثابت ۰.۵٪ در هر معامله
  - حداکثر ضرر روزانه ۲٪
  - R:R حداقل ۱:۳
  - شبیه‌سازی قوانین پراپ فرم
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yaml
import os
import glob
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
from tqdm import tqdm
from tabulate import tabulate

warnings.filterwarnings('ignore')


# ============================
# ENUMS & DATA CLASSES
# ============================

class Direction(Enum):
    LONG = 1
    SHORT = -1

class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_BE = "CLOSED_BE"
    CLOSED_DAILY_LIMIT = "CLOSED_DAILY_LIMIT"

class MarketBias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class SessionType(Enum):
    LONDON = "LONDON"
    NEWYORK = "NEWYORK"
    OVERLAP = "OVERLAP"
    OFF = "OFF"


@dataclass
class SwingPoint:
    time: pd.Timestamp
    price: float
    is_high: bool  # True = swing high, False = swing low
    broken: bool = False


@dataclass
class FVG:
    time: pd.Timestamp
    top: float
    bottom: float
    direction: Direction  # Bullish FVG = Direction.LONG
    filled: bool = False


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: Direction
    sl_price: float
    tp_price: float
    lot_size: float
    risk_amount: float
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    r_multiple: float = 0.0
    commission: float = 0.0
    reason: str = ""
    symbol: str = ""


@dataclass
class DailyStats:
    date: str
    starting_balance: float
    ending_balance: float
    pnl: float
    trades_taken: int
    wins: int
    losses: int
    daily_drawdown_hit: bool = False


# ============================
# DATA LOADER
# ============================

class DataLoader:
    """بارگذاری و آماده‌سازی داده‌های HistData"""

    @staticmethod
    def load_histdata_csv(filepath: str) -> pd.DataFrame:
        """
        فرمت HistData:
        20200102 170000;1.121220;1.121240;1.121190;1.121200;0
        """
        try:
            df = pd.read_csv(
                filepath,
                sep=';',
                header=None,
                names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                dtype={
                    'DateTime': str,
                    'Open': float,
                    'High': float,
                    'Low': float,
                    'Close': float,
                    'Volume': float
                }
            )

            df['DateTime'] = df['DateTime'].str.strip()

            # Try multiple datetime formats
            parsed = False
            for fmt in ['%Y%m%d %H%M%S', '%Y.%m.%d %H:%M', '%Y%m%d %H:%M:%S',
                        '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S']:
                try:
                    df['DateTime'] = pd.to_datetime(df['DateTime'], format=fmt)
                    parsed = True
                    break
                except (ValueError, TypeError):
                    continue

            if not parsed:
                df['DateTime'] = pd.to_datetime(df['DateTime'], infer_datetime_format=True)

            df.set_index('DateTime', inplace=True)
            df.sort_index(inplace=True)
            df.drop_duplicates(inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.dropna(inplace=True)

            return df

        except Exception as e:
            print(f"  [ERROR] Loading {filepath}: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_symbol_data(data_dir: str, symbol: str, years: list,
                         file_pattern: str) -> pd.DataFrame:
        """بارگذاری تمام سال‌های یک نماد"""
        all_dfs = []

        for year in years:
            filename = file_pattern.format(symbol=symbol, year=year)
            filepath = os.path.join(data_dir, filename)

            if os.path.exists(filepath):
                print(f"  Loading {filename}...")
                df = DataLoader.load_histdata_csv(filepath)
                if not df.empty:
                    all_dfs.append(df)
                    print(f"    -> {len(df):,} candles loaded")
            else:
                print(f"  [SKIP] {filename} not found")

        if all_dfs:
            combined = pd.concat(all_dfs)
            combined.sort_index(inplace=True)
            combined = combined[~combined.index.duplicated(keep='first')]
            print(f"  Total: {len(combined):,} M1 candles for {symbol}")
            return combined

        return pd.DataFrame()

    @staticmethod
    def resample_to_timeframe(df_m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """تبدیل M1 به تایم‌فریم‌های بالاتر"""
        rule = f'{minutes}min' if minutes < 60 else f'{minutes // 60}h'

        resampled = df_m1.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        return resampled


# ============================
# MARKET STRUCTURE ANALYZER
# ============================

class MarketStructure:
    """تحلیل ساختار بازار: Swing Points, BOS, CHoCH"""

    @staticmethod
    def find_swing_points(df: pd.DataFrame, lookback: int = 20) -> List[SwingPoint]:
        """شناسایی نقاط سویینگ (سقف و کف‌ها)"""
        swings = []
        highs = df['High'].values
        lows = df['Low'].values
        times = df.index

        half = lookback // 2

        for i in range(half, len(df) - half):
            # Swing High
            window_highs = highs[i - half:i + half + 1]
            if highs[i] == np.max(window_highs):
                swings.append(SwingPoint(
                    time=times[i],
                    price=highs[i],
                    is_high=True
                ))

            # Swing Low
            window_lows = lows[i - half:i + half + 1]
            if lows[i] == np.min(window_lows):
                swings.append(SwingPoint(
                    time=times[i],
                    price=lows[i],
                    is_high=False
                ))

        return swings

    @staticmethod
    def detect_market_bias(swings: List[SwingPoint],
                           current_idx: int = -1) -> MarketBias:
        """تشخیص بایاس بازار بر اساس HH/HL یا LH/LL"""
        if len(swings) < 4:
            return MarketBias.NEUTRAL

        recent = swings[max(0, len(swings) - 6):]

        swing_highs = [s for s in recent if s.is_high]
        swing_lows = [s for s in recent if not s.is_high]

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Higher High + Higher Low = Bullish
            if (swing_highs[-1].price > swing_highs[-2].price and
                    swing_lows[-1].price > swing_lows[-2].price):
                return MarketBias.BULLISH

            # Lower High + Lower Low = Bearish
            if (swing_highs[-1].price < swing_highs[-2].price and
                    swing_lows[-1].price < swing_lows[-2].price):
                return MarketBias.BEARISH

        return MarketBias.NEUTRAL

    @staticmethod
    def detect_choch(df: pd.DataFrame, swings: List[SwingPoint],
                     current_bar_idx: int, pip_size: float) -> Optional[Direction]:
        """
        تشخیص Change of Character (CHoCH)
        - در روند صعودی: شکست آخرین swing low = CHoCH نزولی
        - در روند نزولی: شکست آخرین swing high = CHoCH صعودی
        """
        if len(swings) < 4:
            return None

        bias = MarketStructure.detect_market_bias(swings)
        current_close = df['Close'].iloc[current_bar_idx]
        current_low = df['Low'].iloc[current_bar_idx]
        current_high = df['High'].iloc[current_bar_idx]

        recent_lows = [s for s in swings[-6:] if not s.is_high]
        recent_highs = [s for s in swings[-6:] if s.is_high]

        if bias == MarketBias.BULLISH and recent_lows:
            last_swing_low = recent_lows[-1]
            if current_close < last_swing_low.price:
                return Direction.SHORT  # CHoCH bearish

        elif bias == MarketBias.BEARISH and recent_highs:
            last_swing_high = recent_highs[-1]
            if current_close > last_swing_high.price:
                return Direction.LONG  # CHoCH bullish

        return None


# ============================
# LIQUIDITY ANALYZER
# ============================

class LiquidityAnalyzer:
    """تحلیل نقدینگی: PDH/PDL, Sweeps"""

    @staticmethod
    def get_previous_day_levels(df_m1: pd.DataFrame,
                                current_time: pd.Timestamp) -> Dict:
        """سقف و کف روز قبل"""
        current_date = current_time.date()
        prev_date = current_date - timedelta(days=1)

        # Handle weekends
        for _ in range(5):
            mask = df_m1.index.date == prev_date
            day_data = df_m1[mask]
            if not day_data.empty:
                return {
                    'pdh': day_data['High'].max(),
                    'pdl': day_data['Low'].min(),
                    'date': prev_date
                }
            prev_date -= timedelta(days=1)

        return None

    @staticmethod
    def get_session_levels(df_m1: pd.DataFrame, current_time: pd.Timestamp,
                           session_start: int, session_end: int) -> Optional[Dict]:
        """سقف و کف یک سشن معاملاتی"""
        current_date = current_time.date()
        mask = (df_m1.index.date == current_date) & \
               (df_m1.index.hour >= session_start) & \
               (df_m1.index.hour < session_end)
        session_data = df_m1[mask]

        if not session_data.empty and len(session_data) > 10:
            return {
                'high': session_data['High'].max(),
                'low': session_data['Low'].min()
            }
        return None

    @staticmethod
    def detect_liquidity_sweep(df: pd.DataFrame, bar_idx: int,
                               level: float, is_high: bool,
                               threshold_pips: float, pip_size: float,
                               max_candles: int = 3) -> bool:
        """
        تشخیص Liquidity Sweep:
        - قیمت از سطح عبور می‌کند (sweep)
        - سپس با کندل بازگشتی به زیر/بالای سطح برمی‌گردد
        """
        if bar_idx < 1 or bar_idx >= len(df):
            return False

        threshold = threshold_pips * pip_size
        current = df.iloc[bar_idx]

        if is_high:
            # Sweep above high
            if current['High'] > level + threshold:
                # Check for rejection (close back below)
                if current['Close'] < level:
                    # Bearish rejection candle
                    body = abs(current['Open'] - current['Close'])
                    upper_wick = current['High'] - max(current['Open'], current['Close'])
                    if upper_wick > body * 0.5:
                        return True

                # Check next few candles for rejection
                for j in range(1, min(max_candles + 1, len(df) - bar_idx)):
                    next_bar = df.iloc[bar_idx + j]
                    if next_bar['Close'] < level:
                        return True

        else:
            # Sweep below low
            if current['Low'] < level - threshold:
                # Check for rejection (close back above)
                if current['Close'] > level:
                    body = abs(current['Open'] - current['Close'])
                    lower_wick = min(current['Open'], current['Close']) - current['Low']
                    if lower_wick > body * 0.5:
                        return True

                for j in range(1, min(max_candles + 1, len(df) - bar_idx)):
                    next_bar = df.iloc[bar_idx + j]
                    if next_bar['Close'] > level:
                        return True

        return False


# ============================
# FVG DETECTOR
# ============================

class FVGDetector:
    """شناسایی Fair Value Gap (شکاف ارزش منصفانه)"""

    @staticmethod
    def find_fvgs(df: pd.DataFrame, min_size_pips: float,
                  pip_size: float) -> List[FVG]:
        """شناسایی FVG‌ها"""
        fvgs = []
        min_size = min_size_pips * pip_size

        for i in range(2, len(df)):
            candle1 = df.iloc[i - 2]
            candle2 = df.iloc[i - 1]
            candle3 = df.iloc[i]

            # Bullish FVG: gap between candle1 high and candle3 low
            if candle3['Low'] > candle1['High']:
                gap_size = candle3['Low'] - candle1['High']
                if gap_size >= min_size:
                    fvgs.append(FVG(
                        time=df.index[i - 1],
                        top=candle3['Low'],
                        bottom=candle1['High'],
                        direction=Direction.LONG
                    ))

            # Bearish FVG: gap between candle3 high and candle1 low
            if candle3['High'] < candle1['Low']:
                gap_size = candle1['Low'] - candle3['High']
                if gap_size >= min_size:
                    fvgs.append(FVG(
                        time=df.index[i - 1],
                        top=candle1['Low'],
                        bottom=candle3['High'],
                        direction=Direction.SHORT
                    ))

        return fvgs

    @staticmethod
    def find_recent_fvg(fvgs: List[FVG], direction: Direction,
                        current_time: pd.Timestamp,
                        max_age_bars: int = 50) -> Optional[FVG]:
        """پیدا کردن آخرین FVG پر نشده"""
        matching = [f for f in fvgs
                    if f.direction == direction
                    and not f.filled
                    and f.time < current_time]

        if matching:
            return matching[-1]
        return None


# ============================
# SESSION MANAGER
# ============================

class SessionManager:
    """مدیریت سشن‌های معاملاتی"""

    def __init__(self, config: dict):
        self.london_start = config['strategy']['london_start']
        self.london_end = config['strategy']['london_end']
        self.ny_start = config['strategy']['newyork_start']
        self.ny_end = config['strategy']['newyork_end']

    def get_session(self, timestamp: pd.Timestamp) -> SessionType:
        hour = timestamp.hour
        in_london = self.london_start <= hour < self.london_end
        in_ny = self.ny_start <= hour < self.ny_end

        if in_london and in_ny:
            return SessionType.OVERLAP
        elif in_london:
            return SessionType.LONDON
        elif in_ny:
            return SessionType.NEWYORK
        else:
            return SessionType.OFF

    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        session = self.get_session(timestamp)
        return session != SessionType.OFF


# ============================
# POSITION SIZER
# ============================

class PositionSizer:
    """محاسبه حجم معامله"""

    @staticmethod
    def calculate_lot_size(balance: float, risk_pct: float,
                           sl_pips: float, pip_value: float) -> float:
        """
        محاسبه سایز لات:
        risk_amount = balance * risk_pct
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        """
        if sl_pips <= 0:
            return 0.0

        risk_amount = balance * risk_pct
        pip_value_per_lot = pip_value * 100000  # Standard lot
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)

        # Round to 2 decimal places, min 0.01
        lot_size = max(0.01, round(lot_size, 2))
        return lot_size


# ============================
# MAIN BACKTEST ENGINE
# ============================

class BacktestEngine:
    """موتور اصلی بکتست"""

    def __init__(self, config: dict):
        self.config = config
        self.session_mgr = SessionManager(config)

        # Account
        self.initial_balance = config['account']['initial_balance']
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_balance = self.initial_balance

        # Pip sizes
        self.pip_sizes = {
            'EURUSD': 0.0001,
            'GBPUSD': 0.0001,
            'USDCHF': 0.0001,
            'USDCAD': 0.0001,
            'AUDUSD': 0.0001,
            'NZDUSD': 0.0001,
            'AUDNZD': 0.0001,
            'EURGBP': 0.0001,
            'XAUUSD': 0.10,
            'XAGUSD': 0.001,
        }

        # Pip values (per pip per standard lot)
        self.pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'USDCHF': 10.0,
            'USDCAD': 10.0,
            'AUDUSD': 10.0,
            'NZDUSD': 10.0,
            'AUDNZD': 10.0,
            'EURGBP': 10.0,
            'XAUUSD': 10.0,
            'XAGUSD': 50.0,
        }

        # Results
        self.trades: List[Trade] = []
        self.equity_curve: List[dict] = []
        self.daily_stats: List[DailyStats] = []

        # State
        self.open_trades: List[Trade] = []
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.current_date = None
        self.day_start_balance = 0.0

        # Prop firm tracking
        self.prop_passed_phase1 = False
        self.prop_passed_phase2 = False
        self.prop_blown = False
        self.trading_days = set()

    def get_pip_size(self, symbol: str) -> float:
        return self.pip_sizes.get(symbol, 0.0001)

    def get_pip_value(self, symbol: str) -> float:
        return self.pip_values.get(symbol, 10.0)

    def price_to_pips(self, price_diff: float, symbol: str) -> float:
        return abs(price_diff) / self.get_pip_size(symbol)

    def check_daily_reset(self, current_time: pd.Timestamp):
        """ریست آمار روزانه"""
        current_date_str = str(current_time.date())

        if self.current_date != current_date_str:
            # Save previous day stats
            if self.current_date is not None:
                self.daily_stats.append(DailyStats(
                    date=self.current_date,
                    starting_balance=self.day_start_balance,
                    ending_balance=self.balance,
                    pnl=self.daily_pnl,
                    trades_taken=self.daily_trades_count,
                    wins=sum(1 for t in self.trades
                             if str(t.entry_time.date()) == self.current_date
                             and t.pnl > 0),
                    losses=sum(1 for t in self.trades
                               if str(t.entry_time.date()) == self.current_date
                               and t.pnl < 0),
                    daily_drawdown_hit=self.daily_pnl <=
                                       -self.day_start_balance *
                                       self.config['risk']['max_daily_loss']
                ))

            self.current_date = current_date_str
            self.day_start_balance = self.balance
            self.daily_pnl = 0.0
            self.daily_trades_count = 0

    def check_prop_rules(self) -> bool:
        """بررسی قوانین پراپ فرم"""
        max_dd = self.config['prop_rules']['max_total_drawdown']
        max_daily_dd = self.config['prop_rules']['max_daily_drawdown']

        # Total drawdown check
        total_dd = (self.peak_balance - self.balance) / self.initial_balance
        if total_dd >= max_dd:
            self.prop_blown = True
            return False

        # Daily drawdown check
        if self.day_start_balance > 0:
            daily_dd = (self.day_start_balance - self.balance) / \
                       self.day_start_balance
            if daily_dd >= max_daily_dd:
                return False

        return True

    def can_open_trade(self, current_time: pd.Timestamp) -> bool:
        """آیا می‌توان معامله جدید باز کرد؟"""
        # Check prop rules
        if not self.check_prop_rules():
            return False

        # Check daily loss limit (self-imposed)
        max_daily = self.config['risk']['max_daily_loss']
        if self.daily_pnl <= -self.day_start_balance * max_daily:
            return False

        # Check max concurrent trades
        if len(self.open_trades) >= self.config['risk']['max_concurrent_trades']:
            return False

        # Check session
        if self.config['strategy']['trade_sessions']:
            if not self.session_mgr.is_trading_time(current_time):
                return False

        return True

    def open_trade(self, entry_time: pd.Timestamp, entry_price: float,
                   direction: Direction, sl_price: float, tp_price: float,
                   symbol: str, reason: str = ""):
        """باز کردن معامله جدید"""
        # Apply spread and slippage
        spread = self.config['execution']['spread_pips'] * self.get_pip_size(symbol)
        slippage = self.config['execution']['slippage_pips'] * self.get_pip_size(symbol)

        if direction == Direction.LONG:
            entry_price += (spread / 2 + slippage)
        else:
            entry_price -= (spread / 2 + slippage)

        # Calculate SL distance in pips
        sl_pips = self.price_to_pips(entry_price - sl_price, symbol)

        if sl_pips <= 0:
            return

        # Position sizing
        risk_pct = self.config['risk']['risk_per_trade']
        pip_value = self.get_pip_value(symbol)
        lot_size = PositionSizer.calculate_lot_size(
            self.balance, risk_pct, sl_pips, pip_value / 100000
        )

        risk_amount = self.balance * risk_pct
        commission = self.config['execution']['commission_per_lot'] * lot_size

        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            sl_price=sl_price,
            tp_price=tp_price,
            lot_size=lot_size,
            risk_amount=risk_amount,
            commission=commission,
            symbol=symbol,
            reason=reason
        )

        self.open_trades.append(trade)
        self.daily_trades_count += 1
        self.trading_days.add(str(entry_time.date()))

    def update_open_trades(self, bar: pd.Series, bar_time: pd.Timestamp,
                           symbol: str):
        """بروزرسانی معاملات باز با کندل جدید"""
        closed_trades = []

        for trade in self.open_trades:
            if trade.symbol != symbol:
                continue

            hit_sl = False
            hit_tp = False
            exit_price = 0.0

            if trade.direction == Direction.LONG:
                # Check SL
                if bar['Low'] <= trade.sl_price:
                    hit_sl = True
                    exit_price = trade.sl_price

                # Check TP
                if bar['High'] >= trade.tp_price:
                    hit_tp = True
                    exit_price = trade.tp_price

                # If both hit in same bar, use OHLC order
                if hit_sl and hit_tp:
                    # Assume SL hit first if open is closer to SL
                    if abs(bar['Open'] - trade.sl_price) < \
                       abs(bar['Open'] - trade.tp_price):
                        hit_tp = False
                        exit_price = trade.sl_price
                    else:
                        hit_sl = False
                        exit_price = trade.tp_price

                # Breakeven logic
                if self.config['strategy']['breakeven_at_1r'] and \
                   not hit_sl and not hit_tp:
                    one_r = abs(trade.entry_price - trade.sl_price)
                    if bar['High'] >= trade.entry_price + one_r:
                        if trade.sl_price < trade.entry_price:
                            trade.sl_price = trade.entry_price + \
                                             self.get_pip_size(symbol) * 2

            else:  # SHORT
                if bar['High'] >= trade.sl_price:
                    hit_sl = True
                    exit_price = trade.sl_price

                if bar['Low'] <= trade.tp_price:
                    hit_tp = True
                    exit_price = trade.tp_price

                if hit_sl and hit_tp:
                    if abs(bar['Open'] - trade.sl_price) < \
                       abs(bar['Open'] - trade.tp_price):
                        hit_tp = False
                        exit_price = trade.sl_price
                    else:
                        hit_sl = False
                        exit_price = trade.tp_price

                if self.config['strategy']['breakeven_at_1r'] and \
                   not hit_sl and not hit_tp:
                    one_r = abs(trade.sl_price - trade.entry_price)
                    if bar['Low'] <= trade.entry_price - one_r:
                        if trade.sl_price > trade.entry_price:
                            trade.sl_price = trade.entry_price - \
                                             self.get_pip_size(symbol) * 2

            # Close trade if hit
            if hit_sl or hit_tp:
                trade.exit_time = bar_time
                trade.exit_price = exit_price

                pip_size = self.get_pip_size(symbol)
                pip_val = self.get_pip_value(symbol)

                if trade.direction == Direction.LONG:
                    trade.pnl_pips = (exit_price - trade.entry_price) / pip_size
                else:
                    trade.pnl_pips = (trade.entry_price - exit_price) / pip_size

                trade.pnl = (trade.pnl_pips * pip_val * trade.lot_size) - \
                            trade.commission

                if trade.risk_amount > 0:
                    trade.r_multiple = trade.pnl / trade.risk_amount

                if hit_tp:
                    trade.status = TradeStatus.CLOSED_TP
                elif hit_sl:
                    if abs(trade.sl_price - trade.entry_price) < \
                       pip_size * 5:
                        trade.status = TradeStatus.CLOSED_BE
                    else:
                        trade.status = TradeStatus.CLOSED_SL

                self.balance += trade.pnl
                self.daily_pnl += trade.pnl
                self.peak_balance = max(self.peak_balance, self.balance)

                self.trades.append(trade)
                closed_trades.append(trade)

        for t in closed_trades:
            if t in self.open_trades:
                self.open_trades.remove(t)

    def run_backtest(self, df_m1: pd.DataFrame, symbol: str):
        """اجرای بکتست روی یک نماد"""
        print(f"\n{'=' * 60}")
        print(f"  RUNNING BACKTEST: {symbol}")
        print(f"  Period: {df_m1.index[0]} to {df_m1.index[-1]}")
        print(f"  M1 Candles: {len(df_m1):,}")
        print(f"{'=' * 60}")

        cfg = self.config['strategy']
        pip_size = self.get_pip_size(symbol)

        # Build higher timeframes
        print("  Building timeframes...")
        df_htf = DataLoader.resample_to_timeframe(df_m1, cfg['htf_minutes'])
        df_mtf = DataLoader.resample_to_timeframe(df_m1, cfg['mtf_minutes'])
        df_ltf = DataLoader.resample_to_timeframe(df_m1, cfg['ltf_minutes'])
        print(f"    HTF ({cfg['htf_minutes']}min): {len(df_htf):,} candles")
        print(f"    MTF ({cfg['mtf_minutes']}min): {len(df_mtf):,} candles")
        print(f"    LTF ({cfg['ltf_minutes']}min): {len(df_ltf):,} candles")

        # Pre-compute swing points on HTF
        print("  Computing swing points (HTF)...")
        htf_swings = MarketStructure.find_swing_points(
            df_htf, cfg['swing_lookback']
        )
        print(f"    Found {len(htf_swings)} swing points")

        # Pre-compute FVGs on LTF
        print("  Detecting FVGs (LTF)...")
        ltf_fvgs = FVGDetector.find_fvgs(df_ltf, cfg['fvg_min_size_pips'],
                                          pip_size)
        print(f"    Found {len(ltf_fvgs)} FVGs")

        # Main loop on MTF
        print("  Running strategy on MTF...")
        mtf_times = df_mtf.index

        signal_cooldown = 0

        for i in tqdm(range(cfg['structure_lookback'], len(df_mtf)),
                      desc=f"  {symbol}", ncols=80):

            current_bar = df_mtf.iloc[i]
            current_time = mtf_times[i]

            # Daily reset
            self.check_daily_reset(current_time)

            # Update equity curve
            unrealized = 0
            for ot in self.open_trades:
                if ot.direction == Direction.LONG:
                    unrealized += (current_bar['Close'] - ot.entry_price) / \
                                  pip_size * self.get_pip_value(symbol) * \
                                  ot.lot_size
                else:
                    unrealized += (ot.entry_price - current_bar['Close']) / \
                                  pip_size * self.get_pip_value(symbol) * \
                                  ot.lot_size

            self.equity = self.balance + unrealized
            self.equity_curve.append({
                'time': current_time,
                'balance': self.balance,
                'equity': self.equity,
                'symbol': symbol
            })

            # Update open trades with MTF bar (approximate with M1 in
            # real engine, but for speed we use MTF)
            self.update_open_trades(current_bar, current_time, symbol)

            # Check if account blown
            if self.prop_blown:
                print(f"\n  [!] ACCOUNT BLOWN at {current_time}")
                break

            # Cooldown between signals
            if signal_cooldown > 0:
                signal_cooldown -= 1
                continue

            # Can we trade?
            if not self.can_open_trade(current_time):
                continue

            # ========== STRATEGY LOGIC ==========

            # Step 1: Get previous day levels
            pdl_data = LiquidityAnalyzer.get_previous_day_levels(
                df_m1, current_time
            )
            if pdl_data is None:
                continue

            pdh = pdl_data['pdh']
            pdl = pdl_data['pdl']

            # Step 2: Get relevant swing points up to current time
            relevant_swings = [s for s in htf_swings if s.time <= current_time]
            if len(relevant_swings) < 4:
                continue

            # Step 3: Determine market bias on HTF
            htf_bias = MarketStructure.detect_market_bias(relevant_swings)

            # Step 4: Check for liquidity sweep at PDH or PDL
            sweep_high = False
            sweep_low = False

            # Check sweep of PDH (bearish setup)
            if current_bar['High'] > pdh:
                if current_bar['Close'] < pdh:
                    sweep_high = True

            # Check sweep of PDL (bullish setup)
            if current_bar['Low'] < pdl:
                if current_bar['Close'] > pdl:
                    sweep_low = True

            # Also check swing highs/lows
            recent_swing_highs = [s for s in relevant_swings[-10:]
                                  if s.is_high]
            recent_swing_lows = [s for s in relevant_swings[-10:]
                                 if not s.is_high]

            if not sweep_high and recent_swing_highs:
                for sh in recent_swing_highs[-3:]:
                    if current_bar['High'] > sh.price + \
                       cfg['sweep_threshold_pips'] * pip_size:
                        if current_bar['Close'] < sh.price:
                            sweep_high = True
                            break

            if not sweep_low and recent_swing_lows:
                for sl_point in recent_swing_lows[-3:]:
                    if current_bar['Low'] < sl_point.price - \
                       cfg['sweep_threshold_pips'] * pip_size:
                        if current_bar['Close'] > sl_point.price:
                            sweep_low = True
                            break

            if not sweep_high and not sweep_low:
                continue

            # Step 5: Look for CHoCH/BOS on MTF
            mtf_swings_local = MarketStructure.find_swing_points(
                df_mtf.iloc[max(0, i - cfg['structure_lookback']):i + 1],
                lookback=10
            )

            choch = MarketStructure.detect_choch(
                df_mtf, mtf_swings_local, min(i, len(df_mtf) - 1), pip_size
            )

            # Step 6: Determine trade direction
            trade_direction = None

            if sweep_high and (choch == Direction.SHORT or
                               htf_bias == MarketBias.BEARISH):
                trade_direction = Direction.SHORT

            elif sweep_low and (choch == Direction.LONG or
                                htf_bias == MarketBias.BULLISH):
                trade_direction = Direction.LONG

            if trade_direction is None:
                continue

            # Step 7: Find FVG for entry
            relevant_fvgs = [f for f in ltf_fvgs
                             if f.time <= current_time and not f.filled]
            target_fvg = FVGDetector.find_recent_fvg(
                relevant_fvgs, trade_direction, current_time
            )

            # Step 8: Calculate entry, SL, TP
            entry_price = current_bar['Close']
            sl_buffer = cfg['sl_buffer_pips'] * pip_size
            rr = self.config['risk']['reward_to_risk']

            if trade_direction == Direction.LONG:
                # SL below the sweep low
                if recent_swing_lows:
                    sl_price = min(s.price for s in recent_swing_lows[-3:]) \
                               - sl_buffer
                else:
                    sl_price = pdl - sl_buffer

                sl_price = min(sl_price, current_bar['Low'] - sl_buffer)

                # If FVG exists, try to get better entry
                if target_fvg:
                    potential_entry = target_fvg.top
                    if potential_entry < entry_price:
                        entry_price = potential_entry

                sl_distance = entry_price - sl_price
                if sl_distance <= 0:
                    continue

                tp_price = entry_price + sl_distance * rr

                # Target: opposite liquidity (PDH or next swing high)
                if recent_swing_highs:
                    opposite_liq = max(s.price for s in recent_swing_highs[-3:])
                    tp_price = max(tp_price, opposite_liq)

            else:  # SHORT
                if recent_swing_highs:
                    sl_price = max(s.price for s in recent_swing_highs[-3:]) \
                               + sl_buffer
                else:
                    sl_price = pdh + sl_buffer

                sl_price = max(sl_price, current_bar['High'] + sl_buffer)

                if target_fvg:
                    potential_entry = target_fvg.bottom
                    if potential_entry > entry_price:
                        entry_price = potential_entry

                sl_distance = sl_price - entry_price
                if sl_distance <= 0:
                    continue

                tp_price = entry_price - sl_distance * rr

                if recent_swing_lows:
                    opposite_liq = min(s.price for s in recent_swing_lows[-3:])
                    tp_price = min(tp_price, opposite_liq)

            # Validate SL distance
            sl_pips = self.price_to_pips(sl_distance, symbol)
            if sl_pips < 3 or sl_pips > 100:
                continue

            # Step 9: Open trade
            reason = f"{'Sweep_PDH' if sweep_high else 'Sweep_PDL'}" \
                     f"_{'CHoCH' if choch else 'Bias'}" \
                     f"_{'FVG' if target_fvg else 'Market'}"

            self.open_trade(
                entry_time=current_time,
                entry_price=entry_price,
                direction=trade_direction,
                sl_price=sl_price,
                tp_price=tp_price,
                symbol=symbol,
                reason=reason
            )

            signal_cooldown = 4  # Skip next 4 MTF bars

            # Mark FVG as filled
            if target_fvg:
                target_fvg.filled = True

        # Close any remaining open trades at last price
        if self.open_trades:
            last_bar = df_mtf.iloc[-1]
            last_time = mtf_times[-1]
            for trade in self.open_trades[:]:
                trade.exit_time = last_time
                trade.exit_price = last_bar['Close']
                if trade.direction == Direction.LONG:
                    trade.pnl_pips = (last_bar['Close'] - trade.entry_price) \
                                     / pip_size
                else:
                    trade.pnl_pips = (trade.entry_price - last_bar['Close']) \
                                     / pip_size
                pv = self.get_pip_value(symbol)
                trade.pnl = (trade.pnl_pips * pv * trade.lot_size) - \
                            trade.commission
                if trade.risk_amount > 0:
                    trade.r_multiple = trade.pnl / trade.risk_amount
                trade.status = TradeStatus.CLOSED_SL
                self.balance += trade.pnl
                self.trades.append(trade)
            self.open_trades.clear()


# ============================
# RESULTS ANALYZER
# ============================

class ResultsAnalyzer:
    """تحلیل و نمایش نتایج بکتست"""

    def __init__(self, engine: BacktestEngine, config: dict):
        self.engine = engine
        self.config = config
        self.output_dir = config['output']['output_directory']
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self):
        """تولید گزارش کامل"""
        trades = self.engine.trades
        if not trades:
            print("\n  [!] No trades executed. Check data/parameters.")
            return

        print(f"\n{'=' * 70}")
        print(f"{'BACKTEST RESULTS':^70}")
        print(f"{'=' * 70}")

        # Basic stats
        total = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        breakevens = [t for t in trades if t.pnl == 0]

        win_rate = len(wins) / total * 100 if total > 0 else 0
        total_pnl = sum(t.pnl for t in trades)
        total_pips = sum(t.pnl_pips for t in trades)

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        avg_win_pips = np.mean([t.pnl_pips for t in wins]) if wins else 0
        avg_loss_pips = np.mean([t.pnl_pips for t in losses]) if losses else 0

        profit_factor = abs(sum(t.pnl for t in wins) /
                           sum(t.pnl for t in losses)) \
            if losses and sum(t.pnl for t in losses) != 0 else float('inf')

        avg_r = np.mean([t.r_multiple for t in trades])

        # Drawdown
        equity_values = [e['balance'] for e in self.engine.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            max_dd_pct = 0
            for val in equity_values:
                if val > peak:
                    peak = val
                dd = peak - val
                dd_pct = dd / peak * 100
                if dd_pct > max_dd_pct:
                    max_dd = dd
                    max_dd_pct = dd_pct
        else:
            max_dd = 0
            max_dd_pct = 0

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_streak = 0
        for t in trades:
            if t.pnl > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_consec_wins = max(max_consec_wins, current_streak)
            elif t.pnl < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_consec_losses = max(max_consec_losses, abs(current_streak))

        # Return percentage
        return_pct = (self.engine.balance - self.engine.initial_balance) / \
                     self.engine.initial_balance * 100

        # Table
        stats_table = [
            ["Initial Balance", f"${self.engine.initial_balance:,.2f}"],
            ["Final Balance", f"${self.engine.balance:,.2f}"],
            ["Net P&L", f"${total_pnl:,.2f}"],
            ["Return", f"{return_pct:.2f}%"],
            ["", ""],
            ["Total Trades", total],
            ["Winning Trades", f"{len(wins)} ({win_rate:.1f}%)"],
            ["Losing Trades", len(losses)],
            ["Breakeven Trades", len(breakevens)],
            ["", ""],
            ["Avg Win", f"${avg_win:,.2f} ({avg_win_pips:.1f} pips)"],
            ["Avg Loss", f"${avg_loss:,.2f} ({avg_loss_pips:.1f} pips)"],
            ["Avg R-Multiple", f"{avg_r:.2f}R"],
            ["Profit Factor", f"{profit_factor:.2f}"],
            ["Total Pips", f"{total_pips:.1f}"],
            ["", ""],
            ["Max Drawdown", f"${max_dd:,.2f} ({max_dd_pct:.2f}%)"],
            ["Max Consec. Wins", max_consec_wins],
            ["Max Consec. Losses", max_consec_losses],
            ["Trading Days", len(self.engine.trading_days)],
            ["", ""],
            ["Prop Phase 1 (8%)",
             "PASSED ✅" if return_pct >= 8 else "FAILED ❌"],
            ["Prop Max DD Rule (10%)",
             "OK ✅" if max_dd_pct < 10 else "VIOLATED ❌"],
            ["Prop Daily DD Rule (5%)",
             "OK ✅" if all(not d.daily_drawdown_hit
                           for d in self.engine.daily_stats)
             else "VIOLATED ❌"],
        ]

        print(tabulate(stats_table, headers=["Metric", "Value"],
                        tablefmt="fancy_grid"))

        # Monthly breakdown
        self._print_monthly_breakdown(trades)

        # Save trades to CSV
        if self.config['output']['save_trades']:
            self._save_trades_csv(trades)

        # Plot equity curve
        if self.config['output']['save_equity_curve']:
            self._plot_equity_curve()

        # Plot monthly returns heatmap
        if self.config['output']['save_charts']:
            self._plot_monthly_heatmap(trades)
            self._plot_distribution(trades)

    def _print_monthly_breakdown(self, trades: List[Trade]):
        """خلاصه ماهانه"""
        print(f"\n{'MONTHLY BREAKDOWN':^70}")
        print("-" * 70)

        monthly = {}
        for t in trades:
            if t.exit_time:
                key = t.exit_time.strftime('%Y-%m')
            else:
                key = t.entry_time.strftime('%Y-%m')
            if key not in monthly:
                monthly[key] = {'pnl': 0, 'trades': 0, 'wins': 0}
            monthly[key]['pnl'] += t.pnl
            monthly[key]['trades'] += 1
            if t.pnl > 0:
                monthly[key]['wins'] += 1

        rows = []
        for month in sorted(monthly.keys()):
            d = monthly[month]
            wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
            rows.append([
                month, d['trades'], f"{wr:.0f}%", f"${d['pnl']:,.2f}"
            ])

        print(tabulate(rows,
                        headers=["Month", "Trades", "Win Rate", "P&L"],
                        tablefmt="simple"))

    def _save_trades_csv(self, trades: List[Trade]):
        """ذخیره معاملات در CSV"""
        filepath = os.path.join(self.output_dir, "trades.csv")
        records = []
        for t in trades:
            records.append({
                'Symbol': t.symbol,
                'Direction': t.direction.name,
                'Entry_Time': t.entry_time,
                'Exit_Time': t.exit_time,
                'Entry_Price': t.entry_price,
                'Exit_Price': t.exit_price,
                'SL': t.sl_price,
                'TP': t.tp_price,
                'Lot_Size': t.lot_size,
                'PnL': round(t.pnl, 2),
                'PnL_Pips': round(t.pnl_pips, 1),
                'R_Multiple': round(t.r_multiple, 2),
                'Status': t.status.value,
                'Reason': t.reason
            })
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False)
        print(f"\n  Trades saved to: {filepath}")

    def _plot_equity_curve(self):
        """رسم نمودار اکوئیتی"""
        if not self.engine.equity_curve:
            return

        filepath = os.path.join(self.output_dir, "equity_curve.png")
        df = pd.DataFrame(self.engine.equity_curve)
        df['time'] = pd.to_datetime(df['time'])

        fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                                  gridspec_kw={'height_ratios': [3, 1]})

        # Equity curve
        axes[0].plot(df['time'], df['balance'], color='#2196F3',
                     linewidth=1.5, label='Balance')
        axes[0].axhline(y=self.engine.initial_balance, color='gray',
                         linestyle='--', alpha=0.5, label='Initial Balance')

        # Profit target line
        target = self.engine.initial_balance * 1.08
        axes[0].axhline(y=target, color='green', linestyle='--',
                         alpha=0.5, label='Profit Target (8%)')

        # Max drawdown line
        dd_line = self.engine.initial_balance * 0.90
        axes[0].axhline(y=dd_line, color='red', linestyle='--',
                         alpha=0.5, label='Max DD Limit (10%)')

        axes[0].set_title('Equity Curve - Liquidity Sweep + SMC Strategy',
                           fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Balance ($)')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)

        # Drawdown
        peak = df['balance'].cummax()
        dd = (peak - df['balance']) / peak * 100
        axes[1].fill_between(df['time'], dd, color='red', alpha=0.3)
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('DD %')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Equity curve saved to: {filepath}")

    def _plot_monthly_heatmap(self, trades: List[Trade]):
        """هیت‌مپ بازده ماهانه"""
        if not trades:
            return

        filepath = os.path.join(self.output_dir, "monthly_heatmap.png")

        monthly_pnl = {}
        for t in trades:
            dt = t.exit_time if t.exit_time else t.entry_time
            year = dt.year
            month = dt.month
            key = (year, month)
            monthly_pnl[key] = monthly_pnl.get(key, 0) + t.pnl

        if not monthly_pnl:
            return

        years = sorted(set(k[0] for k in monthly_pnl.keys()))
        months = list(range(1, 13))

        data = np.full((len(years), 12), np.nan)
        for (y, m), pnl in monthly_pnl.items():
            yi = years.index(y)
            data[yi][m - 1] = pnl

        fig, ax = plt.subplots(figsize=(14, max(4, len(years) * 1.5)))
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        sns.heatmap(data, annot=True, fmt='.0f', cmap='RdYlGn',
                    center=0, xticklabels=month_labels,
                    yticklabels=years, ax=ax, linewidths=1)

        ax.set_title('Monthly P&L Heatmap ($)', fontsize=14,
                      fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Monthly heatmap saved to: {filepath}")

    def _plot_distribution(self, trades: List[Trade]):
        """توزیع R-Multiple معاملات"""
        if not trades:
            return

        filepath = os.path.join(self.output_dir, "r_distribution.png")
        r_values = [t.r_multiple for t in trades]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if r > 0 else 'red' for r in r_values]

        ax.bar(range(len(r_values)), r_values, color=colors, alpha=0.7,
               width=1.0)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axhline(y=np.mean(r_values), color='blue', linestyle='--',
                    label=f'Avg R: {np.mean(r_values):.2f}')
        ax.set_title('Trade R-Multiple Distribution', fontsize=14,
                      fontweight='bold')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('R-Multiple')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  R-distribution saved to: {filepath}")


# ============================
# MAIN RUNNER
# ============================

def main():
    """اجرای اصلی بکتست"""
    print("=" * 70)
    print("  LIQUIDITY SWEEP + SMC BACKTEST ENGINE")
    print("  Prop Firm Challenge Simulator")
    print("=" * 70)

    # Load config
    config_path = "config.yml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n  Config loaded: {config_path}")
    print(f"  Initial Balance: ${config['account']['initial_balance']:,}")
    print(f"  Risk per Trade: {config['risk']['risk_per_trade'] * 100}%")
    print(f"  R:R Target: 1:{config['risk']['reward_to_risk']}")

    # Initialize engine
    engine = BacktestEngine(config)

    # Load and process each symbol
    data_dir = config['data']['directory']
    symbols = config['data']['symbols']
    years = config['data']['years']
    file_pattern = config['data']['file_pattern']

    for symbol in symbols:
        print(f"\n{'─' * 50}")
        print(f"  Loading data for {symbol}...")
        print(f"{'─' * 50}")

        df_m1 = DataLoader.load_symbol_data(data_dir, symbol, years,
                                             file_pattern)

        if df_m1.empty:
            print(f"  [SKIP] No data for {symbol}")
            continue

        # Filter weekends
        df_m1 = df_m1[df_m1.index.dayofweek < 5]

        # Run backtest
        engine.run_backtest(df_m1, symbol)

        if engine.prop_blown:
            print("  Account blown - stopping all backtests")
            break

    # Generate results
    print(f"\n{'─' * 50}")
    print("  Generating reports...")
    print(f"{'─' * 50}")

    analyzer = ResultsAnalyzer(engine, config)
    analyzer.generate_report()

    print(f"\n{'=' * 70}")
    print("  BACKTEST COMPLETE")
    print(f"  Results saved to: {config['output']['output_directory']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
