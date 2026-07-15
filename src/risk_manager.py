"""
Risk management layer.

Tracks daily/weekly drawdown, exposure limits, session filters.
All checks are pure functions given current state - no lookahead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# News event database (simplified - major events only)
# ---------------------------------------------------------------------------

# In a real system this would be loaded from an economic calendar.
# For backtesting we approximate: we know which days had major news.
# We use a conservative approach: RBNZ/RBA/ECB/Fed/BoE meetings are
# typically on specific days. We'll use a flag-based approach.

# Format: set of (year, month, day) tuples for known high-impact days
# This is a simplified list. In production use a proper calendar feed.
KNOWN_HIGH_IMPACT_DATES: Set[tuple] = set()  # populated separately


def load_news_calendar(filepath: Optional[str] = None) -> Set[tuple]:
    """
    Load news dates from CSV if available.
    Format: date,time,currency,impact
    Returns set of (year, month, day) for high-impact events.
    """
    if filepath is None:
        return KNOWN_HIGH_IMPACT_DATES

    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        high = df[df["impact"].str.upper() == "HIGH"]
        dates = set()
        for _, row in high.iterrows():
            d = row["date"]
            dates.add((d.year, d.month, d.day))
        return dates
    except Exception as e:
        logger.warning(f"Could not load news calendar: {e}")
        return KNOWN_HIGH_IMPACT_DATES


# ---------------------------------------------------------------------------
# Session checker
# ---------------------------------------------------------------------------

def is_tradeable_session(dt: pd.Timestamp, config) -> bool:
    """
    Returns True if dt falls within an allowed trading session.

    Allowed:
    - London:          08:00–16:00 UTC
    - London-NY overlap: 13:00–16:00 UTC (subset of above)

    Forbidden:
    - Asian session:   00:00–08:00 UTC
    - Friday after 12:00 UTC
    - Weekend
    """
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    hour = dt.hour

    # Friday cutoff
    if dt.weekday() == 4 and hour >= config.friday_cutoff_utc:
        return False

    # Allowed: London session
    if config.london_open_utc <= hour < config.london_close_utc:
        return True

    return False


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

@dataclass
class DailyStats:
    date: object
    start_equity: float
    min_equity: float = 0.0
    trades: int = 0
    drawdown_hit: bool = False


@dataclass
class WeeklyStats:
    week_key: str  # "YYYY-WW"
    start_equity: float
    min_equity: float = 0.0
    disabled: bool = False


class RiskManager:
    """
    Stateful risk manager for the backtest.
    Must be updated at each bar with current equity.
    """

    def __init__(self, config, initial_capital: float):
        self.config  = config
        self.capital = initial_capital

        self._daily: Dict[str, DailyStats]  = {}
        self._weekly: Dict[str, WeeklyStats] = {}

        self._current_date_key: str = ""
        self._current_week_key: str = ""

        self._news_dates: Set[tuple] = KNOWN_HIGH_IMPACT_DATES

        # Track open baskets per pair per direction
        self.open_baskets_count: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, current_time: pd.Timestamp, current_equity: float):
        """Call at every bar with current equity."""
        date_key = current_time.strftime("%Y-%m-%d")
        week_key = current_time.strftime("%Y-%W")

        # --- Daily tracking ---
        if date_key != self._current_date_key:
            self._current_date_key = date_key
            if date_key not in self._daily:
                self._daily[date_key] = DailyStats(
                    date=current_time.date(),
                    start_equity=current_equity,
                    min_equity=current_equity,
                )

        day = self._daily[date_key]
        day.min_equity = min(day.min_equity, current_equity)

        # Check daily DD
        daily_dd = (day.start_equity - current_equity) / day.start_equity
        if daily_dd >= self.config.max_daily_drawdown_pct:
            day.drawdown_hit = True

        # --- Weekly tracking ---
        if week_key != self._current_week_key:
            self._current_week_key = week_key
            if week_key not in self._weekly:
                self._weekly[week_key] = WeeklyStats(
                    week_key=week_key,
                    start_equity=current_equity,
                    min_equity=current_equity,
                )

        week = self._weekly[week_key]
        week.min_equity = min(week.min_equity, current_equity)

        weekly_dd = (week.start_equity - current_equity) / week.start_equity
        if weekly_dd >= self.config.max_weekly_drawdown_pct:
            week.disabled = True

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def is_daily_dd_hit(self, current_time: pd.Timestamp) -> bool:
        date_key = current_time.strftime("%Y-%m-%d")
        if date_key in self._daily:
            return self._daily[date_key].drawdown_hit
        return False

    def is_weekly_disabled(self, current_time: pd.Timestamp) -> bool:
        week_key = current_time.strftime("%Y-%W")
        if week_key in self._weekly:
            return self._weekly[week_key].disabled
        return False

    def is_news_time(self, current_time: pd.Timestamp) -> bool:
        """
        Simple check: is this a known high-impact news date?
        In absence of minute-level news data, we skip entire day.
        """
        key = (current_time.year, current_time.month, current_time.day)
        return key in self._news_dates

    def can_open_new_basket(
        self,
        pair: str,
        direction: str,
        current_time: pd.Timestamp,
        current_equity: float,
        open_baskets: list,
    ) -> tuple:
        """
        Returns (allowed: bool, reason: str).
        """
        # Weekly disabled
        if self.is_weekly_disabled(current_time):
            return False, "weekly_dd_limit"

        # Daily DD hit
        if self.is_daily_dd_hit(current_time):
            return False, "daily_dd_limit"

        # Session check
        if not is_tradeable_session(current_time, self.config):
            return False, "bad_session"

        # News check
        if self.is_news_time(current_time):
            return False, "news_blackout"

        # Weekend
        if current_time.weekday() >= 5:
            return False, "weekend"

        # Max simultaneous baskets
        active_baskets = [b for b in open_baskets if b.status.value == "ACTIVE"]
        if len(active_baskets) >= self.config.max_simultaneous_baskets:
            return False, "max_simultaneous"

        # Max per direction per pair
        same_dir = [
            b for b in active_baskets
            if b.pair == pair and b.direction.value == direction
        ]
        if len(same_dir) >= self.config.max_baskets_per_direction:
            return False, "max_per_direction"

        # Account exposure
        total_lots = sum(b.total_lots() for b in active_baskets)
        lot_usd = total_lots * 100_000  # 1 lot = 100k
        exposure = lot_usd / (current_equity * self.config.leverage)
        if exposure >= self.config.max_account_exposure_pct:
            return False, "max_exposure"

        return True, "ok"

    def calculate_base_lot(self, equity: float, pair_config) -> float:
        """
        Position size for first grid level.
        Risk = 0.20% of equity.
        For a Martingale grid, the "risk" is defined as the
        exposure of the first position only.
        Minimum 0.01 lot.
        """
        risk_amount = equity * self.config.base_risk_pct
        # Approximate: risk_amount / (grid_pips * pip_value)
        # Use default grid pips for sizing
        grid_pips = pair_config.default_grid_pips
        pip_val   = pair_config.point_value  # USD/pip/lot

        if pip_val <= 0 or grid_pips <= 0:
            return 0.01

        lot = risk_amount / (grid_pips * pip_val)
        lot = max(0.01, round(lot, 2))  # minimum 0.01

        return lot
