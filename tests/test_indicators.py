"""
Unit tests for indicators - verify no look-ahead bias.
Run with: pytest tests/
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indicators import rsi, atr, adx, bollinger_bands, ema, detect_candle_patterns


def make_price_series(n=100, seed=42) -> pd.Series:
    np.random.seed(seed)
    prices = 1.30 + np.cumsum(np.random.randn(n) * 0.0001)
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    return pd.Series(prices, index=idx)


def make_ohlcv(n=100, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    close  = 1.30 + np.cumsum(np.random.randn(n) * 0.0005)
    spread = np.abs(np.random.randn(n) * 0.0003)
    high   = close + spread
    low    = close - spread
    open_  = close + np.random.randn(n) * 0.0002
    vol    = np.abs(np.random.randn(n) * 1000)
    idx    = pd.date_range("2020-01-01", periods=n, freq="min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx
    )


class TestNoLookAhead:
    """
    Critical tests: changing bar N+1 must NOT change indicator value at bar N.
    This is the look-ahead bias test.
    """

    def test_rsi_no_lookahead(self):
        series = make_price_series(200)

        # Compute RSI on first 150 bars
        rsi_150 = rsi(series.iloc[:150], 14)
        val_at_100 = rsi_150.iloc[100]

        # Compute RSI on first 200 bars (add 50 more bars)
        rsi_200 = rsi(series.iloc[:200], 14)
        val_at_100_full = rsi_200.iloc[100]

        # Values at bar 100 must be IDENTICAL
        assert abs(val_at_100 - val_at_100_full) < 1e-10, (
            f"RSI look-ahead detected! "
            f"150-bar: {val_at_100:.6f}, 200-bar: {val_at_100_full:.6f}"
        )

    def test_ema_no_lookahead(self):
        series = make_price_series(300)

        ema_200 = ema(series.iloc[:200], 50)
        val_at_150 = ema_200.iloc[150]

        ema_300 = ema(series.iloc[:300], 50)
        val_at_150_full = ema_300.iloc[150]

        assert abs(val_at_150 - val_at_150_full) < 1e-10, (
            f"EMA look-ahead detected!"
        )

    def test_atr_no_lookahead(self):
        df = make_ohlcv(200)

        atr_150 = atr(df.iloc[:150], 14)
        val_at_100 = atr_150.iloc[100]

        atr_200 = atr(df.iloc[:200], 14)
        val_at_100_full = atr_200.iloc[100]

        assert abs(val_at_100 - val_at_100_full) < 1e-10, (
            f"ATR look-ahead detected!"
        )

    def test_adx_no_lookahead(self):
        df = make_ohlcv(200)

        adx_150 = adx(df.iloc[:150], 14)
        val_at_100 = adx_150.iloc[100]

        adx_200 = adx(df.iloc[:200], 14)
        val_at_100_full = adx_200.iloc[100]

        assert abs(val_at_100 - val_at_100_full) < 1e-10, (
            f"ADX look-ahead detected!"
        )

    def test_bollinger_no_lookahead(self):
        series = make_price_series(200)

        u1, m1, l1 = bollinger_bands(series.iloc[:150], 20, 2.0)
        val_at_100_u = u1.iloc[100]

        u2, m2, l2 = bollinger_bands(series.iloc[:200], 20, 2.0)
        val_at_100_u_full = u2.iloc[100]

        assert abs(val_at_100_u - val_at_100_u_full) < 1e-10


class TestRSIBoundaries:
    def test_rsi_range(self):
        series = make_price_series(500)
        r = rsi(series, 14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_trending_up(self):
        """Strongly rising prices should give high RSI."""
        idx = pd.date_range("2020-01-01", periods=100, freq="min")
        series = pd.Series(range(100), index=idx, dtype=float)
        r = rsi(series, 14)
        assert r.iloc[-1] > 80

    def test_rsi_trending_down(self):
        """Strongly falling prices should give low RSI."""
        idx = pd.date_range("2020-01-01", periods=100, freq="min")
        series = pd.Series(range(100, 0, -1), index=idx, dtype=float)
        r = rsi(series, 14)
        assert r.iloc[-1] < 20


class TestATR:
    def test_atr_positive(self):
        df = make_ohlcv(100)
        a = atr(df, 14)
        assert (a.dropna() > 0).all()

    def test_atr_flat_market(self):
        """In flat market, ATR should be close to spread."""
        idx = pd.date_range("2020-01-01", periods=100, freq="min")
        df = pd.DataFrame({
            "open":  [1.3000] * 100,
            "high":  [1.3001] * 100,
            "low":   [1.2999] * 100,
            "close": [1.3000] * 100,
            "volume": [100] * 100,
        }, index=idx)
        a = atr(df, 14)
        assert (a.dropna() < 0.001).all()


class TestCandlePatterns:
    def test_bullish_engulfing_detected(self):
        """Create a textbook bullish engulfing pattern."""
        idx = pd.date_range("2020-01-01", periods=5, freq="min")
        df = pd.DataFrame({
            "open":  [1.305, 1.302, 1.302, 1.301, 1.299],
            "high":  [1.306, 1.303, 1.303, 1.302, 1.304],
            "low":   [1.300, 1.298, 1.298, 1.295, 1.295],
            "close": [1.302, 1.300, 1.300, 1.298, 1.304],
            "volume": [100] * 5,
        }, index=idx)

        patterns = detect_candle_patterns(df)
        # Bar 4: bull close 1.304 > bear open 1.301 (engulfs)
        assert patterns["bullish_engulfing"].iloc[4] == True

    def test_hammer_detected(self):
        """Create a hammer candle."""
        idx = pd.date_range("2020-01-01", periods=3, freq="min")
        df = pd.DataFrame({
            "open":  [1.302, 1.300, 1.2995],
            "high":  [1.303, 1.301, 1.3000],
            "low":   [1.299, 1.297, 1.2960],  # long lower wick
            "close": [1.300, 1.299, 1.2998],  # small body
            "volume": [100] * 3,
        }, index=idx)

        patterns = detect_candle_patterns(df)
        assert patterns["hammer"].iloc[2] == True


class TestBasket:
    """Test basket logic."""

    def test_basket_tp_exit(self):
        from src.basket import Basket, Direction

        basket = Basket(
            basket_id=1,
            pair="EURGBP",
            direction=Direction.BUY,
            open_time=pd.Timestamp("2020-01-01"),
            pip_size=0.0001,
            spread_pips=1.5,
            pip_value_per_lot=12.5,
            grid_distance_pips=30,
            lot_sequence=[1.0, 1.35, 1.80],
            base_lot=0.01,
            max_levels=7,
        )

        # Add first position
        basket.try_add_level(1.3000, pd.Timestamp("2020-01-01"), 1.5)
        assert len(basket.positions) == 1
        assert basket.next_level == 1

        # At entry: no profit
        assert abs(basket.unrealized_usd(1.3000)) < 0.01

        # Price moves up 50 pips
        profit = basket.unrealized_usd(1.3050)
        assert profit > 0

    def test_grid_levels_added_correctly(self):
        from src.basket import Basket, Direction

        basket = Basket(
            basket_id=1,
            pair="EURGBP",
            direction=Direction.BUY,
            open_time=pd.Timestamp("2020-01-01"),
            pip_size=0.0001,
            spread_pips=1.5,
            pip_value_per_lot=12.5,
            grid_distance_pips=30,
            lot_sequence=[1.0, 1.35, 1.80, 2.40],
            base_lot=0.10,
            max_levels=7,
        )

        # Level 0
        basket.try_add_level(1.3000, pd.Timestamp("2020-01-01"), 1.5)
        assert len(basket.positions) == 1

        # Level 1 should trigger when price drops 30 pips = 0.0030
        # At 1.2975 (not enough = 0.0025 drop)
        basket.try_add_level(1.2975, pd.Timestamp("2020-01-01 00:01"), 1.5)
        assert len(basket.positions) == 1  # not triggered yet

        # At 1.2969 (31 pips drop)
        basket.try_add_level(1.2969, pd.Timestamp("2020-01-01 00:02"), 1.5)
        assert len(basket.positions) == 2  # now triggered

    def test_max_levels_enforced(self):
        from src.basket import Basket, Direction

        basket = Basket(
            basket_id=1,
            pair="EURGBP",
            direction=Direction.BUY,
            open_time=pd.Timestamp("2020-01-01"),
            pip_size=0.0001,
            spread_pips=1.5,
            pip_value_per_lot=12.5,
            grid_distance_pips=30,
            lot_sequence=[1.0, 1.35, 1.80, 2.40, 3.20, 4.30],
            base_lot=0.01,
            max_levels=4,  # Hard limit of 4
        )

        price = 1.3000
        t = pd.Timestamp("2020-01-01")
        # Force all levels
        for i in range(10):
            price -= 0.0031  # 31 pips drop
            t += pd.Timedelta(minutes=1)
            basket.try_add_level(price, t, 1.5)

        # Should never exceed max_levels
        assert len(basket.positions) <= 4


class TestRiskManager:
    def test_daily_dd_limit(self):
        from src.risk_manager import RiskManager
        from config import CONFIG

        rm = RiskManager(CONFIG, 10000.0)

        t = pd.Timestamp("2020-01-02 10:00")
        rm.update(t, 10000.0)   # day start

        # Simulate 3.5% loss
        t2 = pd.Timestamp("2020-01-02 14:00")
        rm.update(t2, 9650.0)

        assert rm.is_daily_dd_hit(t2) == True

    def test_weekly_dd_disable(self):
        from src.risk_manager import RiskManager
        from config import CONFIG

        rm = RiskManager(CONFIG, 10000.0)

        t = pd.Timestamp("2020-01-06 10:00")  # Monday
        rm.update(t, 10000.0)

        # Simulate 8.5% loss
        t2 = pd.Timestamp("2020-01-08 14:00")  # Wednesday
        rm.update(t2, 9150.0)

        assert rm.is_weekly_disabled(t2) == True


class TestSessionFilter:
    def test_london_session_allowed(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG

        # Monday 10:00 UTC (London session)
        t = pd.Timestamp("2020-01-06 10:00:00")
        assert is_tradeable_session(t, CONFIG) == True

    def test_asian_session_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG

        # Tuesday 03:00 UTC (Asian session)
        t = pd.Timestamp("2020-01-07 03:00:00")
        assert is_tradeable_session(t, CONFIG) == False

    def test_friday_afternoon_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG

        # Friday 14:00 UTC (after Friday cutoff)
        t = pd.Timestamp("2020-01-10 14:00:00")
        assert is_tradeable_session(t, CONFIG) == False

    def test_weekend_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG

        t = pd.Timestamp("2020-01-11 10:00:00")  # Saturday
        assert is_tradeable_session(t, CONFIG) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
