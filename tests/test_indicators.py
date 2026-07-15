"""
Unit tests for indicators - verify no look-ahead bias.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.indicators import (
    rsi, atr, adx, bollinger_bands, ema, detect_candle_patterns
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_price_series(n=100, seed=42) -> pd.Series:
    np.random.seed(seed)
    prices = 1.30 + np.cumsum(np.random.randn(n) * 0.0001)
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    return pd.Series(prices, index=idx)


def make_ohlcv(n=100, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    close  = 1.30 + np.cumsum(np.random.randn(n) * 0.0005)
    spread = np.abs(np.random.randn(n) * 0.0003) + 0.0001
    high   = close + spread
    low    = close - spread
    open_  = close + np.random.randn(n) * 0.0002
    vol    = np.abs(np.random.randn(n) * 1000) + 100
    idx    = pd.date_range("2020-01-01", periods=n, freq="min")
    return pd.DataFrame(
        {
            "open":   open_,
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": vol,
        },
        index=idx,
    )


def make_trending_up_series(n=100) -> pd.Series:
    """
    Strictly rising prices with small noise so RSI stays high.
    Uses exponential growth to keep gains >> losses.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    # Each bar gains more than it loses → avg_gain >> avg_loss → RSI > 80
    base   = np.linspace(1.0, 2.0, n)          # strong uptrend
    noise  = np.random.default_rng(0).normal(0, 0.002, n)
    prices = base + noise
    # Ensure strictly increasing on average
    prices = np.maximum.accumulate(prices - np.abs(noise) * 0.5)
    return pd.Series(prices, index=idx)


def make_trending_down_series(n=100) -> pd.Series:
    """Strictly falling prices → RSI < 20."""
    idx = pd.date_range("2020-01-01", periods=n, freq="min")
    base   = np.linspace(2.0, 1.0, n)
    noise  = np.random.default_rng(1).normal(0, 0.002, n)
    prices = base + noise
    prices = np.minimum.accumulate(prices + np.abs(noise) * 0.5)
    return pd.Series(prices, index=idx)


# ─────────────────────────────────────────────────────────────
# No Look-Ahead Tests
# ─────────────────────────────────────────────────────────────

class TestNoLookAhead:
    """
    CRITICAL: Adding future bars must NOT change past indicator values.
    This directly tests for look-ahead bias.
    """

    def test_rsi_no_lookahead(self):
        series = make_price_series(300)

        rsi_200 = rsi(series.iloc[:200], 14)
        val_at_100 = rsi_200.iloc[100]

        rsi_300 = rsi(series.iloc[:300], 14)
        val_at_100_full = rsi_300.iloc[100]

        assert abs(val_at_100 - val_at_100_full) < 1e-10, (
            f"RSI look-ahead detected! "
            f"200-bar: {val_at_100:.8f}, "
            f"300-bar: {val_at_100_full:.8f}"
        )

    def test_ema_no_lookahead(self):
        series = make_price_series(400)

        ema_200 = ema(series.iloc[:200], 50)
        val_at_150 = ema_200.iloc[150]

        ema_400 = ema(series.iloc[:400], 50)
        val_at_150_full = ema_400.iloc[150]

        assert abs(val_at_150 - val_at_150_full) < 1e-10, (
            f"EMA look-ahead detected! "
            f"200-bar: {val_at_150:.8f}, "
            f"400-bar: {val_at_150_full:.8f}"
        )

    def test_atr_no_lookahead(self):
        df = make_ohlcv(300)

        atr_150 = atr(df.iloc[:150], 14)
        val_at_100 = atr_150.iloc[100]

        atr_300 = atr(df.iloc[:300], 14)
        val_at_100_full = atr_300.iloc[100]

        assert abs(val_at_100 - val_at_100_full) < 1e-10, (
            f"ATR look-ahead detected!"
        )

    def test_adx_no_lookahead(self):
        df = make_ohlcv(300)

        adx_150 = adx(df.iloc[:150], 14)
        val_at_100 = adx_150.iloc[100]

        adx_300 = adx(df.iloc[:300], 14)
        val_at_100_full = adx_300.iloc[100]

        assert abs(val_at_100 - val_at_100_full) < 1e-10, (
            f"ADX look-ahead detected!"
        )

    def test_bollinger_no_lookahead(self):
        series = make_price_series(300)

        u1, m1, l1 = bollinger_bands(series.iloc[:200], 20, 2.0)
        val_upper_at_100 = u1.iloc[100]

        u2, m2, l2 = bollinger_bands(series.iloc[:300], 20, 2.0)
        val_upper_at_100_full = u2.iloc[100]

        assert abs(val_upper_at_100 - val_upper_at_100_full) < 1e-10, (
            f"Bollinger look-ahead detected!"
        )

    def test_candle_patterns_no_lookahead(self):
        """Pattern at bar N must not change when bar N+1 is added."""
        df = make_ohlcv(50)

        p1 = detect_candle_patterns(df.iloc[:30])
        val_at_20 = p1["bullish_engulfing"].iloc[20]

        p2 = detect_candle_patterns(df.iloc[:50])
        val_at_20_full = p2["bullish_engulfing"].iloc[20]

        assert val_at_20 == val_at_20_full, (
            "Candle pattern look-ahead detected!"
        )


# ─────────────────────────────────────────────────────────────
# RSI Tests
# ─────────────────────────────────────────────────────────────

class TestRSIBoundaries:

    def test_rsi_range(self):
        """RSI must always be in [0, 100]."""
        series = make_price_series(500)
        r = rsi(series, 14)
        valid = r.dropna()
        assert len(valid) > 0, "RSI produced no values"
        assert (valid >= 0).all(), f"RSI below 0: min={valid.min():.4f}"
        assert (valid <= 100).all(), f"RSI above 100: max={valid.max():.4f}"

    def test_rsi_trending_up(self):
        """
        Strongly rising prices must produce RSI > 70.

        FIX: Previous test used range(100) which produces CONSTANT diffs
        causing 0/0 = NaN → fallback to 50.0.
        We use a properly trending float series.
        """
        series = make_trending_up_series(150)
        r = rsi(series, 14)

        # Skip warmup period
        r_valid = r.dropna()
        assert len(r_valid) > 0, "RSI produced no values"

        # Last quarter should be strongly overbought
        last_quarter = r_valid.iloc[len(r_valid) // 2:]
        max_rsi = last_quarter.max()

        assert max_rsi > 70, (
            f"Expected RSI > 70 in strong uptrend, got max={max_rsi:.2f}\n"
            f"Series: first={series.iloc[0]:.4f}, last={series.iloc[-1]:.4f}"
        )

    def test_rsi_trending_down(self):
        """Strongly falling prices must produce RSI < 30."""
        series = make_trending_down_series(150)
        r = rsi(series, 14)

        r_valid = r.dropna()
        assert len(r_valid) > 0

        last_quarter = r_valid.iloc[len(r_valid) // 2:]
        min_rsi = last_quarter.min()

        assert min_rsi < 30, (
            f"Expected RSI < 30 in strong downtrend, got min={min_rsi:.2f}\n"
            f"Series: first={series.iloc[0]:.4f}, last={series.iloc[-1]:.4f}"
        )

    def test_rsi_neutral_random(self):
        """Random walk RSI should hover around 50 on average."""
        np.random.seed(99)
        idx = pd.date_range("2020-01-01", periods=500, freq="min")
        series = pd.Series(
            1.0 + np.cumsum(np.random.randn(500) * 0.001),
            index=idx,
        )
        r = rsi(series, 14).dropna()
        mean_rsi = r.mean()
        assert 35 < mean_rsi < 65, (
            f"Expected RSI mean near 50 for random walk, got {mean_rsi:.2f}"
        )

    def test_rsi_nan_count(self):
        """First (period) bars should be NaN, rest should be valid."""
        series = make_price_series(100)
        r = rsi(series, 14)
        # After warmup, should have values
        assert r.iloc[50:].notna().all(), "RSI has NaN after warmup period"


# ─────────────────────────────────────────────────────────────
# ATR Tests
# ─────────────────────────────────────────────────────────────

class TestATR:

    def test_atr_positive(self):
        """ATR must always be positive."""
        df = make_ohlcv(100)
        a = atr(df, 14)
        valid = a.dropna()
        assert len(valid) > 0
        assert (valid > 0).all(), f"ATR has non-positive values: min={valid.min()}"

    def test_atr_flat_market(self):
        """In flat market, ATR ≈ range of candles."""
        idx = pd.date_range("2020-01-01", periods=100, freq="min")
        df = pd.DataFrame(
            {
                "open":   [1.3000] * 100,
                "high":   [1.3002] * 100,
                "low":    [1.2998] * 100,
                "close":  [1.3000] * 100,
                "volume": [100.0]  * 100,
            },
            index=idx,
        )
        a = atr(df, 14)
        valid = a.dropna()
        # H-L = 0.0004, ATR should be close
        assert (valid < 0.001).all(), (
            f"ATR too large for flat market: max={valid.max():.6f}"
        )

    def test_atr_volatile_vs_flat(self):
        """Volatile market ATR must be larger than flat market ATR."""
        idx = pd.date_range("2020-01-01", periods=100, freq="min")

        df_flat = pd.DataFrame(
            {
                "open":   [1.3000] * 100,
                "high":   [1.3001] * 100,
                "low":    [1.2999] * 100,
                "close":  [1.3000] * 100,
                "volume": [100.0]  * 100,
            },
            index=idx,
        )

        np.random.seed(7)
        close_v = 1.3 + np.cumsum(np.random.randn(100) * 0.005)
        df_vol = pd.DataFrame(
            {
                "open":   close_v + np.random.randn(100) * 0.002,
                "high":   close_v + np.abs(np.random.randn(100) * 0.005),
                "low":    close_v - np.abs(np.random.randn(100) * 0.005),
                "close":  close_v,
                "volume": [100.0] * 100,
            },
            index=idx,
        )

        atr_flat = atr(df_flat, 14).dropna().mean()
        atr_vol  = atr(df_vol,  14).dropna().mean()

        assert atr_vol > atr_flat * 3, (
            f"Volatile ATR ({atr_vol:.6f}) not much larger than flat ({atr_flat:.6f})"
        )


# ─────────────────────────────────────────────────────────────
# Bollinger Bands Tests
# ─────────────────────────────────────────────────────────────

class TestBollingerBands:

    def test_upper_above_lower(self):
        """Upper band must always be above lower band."""
        series = make_price_series(200)
        upper, middle, lower = bollinger_bands(series, 20, 2.0)
        valid = upper.dropna()
        assert (upper.dropna() > lower.dropna()).all()

    def test_middle_is_sma(self):
        """Middle band must equal simple moving average."""
        series = make_price_series(100)
        _, middle, _ = bollinger_bands(series, 20, 2.0)
        sma = series.rolling(20).mean()
        diff = (middle - sma).dropna().abs()
        assert (diff < 1e-10).all(), "Middle band is not SMA"

    def test_price_outside_bands_possible(self):
        """Extreme moves should push price outside bands."""
        idx = pd.date_range("2020-01-01", periods=100, freq="min")
        # Stable then spike
        prices = [1.3] * 80 + [1.35, 1.36, 1.37, 1.38, 1.39,
                               1.40, 1.41, 1.42, 1.43, 1.44,
                               1.45, 1.46, 1.47, 1.48, 1.49,
                               1.50, 1.51, 1.52, 1.53, 1.54]
        series = pd.Series(prices, index=idx)
        upper, middle, lower = bollinger_bands(series, 20, 2.0)
        # At some point the spike should exceed upper band
        above_upper = (series > upper).any()
        assert above_upper, "Price never exceeded upper band during spike"


# ─────────────────────────────────────────────────────────────
# ADX Tests
# ─────────────────────────────────────────────────────────────

class TestADX:

    def test_adx_range(self):
        """ADX must be in [0, 100]."""
        df = make_ohlcv(200)
        a = adx(df, 14)
        valid = a.dropna()
        assert (valid >= 0).all(), f"ADX < 0: {valid.min()}"
        assert (valid <= 100).all(), f"ADX > 100: {valid.max()}"

    def test_adx_trending_higher(self):
        """Strong trend should produce higher ADX than flat market."""
        idx = pd.date_range("2020-01-01", periods=200, freq="min")

        # Strong trend
        close_t = np.linspace(1.0, 1.5, 200)
        df_trend = pd.DataFrame(
            {
                "open":   close_t - 0.001,
                "high":   close_t + 0.002,
                "low":    close_t - 0.002,
                "close":  close_t,
                "volume": [100.0] * 200,
            },
            index=idx,
        )

        # Flat market
        df_flat = pd.DataFrame(
            {
                "open":   [1.3000] * 200,
                "high":   [1.3002] * 200,
                "low":    [1.2998] * 200,
                "close":  [1.3000] * 200,
                "volume": [100.0]  * 200,
            },
            index=idx,
        )

        adx_trend = adx(df_trend, 14).dropna().iloc[-50:].mean()
        adx_flat  = adx(df_flat,  14).dropna().iloc[-50:].mean()

        assert adx_trend > adx_flat, (
            f"Trend ADX ({adx_trend:.2f}) should be > flat ADX ({adx_flat:.2f})"
        )


# ─────────────────────────────────────────────────────────────
# Candle Pattern Tests
# ─────────────────────────────────────────────────────────────

class TestCandlePatterns:

    def test_bullish_engulfing_detected(self):
        """
        Textbook bullish engulfing:
        Bar N-1: bearish (open > close)
        Bar N:   bullish (close > open)
                 Body N completely covers body N-1
                 open_N <= close_N-1  AND  close_N >= open_N-1

        FIX: Previous test data did not satisfy engulfing conditions.
        """
        idx = pd.date_range("2020-01-01", periods=5, freq="min")
        df = pd.DataFrame(
            {
                #         bar0    bar1    bar2    bar3      bar4
                # bar3: bearish: open=1.3050, close=1.3000 (body=50pip down)
                # bar4: bullish: open=1.2990 <= close_3=1.3000 ✓
                #               close=1.3060 >= open_3=1.3050  ✓
                #               body_4=70pip > body_3=50pip    ✓
                "open":  [1.302,  1.303,  1.304,  1.3050, 1.2990],
                "high":  [1.305,  1.306,  1.307,  1.3060, 1.3070],
                "low":   [1.299,  1.300,  1.301,  1.2990, 1.2980],
                "close": [1.303,  1.301,  1.302,  1.3000, 1.3060],
                "volume":[100,    100,    100,    100,    100   ],
            },
            index=idx,
        )

        patterns = detect_candle_patterns(df)

        # Verify bar3 is bearish (prerequisite)
        assert df["close"].iloc[3] < df["open"].iloc[3], (
            "Bar 3 should be bearish"
        )
        # Verify bar4 is bullish
        assert df["close"].iloc[4] > df["open"].iloc[4], (
            "Bar 4 should be bullish"
        )
        # Verify engulfing conditions
        assert df["open"].iloc[4]  <= df["close"].iloc[3], (
            f"open4={df['open'].iloc[4]} should <= close3={df['close'].iloc[3]}"
        )
        assert df["close"].iloc[4] >= df["open"].iloc[3], (
            f"close4={df['close'].iloc[4]} should >= open3={df['open'].iloc[3]}"
        )

        assert patterns["bullish_engulfing"].iloc[4] == True, (
            f"Bullish engulfing not detected at bar 4.\n"
            f"Patterns at bar 4: {patterns.iloc[4].to_dict()}"
        )

    def test_bearish_engulfing_detected(self):
        """
        Textbook bearish engulfing:
        Bar N-1: bullish
        Bar N:   bearish, body covers bar N-1 entirely
        """
        idx = pd.date_range("2020-01-01", periods=5, freq="min")
        df = pd.DataFrame(
            {
                #        bar0    bar1    bar2    bar3      bar4
                # bar3: bullish: open=1.3000, close=1.3050
                # bar4: bearish: open=1.3060 >= close_3=1.3050 ✓
                #               close=1.2990 <= open_3=1.3000  ✓
                "open":  [1.302,  1.301,  1.300,  1.3000, 1.3060],
                "high":  [1.305,  1.304,  1.303,  1.3060, 1.3070],
                "low":   [1.299,  1.298,  1.297,  1.2990, 1.2980],
                "close": [1.301,  1.303,  1.302,  1.3050, 1.2990],
                "volume":[100,    100,    100,    100,    100   ],
            },
            index=idx,
        )

        patterns = detect_candle_patterns(df)
        assert patterns["bearish_engulfing"].iloc[4] == True, (
            f"Bearish engulfing not detected.\n"
            f"Bar3: O={df['open'].iloc[3]} C={df['close'].iloc[3]}\n"
            f"Bar4: O={df['open'].iloc[4]} C={df['close'].iloc[4]}\n"
            f"Patterns: {patterns.iloc[4].to_dict()}"
        )

    def test_hammer_detected(self):
        """
        Hammer: small body near top, long lower wick (>= 2x body), tiny upper wick.
        """
        idx = pd.date_range("2020-01-01", periods=5, freq="min")
        df = pd.DataFrame(
            {
                "open":  [1.302, 1.300, 1.298, 1.299, 1.2998],
                "high":  [1.303, 1.301, 1.299, 1.300, 1.3002],  # tiny upper wick
                "low":   [1.299, 1.297, 1.295, 1.296, 1.2960],  # long lower wick
                "close": [1.301, 1.299, 1.297, 1.298, 1.3000],  # small body
                "volume":[100,   100,   100,   100,   100  ],
            },
            index=idx,
        )

        patterns = detect_candle_patterns(df)

        # Verify hammer geometry at bar 4:
        # body = |close - open| = |1.3000 - 1.2998| = 0.0002
        # lower_wick = open - low = 1.2998 - 1.2960 = 0.0038  (>= 2x body ✓)
        # upper_wick = high - close = 1.3002 - 1.3000 = 0.0002 (<= 30% range)
        body_4   = abs(df["close"].iloc[4] - df["open"].iloc[4])
        lw_4     = df["open"].iloc[4]  - df["low"].iloc[4]
        uw_4     = df["high"].iloc[4]  - df["close"].iloc[4]
        range_4  = df["high"].iloc[4]  - df["low"].iloc[4]

        assert lw_4 >= 2 * body_4, (
            f"Hammer test data invalid: lower_wick={lw_4:.5f} < 2*body={2*body_4:.5f}"
        )
        assert uw_4 <= 0.3 * range_4, (
            f"Hammer test data invalid: upper_wick={uw_4:.5f} > 0.3*range={0.3*range_4:.5f}"
        )

        assert patterns["hammer"].iloc[4] == True, (
            f"Hammer not detected. body={body_4:.5f} lw={lw_4:.5f} uw={uw_4:.5f}"
        )

    def test_no_false_positives_flat(self):
        """Flat candles should not trigger any patterns."""
        idx = pd.date_range("2020-01-01", periods=30, freq="min")
        df = pd.DataFrame(
            {
                "open":   [1.3000] * 30,
                "high":   [1.3001] * 30,
                "low":    [1.2999] * 30,
                "close":  [1.3000] * 30,
                "volume": [100.0]  * 30,
            },
            index=idx,
        )
        patterns = detect_candle_patterns(df)
        # No pattern should fire on perfectly flat doji candles
        for col in ["bullish_engulfing", "bearish_engulfing"]:
            assert not patterns[col].any(), (
                f"False positive {col} on flat candles"
            )


# ─────────────────────────────────────────────────────────────
# Basket Tests
# ─────────────────────────────────────────────────────────────

class TestBasket:

    def _make_basket(self, direction="BUY", max_levels=7, base_lot=0.01):
        from src.basket import Basket, Direction
        d = Direction.BUY if direction == "BUY" else Direction.SELL
        return Basket(
            basket_id=1,
            pair="EURGBP",
            direction=d,
            open_time=pd.Timestamp("2020-01-01"),
            pip_size=0.0001,
            spread_pips=1.5,
            pip_value_per_lot=12.5,
            grid_distance_pips=30,
            lot_sequence=[1.00, 1.35, 1.80, 2.40, 3.20, 4.30],
            base_lot=base_lot,
            max_levels=max_levels,
        )

    def test_entry_at_ask_price(self):
        """
        BUY order enters at ASK = bid + spread.
        So at bid=1.3000, spread=1.5 pips → entry=1.30015.
        Unrealized at bid=1.3000 should be NEGATIVE (spread cost).

        FIX: Previous test wrongly expected 0 P&L at entry bid price.
        """
        basket = self._make_basket()
        bid_price = 1.3000
        spread_pips = 1.5
        pip_size    = 0.0001

        basket.try_add_level(bid_price, pd.Timestamp("2020-01-01"), spread_pips)

        assert len(basket.positions) == 1

        expected_entry = bid_price + spread_pips * pip_size  # 1.30015
        actual_entry   = basket.positions[0].entry_price

        assert abs(actual_entry - expected_entry) < 1e-8, (
            f"Entry price wrong: expected {expected_entry:.5f}, "
            f"got {actual_entry:.5f}"
        )

        # At entry bid, we're down by spread cost
        pnl_at_bid = basket.unrealized_usd(bid_price)
        assert pnl_at_bid < 0, (
            f"BUY at ask should show loss at bid immediately. "
            f"Got P&L={pnl_at_bid:.4f}"
        )

        # Breakeven is at ask price (entry price)
        pnl_at_ask = basket.unrealized_usd(expected_entry)
        assert abs(pnl_at_ask) < 0.01, (
            f"P&L at ask should be ~0. Got {pnl_at_ask:.6f}"
        )

    def test_buy_profit_on_price_rise(self):
        """BUY basket should profit when price rises."""
        basket = self._make_basket()
        basket.try_add_level(1.3000, pd.Timestamp("2020-01-01"), 1.5)

        # Price rises 50 pips from entry ask
        entry_ask = 1.3000 + 1.5 * 0.0001   # 1.30015
        target    = entry_ask + 50 * 0.0001  # 50 pips profit

        pnl = basket.unrealized_usd(target)
        assert pnl > 0, f"BUY should profit on rise. Got {pnl:.4f}"

        # Verify magnitude: 50 pips * $12.5/pip/lot * 0.01 lot = $0.0625
        expected_pnl = 50 * 12.5 * 0.01
        assert abs(pnl - expected_pnl) < 0.001, (
            f"Wrong P&L magnitude: expected {expected_pnl:.4f}, got {pnl:.4f}"
        )

    def test_sell_profit_on_price_fall(self):
        """SELL basket should profit when price falls."""
        basket = self._make_basket("SELL")
        basket.try_add_level(1.3000, pd.Timestamp("2020-01-01"), 1.5)

        # SELL entry = bid = 1.3000
        # Price falls 50 pips
        target = 1.3000 - 50 * 0.0001

        pnl = basket.unrealized_usd(target)
        assert pnl > 0, f"SELL should profit on fall. Got {pnl:.4f}"

    def test_grid_levels_added_correctly(self):
        """Grid level 1 triggers only after price drops grid_distance pips."""
        basket = self._make_basket()
        t0 = pd.Timestamp("2020-01-01")

        # Level 0 at 1.3000
        basket.try_add_level(1.3000, t0, 1.5)
        assert len(basket.positions) == 1, "Should have 1 position"

        # Level 1 triggers at entry_0 - 30 pips = 1.3000 - 0.0030 = 1.2970
        # but entry_0 is 1.30015 (ask), so trigger = 1.30015 - 0.0030 = 1.29715
        trigger = basket.positions[0].entry_price - 30 * 0.0001

        # Not yet triggered (25 pips drop from ask)
        basket.try_add_level(trigger + 0.0005, t0 + pd.Timedelta(minutes=1), 1.5)
        assert len(basket.positions) == 1, "Should still have 1 position"

        # Triggered (price at trigger)
        basket.try_add_level(trigger - 0.0001, t0 + pd.Timedelta(minutes=2), 1.5)
        assert len(basket.positions) == 2, "Should have 2 positions now"

    def test_lot_sequence_applied(self):
        """Each grid level should use the correct lot multiplier."""
        basket = self._make_basket(base_lot=0.10)
        t0 = pd.Timestamp("2020-01-01")

        expected_lots = [
            round(0.10 * m, 2)
            for m in [1.00, 1.35, 1.80, 2.40, 3.20, 4.30]
        ]

        price = 1.3000
        basket.try_add_level(price, t0, 1.5)

        for i in range(1, 6):
            price -= 31 * 0.0001  # 31 pips drop each time
            t = t0 + pd.Timedelta(minutes=i)
            basket.try_add_level(price, t, 1.5)

        assert len(basket.positions) == 6

        for i, pos in enumerate(basket.positions):
            assert abs(pos.lot_size - expected_lots[i]) < 0.001, (
                f"Level {i}: expected lot {expected_lots[i]}, got {pos.lot_size}"
            )

    def test_max_levels_enforced(self):
        """Basket must never exceed max_levels."""
        basket = self._make_basket(max_levels=3)
        t0 = pd.Timestamp("2020-01-01")

        price = 1.3000
        basket.try_add_level(price, t0, 1.5)

        for i in range(10):  # try to add 10 more
            price -= 31 * 0.0001
            basket.try_add_level(price, t0 + pd.Timedelta(minutes=i+1), 1.5)

        assert len(basket.positions) <= 3, (
            f"max_levels=3 violated: got {len(basket.positions)} positions"
        )

    def test_basket_close_records_pnl(self):
        """Closing basket must record P&L on all positions."""
        basket = self._make_basket(base_lot=0.10)
        t0 = pd.Timestamp("2020-01-01")

        basket.try_add_level(1.3000, t0, 1.5)

        exit_price = 1.3100  # 100 pips up from bid entry
        basket.close_all(exit_price, t0 + pd.Timedelta(hours=1), "TAKE_PROFIT", 1.5)

        assert basket.status.value == "CLOSED"
        assert basket.positions[0].is_open == False
        assert basket.positions[0].pnl_usd != 0

    def test_basket_locked_on_large_loss(self):
        """Basket should lock (no new levels) when loss > 2x expected TP."""
        basket = self._make_basket(base_lot=0.10)
        t0 = pd.Timestamp("2020-01-01")

        basket.try_add_level(1.3000, t0, 1.5)

        # expected_tp = $50 (example)
        expected_tp = 50.0

        # Large adverse move
        basket.check_protection(0.5000, expected_tp)  # huge loss

        assert basket.locked == True, "Basket should be locked after large loss"

        # Should not add new level when locked
        price_for_level2 = basket.positions[0].entry_price - 31 * 0.0001
        result = basket.try_add_level(price_for_level2, t0 + pd.Timedelta(minutes=1), 1.5)
        assert result is None, "Locked basket should not add levels"
        assert len(basket.positions) == 1


# ─────────────────────────────────────────────────────────────
# Risk Manager Tests
# ─────────────────────────────────────────────────────────────

class TestRiskManager:

    def _make_rm(self, capital=10000.0):
        from src.risk_manager import RiskManager
        from config import CONFIG
        return RiskManager(CONFIG, capital)

    def test_daily_dd_limit(self):
        """3%+ daily drawdown should halt trading."""
        rm = self._make_rm()

        t1 = pd.Timestamp("2020-01-02 09:00")
        rm.update(t1, 10000.0)  # day start

        t2 = pd.Timestamp("2020-01-02 14:00")
        rm.update(t2, 9650.0)   # 3.5% loss

        assert rm.is_daily_dd_hit(t2) == True, (
            "Daily DD limit should be hit at 3.5% loss"
        )

    def test_daily_dd_not_hit_small_loss(self):
        """Small loss (< 3%) should NOT trigger daily DD."""
        rm = self._make_rm()

        t1 = pd.Timestamp("2020-01-03 09:00")
        rm.update(t1, 10000.0)

        t2 = pd.Timestamp("2020-01-03 14:00")
        rm.update(t2, 9800.0)  # 2% loss - below limit

        assert rm.is_daily_dd_hit(t2) == False, (
            "2% loss should NOT hit daily DD limit"
        )

    def test_weekly_dd_disable(self):
        """8%+ weekly drawdown should disable trading."""
        rm = self._make_rm()

        t1 = pd.Timestamp("2020-01-06 09:00")  # Monday
        rm.update(t1, 10000.0)

        t2 = pd.Timestamp("2020-01-08 14:00")  # Wednesday
        rm.update(t2, 9150.0)  # 8.5% loss

        assert rm.is_weekly_disabled(t2) == True, (
            "Weekly DD limit should be disabled at 8.5% loss"
        )

    def test_weekly_dd_not_hit_moderate_loss(self):
        """7% weekly loss should NOT disable trading."""
        rm = self._make_rm()

        t1 = pd.Timestamp("2020-01-13 09:00")
        rm.update(t1, 10000.0)

        t2 = pd.Timestamp("2020-01-14 12:00")
        rm.update(t2, 9300.0)  # 7% loss

        assert rm.is_weekly_disabled(t2) == False

    def test_daily_resets_each_day(self):
        """Each new day starts fresh for daily DD tracking."""
        rm = self._make_rm()

        # Day 1: lose 3.5% → DD hit
        t1 = pd.Timestamp("2020-01-06 09:00")
        rm.update(t1, 10000.0)
        t2 = pd.Timestamp("2020-01-06 15:00")
        rm.update(t2, 9650.0)
        assert rm.is_daily_dd_hit(t2) == True

        # Day 2: start fresh, small loss → no DD hit
        t3 = pd.Timestamp("2020-01-07 09:00")
        rm.update(t3, 9650.0)  # new day start
        t4 = pd.Timestamp("2020-01-07 12:00")
        rm.update(t4, 9600.0)  # only 0.5% drop today

        assert rm.is_daily_dd_hit(t4) == False, (
            "New day should reset daily DD tracking"
        )

    def test_lot_size_calculation(self):
        """Base lot should be proportional to account size."""
        from src.risk_manager import RiskManager
        from config import CONFIG

        rm1 = RiskManager(CONFIG, 10_000.0)
        rm2 = RiskManager(CONFIG, 20_000.0)

        pc = CONFIG.pair_configs["EURGBP"]
        lot1 = rm1.calculate_base_lot(10_000.0, pc)
        lot2 = rm2.calculate_base_lot(20_000.0, pc)

        assert lot2 > lot1, "Larger account should get larger lot size"
        assert lot1 >= 0.01, "Minimum lot size must be 0.01"


# ─────────────────────────────────────────────────────────────
# Session Filter Tests
# ─────────────────────────────────────────────────────────────

class TestSessionFilter:

    def test_london_open_allowed(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        # Monday 09:00 UTC - London session
        t = pd.Timestamp("2020-01-06 09:00:00")
        assert is_tradeable_session(t, CONFIG) == True

    def test_london_ny_overlap_allowed(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        # Tuesday 14:00 UTC - London/NY overlap
        t = pd.Timestamp("2020-01-07 14:00:00")
        assert is_tradeable_session(t, CONFIG) == True

    def test_asian_session_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        # Wednesday 03:00 UTC - Asian session
        t = pd.Timestamp("2020-01-08 03:00:00")
        assert is_tradeable_session(t, CONFIG) == False

    def test_early_morning_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        # Thursday 06:00 UTC - pre-London
        t = pd.Timestamp("2020-01-09 06:00:00")
        assert is_tradeable_session(t, CONFIG) == False

    def test_friday_afternoon_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        # Friday 14:00 UTC - after Friday cutoff (12:00)
        t = pd.Timestamp("2020-01-10 14:00:00")
        assert is_tradeable_session(t, CONFIG) == False

    def test_friday_morning_allowed(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        # Friday 10:00 UTC - before Friday cutoff
        t = pd.Timestamp("2020-01-10 10:00:00")
        assert is_tradeable_session(t, CONFIG) == True

    def test_saturday_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        t = pd.Timestamp("2020-01-11 10:00:00")
        assert is_tradeable_session(t, CONFIG) == False

    def test_sunday_rejected(self):
        from src.risk_manager import is_tradeable_session
        from config import CONFIG
        t = pd.Timestamp("2020-01-12 10:00:00")
        assert is_tradeable_session(t, CONFIG) == False


# ─────────────────────────────────────────────────────────────
# Integration: Indicator Pipeline
# ─────────────────────────────────────────────────────────────

class TestIndicatorPipeline:
    """Test that compute_all_indicators runs without errors."""

    def test_pipeline_runs(self):
        from src.indicators import compute_all_indicators
        from config import CONFIG

        df = make_ohlcv(300)
        result = compute_all_indicators(df, CONFIG)

        required_cols = [
            "ema200", "atr14", "atr100", "rsi14",
            "bb_upper", "bb_middle", "bb_lower", "adx14",
            "bullish_engulfing", "bearish_engulfing",
            "hammer", "shooting_star",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_pipeline_no_lookahead(self):
        """compute_all_indicators on subset = same as full at same index."""
        from src.indicators import compute_all_indicators
        from config import CONFIG

        df = make_ohlcv(400)

        r1 = compute_all_indicators(df.iloc[:200], CONFIG)
        r2 = compute_all_indicators(df.iloc[:400], CONFIG)

        # Check EMA200 at bar 199 (last bar of first run)
        # Note: EMA200 needs 200 bars, so bar 199 is the first valid bar
        val1 = r1["ema200"].iloc[199]
        val2 = r2["ema200"].iloc[199]

        if not (pd.isna(val1) and pd.isna(val2)):
            assert abs(val1 - val2) < 1e-10, (
                f"Pipeline look-ahead: ema200 at 199 differs! "
                f"{val1:.8f} vs {val2:.8f}"
            )

        # RSI at bar 150 (well past warmup)
        val1_rsi = r1["rsi14"].iloc[150]
        val2_rsi = r2["rsi14"].iloc[150]
        assert abs(val1_rsi - val2_rsi) < 1e-10, (
            f"Pipeline look-ahead in RSI at bar 150"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
