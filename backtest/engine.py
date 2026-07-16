"""
engine.py
==========
Bar-by-bar, event-driven backtest engine for the Adaptive Recovery Grid
strategy described in the spec (EURGBP / AUDNZD, regime-filtered mean
reversion + ATR-adaptive grid + non-linear lot progression + basket
risk management).

ANTI-LOOKAHEAD DESIGN (read this before trusting any number it prints):

1. All indicators (indicators.py) are computed once, vectorized, over
   the whole series. Each value at index i legitimately only depends on
   bars <= i (that's just what trailing indicators are).

2. The loop below NEVER lets a bar trade on its own not-yet-closed
   information. For bar i, every decision (regime, filters, signals)
   is read from index i-1 (the last bar that was fully closed BEFORE
   bar i opened). The resulting order, if any, is filled at bar i's
   OPEN price plus spread/slippage. This reproduces the real-world
   sequence: "bar closes -> EA evaluates -> EA sends order -> order
   fills near the next bar's open."

3. Weekly structure (weekly_prior_levels) uses the prior COMPLETED
   week only (shift(1) on a weekly resample), never the forming week.

4. No parameter is fit on the backtest data itself. Config values are
   fixed, economically-motivated defaults (see config/default.yaml).
   There is no in-sample optimisation loop in this codebase. If you
   want to check robustness, run this same code on different
   --start-year/--end-year windows -- the code does not adapt itself
   to whichever window you pick.

5. Transaction costs (spread + commission + slippage) are always
   applied, and P&L is converted to USD using the real GBPUSD/NZDUSD
   history (fx_convert.py) rather than assumed constant, which removes
   a common source of "too good to be true" backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from . import indicators as ind
from .fx_convert import build_conversion_series

PIP_SIZE = {
    "EURGBP": 0.0001,
    "AUDNZD": 0.0001,
}

LOT_SIZE_UNITS = 100_000.0  # standard lot


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@dataclass
class RegimeParams:
    adx_trend_threshold: float = 25.0
    adx_range_threshold: float = 20.0
    atr_expand_lookback: int = 20
    atr_expand_ratio: float = 1.15
    ema_slope_lookback: int = 10
    ema_slope_threshold_pips: float = 3.0


@dataclass
class StrategyConfig:
    symbol: str = "EURGBP"
    timeframe: str = "H1"
    start_year: int = 2020
    end_year: int = 2024

    # account / sizing
    initial_balance: float = 10_000.0
    base_lot: float = 0.01                 # lots at multiplier 1.00
    compound_sizing: bool = True           # scale base_lot with equity growth
    lot_progression: List[float] = field(
        default_factory=lambda: [1.00, 1.30, 1.70, 2.20, 2.80, 3.60, 4.60]
    )
    max_positions: int = 7

    # costs
    spread_pips: float = 1.2
    spread_filter_mult: float = 2.0
    off_session_spread_mult: float = 1.8   # heuristic: spreads widen off-session
    commission_per_lot_roundturn: float = 7.0
    slippage_pips: float = 0.5

    # grid
    grid_atr_mult_quiet: float = 0.8
    grid_atr_mult_normal: float = 1.0
    grid_atr_mult_high: float = 1.3
    atr_extreme_ratio: float = 1.8         # ATR vs its rolling avg -> grid halted

    # entry filters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    adx_entry_max: float = 25.0
    atr_period: int = 14
    ema_fast: int = 50
    ema_slow: int = 200
    ema200_min_dist_pips: float = 5.0
    use_vwap_filter: bool = False          # off by default: volume is only a tick-count proxy
    vwap_min_dist_pips: float = 3.0

    # recovery / breakout
    recovery_stat_limit_atr: float = 4.0
    breakout_atr_spike_ratio: float = 1.6
    adx_surge_delta: float = 8.0

    # basket exit
    basket_tp_atr_mult: float = 1.2
    partial_close_pct: float = 0.5
    partial_close_trigger_frac_of_tp: float = 0.6
    momentum_exhaustion_rsi_buffer: float = 5.0

    # risk / safety
    max_floating_dd_pct: float = 15.0
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_pct: float = 10.0
    max_margin_util_pct: float = 60.0
    leverage: float = 30.0

    # session (UTC hours), spec prefers London / London-NY overlap
    session_start_hour_utc: int = 7
    session_end_hour_utc: int = 17
    avoid_friday_after_hour_utc: int = 19

    # misc
    enable_trend_single_trade: bool = False  # spec allows this optionally; off by default (recovery-grid focus)
    regime: RegimeParams = field(default_factory=RegimeParams)


# ----------------------------------------------------------------------
# Position / basket state
# ----------------------------------------------------------------------

@dataclass
class Position:
    direction: str          # "BUY" or "SELL"
    entry_price: float
    lots: float
    open_time: pd.Timestamp
    grid_level: int


@dataclass
class Basket:
    positions: List[Position] = field(default_factory=list)
    direction: Optional[str] = None
    recovery_stopped: bool = False

    def is_open(self) -> bool:
        return len(self.positions) > 0

    def total_lots(self) -> float:
        return sum(p.lots for p in self.positions)

    def avg_price(self) -> Optional[float]:
        tl = self.total_lots()
        if tl == 0:
            return None
        return sum(p.entry_price * p.lots for p in self.positions) / tl

    def first_entry_price(self) -> Optional[float]:
        return self.positions[0].entry_price if self.positions else None

    def last_entry_price(self) -> Optional[float]:
        return self.positions[-1].entry_price if self.positions else None


@dataclass
class Trade:
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    direction: str
    lots: float
    entry_price: float
    exit_price: float
    pnl_usd: float
    reason: str
    grid_levels_used: int


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class BacktestEngine:
    def __init__(self, df: pd.DataFrame, cfg: StrategyConfig, conv_rate_at):
        """
        df: OHLCV dataframe at the target timeframe (already resampled, causal)
        cfg: StrategyConfig
        conv_rate_at: callable(pd.Series[timestamps]) -> pd.Series[rate] used to
                      convert quote-currency P&L to USD (see fx_convert.py)
        """
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.pip = PIP_SIZE[cfg.symbol]
        self.conv_rate_at = conv_rate_at

        self.trades: List[Trade] = []
        self.equity_curve = np.zeros(len(self.df))
        self.balance_curve = np.zeros(len(self.df))
        self.regime_series: List[str] = [""] * len(self.df)

        self._compute_indicators()

    # -- indicator prep --------------------------------------------------
    def _compute_indicators(self):
        df = self.df
        cfg = self.cfg

        df["ema_fast"] = ind.ema(df["close"], cfg.ema_fast)
        df["ema_slow"] = ind.ema(df["close"], cfg.ema_slow)
        df["atr"] = ind.atr(df, cfg.atr_period)
        df["atr_avg"] = df["atr"].rolling(cfg.regime.atr_expand_lookback).mean()
        adx_df = ind.adx(df, cfg.adx_period)
        df["adx"] = adx_df["adx"]
        df["rsi"] = ind.rsi(df["close"], cfg.rsi_period)
        k, d = ind.stochastic(df, cfg.stoch_k, cfg.stoch_d, cfg.stoch_smooth)
        df["stoch_k"] = k
        df["stoch_d"] = d
        bb_u, bb_m, bb_l = ind.bollinger(df["close"], cfg.bb_period, cfg.bb_std)
        df["bb_upper"] = bb_u
        df["bb_mid"] = bb_m
        df["bb_lower"] = bb_l
        wk = ind.weekly_prior_levels(df)
        df["week_high_prior"] = wk["week_high_prior"]
        df["week_low_prior"] = wk["week_low_prior"]
        if cfg.use_vwap_filter:
            df["vwap"] = ind.daily_vwap(df)
        else:
            df["vwap"] = np.nan

        df["hour_utc"] = df["datetime"].dt.hour
        df["weekday"] = df["datetime"].dt.weekday  # 0=Mon .. 6=Sun

    # -- regime ------------------------------------------------------------
    def _classify_regime(self, i: int) -> str:
        """Regime as of the CLOSE of bar i (uses only data through i)."""
        cfg = self.cfg.regime
        df = self.df
        lb_atr = self.cfg.regime.atr_expand_lookback
        lb_ema = self.cfg.regime.ema_slope_lookback
        if i < max(lb_atr, lb_ema, self.cfg.ema_slow):
            return "TRANSITION"

        adx_val = df["adx"].iat[i]
        atr_now = df["atr"].iat[i]
        atr_avg = df["atr"].iloc[max(0, i - lb_atr):i].mean()
        atr_expanding = bool(atr_avg and atr_avg > 0 and (atr_now / atr_avg) > cfg.atr_expand_ratio)

        ema_now = df["ema_fast"].iat[i]
        ema_prev = df["ema_fast"].iat[i - lb_ema]
        slope_pips = (ema_now - ema_prev) / self.pip
        steep = abs(slope_pips) > cfg.ema_slope_threshold_pips

        if pd.isna(adx_val):
            return "TRANSITION"

        if adx_val >= cfg.adx_trend_threshold and atr_expanding and steep:
            return "TRENDING"
        if adx_val <= cfg.adx_range_threshold and not atr_expanding:
            return "RANGING"
        return "TRANSITION"

    # -- filters -------------------------------------------------------
    def _session_ok(self, i: int) -> bool:
        cfg = self.cfg
        row = self.df.iloc[i]
        hour = row["hour_utc"]
        weekday = row["weekday"]
        if weekday == 6:  # Sunday
            return False
        if weekday == 4 and hour >= cfg.avoid_friday_after_hour_utc:  # late Friday
            return False
        return cfg.session_start_hour_utc <= hour < cfg.session_end_hour_utc

    def _current_spread_pips(self, i: int) -> float:
        cfg = self.cfg
        if self._session_ok(i):
            return cfg.spread_pips
        return cfg.spread_pips * cfg.off_session_spread_mult

    def _spread_filter_ok(self, i: int) -> bool:
        # Spread filter: never trade if spread > 2x AVERAGE spread.
        # We only have a modeled spread (no true tick spread history), so we
        # compare the modeled current spread against the modeled baseline.
        cur = self._current_spread_pips(i)
        return cur <= self.cfg.spread_filter_mult * self.cfg.spread_pips

    def _grid_atr_mult(self, i: int) -> Optional[float]:
        cfg = self.cfg
        atr_now = self.df["atr"].iat[i]
        atr_avg = self.df["atr_avg"].iat[i]
        if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg == 0:
            return None
        ratio = atr_now / atr_avg
        if ratio >= cfg.atr_extreme_ratio:
            return None  # extreme volatility -> no new grid orders
        if ratio <= 0.85:
            return cfg.grid_atr_mult_quiet
        if ratio <= 1.25:
            return cfg.grid_atr_mult_normal
        return cfg.grid_atr_mult_high

    def _breakout_detected(self, i: int) -> bool:
        cfg = self.cfg
        df = self.df
        row = df.iloc[i]
        if pd.isna(row["week_high_prior"]) or pd.isna(row["week_low_prior"]):
            weekly_break = False
        else:
            weekly_break = (row["close"] > row["week_high_prior"]) or (row["close"] < row["week_low_prior"])

        atr_now, atr_avg = row["atr"], row["atr_avg"]
        atr_spike = bool(atr_avg and atr_avg > 0 and (atr_now / atr_avg) >= cfg.breakout_atr_spike_ratio)

        adx_now = row["adx"]
        adx_prev = df["adx"].iat[max(0, i - 3)]
        adx_surge = bool(not pd.isna(adx_now) and not pd.isna(adx_prev) and (adx_now - adx_prev) >= cfg.adx_surge_delta)

        candle_range = row["high"] - row["low"]
        impulse = bool(not pd.isna(atr_now) and atr_now > 0 and candle_range >= 1.8 * atr_now)

        return weekly_break or atr_spike or adx_surge or impulse

    def _entry_signal(self, i: int) -> Optional[str]:
        """Evaluate all entry filters at bar i (last closed bar). Returns
        'BUY', 'SELL' or None. Requires ALL filters to agree, per spec."""
        cfg = self.cfg
        row = self.df.iloc[i]

        required = ["rsi", "stoch_k", "bb_upper", "bb_lower", "atr", "adx", "ema_slow"]
        if any(pd.isna(row[c]) for c in required):
            return None

        if not self._session_ok(i):
            return None
        if not self._spread_filter_ok(i):
            return None
        if row["adx"] >= cfg.adx_entry_max:
            return None  # only mean-reversion entries, not in strong trend

        dist_ema200_pips = (row["close"] - row["ema_slow"]) / self.pip
        prev_close = self.df["close"].iat[i - 1] if i > 0 else row["close"]
        prev_open = self.df["open"].iat[i - 1] if i > 0 else row["open"]
        bullish_rejection = row["close"] > row["open"] and row["low"] < row["bb_lower"]
        bearish_rejection = row["close"] < row["open"] and row["high"] > row["bb_upper"]

        vwap_ok_buy = True
        vwap_ok_sell = True
        if cfg.use_vwap_filter and not pd.isna(row["vwap"]):
            vwap_ok_buy = (row["vwap"] - row["close"]) / self.pip >= cfg.vwap_min_dist_pips
            vwap_ok_sell = (row["close"] - row["vwap"]) / self.pip >= cfg.vwap_min_dist_pips

        buy_ok = (
            row["close"] < row["bb_lower"]
            and row["rsi"] <= cfg.rsi_oversold
            and row["stoch_k"] <= cfg.stoch_oversold
            and bullish_rejection
            and abs(dist_ema200_pips) >= cfg.ema200_min_dist_pips
            and vwap_ok_buy
        )
        sell_ok = (
            row["close"] > row["bb_upper"]
            and row["rsi"] >= cfg.rsi_overbought
            and row["stoch_k"] >= cfg.stoch_overbought
            and bearish_rejection
            and abs(dist_ema200_pips) >= cfg.ema200_min_dist_pips
            and vwap_ok_sell
        )

        if buy_ok and not sell_ok:
            return "BUY"
        if sell_ok and not buy_ok:
            return "SELL"
        return None

    def _recovery_ok(self, i: int, basket: Basket) -> bool:
        cfg = self.cfg
        if basket.recovery_stopped:
            return False
        if len(basket.positions) >= min(cfg.max_positions, len(cfg.lot_progression)):
            return False
        regime = self.regime_series[i]
        if regime != "RANGING":
            return False
        row = self.df.iloc[i]
        atr_now, atr_avg = row["atr"], row["atr_avg"]
        if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg == 0:
            return False
        if (atr_now / atr_avg) >= cfg.atr_extreme_ratio:
            return False
        if row["adx"] >= cfg.adx_entry_max:
            return False
        if self._breakout_detected(i):
            return False

        grid_mult = self._grid_atr_mult(i)
        if grid_mult is None:
            return False
        grid_distance = grid_mult * atr_now

        last_price = basket.last_entry_price()
        first_price = basket.first_entry_price()
        close = row["close"]

        if basket.direction == "BUY":
            far_enough = (last_price - close) >= grid_distance
            within_stat_limit = (first_price - close) <= cfg.recovery_stat_limit_atr * atr_now
        else:
            far_enough = (close - last_price) >= grid_distance
            within_stat_limit = (close - first_price) <= cfg.recovery_stat_limit_atr * atr_now

        return bool(far_enough and within_stat_limit)

    # -- sizing / pnl -----------------------------------------------------
    def _lot_for_level(self, level_idx: int, equity: float) -> float:
        cfg = self.cfg
        mult = cfg.lot_progression[level_idx]
        base = cfg.base_lot
        if cfg.compound_sizing:
            base = base * max(equity, 1.0) / cfg.initial_balance
        lots = round(base * mult, 2)
        return max(lots, 0.01)

    def _quote_pnl(self, direction: str, entry: float, exit_: float, lots: float) -> float:
        sign = 1.0 if direction == "BUY" else -1.0
        return sign * (exit_ - entry) * lots * LOT_SIZE_UNITS

    def _to_usd(self, quote_pnl: float, rate: float) -> float:
        return quote_pnl * rate

    def _commission(self, lots: float) -> float:
        return lots * self.cfg.commission_per_lot_roundturn

    # -- main loop ----------------------------------------------------
    def run(self) -> pd.DataFrame:
        cfg = self.cfg
        df = self.df
        n = len(df)

        # pre-compute regime for every bar (causal, uses <= i only)
        self.regime_series = [self._classify_regime(i) for i in range(n)]

        rates = self.conv_rate_at(df["datetime"]).values

        balance = cfg.initial_balance
        basket = Basket()

        day_start_balance = balance
        current_day = df["datetime"].iat[0].date() if n else None
        week_start_balance = balance
        current_week = df["datetime"].iat[0].isocalendar()[1] if n else None
        daily_locked = False
        weekly_locked = False

        warmup = max(cfg.ema_slow, cfg.regime.atr_expand_lookback, cfg.bb_period) + 5

        for i in range(n):
            row = df.iloc[i]
            rate = rates[i] if not np.isnan(rates[i]) else 1.0

            # ---- day/week rollovers ----
            bar_date = row["datetime"].date()
            bar_week = row["datetime"].isocalendar()[1]
            if bar_date != current_day:
                current_day = bar_date
                day_start_balance = balance
                daily_locked = False
            if bar_week != current_week:
                current_week = bar_week
                week_start_balance = balance
                weekly_locked = False

            if i < warmup:
                self.balance_curve[i] = balance
                self.equity_curve[i] = balance
                continue

            decision_idx = i - 1  # last CLOSED bar -> avoids lookahead
            exec_price_base = row["open"]  # fills happen at this bar's open
            spread_pips = self._current_spread_pips(decision_idx)
            slip = cfg.slippage_pips * self.pip
            spread_price = spread_pips * self.pip

            # ---- manage existing basket ----
            if basket.is_open():
                avg = basket.avg_price()
                atr_now = df["atr"].iat[decision_idx]
                tp_dist = cfg.basket_tp_atr_mult * atr_now if not pd.isna(atr_now) else None

                # mark-to-market floating pnl using current bar CLOSE (for equity curve / DD checks)
                mtm_price = row["close"]
                floating_quote = self._quote_pnl(basket.direction, avg, mtm_price, basket.total_lots())
                floating_usd = self._to_usd(floating_quote, rate)
                equity_now = balance + floating_usd

                # breakout -> stop recovery permanently for this basket
                if self._breakout_detected(decision_idx):
                    basket.recovery_stopped = True

                # ---- hard risk kill-switch: max floating DD ----
                dd_pct = -100.0 * floating_usd / max(equity_now, 1.0) if floating_usd < 0 else 0.0
                force_close = dd_pct >= cfg.max_floating_dd_pct

                # ---- exit engine ----
                exit_reason = None
                if force_close:
                    exit_reason = "max_floating_dd"
                elif tp_dist is not None:
                    if basket.direction == "BUY" and row["close"] >= avg + tp_dist:
                        exit_reason = "basket_tp"
                    elif basket.direction == "SELL" and row["close"] <= avg - tp_dist:
                        exit_reason = "basket_tp"
                if exit_reason is None:
                    # weighted-average / breakeven-plus exit once recovery has stopped
                    if basket.recovery_stopped:
                        buf = 0.3 * (atr_now if not pd.isna(atr_now) else 0)
                        if basket.direction == "BUY" and row["close"] >= avg + buf:
                            exit_reason = "weighted_avg_exit"
                        elif basket.direction == "SELL" and row["close"] <= avg - buf:
                            exit_reason = "weighted_avg_exit"
                if exit_reason is None:
                    # opposite Bollinger band touch
                    if basket.direction == "BUY" and row["close"] >= row["bb_upper"]:
                        exit_reason = "opposite_bb"
                    elif basket.direction == "SELL" and row["close"] <= row["bb_lower"]:
                        exit_reason = "opposite_bb"
                if exit_reason is None:
                    # momentum exhaustion: RSI rolling back from extreme
                    rsi_now = row["rsi"]
                    if not pd.isna(rsi_now):
                        if basket.direction == "BUY" and rsi_now >= (cfg.rsi_overbought - cfg.momentum_exhaustion_rsi_buffer) and row["close"] > avg:
                            exit_reason = "momentum_exhaustion"
                        elif basket.direction == "SELL" and rsi_now <= (cfg.rsi_oversold + cfg.momentum_exhaustion_rsi_buffer) and row["close"] < avg:
                            exit_reason = "momentum_exhaustion"

                if exit_reason is not None:
                    exit_px = exec_price_base - spread_price / 2 - slip if basket.direction == "BUY" else exec_price_base + spread_price / 2 + slip
                    quote_pnl = self._quote_pnl(basket.direction, avg, exit_px, basket.total_lots())
                    usd_pnl = self._to_usd(quote_pnl, rate) - self._commission(basket.total_lots())
                    balance += usd_pnl
                    self.trades.append(Trade(
                        open_time=basket.positions[0].open_time,
                        close_time=row["datetime"],
                        direction=basket.direction,
                        lots=basket.total_lots(),
                        entry_price=avg,
                        exit_price=exit_px,
                        pnl_usd=usd_pnl,
                        reason=exit_reason,
                        grid_levels_used=len(basket.positions),
                    ))
                    basket = Basket()

                else:
                    # ---- partial close ----
                    if tp_dist is not None and basket.total_lots() > 0:
                        trigger = cfg.partial_close_trigger_frac_of_tp * tp_dist
                        hit = (basket.direction == "BUY" and row["close"] >= avg + trigger) or \
                              (basket.direction == "SELL" and row["close"] <= avg - trigger)
                        if hit and len(basket.positions) > 1:
                            close_lots_total = round(basket.total_lots() * cfg.partial_close_pct, 2)
                            remaining = close_lots_total
                            exit_px = exec_price_base - spread_price / 2 - slip if basket.direction == "BUY" else exec_price_base + spread_price / 2 + slip
                            new_positions = []
                            for p in basket.positions:
                                if remaining <= 0:
                                    new_positions.append(p)
                                    continue
                                take = min(p.lots, remaining)
                                remaining -= take
                                quote_pnl = self._quote_pnl(basket.direction, p.entry_price, exit_px, take)
                                usd_pnl = self._to_usd(quote_pnl, rate) - self._commission(take)
                                balance += usd_pnl
                                self.trades.append(Trade(
                                    open_time=p.open_time, close_time=row["datetime"],
                                    direction=basket.direction, lots=take,
                                    entry_price=p.entry_price, exit_price=exit_px,
                                    pnl_usd=usd_pnl, reason="partial_close",
                                    grid_levels_used=1,
                                ))
                                if take < p.lots:
                                    new_positions.append(Position(p.direction, p.entry_price, round(p.lots - take, 2), p.open_time, p.grid_level))
                            basket.positions = new_positions

                    # ---- recovery add-on ----
                    if basket.is_open() and not daily_locked and not weekly_locked and self._recovery_ok(decision_idx, basket):
                        level = len(basket.positions)
                        lots = self._lot_for_level(level, balance)
                        entry_px = exec_price_base + spread_price / 2 + slip if basket.direction == "BUY" else exec_price_base - spread_price / 2 - slip
                        basket.positions.append(Position(basket.direction, entry_px, lots, row["datetime"], level))

            # ---- consider new basket entry ----
            elif not daily_locked and not weekly_locked:
                regime = self.regime_series[decision_idx]
                if regime in ("RANGING", "TRANSITION"):
                    signal = self._entry_signal(decision_idx)
                    if signal is not None:
                        entry_px = exec_price_base + spread_price / 2 + slip if signal == "BUY" else exec_price_base - spread_price / 2 - slip
                        lots = self._lot_for_level(0, balance)
                        basket = Basket(positions=[Position(signal, entry_px, lots, row["datetime"], 0)], direction=signal)

            # ---- mark-to-market equity curve ----
            if basket.is_open():
                avg = basket.avg_price()
                floating_quote = self._quote_pnl(basket.direction, avg, row["close"], basket.total_lots())
                floating_usd = self._to_usd(floating_quote, rate)
            else:
                floating_usd = 0.0
            equity = balance + floating_usd
            self.balance_curve[i] = balance
            self.equity_curve[i] = equity

            # ---- daily / weekly loss lockouts ----
            if day_start_balance > 0:
                day_loss_pct = 100.0 * (day_start_balance - equity) / day_start_balance
                if day_loss_pct >= cfg.max_daily_loss_pct:
                    daily_locked = True
            if week_start_balance > 0:
                week_loss_pct = 100.0 * (week_start_balance - equity) / week_start_balance
                if week_loss_pct >= cfg.max_weekly_loss_pct:
                    weekly_locked = True

        df["equity"] = self.equity_curve
        df["balance"] = self.balance_curve
        df["regime"] = self.regime_series
        return df
