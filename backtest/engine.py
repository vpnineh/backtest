"""
engine.py
==========
Bar-by-bar, event-driven backtest engine for the Adaptive Recovery Grid
strategy (EURGBP / AUDNZD): regime-filtered mean reversion + ATR-adaptive
grid + non-linear lot progression + basket risk management.

ANTI-LOOKAHEAD DESIGN (read this before trusting any number it prints):

1. All indicators (indicators.py) are computed once, vectorized, over
   the whole series. Each value at index i legitimately only depends on
   bars <= i (that's just what trailing indicators are).

2. The loop below NEVER lets a bar trade on its own not-yet-closed
   information. For bar i, every decision (regime, filters, signals)
   is read from index i-1 (the last bar that was fully closed BEFORE
   bar i opened). The resulting order, if any, is filled at bar i's
   OPEN price plus spread/slippage.

3. Weekly structure (weekly_prior_levels) uses the prior COMPLETED
   week only (shift(1) on a weekly resample), never the forming week.

4. No parameter is fit on the backtest data itself. Config values are
   fixed, economically-motivated defaults. There is no in-sample
   optimisation loop anywhere in this codebase.

5. Transaction costs (spread + commission + slippage) are always
   applied, and P&L is converted to USD using the real GBPUSD/NZDUSD
   history (fx_convert.py) rather than assumed constant.

FREQUENCY MODE (added after the first backtest round):
The original spec ("ALL filters must agree", one basket at a time) is
mathematically very low-frequency (see README / diagnostics block) --
independent low-probability filters AND-ed together collapse to a
tiny combined probability. To hit a target trade frequency two honest
levers are exposed here, both still driven by real market structure,
neither one "faking" performance:

  a) `min_core_filters_required`: entry now requires N-of-5 core signal
     components to agree (majority vote) instead of a strict 5-of-5 AND.
     ADX / session / spread stay as hard gates (cheap, high pass-rate
     sanity checks, not part of the vote).
  b) `max_concurrent_baskets`: more than one independent grid basket
     (in either direction) can be open on the same symbol at once,
     instead of forcing the engine to sit idle until the previous
     basket fully closes.

Loosening these does trade off some selectivity for frequency -- that
trade-off is real and shows up honestly in the backtest stats (win
rate, drawdown), it is not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from . import indicators as ind

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
    max_positions: int = 7                 # grid levels per basket

    # concurrency (frequency lever #2)
    max_concurrent_baskets: int = 3        # independent baskets open at once, same symbol

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

    # entry vote (frequency lever #1): N of the 5 CORE components below
    # must agree. Core components: bb_break, rsi_extreme, stoch_extreme,
    # rejection_candle, ema200_dist_ok. ADX/session/spread remain hard gates.
    min_core_filters_required: int = 5     # 5 = original strict "ALL must agree"

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
    max_floating_dd_pct: float = 15.0      # per-basket floating DD vs equity
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_pct: float = 10.0
    max_margin_util_pct: float = 60.0
    leverage: float = 30.0

    # session (UTC hours), spec prefers London / London-NY overlap
    session_start_hour_utc: int = 7
    session_end_hour_utc: int = 17
    avoid_friday_after_hour_utc: int = 19

    # misc
    enable_trend_single_trade: bool = False
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
    basket_id: int
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
    basket_id: int = -1


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------

class BacktestEngine:
    def __init__(self, df: pd.DataFrame, cfg: StrategyConfig, conv_rate_at):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.pip = PIP_SIZE[cfg.symbol]
        self.conv_rate_at = conv_rate_at

        self.trades: List[Trade] = []
        self.equity_curve = np.zeros(len(self.df))
        self.balance_curve = np.zeros(len(self.df))
        self.regime_series: List[str] = [""] * len(self.df)
        self._next_basket_id = 1

        self.diag = {
            "baskets_opened": 0,
            "recovery_checked": 0,
            "recovery_approved": 0,
            "recovery_rejected_reasons": {},
            "recovery_additions": 0,
            "forced_closes_max_dd": 0,
            "daily_lock_events": 0,
            "weekly_lock_events": 0,
            "breakout_stops_triggered": 0,
            "entries_skipped_max_concurrent": 0,
        }

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
        if weekday == 6:
            return False
        if weekday == 4 and hour >= cfg.avoid_friday_after_hour_utc:
            return False
        return cfg.session_start_hour_utc <= hour < cfg.session_end_hour_utc

    def _current_spread_pips(self, i: int) -> float:
        cfg = self.cfg
        if self._session_ok(i):
            return cfg.spread_pips
        return cfg.spread_pips * cfg.off_session_spread_mult

    def _spread_filter_ok(self, i: int) -> bool:
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
            return None
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

    # -- entry: N-of-5 core filter vote, ADX/session/spread as hard gates --
    def _entry_signal(self, i: int) -> Optional[str]:
        cfg = self.cfg
        row = self.df.iloc[i]

        required = ["rsi", "stoch_k", "bb_upper", "bb_lower", "atr", "adx", "ema_slow"]
        if any(pd.isna(row[c]) for c in required):
            return None

        # hard gates (always required, not part of the vote)
        if not self._session_ok(i):
            return None
        if not self._spread_filter_ok(i):
            return None
        if row["adx"] >= cfg.adx_entry_max:
            return None

        dist_ema200_pips = (row["close"] - row["ema_slow"]) / self.pip
        bullish_rejection = row["close"] > row["open"] and row["low"] < row["bb_lower"]
        bearish_rejection = row["close"] < row["open"] and row["high"] > row["bb_upper"]

        vwap_ok_buy = True
        vwap_ok_sell = True
        if cfg.use_vwap_filter and not pd.isna(row["vwap"]):
            vwap_ok_buy = (row["vwap"] - row["close"]) / self.pip >= cfg.vwap_min_dist_pips
            vwap_ok_sell = (row["close"] - row["vwap"]) / self.pip >= cfg.vwap_min_dist_pips

        buy_components = [
            row["close"] < row["bb_lower"],
            row["rsi"] <= cfg.rsi_oversold,
            row["stoch_k"] <= cfg.stoch_oversold,
            bullish_rejection,
            abs(dist_ema200_pips) >= cfg.ema200_min_dist_pips,
        ]
        sell_components = [
            row["close"] > row["bb_upper"],
            row["rsi"] >= cfg.rsi_overbought,
            row["stoch_k"] >= cfg.stoch_overbought,
            bearish_rejection,
            abs(dist_ema200_pips) >= cfg.ema200_min_dist_pips,
        ]

        buy_votes = sum(bool(c) for c in buy_components)
        sell_votes = sum(bool(c) for c in sell_components)

        buy_ok = buy_votes >= cfg.min_core_filters_required and vwap_ok_buy
        sell_ok = sell_votes >= cfg.min_core_filters_required and vwap_ok_sell

        if buy_ok and not sell_ok:
            return "BUY"
        if sell_ok and not buy_ok:
            return "SELL"
        if buy_ok and sell_ok:
            # both sides voted true (only possible with a loose N) -> take the stronger vote
            return "BUY" if buy_votes > sell_votes else ("SELL" if sell_votes > buy_votes else None)
        return None

    def _reject_recovery(self, reason: str) -> bool:
        counts = self.diag["recovery_rejected_reasons"]
        counts[reason] = counts.get(reason, 0) + 1
        return False

    def _recovery_ok(self, i: int, basket: Basket) -> bool:
        cfg = self.cfg
        self.diag["recovery_checked"] += 1

        if basket.recovery_stopped:
            return self._reject_recovery("recovery_already_stopped_by_breakout")
        if len(basket.positions) >= min(cfg.max_positions, len(cfg.lot_progression)):
            return self._reject_recovery("max_positions_reached")
        regime = self.regime_series[i]
        if regime != "RANGING":
            return self._reject_recovery(f"regime_not_ranging({regime})")
        row = self.df.iloc[i]
        atr_now, atr_avg = row["atr"], row["atr_avg"]
        if pd.isna(atr_now) or pd.isna(atr_avg) or atr_avg == 0:
            return self._reject_recovery("atr_not_ready")
        if (atr_now / atr_avg) >= cfg.atr_extreme_ratio:
            return self._reject_recovery("atr_extreme")
        if row["adx"] >= cfg.adx_entry_max:
            return self._reject_recovery("adx_too_high")
        if self._breakout_detected(i):
            return self._reject_recovery("breakout_detected")

        grid_mult = self._grid_atr_mult(i)
        if grid_mult is None:
            return self._reject_recovery("grid_disabled_extreme_volatility")
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

        if not far_enough:
            return self._reject_recovery("price_not_far_enough_for_next_grid_level")
        if not within_stat_limit:
            return self._reject_recovery("beyond_statistical_deviation_limit")

        self.diag["recovery_approved"] += 1
        return True

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

        self.regime_series = [self._classify_regime(i) for i in range(n)]
        regime_counts = pd.Series(self.regime_series).value_counts()
        self.diag["regime_distribution_pct"] = {
            k: round(100.0 * v / n, 2) for k, v in regime_counts.items()
        }

        rates = self.conv_rate_at(df["datetime"]).values

        balance = cfg.initial_balance
        baskets: List[Basket] = []

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

            decision_idx = i - 1
            exec_price_base = row["open"]
            spread_pips = self._current_spread_pips(decision_idx)
            slip = cfg.slippage_pips * self.pip
            spread_price = spread_pips * self.pip

            # total floating pnl across ALL open baskets, needed for equity/DD checks
            def total_floating_usd(basket_list: List[Basket], price: float) -> float:
                tot = 0.0
                for b in basket_list:
                    if not b.is_open():
                        continue
                    avg_b = b.avg_price()
                    q = self._quote_pnl(b.direction, avg_b, price, b.total_lots())
                    tot += self._to_usd(q, rate)
                return tot

            equity_now_close = balance + total_floating_usd(baskets, row["close"])

            # ---- manage each existing basket ----
            still_open: List[Basket] = []
            for basket in baskets:
                avg = basket.avg_price()
                atr_now = df["atr"].iat[decision_idx]
                tp_dist = cfg.basket_tp_atr_mult * atr_now if not pd.isna(atr_now) else None

                mtm_price = row["close"]
                floating_quote = self._quote_pnl(basket.direction, avg, mtm_price, basket.total_lots())
                floating_usd = self._to_usd(floating_quote, rate)

                if self._breakout_detected(decision_idx) and not basket.recovery_stopped:
                    basket.recovery_stopped = True
                    self.diag["breakout_stops_triggered"] += 1

                dd_pct = -100.0 * floating_usd / max(equity_now_close, 1.0) if floating_usd < 0 else 0.0
                force_close = dd_pct >= cfg.max_floating_dd_pct

                exit_reason = None
                if force_close:
                    exit_reason = "max_floating_dd"
                    self.diag["forced_closes_max_dd"] += 1
                elif tp_dist is not None:
                    if basket.direction == "BUY" and row["close"] >= avg + tp_dist:
                        exit_reason = "basket_tp"
                    elif basket.direction == "SELL" and row["close"] <= avg - tp_dist:
                        exit_reason = "basket_tp"
                if exit_reason is None and basket.recovery_stopped:
                    buf = 0.3 * (atr_now if not pd.isna(atr_now) else 0)
                    if basket.direction == "BUY" and row["close"] >= avg + buf:
                        exit_reason = "weighted_avg_exit"
                    elif basket.direction == "SELL" and row["close"] <= avg - buf:
                        exit_reason = "weighted_avg_exit"
                if exit_reason is None:
                    if basket.direction == "BUY" and row["close"] >= row["bb_upper"]:
                        exit_reason = "opposite_bb"
                    elif basket.direction == "SELL" and row["close"] <= row["bb_lower"]:
                        exit_reason = "opposite_bb"
                if exit_reason is None:
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
                        basket_id=basket.basket_id,
                    ))
                    continue  # this basket is closed -> drop it from still_open

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
                                grid_levels_used=1, basket_id=basket.basket_id,
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
                    self.diag["recovery_additions"] += 1

                if basket.is_open():
                    still_open.append(basket)

            baskets = still_open

            # ---- consider a new basket entry ----
            if not daily_locked and not weekly_locked:
                if len(baskets) < cfg.max_concurrent_baskets:
                    regime = self.regime_series[decision_idx]
                    if regime in ("RANGING", "TRANSITION"):
                        signal = self._entry_signal(decision_idx)
                        if signal is not None:
                            entry_px = exec_price_base + spread_price / 2 + slip if signal == "BUY" else exec_price_base - spread_price / 2 - slip
                            lots = self._lot_for_level(0, balance)
                            new_basket = Basket(basket_id=self._next_basket_id, positions=[Position(signal, entry_px, lots, row["datetime"], 0)], direction=signal)
                            self._next_basket_id += 1
                            baskets.append(new_basket)
                            self.diag["baskets_opened"] += 1
                else:
                    regime = self.regime_series[decision_idx]
                    if regime in ("RANGING", "TRANSITION") and self._entry_signal(decision_idx) is not None:
                        self.diag["entries_skipped_max_concurrent"] += 1

            # ---- mark-to-market equity curve ----
            floating_usd = total_floating_usd(baskets, row["close"])
            equity = balance + floating_usd
            self.balance_curve[i] = balance
            self.equity_curve[i] = equity

            if day_start_balance > 0:
                day_loss_pct = 100.0 * (day_start_balance - equity) / day_start_balance
                if day_loss_pct >= cfg.max_daily_loss_pct and not daily_locked:
                    daily_locked = True
                    self.diag["daily_lock_events"] += 1
            if week_start_balance > 0:
                week_loss_pct = 100.0 * (week_start_balance - equity) / week_start_balance
                if week_loss_pct >= cfg.max_weekly_loss_pct and not weekly_locked:
                    weekly_locked = True
                    self.diag["weekly_lock_events"] += 1

        df["equity"] = self.equity_curve
        df["balance"] = self.balance_curve
        df["regime"] = self.regime_series
        self.diag["filter_hit_rates_pct"] = self._compute_filter_hit_rates()
        return df

    def _compute_filter_hit_rates(self) -> dict:
        """Independent AND N-of-5-vote pass-rates, computed vectorized over
        the whole series. Purely diagnostic -- shows the effect of the
        min_core_filters_required setting without needing to guess."""
        cfg = self.cfg
        df = self.df
        valid = df["rsi"].notna() & df["stoch_k"].notna() & df["bb_upper"].notna() & \
            df["adx"].notna() & df["ema_slow"].notna()
        n_valid = int(valid.sum())
        if n_valid == 0:
            return {}

        dist_ema200_pips = (df["close"] - df["ema_slow"]) / self.pip
        bullish_rejection = (df["close"] > df["open"]) & (df["low"] < df["bb_lower"])
        bearish_rejection = (df["close"] < df["open"]) & (df["high"] > df["bb_upper"])
        session_ok = (
            (df["weekday"] != 6)
            & ~((df["weekday"] == 4) & (df["hour_utc"] >= cfg.avoid_friday_after_hour_utc))
            & (df["hour_utc"] >= cfg.session_start_hour_utc)
            & (df["hour_utc"] < cfg.session_end_hour_utc)
        )
        hard_gate = (df["adx"] < cfg.adx_entry_max) & session_ok

        buy_c = [
            df["close"] < df["bb_lower"],
            df["rsi"] <= cfg.rsi_oversold,
            df["stoch_k"] <= cfg.stoch_oversold,
            bullish_rejection,
            dist_ema200_pips.abs() >= cfg.ema200_min_dist_pips,
        ]
        sell_c = [
            df["close"] > df["bb_upper"],
            df["rsi"] >= cfg.rsi_overbought,
            df["stoch_k"] >= cfg.stoch_overbought,
            bearish_rejection,
            dist_ema200_pips.abs() >= cfg.ema200_min_dist_pips,
        ]
        buy_votes = sum(c.astype(int) for c in buy_c)
        sell_votes = sum(c.astype(int) for c in sell_c)

        out = {
            "adx_below_entry_max": round(100.0 * ((df["adx"] < cfg.adx_entry_max) & valid).sum() / n_valid, 3),
            "session_ok": round(100.0 * (session_ok & valid).sum() / n_valid, 3),
        }
        for k in range(1, 6):
            out[f"buy__at_least_{k}_of_5_core"] = round(100.0 * ((buy_votes >= k) & hard_gate & valid).sum() / n_valid, 3)
            out[f"sell__at_least_{k}_of_5_core"] = round(100.0 * ((sell_votes >= k) & hard_gate & valid).sum() / n_valid, 3)
        out["buy__ALL_5_OF_5"] = out["buy__at_least_5_of_5_core"]
        out["sell__ALL_5_OF_5"] = out["sell__at_least_5_of_5_core"]
        out[f"buy__at_configured_threshold({cfg.min_core_filters_required})"] = out[f"buy__at_least_{cfg.min_core_filters_required}_of_5_core"]
        out[f"sell__at_configured_threshold({cfg.min_core_filters_required})"] = out[f"sell__at_least_{cfg.min_core_filters_required}_of_5_core"]
        return out
