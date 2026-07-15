"""
Main backtesting engine.

Iterates bar by bar over M5 data.
At each bar:
1. Update risk manager with current equity
2. Check emergency exits for all open baskets
3. Check basket TP exits
4. Check for new basket entry signals
5. Check grid level additions for open baskets

NO LOOK-AHEAD: all decisions use only data available
at the START of current bar (i.e., previous bar close).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm

from .basket import Basket, BasketStatus, Direction, Position
from .strategy import SignalGenerator, SignalType, EntrySignal, check_emergency_exit
from .risk_manager import RiskManager
from .indicators import compute_all_indicators

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pip value calculator
# ---------------------------------------------------------------------------

def get_pip_value_per_lot(pair: str) -> float:
    """
    Approximate USD value per pip per standard lot (100,000 units).
    For cross pairs, this is approximate and varies with exchange rate.
    We use a fixed approximation for simplicity.

    In production: recalculate using current rate.
    """
    # For most EUR/GBP type pairs, pip value ≈ $10 per 0.1 lot
    # EURGBP: 1 pip = 0.0001 GBP * 100,000 = 10 GBP ≈ 12.5 USD (at GBPUSD ~1.25)
    # AUDNZD: 1 pip = 0.0001 NZD * 100,000 = 10 NZD ≈ 6 USD (at NZDUSD ~0.60)
    approximations = {
        "EURGBP": 12.5,
        "AUDNZD": 6.0,
    }
    return approximations.get(pair, 10.0)


# ---------------------------------------------------------------------------
# Weekly range tracker (for emergency exit)
# ---------------------------------------------------------------------------

class WeeklyRangeTracker:
    """Tracks rolling weekly high/low for emergency exit check."""

    def __init__(self):
        self._week_data: Dict[str, Dict[str, float]] = {}

    def update(self, current_time: pd.Timestamp, price: float, pair: str):
        week_key = f"{pair}_{current_time.strftime('%Y-%W')}"
        if week_key not in self._week_data:
            self._week_data[week_key] = {"high": price, "low": price}
        self._week_data[week_key]["high"] = max(self._week_data[week_key]["high"], price)
        self._week_data[week_key]["low"]  = min(self._week_data[week_key]["low"],  price)

    def get_prev_week_range(
        self, current_time: pd.Timestamp, pair: str
    ) -> Tuple[float, float]:
        """Get previous week's high/low."""
        # Current week
        current_week_num = int(current_time.strftime("%W"))
        current_year     = current_time.year

        # Previous week
        if current_week_num == 0:
            prev_year = current_year - 1
            prev_week = 52
        else:
            prev_year = current_year
            prev_week = current_week_num - 1

        prev_key = f"{pair}_{prev_year}-{prev_week:02d}"

        if prev_key in self._week_data:
            return (
                self._week_data[prev_key]["high"],
                self._week_data[prev_key]["low"],
            )
        return (float("inf"), 0.0)  # Safe defaults if no previous week


# ---------------------------------------------------------------------------
# Trade log entry
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    basket_id:    int
    pair:         str
    direction:    str
    level:        int
    entry_time:   object
    exit_time:    object
    entry_price:  float
    exit_price:   float
    lot_size:     float
    pnl_pips:     float
    pnl_usd:      float
    close_reason: str


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity_curve:    pd.Series
    basket_log:      List[dict]
    trade_log:       List[TradeRecord]
    config:          object
    initial_capital: float
    final_equity:    float

    def summary(self) -> dict:
        if not self.basket_log:
            return {"error": "no_baskets"}

        df = pd.DataFrame(self.basket_log)
        eq = self.equity_curve

        total_return = (self.final_equity - self.initial_capital) / self.initial_capital
        peak         = eq.cummax()
        drawdown     = (peak - eq) / peak
        max_dd       = drawdown.max()

        wins  = df[df["pnl_usd"] > 0]
        loses = df[df["pnl_usd"] <= 0]

        return {
            "initial_capital":   self.initial_capital,
            "final_equity":      round(self.final_equity, 2),
            "total_return_pct":  round(total_return * 100, 2),
            "max_drawdown_pct":  round(max_dd * 100, 2),
            "total_baskets":     len(df),
            "winning_baskets":   len(wins),
            "losing_baskets":    len(loses),
            "win_rate_pct":      round(len(wins) / max(len(df), 1) * 100, 2),
            "avg_win_usd":       round(wins["pnl_usd"].mean(),  2) if len(wins) > 0 else 0,
            "avg_loss_usd":      round(loses["pnl_usd"].mean(), 2) if len(loses) > 0 else 0,
            "total_pnl_usd":     round(df["pnl_usd"].sum(), 2),
            "profit_factor":     round(
                wins["pnl_usd"].sum() / max(abs(loses["pnl_usd"].sum()), 0.01), 2
            ),
            "avg_levels_per_basket": round(df["levels_used"].mean(), 2),
            "emergency_exits":   len(df[df["status"] == "EMERGENCY"]),
        }


# ---------------------------------------------------------------------------
# Core Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Event-driven backtester iterating over M5 bars.
    """

    def __init__(self, config, data_store):
        self.config      = config
        self.data_store  = data_store
        self.sig_gen     = SignalGenerator(config)
        self.risk_mgr    = RiskManager(config, config.initial_capital)
        self.weekly_tracker = WeeklyRangeTracker()

        self.equity      = config.initial_capital
        self.equity_curve: List[Tuple[object, float]] = []

        self.open_baskets:  List[Basket] = []
        self.closed_baskets: List[Basket] = []
        self.basket_counter = 0

        self.trade_log: List[TradeRecord] = []

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        start_date: Optional[str] = None,
        end_date:   Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest across all configured pairs simultaneously.
        Uses M5 bars as execution timeframe.
        """
        # Pre-compute indicators for all pairs and timeframes
        logger.info("Pre-computing indicators...")
        indicators: Dict[str, Dict[str, pd.DataFrame]] = {}

        for pair in self.config.pairs:
            indicators[pair] = {}
            for tf in ["M5", "M15", "H4"]:
                df_tf = self.data_store.get_tf(pair, tf)
                indicators[pair][tf] = compute_all_indicators(df_tf, self.config)
                logger.info(f"  {pair} {tf}: {len(df_tf):,} bars")

        # Build unified M5 timeline across all pairs
        all_times = pd.Index([])
        for pair in self.config.pairs:
            all_times = all_times.union(indicators[pair]["M5"].index)

        all_times = all_times.sort_values()

        # Date filter
        if start_date:
            all_times = all_times[all_times >= pd.Timestamp(start_date)]
        if end_date:
            all_times = all_times[all_times <= pd.Timestamp(end_date)]

        logger.info(f"Backtest period: {all_times[0]} → {all_times[-1]}")
        logger.info(f"Total M5 bars: {len(all_times):,}")

        # Warm-up: skip first N bars for indicator initialization
        warmup_bars = max(
            self.config.ema_trend_period * 4,  # H4 EMA200 needs 200 H4 bars = 800 M5
            self.config.atr_filter_period * 15,  # M15 ATR100
        )
        logger.info(f"Warming up: skipping first {warmup_bars} bars")

        # Main loop
        for i, current_time in enumerate(tqdm(all_times, desc="Backtesting")):
            # Skip warmup
            if i < warmup_bars:
                self.equity_curve.append((current_time, self.equity))
                # Still update weekly tracker during warmup
                for pair in self.config.pairs:
                    m5 = indicators[pair]["M5"]
                    if current_time in m5.index:
                        price = m5.loc[current_time, "close"]
                        self.weekly_tracker.update(current_time, price, pair)
                continue

            # Update risk manager
            self.risk_mgr.update(current_time, self.equity)

            # Process each pair
            for pair in self.config.pairs:
                m5_ind  = indicators[pair]["M5"]
                m15_ind = indicators[pair]["M15"]
                h4_ind  = indicators[pair]["H4"]
                pair_cfg = self.config.pair_configs[pair]

                # Current M5 bar
                if current_time not in m5_ind.index:
                    continue

                m5_bar    = m5_ind.loc[current_time]
                m15_slice = m15_ind[m15_ind.index <= current_time]
                h4_slice  = h4_ind[h4_ind.index <= current_time]

                current_price = m5_bar["close"]

                # Update weekly tracker
                self.weekly_tracker.update(current_time, current_price, pair)

                # Get last closed M15 bar (for emergency check)
                if len(m15_slice) < 2:
                    continue
                m15_last = m15_slice.iloc[-1]

                # Weekly range
                prev_w_high, prev_w_low = self.weekly_tracker.get_prev_week_range(
                    current_time, pair
                )

                # --- 1. Check emergency exits ---
                pair_baskets = [
                    b for b in self.open_baskets
                    if b.pair == pair and b.status == BasketStatus.ACTIVE
                ]

                emergency = check_emergency_exit(
                    m15_last, prev_w_high, prev_w_low, self.config
                )

                if emergency:
                    for basket in pair_baskets:
                        pnl_before = sum(p.pnl_usd for p in basket.positions)
                        basket.close_all(
                            current_price, current_time, "EMERGENCY",
                            pair_cfg.spread_pips
                        )
                        self.equity += basket.realized_usd() - pnl_before
                        self._finalize_basket(basket)
                        logger.debug(
                            f"EMERGENCY EXIT: {pair} Basket#{basket.basket_id} "
                            f"PnL: {basket.realized_usd():.2f}"
                        )
                    pair_baskets = []

                # --- 2. Check basket TP exits ---
                tp_usd = self.equity * self.config.basket_tp_target

                for basket in list(pair_baskets):
                    if basket.status != BasketStatus.ACTIVE:
                        continue

                    # Check protection
                    basket.check_protection(current_price, tp_usd)

                    unrealized = basket.unrealized_usd(current_price)

                    if unrealized >= tp_usd:
                        pnl_before = sum(p.pnl_usd for p in basket.positions)
                        basket.close_all(
                            current_price, current_time, "TAKE_PROFIT",
                            pair_cfg.spread_pips
                        )
                        pnl_gained = basket.realized_usd()
                        self.equity += pnl_gained
                        self._finalize_basket(basket)
                        pair_baskets.remove(basket)
                        logger.debug(
                            f"TAKE_PROFIT: {pair} Basket#{basket.basket_id} "
                            f"PnL: {pnl_gained:.2f}"
                        )

                # --- 3. Grid level additions for existing baskets ---
                current_spread = pair_cfg.spread_pips

                for basket in pair_baskets:
                    if basket.status != BasketStatus.ACTIVE:
                        continue
                    basket.try_add_level(current_price, current_time, current_spread)

                # --- 4. Check for new basket entry ---
                can_open, reason = self.risk_mgr.can_open_new_basket(
                    pair, "ANY", current_time, self.equity, self.open_baskets
                )

                if can_open:
                    signal = self.sig_gen.evaluate(
                        pair=pair,
                        current_time=current_time,
                        current_price=current_price,
                        m15_indicators=m15_ind,
                        h4_indicators=h4_ind,
                        pair_config=pair_cfg,
                    )

                    if signal.signal != SignalType.NONE:
                        # Double-check direction limit
                        direction = signal.signal.value
                        can_open2, reason2 = self.risk_mgr.can_open_new_basket(
                            pair, direction, current_time, self.equity, self.open_baskets
                        )

                        if can_open2:
                            self._open_basket(
                                signal, pair_cfg, current_time, current_price
                            )

            # Record equity
            # Include unrealized PnL in equity curve
            total_unrealized = sum(
                b.unrealized_usd(
                    # Use last known price (approximate)
                    indicators[b.pair]["M5"].iloc[
                        indicators[b.pair]["M5"].index.get_loc(current_time)
                        if current_time in indicators[b.pair]["M5"].index
                        else -1
                    ]["close"]
                )
                for b in self.open_baskets
                if b.status == BasketStatus.ACTIVE
            )
            self.equity_curve.append((current_time, self.equity + total_unrealized))

        # End of backtest: close all remaining baskets
        logger.info("Closing remaining open baskets at end of backtest...")
        for basket in list(self.open_baskets):
            if basket.status == BasketStatus.ACTIVE:
                last_price = indicators[basket.pair]["M5"]["close"].iloc[-1]
                basket.close_all(
                    last_price, all_times[-1], "END_OF_TEST",
                    self.config.pair_configs[basket.pair].spread_pips
                )
                self.equity += basket.realized_usd()
                self._finalize_basket(basket)

        logger.info(f"Backtest complete. Final equity: ${self.equity:,.2f}")

        eq_series = pd.Series(
            [e for _, e in self.equity_curve],
            index=[t for t, _ in self.equity_curve],
        )

        basket_log = [b.summary() for b in self.closed_baskets]

        return BacktestResult(
            equity_curve=eq_series,
            basket_log=basket_log,
            trade_log=self.trade_log,
            config=self.config,
            initial_capital=self.config.initial_capital,
            final_equity=self.equity,
        )

    # ------------------------------------------------------------------
    # Basket management
    # ------------------------------------------------------------------

    def _open_basket(
        self,
        signal: EntrySignal,
        pair_cfg,
        current_time,
        current_price: float,
    ):
        """Open a new basket and add first position."""
        self.basket_counter += 1
        pip_value = get_pip_value_per_lot(signal.pair)
        base_lot  = self.risk_mgr.calculate_base_lot(self.equity, pair_cfg)

        direction = (
            Direction.BUY if signal.signal == SignalType.BUY else Direction.SELL
        )

        basket = Basket(
            basket_id=self.basket_counter,
            pair=signal.pair,
            direction=direction,
            open_time=current_time,
            pip_size=pair_cfg.pip_size,
            spread_pips=pair_cfg.spread_pips,
            pip_value_per_lot=pip_value,
            grid_distance_pips=signal.grid_distance_pips,
            lot_sequence=self.config.lot_sequence,
            base_lot=base_lot,
            max_levels=self.config.max_grid_levels,
        )

        # Add first level immediately
        basket.try_add_level(current_price, current_time, pair_cfg.spread_pips)

        self.open_baskets.append(basket)

        logger.debug(
            f"NEW BASKET: #{self.basket_counter} {signal.pair} {direction.value} "
            f"@ {current_price:.5f} Grid:{signal.grid_distance_pips:.1f}pips "
            f"BaseLot:{base_lot:.2f}"
        )

    def _finalize_basket(self, basket: Basket):
        """Move basket from open to closed, record trades."""
        if basket in self.open_baskets:
            self.open_baskets.remove(basket)

        if basket not in self.closed_baskets:
            self.closed_baskets.append(basket)

        # Record individual trades
        for pos in basket.positions:
            self.trade_log.append(TradeRecord(
                basket_id=basket.basket_id,
                pair=basket.pair,
                direction=basket.direction.value,
                level=pos.level,
                entry_time=pos.entry_time,
                exit_time=pos.exit_time,
                entry_price=pos.entry_price,
                exit_price=pos.exit_price if pos.exit_price else 0.0,
                lot_size=pos.lot_size,
                pnl_pips=pos.pnl_pips,
                pnl_usd=pos.pnl_usd,
                close_reason=basket.close_reason,
            ))
