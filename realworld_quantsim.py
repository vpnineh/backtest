"""
CorrArb Prop Simulator — v9 Realistic Synthetic Execution
=========================================================
Pairs:
  EURGBP synthetic = EURUSD / GBPUSD
  AUDNZD synthetic = AUDUSD / NZDUSD

v9 changes:
  ① Synthetic commission: 2 legs → $14/lot
  ② Dynamic spread based on session + ATR regime
  ③ More conservative slippage
  ④ Slow Z filter
  ⑤ Fast Hurst mean-reversion filter
  ⑥ Tighter VR / Corr / ATR regime
  ⑦ News-hour filter
  ⑧ Half-Kelly adaptive sizing
  ⑨ Real break-even stop after partial, including commission
  ⑩ Trailing stop after partial
"""

import pandas as pd
import numpy as np
import glob
import zipfile
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── Prop rules ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── Risk management ──
    risk_base_pct      = 0.015
    risk_min_pct       = 0.0075
    consec_loss_n      = 2
    risk_reduce        = 0.5

    # ── Broker costs, more realistic for synthetic 2-leg execution ──
    spread_pips_normal = 1.2
    spread_pips_high   = 3.5
    commission_per_lot = 14.0      # $7 per lot per leg × 2 legs
    slippage_pips      = 0.5

    # ── Instrument specs ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 2.0
    min_lot  = 0.01
    warmup   = 600

    # ── Z-score params ──
    z_fast_period      = 96
    z_slow_period      = 288
    z_entry            = 2.2
    z_exit_partial     = 0.5
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 20.0

    # ── Filters ──
    corr_period        = 96
    corr_min           = 0.82
    hour_start         = 3
    hour_end           = 18
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # ── Emergency exits ──
    sl_pips            = 30.0
    tp_pips            = 90.0
    time_stop_bars     = 36

    # ── ATR filter ──
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 2.5
    atr_min_mult       = 0.5

    # ── Variance Ratio regime ──
    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.88

    # ── Fast Hurst filter ──
    hurst_period       = 200
    hurst_k            = 8
    hurst_max          = 0.47

    # ── Partial / trailing ──
    partial_ratio      = 0.50
    trail_pips         = 18.0

    # ── Simple news-time filter
    # IMPORTANT: These hours must match your data timezone.
    # Common sensitive macro times: London open, NY data, NY open.
    news_hours         = [(8, 30), (9, 0), (13, 30), (14, 0), (15, 0)]
    news_buffer_mins   = 30

    # ── Kelly sizing ──
    kelly_lookback     = 120
    min_trades_kelly   = 40
    default_wr         = 0.58
    default_rr         = 1.20


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_raw(pattern: str, is_zip: bool) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files found: {pattern}")

    frames = []

    for p in paths:
        try:
            if is_zip:
                with zipfile.ZipFile(p, "r") as z:
                    csv_name = next(
                        (f for f in z.namelist() if f.lower().endswith(".csv")),
                        None
                    )
                    if csv_name is None:
                        print(f"  ⚠ No CSV in {os.path.basename(p)}")
                        continue

                    with z.open(csv_name) as f:
                        df = pd.read_csv(
                            f,
                            sep=";",
                            header=None,
                            names=["ts", "o", "h", "l", "c", "v"]
                        )
            else:
                df = pd.read_csv(
                    p,
                    sep=";",
                    header=None,
                    names=["ts", "o", "h", "l", "c", "v"]
                )

            frames.append(df)

        except Exception as e:
            print(f"  ⚠ Skip {os.path.basename(p)}: {e}")

    if not frames:
        raise ValueError(f"No valid data loaded from: {pattern}")

    raw = pd.concat(frames, ignore_index=True)
    raw["ts"] = pd.to_datetime(raw["ts"], format="%Y%m%d %H%M%S")
    raw = raw.sort_values("ts").drop_duplicates("ts").set_index("ts")
    raw[["o", "h", "l", "c"]] = raw[["o", "h", "l", "c"]].astype(float)

    return raw


def to_15min(raw: pd.DataFrame, sfx: str) -> pd.DataFrame:
    return pd.DataFrame({
        f"o_{sfx}": raw["o"].resample("15min").first(),
        f"h_{sfx}": raw["h"].resample("15min").max(),
        f"l_{sfx}": raw["l"].resample("15min").min(),
        f"c_{sfx}": raw["c"].resample("15min").last(),
    }).dropna()


def build_spread_df(
    df_a: pd.DataFrame,
    sfx_a: str,
    df_b: pd.DataFrame,
    sfx_b: str
) -> pd.DataFrame:
    """
    Synthetic cross:
      spread = pair_a / pair_b

    Example:
      EURGBP = EURUSD / GBPUSD

    quote_rate:
      GBPUSD for EURGBP
      NZDUSD for AUDNZD
    """
    merged = df_a.join(df_b, how="inner").dropna()

    if len(merged) == 0:
        raise ValueError(f"No common timestamps for {sfx_a}/{sfx_b}")

    merged["c_spread"] = merged[f"c_{sfx_a}"] / merged[f"c_{sfx_b}"]
    merged["o_spread"] = merged[f"o_{sfx_a}"] / merged[f"o_{sfx_b}"]

    # Conservative synthetic OHLC approximation
    merged["h_spread"] = merged[f"h_{sfx_a}"] / merged[f"l_{sfx_b}"]
    merged["l_spread"] = merged[f"l_{sfx_a}"] / merged[f"h_{sfx_b}"]

    merged["quote_rate"] = merged[f"c_{sfx_b}"]

    return merged[merged.index.weekday < 5].copy()


def load_all_pairs() -> dict:
    print("\n  Loading and syncing datasets...")
    pairs = {}

    try:
        eur = to_15min(load_raw("data/*EURUSD*.csv", is_zip=False), "eur")
        gbp = to_15min(load_raw("data/*GBPUSD*.csv", is_zip=False), "gbp")
        df = build_spread_df(eur, "eur", gbp, "gbp")
        pairs["EURGBP"] = {"df": df, "leg_a": "c_eur", "leg_b": "c_gbp"}
        print(f"  ✅ EURGBP : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ EURGBP : {e}")

    try:
        aud = to_15min(load_raw("data/HISTDATA*AUDUSD*.zip", is_zip=True), "aud")
        nzd = to_15min(load_raw("data/HISTDATA*NZDUSD*.zip", is_zip=True), "nzd")
        df = build_spread_df(aud, "aud", nzd, "nzd")
        pairs["AUDNZD"] = {"df": df, "leg_a": "c_aud", "leg_b": "c_nzd"}
        print(f"  ✅ AUDNZD : {len(df):>7,} candles | {df.index[0].date()} → {df.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ AUDNZD : {e}")

    if not pairs:
        raise RuntimeError("No pairs loaded. Check data/ directory.")

    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def calc_variance_ratio(series: pd.Series, k: int, window: int) -> pd.Series:
    r1 = series.diff(1)
    rk = series.diff(k)

    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()

    return vark / (k * var1.replace(0, np.nan))


def calc_fast_hurst(series: pd.Series, k: int, window: int) -> pd.Series:
    """
    Fast Hurst approximation via variance scaling:

      Var(X_t - X_{t-k}) ≈ k^(2H) Var(X_t - X_{t-1})

      H < 0.5 → mean-reverting tendency
      H ≈ 0.5 → random walk
      H > 0.5 → trending/persistent
    """
    r1 = series.diff(1)
    rk = series.diff(k)

    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()

    ratio = vark / var1.replace(0, np.nan)
    h = 0.5 * np.log(ratio.replace(0, np.nan)) / np.log(k)

    return h.replace([np.inf, -np.inf], np.nan)


def compute_signals(pair_name: str, pair_info: dict) -> tuple:
    C = Config

    df = pair_info["df"]
    leg_a = pair_info["leg_a"]
    leg_b = pair_info["leg_b"]

    log_ratio = np.log(df["c_spread"])

    # Use rolling stats shifted by 1 bar.
    # Signal is generated on closed bar and executed next bar.
    z_mean_fast = log_ratio.rolling(C.z_fast_period).mean().shift(1)
    z_std_fast  = log_ratio.rolling(C.z_fast_period).std().shift(1)
    z_fast = (log_ratio - z_mean_fast) / z_std_fast.replace(0, np.nan)

    z_mean_slow = log_ratio.rolling(C.z_slow_period).mean().shift(1)
    z_std_slow  = log_ratio.rolling(C.z_slow_period).std().shift(1)
    z_slow = (log_ratio - z_mean_slow) / z_std_slow.replace(0, np.nan)

    ret_a = df[leg_a].pct_change()
    ret_b = df[leg_b].pct_change()
    corr = ret_a.rolling(C.corr_period).corr(ret_b).shift(1)
    corr_ok = corr > C.corr_min

    vr = calc_variance_ratio(log_ratio, C.vr_k, C.vr_period).shift(1)
    regime_ok = vr < C.vr_max

    hurst = calc_fast_hurst(log_ratio, C.hurst_k, C.hurst_period).shift(1)
    hurst_ok = hurst < C.hurst_max

    atr = calc_atr(df["h_spread"], df["l_spread"], df["c_spread"], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    atr_ratio = atr / atr_ma.replace(0, np.nan)

    vol_ok = (
        (atr.shift(1) > atr_ma.shift(1) * C.atr_min_mult) &
        (atr.shift(1) < atr_ma.shift(1) * C.atr_max_mult)
    )

    hour = pd.Series(df.index.hour, index=df.index)
    minute = pd.Series(df.index.minute, index=df.index)
    dow = pd.Series(df.index.dayofweek, index=df.index)

    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    # Simple news filter
    minute_of_day = hour * 60 + minute
    news_mask = pd.Series(False, index=df.index)

    for h, m in C.news_hours:
        news_min = h * 60 + m
        news_mask = news_mask | minute_of_day.between(
            news_min - C.news_buffer_mins,
            news_min + C.news_buffer_mins
        )

    time_ok = time_ok & ~news_mask

    long_cond = (
        (z_fast < -C.z_entry) &
        (z_slow < -1.0) &
        vol_ok &
        time_ok &
        corr_ok &
        regime_ok &
        hurst_ok
    )

    short_cond = (
        (z_fast > C.z_entry) &
        (z_slow > 1.0) &
        vol_ok &
        time_ok &
        corr_ok &
        regime_ok &
        hurst_ok
    )

    sig = pd.Series(0, index=df.index)
    sig[long_cond] = 1
    sig[short_cond] = -1

    # Avoid repeated same-side signals on consecutive bars
    sig = sig.where(sig != sig.shift(), 0)

    n = int((sig != 0).sum())
    l = int((sig == 1).sum())
    s = int((sig == -1).sum())
    r = int(regime_ok.sum())
    h_ok = int(hurst_ok.sum())

    print(
        f"    {pair_name}: {n:,} signals "
        f"(L:{l} | S:{s}) | VR OK: {r:,} | Hurst OK: {h_ok:,}"
    )

    return sig, z_fast, atr_ratio


# ═══════════════════════════════════════════════════════════════════════════
# FINANCIAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def get_dynamic_spread(hour: int, atr_ratio: float) -> float:
    C = Config

    spread = C.spread_pips_normal

    # Thin liquidity / rollover style hours
    if hour in [0, 1, 22, 23]:
        spread *= 2.0

    # Sensitive macro / NY hours
    elif hour in [8, 9, 13, 14, 15]:
        spread *= 1.6

    # Volatility adjustment
    if np.isfinite(atr_ratio):
        if atr_ratio > 2.0:
            spread *= 1.5
        elif atr_ratio > 1.5:
            spread *= 1.25
        elif atr_ratio < 0.7:
            spread *= 1.10

    return float(min(spread, C.spread_pips_high))


def calc_pnl(
    direction: int,
    entry_px: float,
    exit_px: float,
    lot: float,
    quote_rate: float
) -> float:
    """
    PnL in USD:

      gross_quote = direction × price_diff × lot × lot_size
      gross_usd   = gross_quote × quote_rate
      commission  = commission_per_lot × lot

    commission_per_lot is synthetic total round-trip equivalent.
    """
    C = Config

    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    gross_usd = gross_quote * quote_rate
    commission = C.commission_per_lot * lot

    return gross_usd - commission


def entry_execution_price(mid_open: float, direction: int, spread_pips: float) -> float:
    C = Config
    cost_pips = spread_pips / 2.0 + C.slippage_pips
    return mid_open + direction * cost_pips * C.pip


def exit_execution_price(mid_price: float, direction: int, spread_pips: float) -> float:
    C = Config
    cost_pips = spread_pips / 2.0 + C.slippage_pips
    return mid_price - direction * cost_pips * C.pip


def breakeven_price(entry_px: float, direction: int, quote_rate: float) -> float:
    """
    Price where remaining trade approximately nets zero after commission.

    Solve:
      direction × (exit - entry) × lot × lot_size × quote_rate
      - commission_per_lot × lot = 0

    lot cancels out.
    """
    C = Config
    offset = C.commission_per_lot / (C.lot_size * quote_rate)
    return entry_px + direction * offset


def get_recent_wr_rr(all_trades: list) -> tuple:
    C = Config

    if len(all_trades) < C.min_trades_kelly:
        return C.default_wr, C.default_rr

    recent = pd.DataFrame(all_trades[-C.kelly_lookback:])

    if len(recent) < C.min_trades_kelly:
        return C.default_wr, C.default_rr

    wins = recent[recent["pnl"] > 0]["pnl"]
    losses = recent[recent["pnl"] < 0]["pnl"]

    if len(wins) == 0 or len(losses) == 0:
        return C.default_wr, C.default_rr

    wr = len(wins) / len(recent)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    if avg_loss <= 0:
        rr = C.default_rr
    else:
        rr = avg_win / avg_loss

    wr = float(np.clip(wr, 0.35, 0.75))
    rr = float(np.clip(rr, 0.50, 3.00))

    return wr, rr


def calc_lot_kelly(
    equity: float,
    sl_pips: float,
    consec_loss: int,
    quote_rate: float,
    all_trades: list
) -> float:
    C = Config

    wr, rr = get_recent_wr_rr(all_trades)

    # Kelly fraction:
    # f = W - (1-W)/R
    if rr <= 0:
        kelly = 0.0
    else:
        kelly = wr - (1.0 - wr) / rr

    half_kelly = max(0.0, kelly * 0.5)

    # Risk cannot exceed base risk.
    # If Kelly is too low, use minimum risk only if still positive.
    if half_kelly <= 0:
        risk_pct = C.risk_min_pct
    else:
        risk_pct = min(C.risk_base_pct, max(C.risk_min_pct, half_kelly))

    if consec_loss >= C.consec_loss_n:
        risk_pct = max(risk_pct * C.risk_reduce, C.risk_min_pct)

    pip_value_usd = C.pip * C.lot_size * quote_rate
    risk_usd = equity * risk_pct

    raw_lot = risk_usd / (sl_pips * pip_value_usd)

    return round(float(np.clip(raw_lot, C.min_lot, C.max_lot)), 2)


def update_trailing_stop(pos: dict, current_mid_price: float) -> dict:
    C = Config

    if not pos.get("partial_done", False):
        return pos

    d = pos["dir"]

    if d == 1:
        new_sl = current_mid_price - C.trail_pips * C.pip
        if new_sl > pos["sl"]:
            pos["sl"] = new_sl
    else:
        new_sl = current_mid_price + C.trail_pips * C.pip
        if new_sl < pos["sl"]:
            pos["sl"] = new_sl

    return pos


def new_acc(ts) -> dict:
    C = Config

    return {
        "equity": C.initial_balance,
        "start_ts": ts,
        "trades": [],
        "blown": False,
        "blown_rsn": "",
        "peak": C.initial_balance,
        "consec_loss": 0,
    }


def _make_rec(
    pos: dict,
    exit_px: float,
    exit_ts,
    pnl: float,
    status: str,
    lot: float
) -> dict:
    return {
        "pair": pos["pair"],
        "dir": pos["dir"],
        "lot": lot,
        "entry": pos["entry"],
        "exit": exit_px,
        "entry_ts": pos["entry_ts"],
        "exit_ts": exit_ts,
        "pnl": pnl,
        "status": status,
        "entry_bar": pos["entry_bar"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, pair_signals: dict) -> dict:
    C = Config
    pair_names = list(pairs.keys())

    common_idx = None

    for name in pair_names:
        idx = pairs[name]["df"].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)

    if n_bars <= C.warmup + 10:
        raise RuntimeError("Not enough common bars after warmup.")

    print(f"  ✅ Common bars: {n_bars:,} | {common_idx[0].date()} → {common_idx[-1].date()}")

    pa = {}

    for name in pair_names:
        df_p = pairs[name]["df"].reindex(common_idx).ffill()

        sig_s, z_s, atrr_s = pair_signals[name]

        pa[name] = {
            "o": df_p["o_spread"].values.astype(float),
            "c": df_p["c_spread"].values.astype(float),
            "qr": df_p["quote_rate"].values.astype(float),
            "sig": sig_s.reindex(common_idx).fillna(0).values.astype(int),
            "z": z_s.reindex(common_idx).values.astype(float),
            "atrr": atrr_s.reindex(common_idx).values.astype(float),
        }

    PROP_FLOOR = C.initial_balance * (1.0 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1.0 + C.profit_target_pct)

    acc = new_acc(common_idx[C.warmup])
    total_withdrawn = 0.0
    acc_num = 1
    day_start_eq = C.initial_balance

    all_trades = []
    acc_logs = []
    eq_curve = []

    positions = {name: None for name in pair_names}
    trades_today = {name: 0 for name in pair_names}
    pending_sig = {name: 0 for name in pair_names}

    print("\n  ▶ Running Multi-Pair Strict Prop Simulator v9...")
    print(f"    Pairs  : {' + '.join(pair_names)}")
    print(
        f"    Target : +{C.profit_target_pct*100:.0f}% | "
        f"Daily DD: -{C.max_daily_loss_pct*100:.0f}% | "
        f"Total DD: -{C.max_total_dd_pct*100:.0f}%"
    )
    print(
        f"    Risk   : Kelly capped at {C.risk_base_pct*100:.1f}% | "
        f"SL: {C.sl_pips}p | TP: {C.tp_pips}p | "
        f"TimeStop: {C.time_stop_bars} bars"
    )
    print(
        f"    Costs  : Dynamic spread | Slippage: {C.slippage_pips}p | "
        f"Commission: ${C.commission_per_lot:.1f}/lot synthetic"
    )

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq_curve.append((ts, round(acc["equity"], 4)))

        if acc["equity"] > acc["peak"]:
            acc["peak"] = acc["equity"]

        # Daily reset
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc["equity"]
            for name in pair_names:
                trades_today[name] = 0

        # Progress
        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(
                f"    Progress: {pct:5.1f}% | "
                f"Eq: ${acc['equity']:,.2f} | "
                f"Bank: ${total_withdrawn:,.2f}",
                end="\r"
            )

        # If account blown, log and reset
        if acc["blown"]:
            acc_logs.append({
                "account": acc_num,
                "start_ts": acc["start_ts"],
                "end_ts": ts,
                "reason": acc["blown_rsn"],
                "pnl": acc["equity"] - C.initial_balance,
            })

            print(
                f"\n    💥 #{acc_num:>3} | {ts.date()} | "
                f"Eq: ${acc['equity']:>8.2f} | {acc['blown_rsn']}"
            )

            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc["equity"]

            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name] = 0
                positions[name] = None

            continue

        # Execute pending entries at current bar open
        for name in pair_names:
            a = pa[name]

            if (
                pending_sig[name] != 0 and
                positions[name] is None and
                trades_today[name] < C.max_trades_day
            ):
                direction = pending_sig[name]
                qr = a["qr"][bar]
                atrr = a["atrr"][bar]
                spread_pips = get_dynamic_spread(ts.hour, atrr)

                lot = calc_lot_kelly(
                    acc["equity"],
                    C.sl_pips,
                    acc["consec_loss"],
                    qr,
                    all_trades
                )

                entry_mid = a["o"][bar]
                entry_px = entry_execution_price(entry_mid, direction, spread_pips)

                sl = entry_px - direction * C.sl_pips * C.pip
                tp = entry_px + direction * C.tp_pips * C.pip

                positions[name] = {
                    "pair": name,
                    "dir": direction,
                    "lot": lot,
                    "lot_remaining": lot,
                    "partial_done": False,
                    "entry": entry_px,
                    "sl": sl,
                    "tp": tp,
                    "entry_ts": ts,
                    "entry_bar": bar,
                }

                trades_today[name] += 1

            pending_sig[name] = 0

        # Floating PnL for prop DD check using liquidation price
        total_float = 0.0

        for name in pair_names:
            pos = positions[name]

            if pos is not None:
                a = pa[name]
                d = pos["dir"]
                qr = a["qr"][bar]
                atrr = a["atrr"][bar]
                spread_pips = get_dynamic_spread(ts.hour, atrr)

                liquidation_px = exit_execution_price(a["c"][bar], d, spread_pips)

                total_float += calc_pnl(
                    d,
                    pos["entry"],
                    liquidation_px,
                    pos["lot_remaining"],
                    qr
                )

        current_eq = acc["equity"] + total_float
        daily_limit = day_start_eq * (1.0 - C.max_daily_loss_pct)

        # Prop DD breach → close all positions
        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            acc["blown"] = True
            acc["blown_rsn"] = "DailyDD" if current_eq <= daily_limit else "TotalDD"

            for name in pair_names:
                pos = positions[name]

                if pos is None:
                    continue

                a = pa[name]
                d = pos["dir"]
                qr = a["qr"][bar]
                atrr = a["atrr"][bar]
                spread_pips = get_dynamic_spread(ts.hour, atrr)
                exit_px = exit_execution_price(a["c"][bar], d, spread_pips)

                pnl = calc_pnl(
                    d,
                    pos["entry"],
                    exit_px,
                    pos["lot_remaining"],
                    qr
                )

                acc["equity"] += pnl

                rec = _make_rec(
                    pos,
                    exit_px,
                    ts,
                    pnl,
                    "BLOWN",
                    pos["lot_remaining"]
                )

                all_trades.append(rec)
                acc["trades"].append(rec)
                positions[name] = None

            continue

        # Manage exits
        for name in pair_names:
            pos = positions[name]

            if pos is None:
                continue

            a = pa[name]

            cp_mid = a["c"][bar]
            qr = a["qr"][bar]
            atrr = a["atrr"][bar]
            spread_pips = get_dynamic_spread(ts.hour, atrr)

            d = pos["dir"]
            ep = pos["entry"]
            zn = a["z"][bar]
            lot_rem = pos["lot_remaining"]

            # Update trailing stop after partial
            pos = update_trailing_stop(pos, cp_mid)

            # Partial exit
            if not pos["partial_done"] and not np.isnan(zn):
                hit_partial = (
                    (d == 1 and zn >= -C.z_exit_partial) or
                    (d == -1 and zn <= C.z_exit_partial)
                )

                if hit_partial:
                    p_lot = round(lot_rem * C.partial_ratio, 2)

                    if p_lot >= C.min_lot:
                        exit_px = exit_execution_price(cp_mid, d, spread_pips)

                        p_pnl = calc_pnl(
                            d,
                            ep,
                            exit_px,
                            p_lot,
                            qr
                        )

                        # Only partial if net positive
                        if p_pnl > 0:
                            acc["equity"] += p_pnl

                            rec = _make_rec(
                                pos,
                                exit_px,
                                ts,
                                p_pnl,
                                "Partial",
                                p_lot
                            )

                            all_trades.append(rec)
                            acc["trades"].append(rec)

                            pos["lot_remaining"] = round(lot_rem - p_lot, 2)
                            pos["partial_done"] = True

                            # Real BE after commission
                            pos["sl"] = breakeven_price(pos["entry"], d, qr)

                            lot_rem = pos["lot_remaining"]

                            if lot_rem < C.min_lot:
                                positions[name] = None
                                continue

            lot_rem = pos["lot_remaining"]

            # Z-stop
            hit_z_stop = (
                not np.isnan(zn) and (
                    (d == 1 and zn <= -C.z_stop_margin) or
                    (d == -1 and zn >= C.z_stop_margin)
                )
            )

            # Z-exit
            hit_z_exit = False

            if not np.isnan(zn):
                z_crossed = (
                    (d == 1 and zn >= -C.z_exit_full) or
                    (d == -1 and zn <= C.z_exit_full)
                )

                if z_crossed:
                    mkt_exit_px = exit_execution_price(cp_mid, d, spread_pips)
                    pnl_check = calc_pnl(d, ep, mkt_exit_px, lot_rem, qr)

                    if pnl_check >= C.min_net_profit_usd or pos["partial_done"]:
                        hit_z_exit = True

            # SL / TP
            hit_sl = (
                (d == 1 and cp_mid <= pos["sl"]) or
                (d == -1 and cp_mid >= pos["sl"])
            )

            hit_tp = (
                (d == 1 and cp_mid >= pos["tp"]) or
                (d == -1 and cp_mid <= pos["tp"])
            )

            # Smart TimeStop
            bars_open = bar - pos["entry_bar"]
            mkt_exit_px = exit_execution_price(cp_mid, d, spread_pips)
            current_pos_pnl = calc_pnl(d, ep, mkt_exit_px, lot_rem, qr)

            hit_time_stop = (
                (bars_open >= C.time_stop_bars and current_pos_pnl < 0) or
                (bars_open >= C.time_stop_bars * 2)
            )

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or hit_time_stop:
                if hit_sl:
                    exit_px = pos["sl"]
                    status = "SL"
                elif hit_tp:
                    exit_px = pos["tp"]
                    status = "TP"
                elif hit_z_stop:
                    exit_px = exit_execution_price(cp_mid, d, spread_pips)
                    status = "Z-Stop"
                elif hit_time_stop:
                    exit_px = exit_execution_price(cp_mid, d, spread_pips)
                    status = "TimeStop"
                else:
                    exit_px = exit_execution_price(cp_mid, d, spread_pips)
                    status = "Z-Exit"

                final_pnl = calc_pnl(
                    d,
                    ep,
                    exit_px,
                    lot_rem,
                    qr
                )

                acc["equity"] += final_pnl

                rec = _make_rec(
                    pos,
                    exit_px,
                    ts,
                    final_pnl,
                    status,
                    lot_rem
                )

                all_trades.append(rec)
                acc["trades"].append(rec)
                positions[name] = None

                if final_pnl > 0:
                    acc["consec_loss"] = 0
                else:
                    acc["consec_loss"] += 1

        # Withdraw profit if target reached and all positions closed
        all_closed = all(positions[name] is None for name in pair_names)

        if acc["equity"] >= PROFIT_LEVEL and all_closed and not acc["blown"]:
            withdrawn = acc["equity"] - C.initial_balance
            total_withdrawn += withdrawn

            acc_logs.append({
                "account": acc_num,
                "start_ts": acc["start_ts"],
                "end_ts": ts,
                "reason": "TARGET_HIT",
                "pnl": withdrawn,
            })

            print(
                f"\n    💰 #{acc_num:>3} | {ts.date()} | "
                f"Target Hit: ${withdrawn:>7.2f} | "
                f"Total Bank: ${total_withdrawn:>9.2f}"
            )

            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc["equity"]

            for name in pair_names:
                trades_today[name] = 0
                pending_sig[name] = 0

            continue

        # Collect new signals for next-bar execution
        for name in pair_names:
            a = pa[name]

            if (
                positions[name] is None and
                not acc["blown"] and
                trades_today[name] < C.max_trades_day and
                a["sig"][bar] != 0
            ):
                pending_sig[name] = int(a["sig"][bar])

    print()

    return {
        "all_trades": all_trades,
        "account_logs": acc_logs,
        "eq_curve": eq_curve,
        "total_withdrawn": total_withdrawn,
        "final_equity": acc["equity"],
        "total_accounts": acc_num,
        "pair_names": pair_names,
    }


# ═══════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results: dict):
    trades = results["all_trades"]
    pair_names = results.get("pair_names", [])

    if not trades:
        print("\n❌ No trades executed.")
        return

    df_t = pd.DataFrame(trades)
    df_t["exit_ts"] = pd.to_datetime(df_t["exit_ts"])
    df_t["month"] = df_t["exit_ts"].dt.to_period("M")

    wins = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] < 0]

    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0.0

    if len(losses) > 0:
        pf = wins["pnl"].sum() / abs(losses["pnl"].sum())
    else:
        pf = float("inf")

    net_realized = df_t["pnl"].sum()

    print("\n" + "═" * 74)
    print(f" ▌  CorrArb Prop Simulator v9 — {'+'.join(pair_names)}  ▐")
    print("═" * 74)
    print(f" Total Trades:     {len(df_t):,}")
    print(f" Win Rate:         {wr:.2f}%")
    print(f" Profit Factor:    {pf:.2f}")
    print(f" Realized Net PnL: ${net_realized:,.2f}")
    print(f" Total Banked:     ${results['total_withdrawn']:,.2f}")
    print(f" Active Equity:    ${results['final_equity']:,.2f}")

    if len(wins):
        print(f" Avg Win:          ${wins['pnl'].mean():.2f}")

    if len(losses):
        print(f" Avg Loss:         ${losses['pnl'].mean():.2f}")

    # Per-pair
    if "pair" in df_t.columns and len(pair_names) > 1:
        print("-" * 74)
        print(" عملکرد هر جفت ارز:")

        for pair in pair_names:
            pt = df_t[df_t["pair"] == pair]

            if len(pt) == 0:
                continue

            pw = pt[pt["pnl"] > 0]
            pl = pt[pt["pnl"] < 0]

            p_wr = len(pw) / len(pt) * 100 if len(pt) else 0.0

            if len(pl) > 0:
                p_pf = pw["pnl"].sum() / abs(pl["pnl"].sum())
            else:
                p_pf = float("inf")

            print(
                f"   {pair}: {len(pt):>5} trades | "
                f"WR: {p_wr:5.1f}% | "
                f"PF: {p_pf:.2f} | "
                f"Net PnL: ${pt['pnl'].sum():>9,.2f}"
            )

    # Exit types
    print("-" * 74)
    print(" خروج‌ها بر اساس نوع:")

    for st, cnt in df_t["status"].value_counts().items():
        print(f"   {st:<12}: {cnt:>5} معامله")

    # Account logs
    logs = results["account_logs"]
    targets = sum(1 for l in logs if l["reason"] == "TARGET_HIT")
    blown = sum(1 for l in logs if l["reason"] != "TARGET_HIT")

    print("-" * 74)
    print(
        f" حساب‌ها: {results['total_accounts']} کل | "
        f"✅ Target Hit: {targets} | 💥 Blown: {blown}"
    )

    # Monthly
    monthly = df_t.groupby("month")["pnl"].sum()

    if len(monthly):
        pos_m = int((monthly > 0).sum())
        neg_m = int((monthly < 0).sum())
        avg_m = monthly.mean()
        best = monthly.max()
        worst = monthly.min()

        print(
            f" ماهانه   : avg ${avg_m:,.2f} | "
            f"Best: ${best:,.2f} | "
            f"Worst: ${worst:,.2f}"
        )
        print(f"            ماه‌های مثبت: {pos_m} | ماه‌های منفی: {neg_m}")

    # Optional: account log detail
    if logs:
        df_l = pd.DataFrame(logs)
        print("-" * 74)
        print(" خلاصه حساب‌ها:")
        print(
            f"   Avg Account PnL: ${df_l['pnl'].mean():,.2f} | "
            f"Best: ${df_l['pnl'].max():,.2f} | "
            f"Worst: ${df_l['pnl'].min():,.2f}"
        )

    print("═" * 74)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()

    pairs = load_all_pairs()

    print("\n  Computing Statistical Signals v9...")
    pair_signals = {}

    for name, info in pairs.items():
        pair_signals[name] = compute_signals(name, info)

    results = run_backtest(pairs, pair_signals)

    print_report(results)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n  ✅ Executed in: {elapsed:.2f}s")
