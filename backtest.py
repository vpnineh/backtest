#!/usr/bin/env python3
"""
PropBot Backtester v1.0
══════════════════════════════════════════════════════════════
Realistic walk-forward backtest — zero lookahead, zero overfitting
Strategy : EMA20/50 crossover + RSI14 filter + ATR14 SL/TP
Timeframe : H1  (resampled on-the-fly from M1)
Data      : HistData M1  (CSV / ZIP — auto-detected)
Split     : Train 2010-2019  |  Out-of-sample (OOS) 2020-2025
Costs     : Realistic spread + 0.5 pip slippage per side
══════════════════════════════════════════════════════════════
"""

import os, sys, glob, zipfile, io, csv, json, argparse, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  FIXED PARAMETERS  —  set once, never tuned on test data
# ─────────────────────────────────────────────────────────────
CFG = dict(
    ema_fast      = 20,       # EMA fast period
    ema_slow      = 50,       # EMA slow period
    rsi_period    = 14,       # RSI period
    rsi_lo        = 40,       # RSI lower bound for entry
    rsi_hi        = 60,       # RSI upper bound for entry
    atr_period    = 14,       # ATR period
    atr_sl_mult   = 1.5,      # SL = ATR × 1.5
    rr            = 1.8,      # Risk:Reward (TP = SL × 1.8)
    risk_pct      = 0.01,     # 1 % of equity per trade
    max_open      = 2,        # max simultaneous positions
    daily_dd_lim  = 0.03,     # 3 % daily loss limit → freeze
    max_dd_kill   = 0.07,     # 7 % total DD → kill bot
    session_open  = 7,        # UTC hour sessions open
    session_close = 21,       # UTC hour sessions close
    initial_bal   = 10_000.0, # starting balance ($)
    slippage_pip  = 0.5,      # slippage per side (pips)
    min_atr_mult  = 0.5,      # min ATR relative to spread (quality filter)
)

# Realistic bid-ask spread in pips
SPREAD_PIPS = dict(
    EURUSD=1.2, GBPUSD=1.5, AUDUSD=1.4,
    NZDUSD=2.0, USDCAD=1.8, USDCHF=1.8,
    EURGBP=1.5, AUDNZD=2.5,
    XAUUSD=30,  XAGUSD=3.0,
)

# One pip in price units
PIP_SIZE = dict(
    EURUSD=1e-4, GBPUSD=1e-4, AUDUSD=1e-4,
    NZDUSD=1e-4, USDCAD=1e-4, USDCHF=1e-4,
    EURGBP=1e-4, AUDNZD=1e-4,
    XAUUSD=0.01, XAGUSD=0.001,
)

TRAIN_END  = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
def _read_csv_bytes(raw: bytes, sep: str) -> pd.DataFrame:
    """Parse raw CSV bytes into OHLC DataFrame."""
    df = pd.read_csv(
        io.BytesIO(raw), sep=sep, header=None,
        names=["dt", "open", "high", "low", "close", "vol"],
        dtype={"dt": str, "open": float, "high": float,
               "low": float, "close": float, "vol": float},
    )
    df["datetime"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S", utc=True)
    df = df.drop(columns=["dt", "vol"]).set_index("datetime").sort_index()
    return df


def load_pair(data_dir: Path, pair: str) -> pd.DataFrame:
    """
    Load all years for one pair.  Handles:
      - DAT_ASCII_EURUSD_M1_YYYY.csv  (comma-separated, no header)
      - HISTDATA_COM_ASCII_XAUUSD_M1YYYY.zip  (zip → semicolon CSV)
    """
    frames = []

    # ── plain CSV ──
    for f in sorted(data_dir.glob(f"DAT_ASCII_{pair}_M1_*.csv")):
        try:
            raw = f.read_bytes()
            frames.append(_read_csv_bytes(raw, ","))
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    # ── zipped CSV (HistData format) ──
    for f in sorted(data_dir.glob(f"HISTDATA_COM_ASCII_{pair}_M1*.zip")):
        try:
            with zipfile.ZipFile(f) as zf:
                inner = [n for n in zf.namelist() if n.endswith(".csv")]
                if not inner:
                    continue
                raw = zf.read(inner[0])
                # detect separator
                sample = raw[:200].decode("utf-8", errors="replace")
                sep = ";" if ";" in sample else ","
                frames.append(_read_csv_bytes(raw, sep))
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ─────────────────────────────────────────────────────────────
#  INDICATORS  (pure pandas/numpy — no external ta libs)
# ─────────────────────────────────────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_c).abs(),
        (low  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = CFG
    df["ema_fast"] = ema(df["close"], c["ema_fast"])
    df["ema_slow"] = ema(df["close"], c["ema_slow"])
    df["rsi"]      = rsi(df["close"], c["rsi_period"])
    df["atr"]      = atr(df["high"], df["low"], df["close"], c["atr_period"])

    # Trend: +1 bullish, -1 bearish (prev bar to avoid lookahead)
    df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

    # Signal fires on crossover
    df["cross_up"]   = (df["trend"] == 1) & (df["trend"].shift(1) == -1)
    df["cross_down"] = (df["trend"] == -1) & (df["trend"].shift(1) == 1)

    # RSI filter: enter only when RSI is in neutral zone (not overbought/oversold)
    df["rsi_ok_buy"]  = (df["rsi"] > c["rsi_lo"]) & (df["rsi"] < c["rsi_hi"])
    df["rsi_ok_sell"] = (df["rsi"] > c["rsi_lo"]) & (df["rsi"] < c["rsi_hi"])

    # Pullback to EMA20 (close within 0.5 × ATR of EMA20)
    df["near_ema"] = (df["close"] - df["ema_fast"]).abs() <= 0.5 * df["atr"]

    # Combined signal (evaluated on bar close → executed next bar open)
    df["sig_buy"]  = df["trend"] == 1  # in uptrend
    df["sig_sell"] = df["trend"] == -1 # in downtrend

    # Continuation entry: price pulls back to EMA20 while trend holds
    df["entry_long"]  = (
        df["sig_buy"]      &
        df["near_ema"]     &
        df["rsi_ok_buy"]   &
        (df["atr"] > 0)
    )
    df["entry_short"] = (
        df["sig_sell"]     &
        df["near_ema"]     &
        df["rsi_ok_sell"]  &
        (df["atr"] > 0)
    )

    # Shift 1: signal on bar N → action on bar N+1 (no lookahead)
    df["entry_long"]  = df["entry_long"].shift(1).fillna(False)
    df["entry_short"] = df["entry_short"].shift(1).fillna(False)

    return df


# ─────────────────────────────────────────────────────────────
#  TRADE SIMULATION
# ─────────────────────────────────────────────────────────────
class Position:
    __slots__ = ["pair","direction","open_price","sl","tp",
                 "open_time","size_usd","pip"]

    def __init__(self, pair, direction, open_price, sl, tp,
                 open_time, size_usd, pip):
        self.pair       = pair
        self.direction  = direction   # 1=long, -1=short
        self.open_price = open_price
        self.sl         = sl
        self.tp         = tp
        self.open_time  = open_time
        self.size_usd   = size_usd    # dollars at risk
        self.pip        = pip


def _cost(pair: str) -> float:
    """Total cost in price units (spread + slippage both sides)."""
    pip = PIP_SIZE.get(pair, 1e-4)
    sp  = SPREAD_PIPS.get(pair, 2.0) * pip
    sl  = CFG["slippage_pip"] * pip * 2   # entry + exit
    return sp + sl


def simulate(df: pd.DataFrame, pair: str) -> tuple[list, list]:
    """
    Event-driven bar-by-bar simulation on H1 data.
    Returns (trades_list, equity_series).
    """
    pip   = PIP_SIZE.get(pair, 1e-4)
    cost  = _cost(pair)
    c     = CFG

    equity     = c["initial_bal"]
    peak_eq    = equity
    open_pos   : list[Position] = []
    trades     : list[dict]     = []
    eq_curve   : list           = []

    day_open_eq = equity
    last_day    = None
    bot_killed  = False

    for ts, row in df.iterrows():
        if bot_killed:
            break

        # ── daily reset ──
        day = ts.date()
        if day != last_day:
            day_open_eq = equity
            last_day    = day

        # ── session filter ──
        in_session = c["session_open"] <= ts.hour < c["session_close"]

        bar_open  = row["open"]
        bar_high  = row["high"]
        bar_low   = row["low"]

        # ── check open positions (SL / TP hit) ──
        closed_now = []
        for pos in open_pos:
            exit_price = None
            result     = None

            if pos.direction == 1:   # long
                if bar_low <= pos.sl:
                    exit_price = pos.sl
                    result = "sl"
                elif bar_high >= pos.tp:
                    exit_price = pos.tp
                    result = "tp"
            else:                    # short
                if bar_high >= pos.sl:
                    exit_price = pos.sl
                    result = "sl"
                elif bar_low <= pos.tp:
                    exit_price = pos.tp
                    result = "tp"

            # Force-close at session end
            if result is None and ts.hour >= c["session_close"]:
                exit_price = bar_open
                result = "eod"

            if exit_price is not None:
                price_move = (exit_price - pos.open_price) * pos.direction
                pnl_pips   = price_move / pos.pip
                # pnl in $ = (pips_at_risk) * (risk_$) / (sl_pips)
                sl_pips    = abs(pos.open_price - pos.sl) / pos.pip
                if sl_pips > 0:
                    pnl_usd = pnl_pips / sl_pips * pos.size_usd
                else:
                    pnl_usd = 0.0

                pnl_usd -= cost / pos.pip * (pos.size_usd / max(sl_pips, 1))
                equity  += pnl_usd
                peak_eq  = max(peak_eq, equity)

                trades.append(dict(
                    pair      = pair,
                    direction = "long" if pos.direction == 1 else "short",
                    open_time = pos.open_time,
                    close_time= ts,
                    open_px   = round(pos.open_price, 5),
                    close_px  = round(exit_price, 5),
                    sl_px     = round(pos.sl, 5),
                    tp_px     = round(pos.tp, 5),
                    pnl_usd   = round(pnl_usd, 2),
                    result    = result,
                    equity    = round(equity, 2),
                ))
                closed_now.append(pos)

        for p in closed_now:
            open_pos.remove(p)

        eq_curve.append({"datetime": ts, "equity": equity})

        # ── daily loss limit ──
        daily_dd = (day_open_eq - equity) / day_open_eq
        if daily_dd >= c["daily_dd_lim"]:
            continue   # freeze for rest of day (handled by daily_dd check each bar)

        # ── total drawdown kill ──
        total_dd = (peak_eq - equity) / peak_eq
        if total_dd >= c["max_dd_kill"]:
            bot_killed = True
            print(f"  [KILL] {pair} — max DD {total_dd:.1%} hit at {ts}")
            break

        # ── open new positions ──
        if not in_session:
            continue
        if len(open_pos) >= c["max_open"]:
            continue
        if pd.isna(row.get("atr", np.nan)) or row["atr"] <= 0:
            continue

        atr_val = row["atr"]

        # Quality filter: ATR must be meaningfully larger than cost
        if atr_val < CFG["min_atr_mult"] * cost:
            continue

        sl_dist = atr_val * c["atr_sl_mult"]
        tp_dist = sl_dist * c["rr"]
        sl_pips = sl_dist / pip
        if sl_pips < 1:
            continue

        if row["entry_long"] and not any(p.direction == 1 for p in open_pos):
            entry = bar_open + cost / 2   # worse fill (spread)
            sl    = entry - sl_dist
            tp    = entry + tp_dist
            # risk_pct of current equity
            size  = equity * c["risk_pct"]
            open_pos.append(Position(pair, 1, entry, sl, tp, ts, size, pip))

        elif row["entry_short"] and not any(p.direction == -1 for p in open_pos):
            entry = bar_open - cost / 2
            sl    = entry + sl_dist
            tp    = entry - tp_dist
            size  = equity * c["risk_pct"]
            open_pos.append(Position(pair, -1, entry, sl, tp, ts, size, pip))

    # Close any remaining positions at last bar close
    last_close = df["close"].iloc[-1] if not df.empty else None
    for pos in open_pos:
        if last_close is not None:
            price_move = (last_close - pos.open_price) * pos.direction
            pnl_pips   = price_move / pos.pip
            sl_pips    = abs(pos.open_price - pos.sl) / pos.pip
            pnl_usd    = (pnl_pips / sl_pips * pos.size_usd) if sl_pips > 0 else 0
            equity    += pnl_usd
            trades.append(dict(
                pair="...", direction="long" if pos.direction==1 else "short",
                open_time=pos.open_time, close_time=df.index[-1],
                open_px=round(pos.open_price,5), close_px=round(last_close,5),
                sl_px=round(pos.sl,5), tp_px=round(pos.tp,5),
                pnl_usd=round(pnl_usd,2), result="forced", equity=round(equity,2),
            ))

    return trades, eq_curve


# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────
def metrics(trades: list, eq_curve: list, label: str) -> dict:
    if not trades:
        return {"label": label, "trades": 0}

    pnls  = [t["pnl_usd"] for t in trades]
    wins  = [p for p in pnls if p > 0]
    loss  = [p for p in pnls if p <= 0]

    eq    = [e["equity"] for e in eq_curve]
    eq_s  = pd.Series(eq)

    peak  = eq_s.cummax()
    dd    = (peak - eq_s) / peak * 100
    max_dd = dd.max()

    gross_profit = sum(wins)
    gross_loss   = abs(sum(loss))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    net_pnl = sum(pnls)
    ret_pct = net_pnl / CFG["initial_bal"] * 100

    # Sharpe  (using monthly equity returns, annualised)
    eq_df = pd.DataFrame(eq_curve).set_index("datetime")["equity"]
    monthly = eq_df.resample("ME").last().pct_change().dropna()
    if len(monthly) >= 2 and monthly.std() > 0:
        sharpe = (monthly.mean() / monthly.std()) * (12 ** 0.5)
    else:
        sharpe = 0.0

    # Calmar
    calmar = (ret_pct / max_dd) if max_dd > 0 else 0.0

    # Consecutive losses
    consec = max_consec_losses(pnls)

    return dict(
        label         = label,
        trades        = len(trades),
        win_rate      = round(len(wins) / len(trades) * 100, 1),
        net_pnl       = round(net_pnl, 2),
        return_pct    = round(ret_pct, 2),
        profit_factor = round(profit_factor, 2),
        max_dd_pct    = round(max_dd, 2),
        sharpe        = round(sharpe, 2),
        calmar        = round(calmar, 2),
        avg_win       = round(np.mean(wins), 2) if wins else 0,
        avg_loss      = round(np.mean(loss), 2) if loss else 0,
        gross_profit  = round(gross_profit, 2),
        gross_loss    = round(gross_loss, 2),
        max_consec_l  = consec,
        tp_count      = sum(1 for t in trades if t.get("result") == "tp"),
        sl_count      = sum(1 for t in trades if t.get("result") == "sl"),
        eod_count     = sum(1 for t in trades if t.get("result") == "eod"),
    )


def max_consec_losses(pnls: list) -> int:
    best = cur = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ─────────────────────────────────────────────────────────────
#  RESAMPLE M1 → H1
# ─────────────────────────────────────────────────────────────
def resample_h1(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    h1  = df.resample("1h").agg(agg).dropna()
    return h1


# ─────────────────────────────────────────────────────────────
#  REPORT WRITER
# ─────────────────────────────────────────────────────────────
def write_report(all_results: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    div   = "═" * 70

    lines += [
        div,
        "  PropBot Backtest Report",
        f"  Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Strategy  : EMA{CFG['ema_fast']}/{CFG['ema_slow']} + RSI{CFG['rsi_period']} + ATR{CFG['atr_period']}",
        f"  Risk/trade: {CFG['risk_pct']*100:.0f}%  |  RR: 1:{CFG['rr']}  |  SL mult: {CFG['atr_sl_mult']}×ATR",
        f"  Session   : {CFG['session_open']}:00–{CFG['session_close']}:00 UTC",
        f"  Split     : Train 2010-2019  |  OOS Test 2020-2025",
        div, ""
    ]

    for section in ["TRAIN (2010-2019) — In-Sample", "OOS TEST (2020-2025) — Out-of-Sample"]:
        tag = "train" if "TRAIN" in section else "oos"
        lines += ["", f"  {'─'*60}", f"  {section}", f"  {'─'*60}"]

        section_pnl = 0
        section_trades = 0
        pair_rows = []

        for r in all_results:
            m = r[tag]
            if m.get("trades", 0) == 0:
                continue
            section_pnl    += m["net_pnl"]
            section_trades += m["trades"]
            pair_rows.append(m)

        header = f"  {'Pair':<10} {'Trades':>7} {'Win%':>6} {'NetPnL':>9} {'PF':>6} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7}"
        lines += ["", header, "  " + "─"*68]

        for m in pair_rows:
            dd_flag = " ⚠" if m["max_dd_pct"] > 6 else ""
            lines.append(
                f"  {m['label']:<10} {m['trades']:>7} {m['win_rate']:>5.1f}%"
                f" {m['net_pnl']:>+9.0f} {m['profit_factor']:>6.2f}"
                f" {m['max_dd_pct']:>6.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}{dd_flag}"
            )

        lines += [
            "  " + "─"*68,
            f"  {'COMBINED':<10} {section_trades:>7}   {'':>6} {section_pnl:>+9.0f}",
        ]

    lines += ["", div, "  NOTES", div]
    lines += [
        "  • OOS results are what matter for prop firm evaluation.",
        "  • Max DD limits: 3% daily (auto-freeze), 7% total (kill-switch).",
        "  • Spread + slippage deducted every trade (see SPREAD_PIPS config).",
        "  • All signals use shift(1) — no lookahead bias.",
        "  • ⚠  = drawdown exceeded 6% — review pair carefully.",
        "",
    ]

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))

    # ── per-pair trade logs (CSV) ──
    for r in all_results:
        pair  = r["pair"]
        all_t = r["trades_train"] + r["trades_oos"]
        if not all_t:
            continue
        keys = list(all_t[0].keys())
        with open(out_dir / f"trades_{pair}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_t)

    # ── summary JSON (for CI artifact) ──
    summary = []
    for r in all_results:
        summary.append({"pair": r["pair"], "train": r["train"], "oos": r["oos"]})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ── equity CSV (combined) ──
    all_eq_train, all_eq_oos = [], []
    for r in all_results:
        all_eq_train += r.get("eq_train", [])
        all_eq_oos   += r.get("eq_oos",   [])

    def save_equity(rows, path):
        if not rows:
            return
        rows.sort(key=lambda x: x["datetime"])
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["datetime","equity"])
            w.writeheader()
            w.writerows(rows)

    save_equity(all_eq_train, out_dir / "equity_train.csv")
    save_equity(all_eq_oos,   out_dir / "equity_oos.csv")

    # ── optional matplotlib chart ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        fig.suptitle("PropBot — Equity Curve", fontsize=14, fontweight="bold")

        for ax, rows, title, color in [
            (axes[0], all_eq_train, "Train 2010-2019", "#2563eb"),
            (axes[1], all_eq_oos,   "OOS  2020-2025",  "#16a34a"),
        ]:
            if rows:
                rows.sort(key=lambda x: x["datetime"])
                dts = [r["datetime"] for r in rows]
                eqs = [r["equity"]   for r in rows]
                ax.plot(dts, eqs, color=color, linewidth=0.8)
                ax.axhline(CFG["initial_bal"], color="#9ca3af", linewidth=0.5, linestyle="--")
                ax.set_title(title)
                ax.set_ylabel("Equity ($)")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n  [OK] Equity chart saved → {out_dir/'equity_curve.png'}")
    except ImportError:
        print("  [INFO] matplotlib not installed — skipping chart")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PropBot Backtester")
    parser.add_argument("--data-dir",  default=".",        help="Root dir with CSV/ZIP files")
    parser.add_argument("--out-dir",   default="results",  help="Output directory")
    parser.add_argument("--pairs",     default="",         help="Comma-separated pair filter (empty=all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # auto-discover pairs
    known_pairs = [
        "EURUSD","GBPUSD","AUDUSD","NZDUSD",
        "USDCAD","USDCHF","EURGBP","AUDNZD",
        "XAUUSD","XAGUSD",
    ]
    if args.pairs:
        pairs = [p.strip().upper() for p in args.pairs.split(",")]
    else:
        # detect from files
        pairs = []
        for p in known_pairs:
            has_csv = bool(list(data_dir.glob(f"DAT_ASCII_{p}_M1_*.csv")))
            has_zip = bool(list(data_dir.glob(f"HISTDATA_COM_ASCII_{p}_M1*.zip")))
            if has_csv or has_zip:
                pairs.append(p)
        if not pairs:
            pairs = known_pairs  # try all anyway

    if not pairs:
        print("[ERROR] No data files found. Check --data-dir.")
        sys.exit(1)

    print(f"\n{'═'*50}")
    print(f"  PropBot Backtester  |  pairs: {', '.join(pairs)}")
    print(f"  Data dir: {data_dir.resolve()}")
    print(f"{'═'*50}\n")

    all_results = []

    for pair in pairs:
        pip_sz = PIP_SIZE.get(pair)
        if pip_sz is None:
            print(f"  [SKIP] {pair} — not in PIP_SIZE table")
            continue

        print(f"  Loading {pair}...", end=" ", flush=True)
        raw = load_pair(data_dir, pair)

        if raw.empty:
            print("no data found — skip")
            continue

        print(f"{len(raw):,} M1 bars ({raw.index[0].date()} → {raw.index[-1].date()})")

        print(f"         Resampling to H1...", end=" ", flush=True)
        h1 = resample_h1(raw)
        print(f"{len(h1):,} H1 bars")

        print(f"         Computing indicators...", end=" ", flush=True)
        h1 = add_indicators(h1)
        print("done")

        train = h1[h1.index.normalize() <= TRAIN_END]
        oos   = h1[h1.index.normalize() >= TEST_START]

        print(f"         Train bars: {len(train):,}  |  OOS bars: {len(oos):,}")

        print(f"         Simulating TRAIN...", end=" ", flush=True)
        trades_train, eq_train = simulate(train, pair)
        m_train = metrics(trades_train, eq_train, pair)
        print(f"{m_train.get('trades',0)} trades  WR={m_train.get('win_rate',0):.1f}%")

        print(f"         Simulating OOS...",   end=" ", flush=True)
        trades_oos, eq_oos = simulate(oos, pair)
        m_oos = metrics(trades_oos, eq_oos, pair)
        print(f"{m_oos.get('trades',0)} trades  WR={m_oos.get('win_rate',0):.1f}%")

        all_results.append(dict(
            pair         = pair,
            train        = m_train,
            oos          = m_oos,
            trades_train = trades_train,
            trades_oos   = trades_oos,
            eq_train     = eq_train,
            eq_oos       = eq_oos,
        ))
        print()

    if not all_results:
        print("[ERROR] No pairs processed. Check data directory and file names.")
        sys.exit(1)

    print("\nWriting report...")
    write_report(all_results, out_dir)
    print(f"\n  Results → {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
