#!/usr/bin/env python3
"""
PropBot Backtester v1.1
══════════════════════════════════════════════════════════════
Walk-forward backtest — zero lookahead, zero overfitting
Strategy  : EMA20/50 trend + RSI14 filter + ATR14 SL/TP
Timeframe : H4  (resampled from M1 — better signal quality)
Pairs     : EURUSD, GBPUSD, XAUUSD, USDCAD, AUDUSD
            (trend-following works on these; range pairs excluded)
Split     : Train 2010-2019  |  OOS test 2020-2025
Costs     : Realistic spread + 0.5 pip slippage per side
Fix v1.1  : auto-detect CSV separator | UTC-aware timestamps
══════════════════════════════════════════════════════════════
"""

import sys, zipfile, io, csv, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAIRS — only where trend-following has structural edge
#  Excluded:  EURGBP (too range-bound)
#             AUDNZD (low volatility / choppy)
#             NZDUSD (correlated + less liquid)
#             USDCHF (correlated w/ EURUSD, adds no info)
#             XAGUSD (extreme spread kills edge)
# ─────────────────────────────────────────────────────────────
TARGET_PAIRS = ["EURUSD", "GBPUSD", "XAUUSD", "USDCAD", "AUDUSD"]

# ─────────────────────────────────────────────────────────────
#  FIXED PARAMETERS — set once, never tuned on test data
# ─────────────────────────────────────────────────────────────
CFG = dict(
    ema_fast      = 20,
    ema_slow      = 50,
    rsi_period    = 14,
    rsi_lo        = 40,    # RSI entry zone low
    rsi_hi        = 60,    # RSI entry zone high
    atr_period    = 14,
    atr_sl_mult   = 1.5,   # SL = 1.5 × ATR
    rr            = 1.8,   # TP = SL × 1.8
    risk_pct      = 0.01,  # 1% equity per trade
    max_open      = 2,
    daily_dd_lim  = 0.03,  # 3% daily → freeze
    max_dd_kill   = 0.07,  # 7% total → kill
    session_open  = 7,     # UTC
    session_close = 21,    # UTC
    initial_bal   = 10_000.0,
    slippage_pip  = 0.5,
    timeframe     = "4h",  # H4 — better trend quality than H1
)

SPREAD_PIPS = dict(
    EURUSD=1.2, GBPUSD=1.5, AUDUSD=1.4,
    USDCAD=1.8, XAUUSD=30,
)

PIP_SIZE = dict(
    EURUSD=1e-4, GBPUSD=1e-4, AUDUSD=1e-4,
    USDCAD=1e-4, XAUUSD=0.01,
)

# ── FIX: tz-aware timestamps for comparison with UTC index ──
TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ─────────────────────────────────────────────────────────────
#  DATA LOADING  — auto-detects semicolon or comma
# ─────────────────────────────────────────────────────────────
def _detect_sep(raw: bytes) -> str:
    """Sample first 300 bytes to detect field separator."""
    sample = raw[:300].decode("utf-8", errors="replace")
    return ";" if sample.count(";") > sample.count(",") else ","


def _parse_csv(raw: bytes) -> pd.DataFrame:
    sep = _detect_sep(raw)
    df = pd.read_csv(
        io.BytesIO(raw),
        sep=sep,
        header=None,
        names=["dt", "open", "high", "low", "close", "vol"],
        dtype={"open": float, "high": float, "low": float,
               "close": float, "vol": float},
        on_bad_lines="skip",
    )
    # HistData datetime: "20100103 170000"
    df["datetime"] = pd.to_datetime(
        df["dt"].astype(str).str.strip(),
        format="%Y%m%d %H%M%S",
        utc=True,
        errors="coerce",
    )
    df = (df.dropna(subset=["datetime"])
            .drop(columns=["dt", "vol"])
            .set_index("datetime")
            .sort_index())
    return df


def load_pair(data_dir: Path, pair: str) -> pd.DataFrame:
    frames = []

    # plain CSV  (DAT_ASCII_EURUSD_M1_YYYY.csv)
    for f in sorted(data_dir.glob(f"DAT_ASCII_{pair}_M1_*.csv")):
        try:
            frames.append(_parse_csv(f.read_bytes()))
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    # zipped CSV  (HISTDATA_COM_ASCII_XAUUSD_M1YYYY.zip)
    for f in sorted(data_dir.glob(f"HISTDATA_COM_ASCII_{pair}_M1*.zip")):
        try:
            with zipfile.ZipFile(f) as zf:
                inner = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if inner:
                    frames.append(_parse_csv(zf.read(inner[0])))
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # basic sanity: drop zero/negative prices
    df = df[(df[["open","high","low","close"]] > 0).all(axis=1)]
    return df


# ─────────────────────────────────────────────────────────────
#  INDICATORS  (pure pandas/numpy)
# ─────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(com=n-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n-1, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = CFG
    df = df.copy()
    df["ema_fast"] = _ema(df["close"], c["ema_fast"])
    df["ema_slow"] = _ema(df["close"], c["ema_slow"])
    df["rsi"]      = _rsi(df["close"], c["rsi_period"])
    df["atr"]      = _atr(df["high"], df["low"], df["close"], c["atr_period"])

    # trend direction (current bar)
    df["bull"] = df["ema_fast"] > df["ema_slow"]

    # entry condition: trend holds + price pulls back into EMA20 ± 0.4×ATR
    #                 + RSI in neutral zone (not already extended)
    near = (df["close"] - df["ema_fast"]).abs() <= 0.4 * df["atr"]
    rsi_ok = df["rsi"].between(c["rsi_lo"], c["rsi_hi"])

    raw_long  = df["bull"]  & near & rsi_ok
    raw_short = ~df["bull"] & near & rsi_ok

    # ── STRICT no-lookahead: signal on bar N → execute on bar N+1 open ──
    df["sig_long"]  = raw_long.shift(1).fillna(False)
    df["sig_short"] = raw_short.shift(1).fillna(False)

    return df.dropna(subset=["ema_fast","ema_slow","rsi","atr"])


# ─────────────────────────────────────────────────────────────
#  TRADE SIMULATION
# ─────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["d","entry","sl","tp","t0","risk_usd","pip"]
    def __init__(self, d, entry, sl, tp, t0, risk_usd, pip):
        self.d, self.entry, self.sl, self.tp = d, entry, sl, tp
        self.t0, self.risk_usd, self.pip = t0, risk_usd, pip


def _total_cost(pair: str) -> float:
    pip  = PIP_SIZE[pair]
    sp   = SPREAD_PIPS.get(pair, 2.0) * pip
    slp  = CFG["slippage_pip"] * pip * 2
    return sp + slp


def simulate(df: pd.DataFrame, pair: str):
    pip    = PIP_SIZE[pair]
    cost   = _total_cost(pair)
    c      = CFG

    equity   = c["initial_bal"]
    peak     = equity
    positions: list[Pos] = []
    trades   : list[dict] = []
    eq_curve : list[dict] = []

    day_eq   = equity
    last_day = None
    frozen   = False   # daily freeze flag

    for ts, row in df.iterrows():
        # ── daily reset ──
        d = ts.date()
        if d != last_day:
            day_eq   = equity
            last_day = d
            frozen   = False

        o, h, l = row["open"], row["high"], row["low"]

        # ── check existing positions ──
        closed = []
        for pos in positions:
            res = ep = None
            if pos.d == 1:
                if l <= pos.sl: ep, res = pos.sl, "sl"
                elif h >= pos.tp: ep, res = pos.tp, "tp"
            else:
                if h >= pos.sl: ep, res = pos.sl, "sl"
                elif l <= pos.tp: ep, res = pos.tp, "tp"

            # force-close at session end
            if res is None and ts.hour >= c["session_close"]:
                ep, res = o, "eod"

            if ep is not None:
                move    = (ep - pos.entry) * pos.d
                pnl_pip = move / pos.pip
                sl_pip  = abs(pos.entry - pos.sl) / pos.pip
                pnl_usd = (pnl_pip / sl_pip * pos.risk_usd) if sl_pip > 0 else 0
                # deduct cost (spread+slippage) proportional to risk
                pnl_usd -= (cost / pos.pip) * (pos.risk_usd / max(sl_pip, 1))
                equity  += pnl_usd
                peak     = max(peak, equity)
                trades.append(dict(
                    pair=pair, dir="long" if pos.d==1 else "short",
                    open_time=pos.t0, close_time=ts,
                    open_px=round(pos.entry,5), close_px=round(ep,5),
                    sl=round(pos.sl,5), tp=round(pos.tp,5),
                    pnl=round(pnl_usd,2), result=res,
                    equity=round(equity,2),
                ))
                closed.append(pos)

        for p in closed:
            positions.remove(p)

        eq_curve.append({"datetime": ts, "equity": equity})

        # ── kill switch ──
        if (peak - equity) / peak >= c["max_dd_kill"]:
            print(f"  [KILL] {pair} max DD hit at {ts.date()}")
            break

        # ── daily freeze ──
        if not frozen and (day_eq - equity) / day_eq >= c["daily_dd_lim"]:
            frozen = True

        if frozen:
            continue

        # ── session guard ──
        if not (c["session_open"] <= ts.hour < c["session_close"]):
            continue

        # ── new entries ──
        if len(positions) >= c["max_open"]:
            continue

        atr_v = row.get("atr", 0)
        if not (atr_v > 0) or pd.isna(atr_v):
            continue

        # Quality gate: ATR must dwarf the transaction cost
        if atr_v < cost * 2:
            continue

        sl_d = atr_v * c["atr_sl_mult"]
        tp_d = sl_d  * c["rr"]
        if sl_d / pip < 5:   # minimum 5-pip SL
            continue

        has_long  = any(p.d ==  1 for p in positions)
        has_short = any(p.d == -1 for p in positions)

        if row["sig_long"] and not has_long:
            entry = o + cost / 2
            positions.append(Pos(1, entry, entry-sl_d, entry+tp_d,
                                 ts, equity*c["risk_pct"], pip))

        elif row["sig_short"] and not has_short:
            entry = o - cost / 2
            positions.append(Pos(-1, entry, entry+sl_d, entry-tp_d,
                                 ts, equity*c["risk_pct"], pip))

    return trades, eq_curve


# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────
def calc_metrics(trades: list, eq_curve: list, label: str) -> dict:
    if not trades:
        return {"label": label, "trades": 0}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]

    eq_s  = pd.Series([e["equity"] for e in eq_curve])
    dd    = (eq_s.cummax() - eq_s) / eq_s.cummax() * 100
    max_dd = dd.max()

    net   = sum(pnls)
    gp    = sum(wins)
    gl    = abs(sum(loss))
    pf    = gp / gl if gl > 0 else float("inf")

    # Sharpe on monthly returns
    eq_df   = pd.DataFrame(eq_curve).set_index("datetime")["equity"]
    monthly = eq_df.resample("ME").last().pct_change().dropna()
    sharpe  = (monthly.mean() / monthly.std() * 12**0.5
               if len(monthly) >= 3 and monthly.std() > 0 else 0.0)

    ret_pct = net / CFG["initial_bal"] * 100
    calmar  = ret_pct / max_dd if max_dd > 0 else 0.0

    # consecutive losses
    best = cur = 0
    for p in pnls:
        cur = cur + 1 if p <= 0 else 0
        best = max(best, cur)

    return dict(
        label=label, trades=len(trades),
        win_rate=round(len(wins)/len(trades)*100, 1),
        net_pnl=round(net, 2), ret_pct=round(ret_pct, 2),
        pf=round(pf, 2), max_dd=round(max_dd, 2),
        sharpe=round(sharpe, 2), calmar=round(calmar, 2),
        avg_win=round(np.mean(wins),2) if wins else 0,
        avg_loss=round(np.mean(loss),2) if loss else 0,
        tp_count=sum(1 for t in trades if t["result"]=="tp"),
        sl_count=sum(1 for t in trades if t["result"]=="sl"),
        eod_count=sum(1 for t in trades if t["result"]=="eod"),
        max_consec_l=best,
    )


# ─────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────
def write_report(results: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    div = "═" * 72
    lines = [
        div,
        "  PropBot Backtest Report  v1.1",
        f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Strategy : EMA{CFG['ema_fast']}/{CFG['ema_slow']} + RSI{CFG['rsi_period']}"
        f" + ATR{CFG['atr_period']}  |  TF: {CFG['timeframe'].upper()}",
        f"  Risk     : {CFG['risk_pct']*100:.0f}%/trade  RR 1:{CFG['rr']}"
        f"  SL {CFG['atr_sl_mult']}×ATR",
        f"  Session  : {CFG['session_open']}:00–{CFG['session_close']}:00 UTC"
        f"  |  DailyDD freeze: {CFG['daily_dd_lim']*100:.0f}%"
        f"  |  MaxDD kill: {CFG['max_dd_kill']*100:.0f}%",
        f"  Pairs    : {', '.join(TARGET_PAIRS)}",
        div, "",
    ]

    hdr = (f"  {'Pair':<8} {'Trades':>7} {'WinR':>6} {'NetPnL':>9}"
           f" {'PF':>5} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7}"
           f" {'TP':>5} {'SL':>5}")
    sep = "  " + "─" * 70

    for section, key in [
        ("TRAIN  2010–2019  (in-sample)", "train"),
        ("OOS    2020–2025  (out-of-sample — what counts)", "oos"),
    ]:
        lines += ["", f"  ┌── {section} {'─'*(54-len(section))}┐", hdr, sep]
        total_pnl = total_trades = 0
        for r in results:
            m = r[key]
            if m.get("trades", 0) == 0:
                lines.append(f"  {r['pair']:<8} {'no trades':>7}")
                continue
            flag = " ⚠" if m["max_dd"] > 6 else ""
            lines.append(
                f"  {m['label']:<8} {m['trades']:>7} {m['win_rate']:>5.1f}%"
                f" {m['net_pnl']:>+9.0f} {m['pf']:>5.2f}"
                f" {m['max_dd']:>6.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}"
                f" {m['tp_count']:>5} {m['sl_count']:>5}{flag}"
            )
            total_pnl    += m["net_pnl"]
            total_trades += m["trades"]

        lines += [sep,
                  f"  {'TOTAL':<8} {total_trades:>7} {'':>6} {total_pnl:>+9.0f}",
                  f"  └{'─'*70}┘"]

    lines += [
        "", div, "  METHODOLOGY NOTES", div,
        "  • H4 bars resampled from M1 — reduces noise vs H1.",
        "  • Signal computed at bar-close; executed at NEXT bar-open (shift=1).",
        "  • Spread + 0.5-pip slippage deducted every trade (both sides).",
        "  • ATR must be ≥ 2× transaction cost — low-volatility bars skipped.",
        "  • Pairs chosen for structural trend-following suitability only.",
        "  • ⚠ = OOS max drawdown > 6% — caution before live use on that pair.",
        "",
    ]

    txt = "\n".join(lines)
    print(txt)
    (out_dir / "report.txt").write_text(txt)

    # per-pair trade CSVs
    for r in results:
        all_t = r["trades_train"] + r["trades_oos"]
        if not all_t:
            continue
        with open(out_dir / f"trades_{r['pair']}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_t[0].keys()))
            w.writeheader(); w.writerows(all_t)

    # summary JSON
    summary = [{"pair": r["pair"], "train": r["train"], "oos": r["oos"]}
               for r in results]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # equity CSVs
    for r in results:
        for tag, key in [("train", "eq_train"), ("oos", "eq_oos")]:
            rows = r.get(key, [])
            if not rows:
                continue
            with open(out_dir / f"equity_{r['pair']}_{tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["datetime","equity"])
                w.writeheader(); w.writerows(rows)

    # optional chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.dates as mdates

        fig, axes = plt.subplots(len(results), 2,
                                 figsize=(16, 4 * len(results)),
                                 squeeze=False)
        fig.suptitle("PropBot — Equity Curves per Pair", fontsize=13)

        for row_i, r in enumerate(results):
            for col_i, (key, title, color) in enumerate([
                ("eq_train", "Train 2010-2019", "#2563eb"),
                ("eq_oos",   "OOS  2020-2025",  "#16a34a"),
            ]):
                ax  = axes[row_i][col_i]
                eq  = r.get(key, [])
                if eq:
                    dts = [e["datetime"] for e in eq]
                    eqs = [e["equity"]   for e in eq]
                    ax.plot(dts, eqs, color=color, linewidth=0.7)
                    ax.axhline(CFG["initial_bal"], color="#9ca3af",
                               linewidth=0.5, linestyle="--")
                ax.set_title(f"{r['pair']} — {title}", fontsize=9)
                ax.set_ylabel("Equity ($)", fontsize=8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n  [OK] Chart → {out_dir / 'equity_curve.png'}")
    except ImportError:
        print("  [INFO] matplotlib not found — chart skipped")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir",  default="results")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    print(f"\n{'═'*52}")
    print(f"  PropBot Backtester v1.1")
    print(f"  Pairs    : {', '.join(TARGET_PAIRS)}")
    print(f"  Data dir : {data_dir.resolve()}")
    print(f"{'═'*52}\n")

    results = []

    for pair in TARGET_PAIRS:
        print(f"  ── {pair} ──────────────────────────────")
        print(f"     Loading...", end=" ", flush=True)
        raw = load_pair(data_dir, pair)

        if raw.empty:
            print("NO DATA — check filename in /data folder")
            continue
        print(f"{len(raw):,} M1 bars  "
              f"({raw.index[0].date()} → {raw.index[-1].date()})")

        # resample to configured timeframe
        tf  = CFG["timeframe"]
        agg = {"open":"first","high":"max","low":"min","close":"last"}
        print(f"     Resampling to {tf.upper()}...", end=" ", flush=True)
        bars = raw.resample(tf).agg(agg).dropna()
        print(f"{len(bars):,} bars")

        print(f"     Indicators...", end=" ", flush=True)
        bars = add_indicators(bars)
        print("done")

        train = bars[bars.index <= TRAIN_END]
        oos   = bars[bars.index >= TEST_START]
        print(f"     Train: {len(train):,}  |  OOS: {len(oos):,} bars")

        print(f"     Simulating TRAIN...", end=" ", flush=True)
        t_tr, eq_tr = simulate(train, pair)
        m_tr = calc_metrics(t_tr, eq_tr, pair)
        print(f"{m_tr.get('trades',0)} trades  "
              f"WR={m_tr.get('win_rate',0):.1f}%  "
              f"PnL={m_tr.get('net_pnl',0):+.0f}$")

        print(f"     Simulating OOS ...", end=" ", flush=True)
        t_oo, eq_oo = simulate(oos, pair)
        m_oo = calc_metrics(t_oo, eq_oo, pair)
        print(f"{m_oo.get('trades',0)} trades  "
              f"WR={m_oo.get('win_rate',0):.1f}%  "
              f"PnL={m_oo.get('net_pnl',0):+.0f}$")
        print()

        results.append(dict(pair=pair,
                            train=m_tr, oos=m_oo,
                            trades_train=t_tr, trades_oos=t_oo,
                            eq_train=eq_tr, eq_oos=eq_oo))

    if not results:
        print("[ERROR] No pairs processed.")
        sys.exit(1)

    print("Writing report...\n")
    write_report(results, out_dir)
    print(f"\n  Done — results in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
