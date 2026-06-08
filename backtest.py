#!/usr/bin/env python3
"""
PropBot Backtester v4.0  —  Execution Timeframe Optimization
══════════════════════════════════════════════════════════════════
Strategy : London Breakout (Hyper-Optimized GBPUSD + BE)
Timeframes: M5, M15, H1
Account  : $5,000 prop firm
Split    : Train 2010-2019  |  OOS 2020-2025
Costs    : Dynamic Spread + Breakout Slippage included
Logic    : 0.5% Risk | R:R 1.5 | Break-Even at 1.0R
══════════════════════════════════════════════════════════════════
"""

import sys, zipfile, io, csv, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  ACCOUNT  —  $5K prop firm (Risk adjusted for DD control)
# ─────────────────────────────────────────────────────────────
ACCOUNT = dict(
    initial_bal  = 5_000.0,
    risk_pct     = 0.005,     # Risk 0.5%
    max_open     = 2,
    daily_dd_lim = 0.03,      
    max_dd_kill  = 0.07,      
)

# ─────────────────────────────────────────────────────────────
#  PESSIMISTIC COST CONFIGURATION
# ─────────────────────────────────────────────────────────────
SPREAD_PIPS = dict(
    EURUSD=1.2, GBPUSD=1.5, AUDUSD=1.4, USDCAD=1.8, XAUUSD=30,
)
PIP_SIZE = dict(
    EURUSD=1e-4, GBPUSD=1e-4, AUDUSD=1e-4, USDCAD=1e-4, XAUUSD=0.01,
)

SLIPPAGE_PIP_STD = 0.5        
SLIPPAGE_PIP_LBO = 1.5        # Heavy slippage for stop orders
LONDON_SPREAD_MULTIPLIER = 2.5 # Morning liquidity sweep simulation

TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ─────────────────────────────────────────────────────────────
#  STRATEGY PARAMS (LONDON BREAKOUT ONLY)
# ─────────────────────────────────────────────────────────────
LBO_CFG = dict(
    pairs         = ["GBPUSD"], # Laser focus on the Cable
    asian_start   = 0,    
    asian_end     = 7,    
    entry_open    = 7,    
    entry_close   = 10,   
    force_close   = 16,       
    min_range_pip = 6,        
    max_range_pip = 45,       
    buffer_pip    = 2,    
    sl_inside_pip = 3,    
    rr            = 1.5,      # Realistic targets
)

# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────
def _detect_sep(raw: bytes) -> str:
    s = raw[:400].decode("utf-8", errors="replace")
    return ";" if s.count(";") > s.count(",") else ","

def _parse(raw: bytes) -> pd.DataFrame:
    sep = _detect_sep(raw)
    df = pd.read_csv(
        io.BytesIO(raw), sep=sep, header=None,
        names=["dt","open","high","low","close","vol"],
        dtype={"open":float,"high":float,"low":float,"close":float,"vol":float},
        on_bad_lines="skip",
    )
    df["datetime"] = pd.to_datetime(
        df["dt"].astype(str).str.strip(),
        format="%Y%m%d %H%M%S", utc=True, errors="coerce",
    )
    return (df.dropna(subset=["datetime"])
              .drop(columns=["dt","vol"])
              .set_index("datetime")
              .sort_index())

def load_pair(data_dir: Path, pair: str) -> pd.DataFrame:
    frames = []
    for f in sorted(data_dir.glob(f"DAT_ASCII_{pair}_M1_*.csv")):
        try: frames.append(_parse(f.read_bytes()))
        except Exception as e: print(f"  [W] {f.name}: {e}")
    for f in sorted(data_dir.glob(f"HISTDATA_COM_ASCII_{pair}_M1*.zip")):
        try:
            with zipfile.ZipFile(f) as z:
                inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if inner: frames.append(_parse(z.read(inner[0])))
        except Exception as e: print(f"  [W] {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df[["open","high","low","close"]] > 0).all(axis=1)]
    return df

def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    agg = {"open":"first","high":"max","low":"min","close":"last"}
    return df.resample(tf).agg(agg).dropna()

# ─────────────────────────────────────────────────────────────
#  TRADE ENGINE  (Pessimistic + Fast Break-Even Logic)
# ─────────────────────────────────────────────────────────────
def get_trade_cost(pair: str, ts: pd.Timestamp) -> float:
    pip = PIP_SIZE[pair]
    base_spread = SPREAD_PIPS.get(pair, 2.0)
    
    if ts.hour == 7:
        spread = base_spread * LONDON_SPREAD_MULTIPLIER
    else:
        spread = base_spread
        
    slippage = SLIPPAGE_PIP_LBO
    return (spread + slippage * 2) * pip

class Pos:
    __slots__ = ["d","entry","sl","tp","t0","risk_usd","pip","c_cost","initial_risk","be_triggered"]
    def __init__(self, d, entry, sl, tp, t0, risk_usd, pip, c_cost):
        self.d, self.entry, self.sl, self.tp = d, entry, sl, tp
        self.t0, self.risk_usd, self.pip = t0, risk_usd, pip
        self.c_cost = c_cost 
        self.initial_risk = abs(entry - sl)
        self.be_triggered = False

def _close_pos(pos, ep, ts, pair):
    move    = (ep - pos.entry) * pos.d
    pnl_pip = move / pos.pip
    sl_pip  = abs(pos.entry - pos.sl) / pos.pip
    if sl_pip < 1e-5: return 0.0
    
    initial_sl_pip = pos.initial_risk / pos.pip
    pnl_usd = pnl_pip / initial_sl_pip * pos.risk_usd
    pnl_usd -= (pos.c_cost / pos.pip) * (pos.risk_usd / initial_sl_pip)
    return pnl_usd

def run_sim(bars: pd.DataFrame, pair: str, signals: pd.DataFrame,
            force_close_hour: int = 20) -> tuple:
    pip      = PIP_SIZE[pair]
    acct     = ACCOUNT

    equity   = acct["initial_bal"]
    peak     = equity
    positions: list[Pos] = []
    trades   : list[dict] = []
    eq_curve : list[dict] = []

    day_eq   = equity
    last_day = None
    frozen   = False
    killed   = False

    for ts in bars.index:
        if killed: break
        row = bars.loc[ts]
        sig = signals.loc[ts] if ts in signals.index else None

        d = ts.date()
        if d != last_day:
            day_eq, last_day, frozen = equity, d, False

        o, h, l = row["open"], row["high"], row["low"]

        # ── 1. Close positions ──
        closed = []
        for pos in positions:
            ep = res = None
            force_exit = (sig is not None and
                          ts.hour >= force_close_hour and
                          force_close_hour < 23)

            if pos.d == 1:
                if l <= pos.sl: ep, res = pos.sl, "sl"
                elif h >= pos.tp: ep, res = pos.tp, "tp"
            else:
                if h >= pos.sl: ep, res = pos.sl, "sl"
                elif l <= pos.tp: ep, res = pos.tp, "tp"

            if res is None and force_exit:
                ep, res = o, "eod"

            if ep is not None:
                pnl = _close_pos(pos, ep, ts, pair)
                equity += pnl
                peak    = max(peak, equity)
                
                # Tag BE hits
                if res == "sl" and pos.be_triggered:
                    res = "be"
                    
                trades.append(dict(
                    pair=pair, dir="long" if pos.d==1 else "short",
                    open_time=str(pos.t0), close_time=str(ts),
                    open_px=round(pos.entry,5), close_px=round(ep,5),
                    sl=round(pos.sl,5), tp=round(pos.tp,5),
                    cost_pips=round(pos.c_cost/pip, 1), 
                    pnl=round(pnl,2), result=res, equity=round(equity,2),
                ))
                closed.append(pos)
                
        for p in closed: positions.remove(p)

        # ── 2. Fast Break-Even Logic (At 1.0R) ──
        for pos in positions:
            if not pos.be_triggered:
                if pos.d == 1 and row["high"] >= pos.entry + 1.0 * pos.initial_risk:
                    pos.sl = pos.entry + pos.c_cost
                    pos.be_triggered = True
                elif pos.d == -1 and row["low"] <= pos.entry - 1.0 * pos.initial_risk:
                    pos.sl = pos.entry - pos.c_cost
                    pos.be_triggered = True

        eq_curve.append({"datetime": ts, "equity": equity})

        # ── 3. Kill / Freeze Check ──
        if (peak - equity) / peak >= acct["max_dd_kill"]:
            killed = True
            print(f"  [KILL] {pair} DD={((peak-equity)/peak*100):.1f}% at {ts.date()}")
            break
        if not frozen and (day_eq - equity) / max(day_eq,1) >= acct["daily_dd_lim"]:
            frozen = True
        if frozen or sig is None: continue

        # ── 4. Open new positions ──
        if len(positions) >= acct["max_open"]: continue

        has_long  = any(p.d== 1 for p in positions)
        has_short = any(p.d==-1 for p in positions)
        risk_usd  = equity * acct["risk_pct"]

        if sig.get("sig_long") and not has_long:
            c_cost = get_trade_cost(pair, ts)
            entry = o + c_cost / 2
            sl, tp = sig["force_sl"], sig["force_tp_l"]
            if abs(entry - sl) / pip < 5: continue
            positions.append(Pos(1, entry, sl, tp, ts, risk_usd, pip, c_cost))

        elif sig.get("sig_short") and not has_short:
            c_cost = get_trade_cost(pair, ts)
            entry = o - c_cost / 2
            sl, tp = sig["force_sl"], sig["force_tp_s"]
            if abs(entry - sl) / pip < 5: continue
            positions.append(Pos(-1, entry, sl, tp, ts, risk_usd, pip, c_cost))

    return trades, eq_curve

# ─────────────────────────────────────────────────────────────
#  STRATEGY B  —  London Breakout (Dynamic Timeframe)
# ─────────────────────────────────────────────────────────────
def strategy_lbo(m1: pd.DataFrame, pair: str, tf_code: str) -> tuple:
    cfg = LBO_CFG
    pip = PIP_SIZE[pair]

    bars = resample(m1, tf_code)
    bars["date"] = bars.index.date

    asian_mask = (bars.index.hour >= cfg["asian_start"]) & \
                 (bars.index.hour <  cfg["asian_end"])
    asian = bars[asian_mask].groupby("date").agg(
        asian_high=("high","max"),
        asian_low =("low","min"),
    )
    asian["range_pip"] = (asian["asian_high"] - asian["asian_low"]) / pip

    asian = asian[asian["range_pip"].between(cfg["min_range_pip"], cfg["max_range_pip"])]

    london_mask = (bars.index.hour >= cfg["entry_open"]) & \
                  (bars.index.hour <= cfg["entry_close"])
    london = bars[london_mask].copy()
    london["date"] = london.index.date
    london = london.join(asian, on="date", how="inner")
    london = london.dropna(subset=["asian_high","asian_low"])

    buf      = cfg["buffer_pip"] * pip
    sl_buf   = cfg["sl_inside_pip"] * pip

    # Trigger based on the CLOSE of the specific timeframe
    london["raw_long"]  = london["close"] > (london["asian_high"] + buf)
    london["raw_short"] = london["close"] < (london["asian_low"]  - buf)

    london["force_sl"]   = np.where(
        london["raw_long"],
        london["asian_high"] - sl_buf,
        london["asian_low"]  + sl_buf,
    )
    r_pip = london["asian_high"] - london["asian_low"]
    london["force_tp_l"] = london["asian_high"] + buf + r_pip * cfg["rr"]
    london["force_tp_s"] = london["asian_low"]  - buf - r_pip * cfg["rr"]

    sigs = london[["raw_long","raw_short","force_sl","force_tp_l","force_tp_s"]].copy()
    sigs["sig_long"]  = sigs["raw_long"].shift(1).fillna(False)
    sigs["sig_short"] = sigs["raw_short"].shift(1).fillna(False)

    sigs["date"] = sigs.index.date
    sigs = sigs[~sigs.duplicated(subset=["date","sig_long","sig_short"], keep="first")]

    sigs_full = sigs.reindex(bars.index)
    sigs_full[["sig_long","sig_short"]] = sigs_full[["sig_long","sig_short"]].fillna(False)

    bars_train = bars[bars.index <= TRAIN_END]
    bars_oos   = bars[bars.index >= TEST_START]
    s_train  = sigs_full[sigs_full.index <= TRAIN_END]
    s_oos    = sigs_full[sigs_full.index >= TEST_START]

    t_tr, eq_tr = run_sim(bars_train, pair, s_train, cfg["force_close"])
    t_oo, eq_oo = run_sim(bars_oos,   pair, s_oos,   cfg["force_close"])
    return t_tr, eq_tr, t_oo, eq_oo

# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────
def metrics(trades, eq_curve, label):
    if not trades:
        return dict(label=label, trades=0, net_pnl=0, win_rate=0,
                    pf=0, max_dd=0, sharpe=0, calmar=0,
                    tp_count=0, sl_count=0, be_count=0, eod_count=0, max_consec_l=0)
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    eqs  = pd.Series([e["equity"] for e in eq_curve])
    dd   = (eqs.cummax() - eqs) / eqs.cummax() * 100
    net  = sum(pnls)
    gp   = sum(wins); gl = abs(sum(loss))
    pf   = gp/gl if gl>0 else float("inf")

    eq_df   = pd.DataFrame(eq_curve).set_index("datetime")["equity"]
    monthly = eq_df.resample("ME").last().pct_change().dropna()
    sharpe  = (monthly.mean()/monthly.std()*12**0.5
               if len(monthly)>=3 and monthly.std()>0 else 0.0)
    ret_pct = net / ACCOUNT["initial_bal"] * 100
    calmar  = ret_pct / dd.max() if dd.max()>0 else 0.0

    best=cur=0
    for p in pnls:
        cur = cur+1 if p<=0 else 0; best=max(best,cur)

    return dict(
        label=label, trades=len(trades),
        win_rate=round(len(wins)/len(trades)*100,1),
        net_pnl=round(net,2), ret_pct=round(ret_pct,2),
        pf=round(pf,2), max_dd=round(dd.max(),2),
        sharpe=round(sharpe,2), calmar=round(calmar,2),
        tp_count=sum(1 for t in trades if t["result"]=="tp"),
        sl_count=sum(1 for t in trades if t["result"]=="sl"),
        be_count=sum(1 for t in trades if t["result"]=="be"),
        eod_count=sum(1 for t in trades if t["result"]=="eod"),
        max_consec_l=best,
        avg_win=round(np.mean(wins),2) if wins else 0,
        avg_loss=round(np.mean(loss),2) if loss else 0,
    )

# ─────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────
def write_report(tf_results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    div = "═"*72
    lines = [
        div,
        f"  PropBot Backtester v4.0  —  Account: $5,000",
        f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Split : Train 2010-2019  |  OOS 2020-2025",
        f"  Costs : Realistic Spread & Slippage Penalty",
        f"  Focus : GBPUSD Execution Timeframe Comparison",
        div,"",
    ]

    HDR = (f"  {'Timeframe':<10} {'#':>5} {'WinR':>6} {'NetPnL':>9}"
           f" {'PF':>5} {'MaxDD':>7} {'Sharpe':>7}"
           f" {'TP':>4} {'SL':>3} {'BE':>3} {'EOD':>3}")
    SEP = "  " + "─"*68

    lines += [f"  ┌── OOS PERFORMANCE (2020-2025) {'─'*33}┐", HDR, SEP]
    
    best_pnl = -float('inf')
    best_tf = ""
    
    for r in tf_results:
        m = r["oos"]
        flag = " ⚠" if m["max_dd"] > 6 else ""
        lines.append(
            f"  {m['label']:<10} {m['trades']:>5} {m['win_rate']:>5.1f}%"
            f" {m['net_pnl']:>+9.0f} {m['pf']:>5.2f}"
            f" {m['max_dd']:>6.1f}% {m['sharpe']:>7.2f}"
            f" {m['tp_count']:>4} {m['sl_count']:>3} {m['be_count']:>3} {m['eod_count']:>3}{flag}"
        )
        if m["net_pnl"] > best_pnl:
            best_pnl = m["net_pnl"]
            best_tf = m['label']
            
    lines += [f"  └{'─'*68}┘", ""]
    
    lines += [
        div,
        f"  VERDICT (OOS)",
        div,
        f"  Optimal Execution Timeframe: {best_tf}",
        f"  Max Profit: {best_pnl:+.0f}$",
        ""
    ]

    txt = "\n".join(lines)
    print(txt)
    (out_dir / "report.txt").write_text(txt)

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

    print(f"\n{'═'*60}")
    print(f"  PropBot v4.0  —  Execution Timeframe Optimization")
    print(f"  Testing M5 vs M15 vs H1 for Breakout Confirmation")
    print(f"{'═'*60}\n")

    pair = LBO_CFG["pairs"][0]
    print(f"  Loading {pair} M1 data...", end=" ", flush=True)
    m1_data = load_pair(data_dir, pair)
    if m1_data.empty:
        print("NO DATA"); sys.exit(1)
    print(f"{len(m1_data):,} records loaded.\n")

    timeframes = [
        ("M5", "5min"),
        ("M15", "15min"),
        ("H1", "1h")
    ]
    
    tf_results = []
    
    for tf_name, tf_code in timeframes:
        print(f"  Running Backtest for Timeframe: {tf_name} ...", end=" ", flush=True)
        t_tr, eq_tr, t_oo, eq_oo = strategy_lbo(m1_data, pair, tf_code)
        
        m_tr = metrics(t_tr, eq_tr, tf_name)
        m_oo = metrics(t_oo, eq_oo, tf_name)
        
        print(f"Done. (OOS Trades: {m_oo['trades']})")
        tf_results.append({
            "tf_name": tf_name,
            "train": m_tr,
            "oos": m_oo,
            "trades_train": t_tr,
            "trades_oos": t_oo,
            "eq_train": eq_tr,
            "eq_oos": eq_oo
        })

    print("\nWriting comparative report...\n")
    write_report(tf_results, out_dir)
    print(f"\n  Done  →  {out_dir.resolve()}\n")

if __name__ == "__main__":
    main()
