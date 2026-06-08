#!/usr/bin/env python3
"""
PropBot Backtester v8.0  —  The Portfolio Scaler
══════════════════════════════════════════════════════════════════
Strategy : Institutional Liquidity Sweep (Macro Aligned)
Pairs    : EURUSD, GBPUSD, AUDUSD, USDCAD
Account  : $5,000 prop firm
Split    : Train 2010-2019  |  OOS 2020-2025
Costs    : Raw Spread + Commission ($6/lot) + Slippage
Logic    : D1 Trend Alignment + Mid-Range BE + No Fridays
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
#  ACCOUNT  —  $5K prop firm 
# ─────────────────────────────────────────────────────────────
ACCOUNT = dict(
    initial_bal  = 5_000.0,
    risk_pct     = 0.005,     # ریسک 0.5% برای هر معامله
    max_open     = 3,         # اجازه باز بودن حداکثر 3 معامله همزمان روی ارزهای مختلف
    daily_dd_lim = 0.03,      
    max_dd_kill  = 0.07,      
)

# ─────────────────────────────────────────────────────────────
#  REALISTIC COST CONFIGURATION (RAW SPREAD ACCOUNT)
# ─────────────────────────────────────────────────────────────
SPREAD_PIPS = dict(EURUSD=0.2, GBPUSD=0.5, AUDUSD=0.4, USDCAD=0.6)
COMMISSION_PIP = 0.6          
SLIPPAGE_PIP = 0.2            
LONDON_SPREAD_MULTIPLIER = 2.0 
PIP_SIZE = dict(EURUSD=1e-4, GBPUSD=1e-4, AUDUSD=1e-4, USDCAD=1e-4)

TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ─────────────────────────────────────────────────────────────
#  STRATEGY PARAMS (MACRO-ALIGNED SWEEP)
# ─────────────────────────────────────────────────────────────
SWEEP_CFG = dict(
    pairs         = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD"], # سبد ارزی
    asian_start   = 0,    
    asian_end     = 7,    
    entry_open    = 7,    
    entry_close   = 11,       
    force_close   = 17,       
    min_range_pip = 8,        # کاهش ملایم برای فرکانس بالاتر
    max_range_pip = 50,       
    atr_sl_mult   = 1.5,      
    macro_ema     = 50,       
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
        except Exception: pass
    for f in sorted(data_dir.glob(f"HISTDATA_COM_ASCII_{pair}_M1*.zip")):
        try:
            with zipfile.ZipFile(f) as z:
                inner = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if inner: frames.append(_parse(z.read(inner[0])))
        except Exception: pass
    if not frames: return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df[["open","high","low","close"]] > 0).all(axis=1)]
    return df

def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    agg = {"open":"first","high":"max","low":"min","close":"last"}
    return df.resample(tf).agg(agg).dropna()

# ─────────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────────
def atr(h, l, c, n):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(com=n-1, adjust=False).mean()

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

# ─────────────────────────────────────────────────────────────
#  TRADE ENGINE (With Mid-Range BE)
# ─────────────────────────────────────────────────────────────
def get_trade_cost(pair: str, ts: pd.Timestamp) -> float:
    pip = PIP_SIZE[pair]
    base_spread = SPREAD_PIPS.get(pair, 0.5)
    if 7 <= ts.hour <= 8:
        spread = base_spread * LONDON_SPREAD_MULTIPLIER
    else:
        spread = base_spread
    return (spread + COMMISSION_PIP + SLIPPAGE_PIP * 2) * pip

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

def run_sim(bars: pd.DataFrame, pair: str, signals: pd.DataFrame, force_close_hour: int) -> tuple:
    pip      = PIP_SIZE[pair]
    acct     = ACCOUNT
    equity   = acct["initial_bal"]
    peak     = equity
    positions: list[Pos] = []
    trades   : list[dict] = []
    eq_curve : list[dict] = []

    day_eq, last_day, frozen, killed = equity, None, False, False

    for ts in bars.index:
        if killed: break
        row = bars.loc[ts]
        sig = signals.loc[ts] if ts in signals.index else None
        d = ts.date()
        if d != last_day: day_eq, last_day, frozen = equity, d, False

        o, h, l = row["open"], row["high"], row["low"]

        closed = []
        for pos in positions:
            ep = res = None
            force_exit = (sig is not None and ts.hour >= force_close_hour and force_close_hour < 23)

            if pos.d == 1:
                if l <= pos.sl: ep, res = pos.sl, "sl"
                elif h >= pos.tp: ep, res = pos.tp, "tp"
            else:
                if h >= pos.sl: ep, res = pos.sl, "sl"
                elif l <= pos.tp: ep, res = pos.tp, "tp"

            if res is None and force_exit: ep, res = o, "eod"

            if ep is not None:
                pnl = _close_pos(pos, ep, ts, pair)
                equity += pnl; peak = max(peak, equity)
                if res == "sl" and pos.be_triggered: res = "be"
                trades.append(dict(
                    pair=pair, dir="long" if pos.d==1 else "short",
                    open_time=str(pos.t0), close_time=str(ts),
                    open_px=round(pos.entry,5), close_px=round(ep,5),
                    sl=round(pos.sl,5), tp=round(pos.tp,5),
                    pnl=round(pnl,2), result=res, equity=round(equity,2),
                ))
                closed.append(pos)
                
        for p in closed: positions.remove(p)
        
        for pos in positions:
            if not pos.be_triggered:
                trigger_dist = abs(pos.tp - pos.entry) * 0.5
                if pos.d == 1 and row["high"] >= pos.entry + trigger_dist:
                    pos.sl = pos.entry + pos.c_cost
                    pos.be_triggered = True
                elif pos.d == -1 and row["low"] <= pos.entry - trigger_dist:
                    pos.sl = pos.entry - pos.c_cost
                    pos.be_triggered = True

        eq_curve.append({"datetime": ts, "equity": equity})

        if (peak - equity) / peak >= acct["max_dd_kill"]:
            killed = True; break
        if not frozen and (day_eq - equity) / max(day_eq,1) >= acct["daily_dd_lim"]:
            frozen = True
        if frozen or sig is None: continue

        if len(positions) >= acct["max_open"]: continue

        has_long  = any(p.d== 1 for p in positions)
        has_short = any(p.d==-1 for p in positions)
        risk_usd  = equity * acct["risk_pct"]

        if sig.get("sig_long") and not has_long:
            c_cost = get_trade_cost(pair, ts)
            entry = o + c_cost / 2
            sl, tp = sig["force_sl"], sig["force_tp"]
            if abs(entry - sl) / pip < 3: continue 
            positions.append(Pos(1, entry, sl, tp, ts, risk_usd, pip, c_cost))

        elif sig.get("sig_short") and not has_short:
            c_cost = get_trade_cost(pair, ts)
            entry = o - c_cost / 2
            sl, tp = sig["force_sl"], sig["force_tp"]
            if abs(entry - sl) / pip < 3: continue
            positions.append(Pos(-1, entry, sl, tp, ts, risk_usd, pip, c_cost))

    return trades, eq_curve

# ─────────────────────────────────────────────────────────────
#  STRATEGY — MACRO ALIGNED SWEEP
# ─────────────────────────────────────────────────────────────
def strategy_prop_killer(m1: pd.DataFrame, pair: str) -> tuple:
    cfg = SWEEP_CFG; pip = PIP_SIZE[pair]
    
    d1 = resample(m1, "1D")
    d1["ema_macro"] = ema(d1["close"], cfg["macro_ema"])
    d1["ema_macro"] = d1["ema_macro"].shift(1) 
    d1.index = d1.index.date 
    
    bars = resample(m1, "15min")
    bars["date"] = bars.index.date
    bars["atr"] = atr(bars["high"], bars["low"], bars["close"], 14)
    
    bars = bars.join(d1[["ema_macro"]], on="date")
    bars["ema_macro"] = bars["ema_macro"].ffill()

    asian_mask = (bars.index.hour >= cfg["asian_start"]) & (bars.index.hour < cfg["asian_end"])
    asian = bars[asian_mask].groupby("date").agg(asian_high=("high","max"), asian_low=("low","min"))
    asian["range_pip"] = (asian["asian_high"] - asian["asian_low"]) / pip
    asian = asian[asian["range_pip"].between(cfg["min_range_pip"], cfg["max_range_pip"])]

    london_mask = (bars.index.hour >= cfg["entry_open"]) & (bars.index.hour <= cfg["entry_close"])
    london = bars[london_mask].copy()
    london["date"] = london.index.date
    london = london.join(asian, on="date", how="inner").dropna(subset=["asian_high", "ema_macro"])

    london["is_friday"] = london.index.dayofweek == 4

    london["macro_bull"] = london["close"] > london["ema_macro"]
    london["macro_bear"] = london["close"] < london["ema_macro"]

    london["session_high"] = london.groupby("date")["high"].cummax()
    london["session_low"]  = london.groupby("date")["low"].cummin()

    london["is_sweep_high"] = london["high"] > london["asian_high"]
    london["is_sweep_low"]  = london["low"] < london["asian_low"]
    london["has_swept_high"] = london.groupby("date")["is_sweep_high"].cummax()
    london["has_swept_low"]  = london.groupby("date")["is_sweep_low"].cummax()

    london["raw_short"] = london["macro_bear"] & london["has_swept_high"] & (london["close"] < london["asian_high"]) & ~london["is_friday"]
    london["raw_long"]  = london["macro_bull"] & london["has_swept_low"]  & (london["close"] > london["asian_low"]) & ~london["is_friday"]

    atr_buf = london["atr"] * cfg["atr_sl_mult"]
    london["force_sl_short"] = london["session_high"] + atr_buf
    london["force_sl_long"]  = london["session_low"]  - atr_buf

    sigs = pd.DataFrame(index=london.index)
    sigs["date"] = london["date"]
    sigs["raw_long"] = london["raw_long"]
    sigs["raw_short"] = london["raw_short"]

    sigs["force_sl"] = np.where(london["raw_long"], london["force_sl_long"],
                       np.where(london["raw_short"], london["force_sl_short"], np.nan))
    sigs["force_tp"] = np.where(london["raw_long"], london["asian_high"],
                       np.where(london["raw_short"], london["asian_low"], np.nan))

    shifted = sigs.shift(1); shifted["date"] = sigs["date"] 
    active_sigs = shifted[(shifted["raw_long"] == True) | (shifted["raw_short"] == True)].copy()
    first_sigs = active_sigs.drop_duplicates(subset=["date"], keep="first")
    first_sigs = first_sigs.rename(columns={"raw_long": "sig_long", "raw_short": "sig_short"})

    sigs_full = pd.DataFrame(index=bars.index).join(first_sigs[["sig_long", "sig_short", "force_sl", "force_tp"]])
    sigs_full[["sig_long", "sig_short"]] = sigs_full[["sig_long", "sig_short"]].fillna(False)

    b_tr = bars[bars.index <= TRAIN_END]; b_oo = bars[bars.index >= TEST_START]
    s_tr = sigs_full[sigs_full.index <= TRAIN_END]; s_oo = sigs_full[sigs_full.index >= TEST_START]

    return run_sim(b_tr, pair, s_tr, cfg["force_close"]) + run_sim(b_oo, pair, s_oo, cfg["force_close"])

# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────
def metrics(trades, eq_curve, label):
    if not trades:
        return dict(label=label, trades=0, net_pnl=0, win_rate=0, pf=0, max_dd=0,
                    tp_count=0, sl_count=0, be_count=0, eod_count=0)
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]; loss = [p for p in pnls if p <= 0]
    eqs  = pd.Series([e["equity"] for e in eq_curve])
    dd   = (eqs.cummax() - eqs) / eqs.cummax() * 100
    net  = sum(pnls)
    gp   = sum(wins); gl = abs(sum(loss))
    pf   = gp/gl if gl>0 else float("inf")

    return dict(
        label=label, trades=len(trades), win_rate=round(len(wins)/len(trades)*100,1),
        net_pnl=round(net,2), pf=round(pf,2), max_dd=round(dd.max(),2),
        tp_count=sum(1 for t in trades if t["result"]=="tp"),
        sl_count=sum(1 for t in trades if t["result"]=="sl"),
        be_count=sum(1 for t in trades if t["result"]=="be"),
        eod_count=sum(1 for t in trades if t["result"]=="eod")
    )

# ─────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────
def write_report(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    div = "═"*78
    lines = [
        div, f"  PropBot Backtester v8.0  —  THE PORTFOLIO SCALER",
        f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Logic : Macro Trend (D1) + Mid-Range BE + No Fridays", div, ""
    ]

    HDR = (f"  {'Pair':<8} {'#':>5} {'WinR':>6} {'NetPnL':>9}"
           f" {'PF':>5} {'MaxDD':>7} {'TP':>4} {'SL':>4} {'BE':>3} {'EOD':>4}")
    SEP = "  " + "─"*74

    lines += [f"  ┌── OOS PERFORMANCE (2020-2025) {'─'*40}┐", HDR, SEP]
    
    tot_trades = tot_pnl = 0
    max_portfolio_dd = 0

    for r in results:
        m = r["oos"]
        flag = " ⚠" if m["max_dd"] > 6 else ""
        lines.append(
            f"  {m['label']:<8} {m['trades']:>5} {m['win_rate']:>5.1f}%"
            f" {m['net_pnl']:>+9.0f} {m['pf']:>5.2f}"
            f" {m['max_dd']:>6.1f}% {m['tp_count']:>4} {m['sl_count']:>4} {m['be_count']:>3} {m['eod_count']:>4}{flag}"
        )
        tot_trades += m["trades"]
        tot_pnl += m["net_pnl"]
        max_portfolio_dd = max(max_portfolio_dd, m["max_dd"]) # Estimate
            
    lines += [SEP, f"  {'TOTAL':<8} {tot_trades:>5} {'-':>6} {tot_pnl:>+9.0f} {'-':>5} {max_portfolio_dd:>6.1f}%"]
    lines += [f"  └{'─'*74}┘", ""]
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "report.txt").write_text(txt)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir",  default="results")
    args = p.parse_args()
    data_dir = Path(args.data_dir); out_dir  = Path(args.out_dir)

    print(f"\n{'═'*60}")
    print(f"  PropBot v8.0  —  The Portfolio Scaler")
    print(f"{'═'*60}\n")

    all_results = []
    
    for pair in SWEEP_CFG["pairs"]:
        print(f"  Loading {pair} M1 data...", end=" ", flush=True)
        m1_data = load_pair(data_dir, pair)
        if m1_data.empty: 
            print("NO DATA - Skipping")
            continue
        print(f"OK. Running Strategy...", end=" ", flush=True)
        
        t_tr, eq_tr, t_oo, eq_oo = strategy_prop_killer(m1_data, pair)
        m_res = {"train": metrics(t_tr, eq_tr, pair), "oos": metrics(t_oo, eq_oo, pair)}
        all_results.append(m_res)
        print(f"Done. (OOS: {m_res['oos']['trades']} trades)")

    print("\nWriting Portfolio report...\n")
    write_report(all_results, out_dir)
    print(f"\n  Done  →  {out_dir.resolve()}\n")

if __name__ == "__main__": main()
