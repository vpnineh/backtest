#!/usr/bin/env python3
"""
PropBot Backtester v10.2 — The Gold Momentum Scalper (High-Frequency)
══════════════════════════════════════════════════════════════════
Strategy : Micro-Pullback Breakout on Steep Momentum
Asset    : XAUUSD (Gold)
Timeframe: M5 (5-Minute)
Costs    : Realistic Gold Spread (15 pips) + Slippage
Logic    : Relaxed Momentum Filter -> Pullback -> Pending Order
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
#  ACCOUNT & GOLD COSTS
# ─────────────────────────────────────────────────────────────
ACCOUNT = dict(
    initial_bal  = 5_000.0,
    risk_pct     = 0.005,
    max_open     = 1,
    daily_dd_lim = 0.03,
    max_dd_kill  = 0.07,
)

SPREAD_PIPS = dict(XAUUSD=15.0)   
COMMISSION_PIP = 6.0              
SLIPPAGE_PIP = 5.0                
PIP_SIZE = dict(XAUUSD=0.01)

TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ─────────────────────────────────────────────────────────────
#  STRATEGY PARAMS
# ─────────────────────────────────────────────────────────────
GOLD_CFG = dict(
    pairs         = ["XAUUSD"],
    session_open  = 7,        
    session_close = 21,       
    force_close   = 22,       
    ema_fast      = 9,
    ema_slow      = 20,
    momentum_atr  = 0.1,      # 🔴 کاهش شدید فیلتر: اجازه نفس کشیدن به استراتژی 🔴
    buffer_pip    = 3.0,      # بافر کمتر برای فعال شدن سریع‌تر
    rr            = 1.0,      
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
#  PENDING ORDER TRADE ENGINE
# ─────────────────────────────────────────────────────────────
def get_trade_cost(pair: str) -> float:
    pip = PIP_SIZE[pair]
    spread = SPREAD_PIPS.get(pair, 15.0)
    return (spread + COMMISSION_PIP + SLIPPAGE_PIP * 2) * pip

class Pos:
    __slots__ = ["d","entry","sl","tp","t0","risk_usd","pip","c_cost","initial_risk"]
    def __init__(self, d, entry, sl, tp, t0, risk_usd, pip, c_cost):
        self.d, self.entry, self.sl, self.tp = d, entry, sl, tp
        self.t0, self.risk_usd, self.pip = t0, risk_usd, pip
        self.c_cost = c_cost
        self.initial_risk = abs(entry - sl)

def _close_pos(pos, ep, ts, pair):
    move    = (ep - pos.entry) * pos.d
    pnl_pip = move / pos.pip
    sl_pip  = abs(pos.entry - pos.sl) / pos.pip
    if sl_pip < 1e-5: return 0.0
    initial_sl_pip = pos.initial_risk / pos.pip
    pnl_usd = pnl_pip / initial_sl_pip * pos.risk_usd
    pnl_usd -= (pos.c_cost / pos.pip) * (pos.risk_usd / initial_sl_pip)
    return pnl_usd

def run_sim_pending(bars: pd.DataFrame, pair: str, signals: pd.DataFrame, force_close_hour: int) -> tuple:
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
                trades.append(dict(
                    pair=pair, dir="long" if pos.d==1 else "short",
                    open_time=str(pos.t0), close_time=str(ts),
                    open_px=round(pos.entry,2), close_px=round(ep,2),
                    sl=round(pos.sl,2), tp=round(pos.tp,2),
                    pnl=round(pnl,2), result=res, equity=round(equity,2),
                ))
                closed.append(pos)
                
        for p in closed: positions.remove(p)
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
        c_cost    = get_trade_cost(pair)

        if sig.get("sig_long") and not has_long:
            buy_stop = sig["entry_l"]
            if h >= buy_stop:
                entry = max(o, buy_stop) + (c_cost / 2)
                sl = sig["force_sl_l"]
                tp = entry + (entry - sl) * GOLD_CFG["rr"]  
                
                # جلوگیری از اردرهای مایکرو که توسط اسپرد بلعیده می‌شوند
                if abs(entry - sl) / pip > 15: 
                    positions.append(Pos(1, entry, sl, tp, ts, risk_usd, pip, c_cost))

        elif sig.get("sig_short") and not has_short:
            sell_stop = sig["entry_s"]
            if l <= sell_stop:
                entry = min(o, sell_stop) - (c_cost / 2)
                sl = sig["force_sl_s"]
                tp = entry - (sl - entry) * GOLD_CFG["rr"]
                
                if abs(entry - sl) / pip > 15:
                    positions.append(Pos(-1, entry, sl, tp, ts, risk_usd, pip, c_cost))

    return trades, eq_curve

# ─────────────────────────────────────────────────────────────
#  STRATEGY — THE GOLD MOMENTUM SCALPER (M5) - HIGH FREQUENCY
# ─────────────────────────────────────────────────────────────
def strategy_gold_momentum(m1: pd.DataFrame, pair: str) -> tuple:
    cfg = GOLD_CFG; pip = PIP_SIZE[pair]
    
    bars = resample(m1, "5min")
    bars["atr"] = atr(bars["high"], bars["low"], bars["close"], 14)
    bars["ema_fast"] = ema(bars["close"], cfg["ema_fast"])
    bars["ema_slow"] = ema(bars["close"], cfg["ema_slow"])
    
    in_session = (bars.index.hour >= cfg["session_open"]) & (bars.index.hour < cfg["session_close"])
    
    # 1. تعریف شیب تند (Relaxed Momentum)
    min_dist = bars["atr"] * cfg["momentum_atr"]
    
    # اضافه شدن شرطِ: EMA سریع در 3 کندل اخیر صعودی بوده باشد (برای تایید مومنتوم)
    ema_bull_trend = (bars["ema_fast"] > bars["ema_fast"].shift(1)) & (bars["ema_fast"].shift(1) > bars["ema_fast"].shift(2))
    ema_bear_trend = (bars["ema_fast"] < bars["ema_fast"].shift(1)) & (bars["ema_fast"].shift(1) < bars["ema_fast"].shift(2))

    steep_bull = (bars["ema_fast"] > bars["ema_slow"]) & ((bars["ema_fast"] - bars["ema_slow"]) > min_dist) & ema_bull_trend
    steep_bear = (bars["ema_fast"] < bars["ema_slow"]) & ((bars["ema_slow"] - bars["ema_fast"]) > min_dist) & ema_bear_trend
    
    # 2. پیدا کردن اولین کندل اصلاحی
    # 🔴 اصلاح: هر کندلی که Low آن کوچکتر یا مساوی کندل قبلی باشد یک پولبک محسوب می‌شود
    pullback_bull = steep_bull.shift(1) & (bars["low"] <= bars["low"].shift(1))
    pullback_bear = steep_bear.shift(1) & (bars["high"] >= bars["high"].shift(1))
    
    # اطمینان از اینکه کندل فعلی هنوز در جهت روند بسته شده یا حداقل خیلی خلاف آن نرفته است
    first_pullback_bull = pullback_bull & in_session & (bars["close"] > bars["ema_slow"])
    first_pullback_bear = pullback_bear & in_session & (bars["close"] < bars["ema_slow"])
    
    # 3. مقادیر سفارشات پندینگ
    buf = cfg["buffer_pip"] * pip
    bars["entry_l"] = bars["high"] + buf
    bars["force_sl_l"] = bars["low"] - buf
    
    bars["entry_s"] = bars["low"] - buf
    bars["force_sl_s"] = bars["high"] + buf

    sigs = pd.DataFrame(index=bars.index)
    sigs["sig_long"]  = first_pullback_bull.shift(1).fillna(False)
    sigs["sig_short"] = first_pullback_bear.shift(1).fillna(False)
    
    sigs["entry_l"] = bars["entry_l"].shift(1)
    sigs["force_sl_l"] = bars["force_sl_l"].shift(1)
    
    sigs["entry_s"] = bars["entry_s"].shift(1)
    sigs["force_sl_s"] = bars["force_sl_s"].shift(1)

    b_tr = bars[bars.index <= TRAIN_END]; b_oo = bars[bars.index >= TEST_START]
    s_tr = sigs[sigs.index <= TRAIN_END]; s_oo = sigs[sigs.index >= TEST_START]

    return run_sim_pending(b_tr, pair, s_tr, cfg["force_close"]) + run_sim_pending(b_oo, pair, s_oo, cfg["force_close"])

# ─────────────────────────────────────────────────────────────
#  METRICS & REPORTING
# ─────────────────────────────────────────────────────────────
def metrics(trades, eq_curve, label):
    if not trades:
        return dict(label=label, trades=0, net_pnl=0, win_rate=0, pf=0, max_dd=0,
                    tp_count=0, sl_count=0, eod_count=0)
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
        eod_count=sum(1 for t in trades if t["result"]=="eod")
    )

def write_report(m_res, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    div = "═"*75
    lines = [
        div, f"  PropBot Backtester v10.2  —  THE GOLD SCALPER (High-Freq)",
        f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Logic : Relaxed Momentum Surge -> Pullback -> Pending Stop Order",
        div, ""
    ]

    HDR = (f"  {'Period':<10} {'#':>5} {'WinR':>6} {'NetPnL':>9}"
           f" {'PF':>5} {'MaxDD':>7} {'TP':>4} {'SL':>4} {'EOD':>3}")
    SEP = "  " + "─"*67

    lines += [f"  ┌── PERFORMANCE REPORT (XAUUSD Exclusive) {'─'*25}┐", HDR, SEP]
    
    for key, name in [("train", "2010-2019"), ("oos", "2020-2025")]:
        m = m_res[key]
        flag = " ⚠" if m["max_dd"] > 6 else ""
        lines.append(
            f"  {name:<10} {m['trades']:>5} {m['win_rate']:>5.1f}%"
            f" {m['net_pnl']:>+9.0f} {m['pf']:>5.2f}"
            f" {m['max_dd']:>6.1f}% {m['tp_count']:>4} {m['sl_count']:>4} {m['eod_count']:>3}{flag}"
        )
            
    lines += [f"  └{'─'*67}┘", ""]
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
    print(f"  PropBot v10.2  —  The Gold Momentum Scalper (High-Freq)")
    print(f"{'═'*60}\n")

    pair = GOLD_CFG["pairs"][0]
    print(f"  Loading {pair} M1 data...", end=" ", flush=True)
    m1_data = load_pair(data_dir, pair)
    if m1_data.empty: sys.exit("NO DATA")
    print(f"{len(m1_data):,} records loaded.\n")

    print(f"  Executing Strategy...", end=" ", flush=True)
    t_tr, eq_tr, t_oo, eq_oo = strategy_gold_momentum(m1_data, pair)
    m_res = {"train": metrics(t_tr, eq_tr, "Train"), "oos": metrics(t_oo, eq_oo, "OOS")}
    print(f"Done. (OOS Trades: {m_res['oos']['trades']})")

    print("\nWriting report...\n")
    write_report(m_res, out_dir)
    print(f"\n  Done  →  {out_dir.resolve()}\n")

if __name__ == "__main__": main()
