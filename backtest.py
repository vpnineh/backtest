#!/usr/bin/env python3
"""
PropBot Backtester v5.0
═══════════════════════════════════════════════════════════════════════
Lessons applied:
  v3: XAUUSD H1 EMA gave WR=46%, PF=1.21, E[t]=+$6.76 OOS → KEEP IT
  v4: H4 filter on XAUUSD reduced trades from 37 to 20 and WR 46→35% → REMOVE
  v4: LBO stop entry never triggered → fix: check stop fill inside BAR, not next

MT4 simulation:
  - Stop-entry orders checked against bar High/Low
  - Gap fill at open (not at SL price)
  - Same-bar: bar direction decides SL vs TP order
  - Round-trip spread + slippage + overnight swap

Strategy A: XAUUSD H1 EMA (H1 only — clean, no H4 noise)
Strategy B: LBO EURUSD + GBPUSD (fixed stop-entry mechanics)

Account: $5,000 — risk 1%/trade
═══════════════════════════════════════════════════════════════════════
"""

import sys, zipfile, io, csv, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
#  ACCOUNT
# ──────────────────────────────────────────────────────────────
ACCT = dict(
    balance  = 5_000.0,
    risk_pct = 0.01,      # $50 per trade
    max_pos  = 2,
    freeze   = 0.03,      # 3% daily → freeze
    kill     = 0.07,      # 7% total → kill
)
PIP  = dict(EURUSD=1e-4, GBPUSD=1e-4, XAUUSD=0.01)
SPR  = dict(EURUSD=1.2,  GBPUSD=1.5,  XAUUSD=30)   # pips
SWAP = dict(                                          # pips/night (long, short)
    EURUSD=(-0.5,0.1), GBPUSD=(-0.7,-0.2), XAUUSD=(-2.5,-1.5)
)
SLIP = 0.5   # extra pips slippage
MIN_SL = 5  # broker minimum SL (pips)

TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ──────────────────────────────────────────────────────────────
#  STRATEGY PARAMS
# ──────────────────────────────────────────────────────────────

# Strategy A: XAUUSD — same setup that gave PF=1.21 in v3
GOLD = dict(
    ema_fast    = 20,
    ema_slow    = 50,
    rsi_period  = 14,
    rsi_max_l   = 65,   # long only if RSI < 65 (not overbought)
    rsi_min_s   = 35,   # short only if RSI > 35 (not oversold)
    atr_period  = 14,
    pullback    = 1.0,  # enter when low touched EMA ± pullback×ATR
    sl_mult     = 1.5,
    rr          = 1.8,
    sess_open   = 7,    # UTC hour
    sess_close  = 21,
)

# Strategy B: London Breakout — fixed
LBO = dict(
    asian_hours = list(range(0, 7)),    # 00-06 UTC: Asian range
    entry_hours = list(range(7, 12)),   # 07-11 UTC: breakout window
    force_cls   = 15,                   # close before NY (15:00 UTC)
    min_rng_pip = 5,
    max_rng_pip = 60,
    buf_pip     = 1.0,   # buffer beyond range for stop entry
    sl_inside   = 3,     # SL inside range from breakout level (pips)
    rr          = 1.8,
)

# ──────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────
def _sep(b): return ";" if b[:300].decode("utf-8","replace").count(";")>2 else ","

def _parse(raw):
    df = pd.read_csv(io.BytesIO(raw), sep=_sep(raw), header=None,
                     names=["dt","O","H","L","C","V"],
                     dtype={"O":float,"H":float,"L":float,"C":float,"V":float},
                     on_bad_lines="skip")
    df["T"] = pd.to_datetime(df["dt"].astype(str).str.strip(),
                              format="%Y%m%d %H%M%S", utc=True, errors="coerce")
    return (df.dropna(subset=["T"])
              .rename(columns={"O":"open","H":"high","L":"low","C":"close"})
              .drop(columns=["dt","V"]).set_index("T").sort_index()
              .loc[lambda d: (d > 0).all(axis=1)])

def load(data_dir, pair):
    frames = []
    for f in sorted(Path(data_dir).glob(f"DAT_ASCII_{pair}_M1_*.csv")):
        try: frames.append(_parse(f.read_bytes()))
        except Exception as e: print(f"  [W] {f.name}: {e}")
    for f in sorted(Path(data_dir).glob(f"HISTDATA_COM_ASCII_{pair}_M1*.zip")):
        try:
            with zipfile.ZipFile(f) as z:
                inn = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if inn: frames.append(_parse(z.read(inn[0])))
        except Exception as e: print(f"  [W] {f.name}: {e}")
    if not frames: return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]

def rs(df, tf):
    return df.resample(tf).agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()

# ──────────────────────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(c, n):
    d=c.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    return 100-100/(1+g.ewm(com=n-1,adjust=False).mean()/
                       l.ewm(com=n-1,adjust=False).mean().replace(0,np.nan))
def atr(h,l,c,n):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(com=n-1,adjust=False).mean()

# ──────────────────────────────────────────────────────────────
#  MT4 BAR CHECKER  (shared by both strategies)
# ──────────────────────────────────────────────────────────────
def bar_check(d, entry, sl, tp, O, H, L, C):
    """
    d: direction +1 long / -1 short
    Returns (exit_price, tag) or (None, None)

    Gap rule: if bar opens past SL → fill at open
    Same-bar: bullish bar path = Low→High (check SL first for long)
              bearish bar path = High→Low (check SL first for short)
    """
    if d == 1:
        if O <= sl: return O, "sl_gap"
        sl_hit, tp_hit = L <= sl, H >= tp
    else:
        if O >= sl: return O, "sl_gap"
        sl_hit, tp_hit = H >= sl, L <= tp

    if not sl_hit and not tp_hit: return None, None
    if sl_hit and not tp_hit:     return sl,  "sl"
    if tp_hit and not sl_hit:     return tp,  "tp"
    # both hit same bar
    bull = C > O
    if d == 1:  return (sl if bull  else tp), ("sl" if bull  else "tp")
    else:       return (sl if not bull else tp), ("sl" if not bull else "tp")

# ──────────────────────────────────────────────────────────────
#  POSITION & P&L
# ──────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["d","entry","sl","tp","t0","risk","pip","swap_day"]
    def __init__(self, d, entry, sl, tp, t0, risk, pip):
        self.d=d; self.entry=entry; self.sl=sl; self.tp=tp
        self.t0=t0; self.risk=risk; self.pip=pip; self.swap_day=t0.date()

def calc_pnl(pos, ep, pair):
    pip    = PIP[pair]
    sl_pip = abs(pos.entry - pos.sl) / pip
    if sl_pip < 1: return 0.0
    move   = (ep - pos.entry) * pos.d / pip
    profit = move / sl_pip * pos.risk
    cost   = (SPR[pair] + SLIP*2) / sl_pip * pos.risk
    return profit - cost

def apply_swap(pos, today, pair):
    if today <= pos.swap_day: return 0.0
    n = (today - pos.swap_day).days; pos.swap_day = today
    sw = SWAP[pair][0 if pos.d==1 else 1] * n
    sl_pip = abs(pos.entry - pos.sl) / PIP[pair]
    return sw / sl_pip * pos.risk if sl_pip > 0 else 0.0

# ──────────────────────────────────────────────────────────────
#  CORE SIMULATOR
# ──────────────────────────────────────────────────────────────
def simulate(bars, pair, signals, force_cls_h=99):
    """
    signals columns (always):
        sig_long  : bool  — attempt a long entry this bar
        sig_short : bool  — attempt a short entry this bar

    For EMA market entries (no fixed SL/TP in signal):
        atr column must exist in bars for position sizing

    For LBO stop entries:
        stop_l, sl_l, tp_l  — long stop level, SL, TP
        stop_s, sl_s, tp_s  — short stop level, SL, TP
    """
    pip      = PIP[pair]
    lbo_mode = "stop_l" in signals.columns

    eq, peak = ACCT["balance"], ACCT["balance"]
    day_eq, last_day = eq, None
    frozen = killed = False
    positions, trades, equity_curve = [], [], []

    for ts, bar in bars.iterrows():
        if killed: break
        d = ts.date()
        if d != last_day:
            day_eq, last_day, frozen = eq, d, False
        O, H, L, C = bar["open"], bar["high"], bar["low"], bar["close"]

        # swap
        for pos in positions:
            eq += apply_swap(pos, d, pair)

        # manage open positions
        closed = []
        for pos in positions:
            ep, res = bar_check(pos.d, pos.entry, pos.sl, pos.tp, O, H, L, C)
            if res is None and ts.hour >= force_cls_h < 23:
                ep, res = O, "eod"
            if ep is not None:
                pnl = calc_pnl(pos, ep, pair)
                eq += pnl; peak = max(peak, eq)
                trades.append(dict(
                    pair=pair, d="L" if pos.d==1 else "S",
                    t0=str(pos.t0)[:16], t1=str(ts)[:16],
                    entry=round(pos.entry,5), exit=round(ep,5),
                    sl=round(pos.sl,5), tp=round(pos.tp,5),
                    pnl=round(pnl,2), res=res, eq=round(eq,2)
                ))
                closed.append(pos)
        for p in closed: positions.remove(p)

        equity_curve.append({"dt": ts, "eq": eq})

        # kill / freeze
        if (peak-eq)/max(peak,1) >= ACCT["kill"]:
            killed=True; print(f"  [KILL] {pair} DD={(peak-eq)/peak*100:.1f}% @ {ts.date()}"); break
        if not frozen and (day_eq-eq)/max(day_eq,1) >= ACCT["freeze"]:
            frozen=True
        if frozen: continue

        # get signal for this bar
        if ts not in signals.index: continue
        sig = signals.loc[ts]
        if len(positions) >= ACCT["max_pos"]: continue
        has_l = any(p.d== 1 for p in positions)
        has_s = any(p.d==-1 for p in positions)
        risk  = eq * ACCT["risk_pct"]

        go_long  = bool(sig.get("sig_long",  False))
        go_short = bool(sig.get("sig_short", False))

        if lbo_mode:
            # ── LBO STOP ENTRY ──
            # Stop orders are checked against THIS bar's H/L
            # (the signal was set at 07:00, we check if THIS bar breaks the level)
            if go_long and not has_l:
                lvl = sig.get("stop_l", np.nan)
                if pd.notna(lvl) and H > lvl:
                    entry = lvl + SPR[pair]*pip/2   # buy ask
                    sl_p  = sig["sl_l"]
                    tp_p  = sig["tp_l"]
                    if abs(entry - sl_p)/pip >= MIN_SL:
                        positions.append(Pos(1, entry, sl_p, tp_p, ts, risk, pip))

            if go_short and not has_s:
                lvl = sig.get("stop_s", np.nan)
                if pd.notna(lvl) and L < lvl:
                    entry = lvl - SPR[pair]*pip/2   # sell bid
                    sl_p  = sig["sl_s"]
                    tp_p  = sig["tp_s"]
                    if abs(entry - sl_p)/pip >= MIN_SL:
                        positions.append(Pos(-1, entry, sl_p, tp_p, ts, risk, pip))
        else:
            # ── EMA MARKET ENTRY ──
            atr_v = float(bar.get("atr", 0))
            if atr_v < SPR[pair]*pip*2: continue   # too narrow

            if go_long and not has_l:
                entry = O + SPR[pair]*pip/2 + SLIP*pip
                sl_p  = entry - atr_v * GOLD["sl_mult"]
                tp_p  = entry + atr_v * GOLD["sl_mult"] * GOLD["rr"]
                if abs(entry-sl_p)/pip >= MIN_SL:
                    positions.append(Pos(1, entry, sl_p, tp_p, ts, risk, pip))

            elif go_short and not has_s:
                entry = O - SPR[pair]*pip/2 - SLIP*pip
                sl_p  = entry + atr_v * GOLD["sl_mult"]
                tp_p  = entry - atr_v * GOLD["sl_mult"] * GOLD["rr"]
                if abs(entry-sl_p)/pip >= MIN_SL:
                    positions.append(Pos(-1, entry, sl_p, tp_p, ts, risk, pip))

    return trades, equity_curve

# ──────────────────────────────────────────────────────────────
#  STRATEGY A: XAUUSD H1 EMA  (v3 approach that worked)
# ──────────────────────────────────────────────────────────────
def strat_xau(m1):
    pair = "XAUUSD"; p = GOLD
    h1 = rs(m1, "1h")
    h1["ef"]  = ema(h1["close"], p["ema_fast"])
    h1["es"]  = ema(h1["close"], p["ema_slow"])
    h1["rsi"] = rsi(h1["close"], p["rsi_period"])
    h1["atr"] = atr(h1["high"], h1["low"], h1["close"], p["atr_period"])

    bull  = h1["ef"] > h1["es"]
    bear  = h1["ef"] < h1["es"]
    sess  = (h1.index.hour >= p["sess_open"]) & (h1.index.hour < p["sess_close"])

    # Long: uptrend + bar's Low touched EMA20 band + RSI not overbought
    pull_l = h1["low"]  <= h1["ef"] + h1["atr"] * p["pullback"]
    pull_s = h1["high"] >= h1["ef"] - h1["atr"] * p["pullback"]

    raw_l = bull & pull_l & (h1["rsi"] < p["rsi_max_l"]) & sess
    raw_s = bear & pull_s & (h1["rsi"] > p["rsi_min_s"]) & sess

    # shift 1: signal on bar N → entry on bar N+1 open
    sigs = pd.DataFrame({
        "sig_long":  raw_l.shift(1).fillna(False),
        "sig_short": raw_s.shift(1).fillna(False),
    }, index=h1.index)

    tr,eq = simulate(h1[h1.index<=TRAIN_END], pair, sigs[sigs.index<=TRAIN_END])
    to,eo = simulate(h1[h1.index>=TEST_START], pair, sigs[sigs.index>=TEST_START])
    return tr, eq, to, eo

# ──────────────────────────────────────────────────────────────
#  STRATEGY B: LONDON BREAKOUT  (fixed stop-entry mechanics)
# ──────────────────────────────────────────────────────────────
def strat_lbo(m1, pair):
    p   = LBO
    pip = PIP[pair]
    h1  = rs(m1, "1h")

    # ── DIAGNOSTIC: check data timezone ──
    # If first bars are around hour 17-18 → data is in EST (UTC-5)
    # If around hour 21-23 → data is UTC
    first_hour = h1.index.hour[0]
    print(f"    [diag] first bar hour={first_hour} (0=UTC midnight, 17=EST market open)")

    # ── Build Asian range per day ──
    # Group H1 bars by calendar date, only hours 00-06
    h1["_date"] = h1.index.normalize()   # midnight of each day (UTC)
    asian_bars  = h1[h1.index.hour.isin(p["asian_hours"])]
    asian = asian_bars.groupby("_date").agg(
        a_hi=("high","max"), a_lo=("low","min")
    )
    asian["rng_pip"] = (asian["a_hi"] - asian["a_lo"]) / pip
    print(f"    [diag] total Asian dates: {len(asian)}  "
          f"mean range: {asian['rng_pip'].mean():.1f} pips  "
          f"in filter [{p['min_rng_pip']},{p['max_rng_pip']}]: "
          f"{asian['rng_pip'].between(p['min_rng_pip'],p['max_rng_pip']).sum()}")

    asian = asian[asian["rng_pip"].between(p["min_rng_pip"], p["max_rng_pip"])]
    if asian.empty:
        print(f"    [diag] ASIAN IS EMPTY — no valid range days found!")
        return [], [], [], []

    # ── Build stop-entry signals for each valid London-open bar ──
    buf = p["buf_pip"] * pip
    sl_in = p["sl_inside"] * pip

    # For every H1 bar at hour 07:00-11:00, look up that day's Asian range
    london_bars = h1[h1.index.hour.isin(p["entry_hours"])].copy()
    london_bars["_date"] = london_bars.index.normalize()
    london_bars = london_bars.join(asian, on="_date").dropna(subset=["a_hi","a_lo"])

    print(f"    [diag] London bars after range join: {len(london_bars)}")

    if london_bars.empty:
        print(f"    [diag] LONDON BARS EMPTY — join with Asian range failed!")
        return [], [], [], []

    rng = london_bars["a_hi"] - london_bars["a_lo"]

    london_bars["stop_l"] = london_bars["a_hi"] + buf
    london_bars["sl_l"]   = london_bars["a_hi"] - sl_in
    london_bars["tp_l"]   = london_bars["a_hi"] + buf + rng * p["rr"]

    london_bars["stop_s"] = london_bars["a_lo"] - buf
    london_bars["sl_s"]   = london_bars["a_lo"] + sl_in
    london_bars["tp_s"]   = london_bars["a_lo"] - buf - rng * p["rr"]

    london_bars["sig_long"]  = True
    london_bars["sig_short"] = True

    # Keep all London bars (not just 07:00) — each bar independently checks its stop
    # BUT: cancel both directions once ONE entry fires per day
    # We implement this by keeping all London bars and letting the engine handle max_pos
    sig_cols = ["sig_long","sig_short","stop_l","sl_l","tp_l","stop_s","sl_s","tp_s"]
    sigs_full = london_bars[sig_cols].reindex(h1.index)
    sigs_full[["sig_long","sig_short"]] = sigs_full[["sig_long","sig_short"]].fillna(False)

    print(f"    [diag] signal rows with sig_long=True: "
          f"{sigs_full['sig_long'].sum()}")

    tr,eq = simulate(h1[h1.index<=TRAIN_END], pair,
                     sigs_full[sigs_full.index<=TRAIN_END], p["force_cls"])
    to,eo = simulate(h1[h1.index>=TEST_START], pair,
                     sigs_full[sigs_full.index>=TEST_START], p["force_cls"])
    return tr, eq, to, eo

# ──────────────────────────────────────────────────────────────
#  METRICS
# ──────────────────────────────────────────────────────────────
def metrics(trades, eq_curve, label):
    z = dict(label=label, n=0, wr=0, pnl=0, pf=0, dd=0,
             sharpe=0, calmar=0, tp=0, sl=0, gap=0, eod=0,
             aw=0, al=0, exp=0, ret=0, cl=0)
    if not trades: return z
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    eqs  = pd.Series([e["eq"] for e in eq_curve])
    dd   = (eqs.cummax()-eqs)/eqs.cummax()*100
    net  = sum(pnls); gp=sum(wins); gl=abs(sum(loss))

    eq_s = (pd.DataFrame(eq_curve).rename(columns={"dt":"T","eq":"E"})
              .set_index("T")["E"])
    mon  = eq_s.resample("ME").last().pct_change().dropna()
    sh   = (mon.mean()/mon.std()*12**0.5
            if len(mon)>=3 and mon.std()>0 else 0.0)
    ret  = net/ACCT["balance"]*100
    cal  = ret/dd.max() if dd.max()>0 else 0.0
    aw   = np.mean(wins) if wins else 0
    al   = np.mean(loss) if loss else 0
    exp  = len(wins)/len(pnls)*aw + len(loss)/len(pnls)*al
    best=cur=0
    for x in pnls: cur=cur+1 if x<=0 else 0; best=max(best,cur)
    return dict(
        label=label, n=len(trades),
        wr=round(len(wins)/len(trades)*100,1),
        pnl=round(net,2), ret=round(ret,2),
        pf=round(gp/gl if gl>0 else float("inf"),2),
        dd=round(dd.max(),2), sharpe=round(sh,2), calmar=round(cal,2),
        tp=sum(1 for t in trades if t["res"]=="tp"),
        sl=sum(1 for t in trades if t["res"] in("sl","sl_gap")),
        gap=sum(1 for t in trades if t["res"]=="sl_gap"),
        eod=sum(1 for t in trades if t["res"]=="eod"),
        cl=best, aw=round(aw,1), al=round(al,1), exp=round(exp,2),
    )

# ──────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────
def report(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    D = "═"*72
    HDR = (f"  {'Strategy':<18}{'#':>5}{'WinR':>7}{'PnL$':>9}"
           f"{'PF':>6}{'MaxDD':>7}{'Sharpe':>8}{'Calmar':>8}"
           f"{'TP':>5}{'SL':>5}{'GAP':>5}{'EOD':>5}{'E[t]':>8}")
    SEP = "  "+"─"*70
    lines = [D,
             "  PropBot v5.0  —  $5,000 Prop  |  MT4-Realistic",
             f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
             "  XAUUSD EMA H1  +  LBO EURUSD/GBPUSD",
             D,""]
    summary = {}

    for period, key in [("TRAIN 2010-2019","train"), ("OOS  2020-2025  ★","oos")]:
        lines += [f"  ┌── {period} {'─'*(56-len(period))}┐", HDR, SEP]
        tot_pnl = tot_n = 0
        summary[key] = []
        for r in results:
            m = r[key]
            summary[key].append({
                "strategy":r["name"], "pair":r["pair"],
                "n":m["n"], "wr":m["wr"], "pf":m["pf"],
                "pnl":m["pnl"], "dd":m["dd"],
                "sharpe":m["sharpe"], "exp":m["exp"]
            })
            if m["n"]==0:
                lines.append(f"  {r['name']:<18} — no trades generated"); continue
            ok = ("✓" if m["pf"]>1.3 and m["n"]>=15 else
                  "?" if m["pf"]>1.0 else "✗")
            dw = " ⚠" if m["dd"]>6 else ""
            lines.append(
                f"  {m['label']:<18}{m['n']:>5}{m['wr']:>6.1f}%{m['pnl']:>+9.0f}"
                f"{m['pf']:>6.2f}{m['dd']:>6.1f}%"
                f"{m['sharpe']:>8.2f}{m['calmar']:>8.2f}"
                f"{m['tp']:>5}{m['sl']:>5}{m['gap']:>5}{m['eod']:>5}"
                f"{m['exp']:>+8.2f}  {ok}{dw}")
            tot_pnl+=m["pnl"]; tot_n+=m["n"]
        lines += [SEP, f"  {'TOTAL':<18}{tot_n:>5}{'':>7}{tot_pnl:>+9.0f}",""]

    lines += [D,"  VERDICT  (OOS only)","─"*72]
    for r in results:
        m = r["oos"]
        n,pf,dd,sh,exp = m["n"],m["pf"],m["dd"],m["sharpe"],m["exp"]
        if n<15:  v="⚠  NEED MORE DATA (≥15 OOS trades)"
        elif pf>=1.3 and dd<5 and sh>=0.5: v="✅ PASS — run 1-month forward test, then live"
        elif pf>=1.0 and dd<7:             v="🟡 MARGINAL — forward test only, not ready for live"
        else:                               v="❌ FAIL — strategy has no edge"
        lines.append(f"  {r['name']:<20}: PF={pf:.2f}  DD={dd:.1f}%  "
                     f"E[t]={exp:+.2f}$  n={n}  → {v}")
    lines += ["",
              "  Breakeven WR at RR=1.8 → 35.7%  |  at RR=1.5 → 40.0%",
              "  E[trade] > 0 = positive expected value per trade",
              "  ⚠ = drawdown > 6% (close to prop 7% kill limit)",
              ""]
    txt = "\n".join(lines)
    print(txt)
    (out_dir/"report.txt").write_text(txt)
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))

    for r in results:
        nm = r["name"].replace(" ","_")
        for tag, tk, ek in [("train","t_tr","e_tr"), ("oos","t_oo","e_oo")]:
            if r.get(tk):
                with open(out_dir/f"trades_{nm}_{tag}.csv","w",newline="") as f:
                    w=csv.DictWriter(f, fieldnames=list(r[tk][0].keys()))
                    w.writeheader(); w.writerows(r[tk])
            if r.get(ek):
                with open(out_dir/f"equity_{nm}_{tag}.csv","w",newline="") as f:
                    w=csv.DictWriter(f, fieldnames=["dt","eq"])
                    w.writeheader(); w.writerows(r[ek])

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.dates as mdates
        fig,axes=plt.subplots(1,len(results),figsize=(6*len(results),5),squeeze=False)
        fig.suptitle("PropBot v5.0 — OOS 2020-2025",fontsize=12,fontweight="bold")
        for i,r in enumerate(results):
            ax=axes[0][i]; m=r["oos"]; eq=r.get("e_oo",[])
            if eq:
                dts=[e["dt"] for e in eq]; eqs=[e["eq"] for e in eq]
                clr="#16a34a" if m["pnl"]>0 else "#dc2626"
                ax.plot(dts,eqs,color=clr,lw=0.9)
                ax.axhline(ACCT["balance"],color="#94a3b8",lw=0.5,ls="--")
                ax.fill_between(dts,ACCT["balance"],eqs,
                    where=[e>=ACCT["balance"] for e in eqs],color="#16a34a",alpha=0.1)
                ax.fill_between(dts,ACCT["balance"],eqs,
                    where=[e<ACCT["balance"] for e in eqs],color="#ef4444",alpha=0.15)
            ax.set_title(f"{r['name']}\n"
                         f"#{m['n']} WR={m['wr']:.0f}% PF={m['pf']:.2f} "
                         f"DD={m['dd']:.1f}%\nE[t]={m['exp']:+.2f}$ PnL={m['pnl']:+.0f}$",
                         fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir/"equity_oos.png",dpi=140,bbox_inches="tight")
        plt.close(); print("  [OK] chart saved")
    except Exception as e: print(f"  [chart] {e}")

# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="data")
    ap.add_argument("--out-dir", default="results")
    args=ap.parse_args()
    dd=Path(args.data_dir); od=Path(args.out_dir)

    print(f"\n{'═'*58}")
    print(f"  PropBot v5.0 — $5K Prop  |  MT4-Realistic")
    print(f"  Data: {dd.resolve()}")
    print(f"{'═'*58}\n")

    cache={}
    for pair in ["XAUUSD","EURUSD","GBPUSD"]:
        print(f"  Loading {pair}...",end=" ",flush=True)
        df=load(dd,pair)
        if df.empty: print("NO DATA"); continue
        cache[pair]=df
        print(f"{len(df):,} M1  ({df.index[0].date()} → {df.index[-1].date()})")
    print()

    results=[]

    if "XAUUSD" in cache:
        print("  ── XAUUSD EMA H1 ─────────────────────────────────")
        tr,eq,to,eo = strat_xau(cache["XAUUSD"])
        mtr=metrics(tr,eq,"XAUUSD EMA"); moo=metrics(to,eo,"XAUUSD EMA")
        print(f"  Train: {mtr['n']} trades  WR={mtr['wr']:.0f}%  PF={mtr['pf']:.2f}")
        print(f"  OOS  : {moo['n']} trades  WR={moo['wr']:.0f}%  PF={moo['pf']:.2f}  E[t]={moo['exp']:+.2f}$")
        results.append(dict(name="XAUUSD EMA", pair="XAUUSD",
                            train=mtr, oos=moo,
                            t_tr=tr, e_tr=eq, t_oo=to, e_oo=eo))
        print()

    for pair in ["EURUSD","GBPUSD"]:
        if pair not in cache: continue
        print(f"  ── LBO {pair} ──────────────────────────────────")
        tr,eq,to,eo = strat_lbo(cache[pair], pair)
        mtr=metrics(tr,eq,f"LBO {pair}"); moo=metrics(to,eo,f"LBO {pair}")
        print(f"  Train: {mtr['n']} trades  WR={mtr['wr']:.0f}%  PF={mtr['pf']:.2f}")
        print(f"  OOS  : {moo['n']} trades  WR={moo['wr']:.0f}%  PF={moo['pf']:.2f}  E[t]={moo['exp']:+.2f}$")
        results.append(dict(name=f"LBO {pair}", pair=pair,
                            train=mtr, oos=moo,
                            t_tr=tr, e_tr=eq, t_oo=to, e_oo=eo))
        print()

    print("Writing report...\n")
    report(results, od)
    print(f"\n  Done → {od.resolve()}\n")

if __name__=="__main__":
    main()
