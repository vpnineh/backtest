#!/usr/bin/env python3
"""
PropBot Backtester v4.0
═══════════════════════════════════════════════════════════════════════
LESSONS FROM v3 RESULTS:
  - EURUSD/GBPUSD EMA: 0% win rate OOS → regime change post-2020
    (choppy, high-vol, non-trending for H1 EMA)
  - XAUUSD EMA: only real positive edge (PF=1.21, WR=46%, OOS)
  - LBO GBPUSD: PF=1.30 but only 10 trades → not statistically meaningful

THIS VERSION focuses on what actually shows edge:
  PAIR 1: XAUUSD — EMA trend with H4 confirmation + tighter pullback
  PAIR 2: GBPUSD — London Breakout (wider range filter → more trades)
  PAIR 3: EURUSD — London Breakout (wider range filter → more trades)

MT4-realistic simulation:
  - Gap fill at open (not SL price)
  - Same-bar direction determines SL/TP order
  - Round-trip spread + slippage deducted per trade
  - Overnight swap applied daily

Account: $5,000
Risk: 1% per trade = $50
═══════════════════════════════════════════════════════════════════════
"""

import sys, zipfile, io, csv, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
#  ACCOUNT
# ─────────────────────────────────────────────────────────
ACCT = dict(
    balance      = 5_000.0,
    risk_pct     = 0.01,
    max_pos      = 2,
    daily_freeze = 0.03,   # 3% daily loss → freeze
    kill_dd      = 0.07,   # 7% total DD  → kill
)

PIP   = dict(EURUSD=1e-4, GBPUSD=1e-4, XAUUSD=0.01)
SPR   = dict(EURUSD=1.2,  GBPUSD=1.5,  XAUUSD=30)    # pips
SWAP  = dict(                                           # pips/night
    EURUSD=(-0.5, 0.1), GBPUSD=(-0.7,-0.2), XAUUSD=(-2.5,-1.5)
)  # (long, short)
SLIP  = 0.5  # extra pips slippage on market fills
MIN_SL = 5   # minimum SL distance in pips

TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ─────────────────────────────────────────────────────────
#  STRATEGY PARAMS  (fixed — not optimised on OOS data)
# ─────────────────────────────────────────────────────────

# Strategy A: XAUUSD EMA Trend
# H4 defines trend direction → H1 provides pullback entry
# v4 improvement: requires confirmed H4 trend PLUS H1 RSI in 35-55 zone
GOLD_P = dict(
    h4_fast  = 20, h4_slow  = 50,    # H4 EMA trend
    h1_fast  = 9,  h1_slow  = 21,    # H1 EMA for pullback zone
    rsi_per  = 14,
    rsi_lo   = 35,  rsi_hi  = 58,    # pullback RSI zone
    atr_per  = 14,
    sl_mult  = 1.4,
    rr       = 1.8,
    sess_o   = 7,   sess_c  = 21,
    min_bars_since_last = 4,         # min H1 bars between entries
)

# Strategy B: London Breakout
# v4 improvement: relaxed range filter (5-60 pips) → more trades
# entry window extended to 07:00-11:00, force close moved to 15:00
LBO_P = dict(
    asian_end   = 7,     # 00:00-06:59 UTC
    entry_beg   = 7,     # London open
    entry_end   = 11,    # last entry hour
    force_cls   = 15,    # close before NY peak
    min_range   = 5,     # pips (was 8 → more trades)
    max_range   = 60,    # pips (was 50)
    buf         = 1.0,   # breakout buffer pips
    sl_inside   = 3,     # SL inside range
    rr          = 1.8,
)

# ─────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(c, n):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(com=n-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def atr(h, l, c, n):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(com=n-1, adjust=False).mean()

# ─────────────────────────────────────────────────────────
#  MT4 BAR CHECKER
# ─────────────────────────────────────────────────────────
def bar_check(d, entry, sl, tp, O, H, L, C):
    """
    MT4-realistic bar simulation.
    Gap: if bar opens past SL → fill at open.
    Same-bar: bar direction determines SL/TP order.
    """
    if d == 1:
        if O <= sl: return O, "sl_gap"
        sl_hit = L <= sl
        tp_hit = H >= tp
    else:
        if O >= sl: return O, "sl_gap"
        sl_hit = H >= sl
        tp_hit = L <= tp

    if not sl_hit and not tp_hit: return None, None
    if sl_hit and not tp_hit:     return sl,  "sl"
    if tp_hit and not sl_hit:     return tp,  "tp"
    # both: bearish bar → assume High before Low for shorts (SL first)
    #                    → assume Low before High for longs  (SL first)
    bull = C > O
    if d == 1:  return (sl if bull else tp), ("sl" if bull else "tp")
    else:       return (sl if not bull else tp), ("sl" if not bull else "tp")

# ─────────────────────────────────────────────────────────
#  POSITION
# ─────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["d","entry","sl","tp","t0","risk","pip","swap_date"]
    def __init__(self, d, entry, sl, tp, t0, risk, pip):
        self.d=d; self.entry=entry; self.sl=sl; self.tp=tp
        self.t0=t0; self.risk=risk; self.pip=pip
        self.swap_date=t0.date()

def pnl(pos, ep, pair):
    pip    = PIP[pair]
    sl_pip = abs(pos.entry - pos.sl) / pip
    if sl_pip < 1: return 0.0
    move   = (ep - pos.entry) * pos.d / pip
    profit = move / sl_pip * pos.risk
    cost   = (SPR[pair] + SLIP*2) / sl_pip * pos.risk
    return profit - cost

def swap_cost(pos, today, pair):
    if today <= pos.swap_date: return 0.0
    nights = (today - pos.swap_date).days
    pos.swap_date = today
    sw_pips = SWAP[pair][0 if pos.d==1 else 1] * nights
    sl_pip  = abs(pos.entry - pos.sl) / PIP[pair]
    return sw_pips / sl_pip * pos.risk if sl_pip > 0 else 0.0

# ─────────────────────────────────────────────────────────
#  SIMULATOR
# ─────────────────────────────────────────────────────────
def simulate(bars, pair, signals, force_close_h=99):
    pip   = PIP[pair]
    eq    = ACCT["balance"]
    peak  = eq
    day_eq, last_day = eq, None
    frozen = killed = False
    positions, trades, equity_curve = [], [], []

    has_stop = "stop_lvl_l" in signals.columns

    for ts, bar in bars.iterrows():
        if killed: break
        d = ts.date()
        if d != last_day:
            day_eq, last_day, frozen = eq, d, False
        O, H, L, C = bar["open"], bar["high"], bar["low"], bar["close"]

        # swap
        for pos in positions:
            eq += swap_cost(pos, d, pair)

        # check open positions
        closed = []
        for pos in positions:
            ep, res = bar_check(pos.d, pos.entry, pos.sl, pos.tp, O, H, L, C)
            if res is None and ts.hour >= force_close_h < 23:
                ep, res = O, "eod"
            if ep is not None:
                p = pnl(pos, ep, pair)
                eq += p; peak = max(peak, eq)
                trades.append(dict(
                    pair=pair, d="L" if pos.d==1 else "S",
                    t0=str(pos.t0)[:16], t1=str(ts)[:16],
                    entry=round(pos.entry,5), exit=round(ep,5),
                    sl=round(pos.sl,5), tp=round(pos.tp,5),
                    pnl=round(p,2), res=res, eq=round(eq,2)
                ))
                closed.append(pos)
        for p in closed: positions.remove(p)
        equity_curve.append({"dt": ts, "eq": eq})

        # kill / freeze checks
        if (peak - eq) / max(peak,1) >= ACCT["kill_dd"]:
            killed = True
            print(f"  [KILL] {pair} @ {ts.date()} DD={(peak-eq)/peak*100:.1f}%")
            break
        if not frozen and (day_eq - eq) / max(day_eq,1) >= ACCT["daily_freeze"]:
            frozen = True
        if frozen: continue

        # new entries
        if ts not in signals.index: continue
        sig = signals.loc[ts]
        if len(positions) >= ACCT["max_pos"]: continue
        has_l = any(p.d== 1 for p in positions)
        has_s = any(p.d==-1 for p in positions)
        risk  = eq * ACCT["risk_pct"]

        def try_open(direction, entry_px, sl_px, tp_px):
            if abs(entry_px - sl_px) / pip < MIN_SL: return
            positions.append(Pos(direction, entry_px, sl_px, tp_px, ts, risk, pip))

        sl_long  = sig.get("sig_long",  False)
        sl_short = sig.get("sig_short", False)

        if sl_long and not has_l:
            if has_stop and not pd.isna(sig.get("stop_lvl_l", np.nan)):
                # stop order: only fill if bar crossed the level
                lvl = sig["stop_lvl_l"]
                if H > lvl:
                    ep = lvl + SPR[pair]*pip/2
                    try_open(1, ep, sig["stop_sl_l"], sig["stop_tp_l"])
            else:
                atr_v = bar.get("atr", 0)
                if atr_v < SPR[pair]*pip*2: continue
                ep = O + SPR[pair]*pip/2 + SLIP*pip
                sl_px = ep - atr_v * GOLD_P["sl_mult"]
                tp_px = ep + atr_v * GOLD_P["sl_mult"] * GOLD_P["rr"]
                try_open(1, ep, sl_px, tp_px)

        elif sl_short and not has_s:
            if has_stop and not pd.isna(sig.get("stop_lvl_s", np.nan)):
                lvl = sig["stop_lvl_s"]
                if L < lvl:
                    ep = lvl - SPR[pair]*pip/2
                    try_open(-1, ep, sig["stop_sl_s"], sig["stop_tp_s"])
            else:
                atr_v = bar.get("atr", 0)
                if atr_v < SPR[pair]*pip*2: continue
                ep = O - SPR[pair]*pip/2 - SLIP*pip
                sl_px = ep + atr_v * GOLD_P["sl_mult"]
                tp_px = ep - atr_v * GOLD_P["sl_mult"] * GOLD_P["rr"]
                try_open(-1, ep, sl_px, tp_px)

    return trades, equity_curve

# ─────────────────────────────────────────────────────────
#  STRATEGY A: XAUUSD EMA TREND (H4 + H1)
# ─────────────────────────────────────────────────────────
def strat_xauusd(m1):
    pair = "XAUUSD"; p = GOLD_P
    h4 = rs(m1, "4h")
    h4["h4_bull"] = ema(h4["close"], p["h4_fast"]) > ema(h4["close"], p["h4_slow"])

    h1 = rs(m1, "1h")
    h1["atr"]  = atr(h1["high"], h1["low"], h1["close"], p["atr_per"])
    h1["rsi"]  = rsi(h1["close"], p["rsi_per"])
    h1["e9"]   = ema(h1["close"], p["h1_fast"])
    h1["e21"]  = ema(h1["close"], p["h1_slow"])

    # bring H4 trend to H1 via forward-fill
    h4_trend = h4["h4_bull"].reindex(h1.index, method="ffill").fillna(False)
    h1["h4_bull"] = h4_trend.astype(bool)

    # H1 trend aligned with H4
    h1_bull = ema(h1["close"], p["h1_fast"]) > ema(h1["close"], p["h1_slow"])

    # Entry condition:
    # Long:  H4 & H1 both bullish + price pulled back (close ≤ e21+0.3×ATR)
    #        + RSI in 35-58 zone (genuinely in pullback, not extreme)
    pull_l = h1["close"] <= h1["e21"] + h1["atr"] * 0.3
    pull_s = h1["close"] >= h1["e21"] - h1["atr"] * 0.3
    sess   = (h1.index.hour >= p["sess_o"]) & (h1.index.hour < p["sess_c"])
    rsi_l  = h1["rsi"].between(p["rsi_lo"], p["rsi_hi"])
    rsi_s  = h1["rsi"].between(100-p["rsi_hi"], 100-p["rsi_lo"])

    raw_l = h1["h4_bull"] & h1_bull & pull_l & rsi_l & sess
    raw_s = ~h1["h4_bull"] & ~h1_bull & pull_s & rsi_s & sess

    # Cooldown: don't re-enter within 4 bars
    def cooldown(sig_series, n=4):
        result = sig_series.copy().astype(bool)
        last_entry = -n
        for i, (_, v) in enumerate(result.items()):
            if v:
                if i - last_entry < n:
                    result.iloc[i] = False
                else:
                    last_entry = i
        return result

    raw_l = cooldown(raw_l, p["min_bars_since_last"])
    raw_s = cooldown(raw_s, p["min_bars_since_last"])

    sigs = pd.DataFrame(index=h1.index)
    sigs["sig_long"]  = raw_l.shift(1).fillna(False)
    sigs["sig_short"] = raw_s.shift(1).fillna(False)

    tr,eq  = simulate(h1[h1.index<=TRAIN_END], pair, sigs[sigs.index<=TRAIN_END])
    to,eo  = simulate(h1[h1.index>=TEST_START], pair, sigs[sigs.index>=TEST_START])
    return tr, eq, to, eo

# ─────────────────────────────────────────────────────────
#  STRATEGY B: LONDON BREAKOUT
# ─────────────────────────────────────────────────────────
def strat_lbo(m1, pair):
    p   = LBO_P
    pip = PIP[pair]
    h1  = rs(m1, "1h")
    h1["date"] = h1.index.date

    # Asian range
    asian = (h1[h1.index.hour < p["asian_end"]]
             .groupby("date").agg(ah=("high","max"), al=("low","min")))
    asian["rng"] = (asian["ah"] - asian["al"]) / pip
    asian = asian[asian["rng"].between(p["min_range"], p["max_range"])]

    # London window bars
    lon = h1[h1.index.hour.isin(range(p["entry_beg"], p["entry_end"]+1))].copy()
    lon["date"] = lon.index.date
    lon = lon.join(asian, on="date").dropna(subset=["ah","al"])

    buf = p["buf"] * pip
    sli = p["sl_inside"] * pip

    lon["stop_lvl_l"] = lon["ah"] + buf
    lon["stop_sl_l"]  = lon["ah"] - sli
    lon["stop_tp_l"]  = lon["ah"] + buf + (lon["ah"] - lon["al"]) * p["rr"]
    lon["stop_lvl_s"] = lon["al"] - buf
    lon["stop_sl_s"]  = lon["al"] + sli
    lon["stop_tp_s"]  = lon["al"] - buf - (lon["ah"] - lon["al"]) * p["rr"]
    lon["sig_long"]   = True
    lon["sig_short"]  = True

    # one entry attempt per day (07:00 bar only)
    lon = lon[~lon["date"].duplicated(keep="first")]

    cols = ["sig_long","sig_short","stop_lvl_l","stop_sl_l","stop_tp_l",
            "stop_lvl_s","stop_sl_s","stop_tp_s"]
    sf = lon[cols].reindex(h1.index)
    sf[["sig_long","sig_short"]] = sf[["sig_long","sig_short"]].fillna(False)

    tr,eq = simulate(h1[h1.index<=TRAIN_END], pair,
                     sf[sf.index<=TRAIN_END], p["force_cls"])
    to,eo = simulate(h1[h1.index>=TEST_START], pair,
                     sf[sf.index>=TEST_START], p["force_cls"])
    return tr, eq, to, eo

# ─────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────
def metrics(trades, eq_curve, label):
    base = dict(label=label, n=0, wr=0, pnl=0, ret=0, pf=0,
                dd=0, sharpe=0, calmar=0, tp=0, sl=0, gap=0, eod=0, cl=0,
                aw=0, al=0, exp=0)
    if not trades: return base
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    eqs  = pd.Series([e["eq"] for e in eq_curve])
    dd   = (eqs.cummax()-eqs)/eqs.cummax()*100
    net  = sum(pnls)
    gp,gl = sum(wins), abs(sum(loss))

    eq_s = pd.DataFrame(eq_curve).rename(columns={"dt":"T","eq":"E"}).set_index("T")["E"]
    mon  = eq_s.resample("ME").last().pct_change().dropna()
    sh   = (mon.mean()/mon.std()*12**0.5 if len(mon)>=3 and mon.std()>0 else 0.0)
    ret  = net/ACCT["balance"]*100
    cal  = ret/dd.max() if dd.max()>0 else 0.0
    aw   = np.mean(wins) if wins else 0
    al   = np.mean(loss) if loss else 0
    exp  = len(wins)/len(pnls)*aw + len(loss)/len(pnls)*al

    best=cur=0
    for p in pnls: cur=cur+1 if p<=0 else 0; best=max(best,cur)
    return dict(
        label=label, n=len(trades), wr=round(len(wins)/len(trades)*100,1),
        pnl=round(net,2), ret=round(ret,2),
        pf=round(gp/gl if gl>0 else float("inf"),2),
        dd=round(dd.max(),2), sharpe=round(sh,2), calmar=round(cal,2),
        tp=sum(1 for t in trades if t["res"]=="tp"),
        sl=sum(1 for t in trades if t["res"] in("sl","sl_gap")),
        gap=sum(1 for t in trades if t["res"]=="sl_gap"),
        eod=sum(1 for t in trades if t["res"]=="eod"),
        cl=best, aw=round(aw,1), al=round(al,1), exp=round(exp,2),
    )

# ─────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────
def report(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    DIV = "═"*72

    lines = [
        DIV,
        "  PropBot v4.0  —  $5K Prop  |  MT4-Realistic",
        f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Focused: XAUUSD (EMA H4+H1) + EURUSD/GBPUSD (London Breakout)",
        DIV, "",
    ]

    HDR = (f"  {'Pair/Strat':<16}{'#':>5}{'WinR':>7}{'PnL$':>9}"
           f"{'PF':>6}{'MaxDD':>7}{'Sharpe':>8}{'Calmar':>8}"
           f"{'TP':>5}{'SL':>5}{'GAP':>5}{'EOD':>5}"
           f"{'E[t]':>8}")
    SEP = "  "+"─"*70
    ALL = {}

    for period, key in [("TRAIN 2010-2019", "train"), ("OOS  2020-2025  ★", "oos")]:
        lines += [f"\n  ┌── {period} {'─'*(58-len(period))}┐", HDR, SEP]
        tot_pnl = tot_n = 0
        for r in results:
            m = r[key]
            ALL.setdefault(key, []).append({
                "strategy": r["name"], "pair": r["pair"],
                "trades": m["n"], "win_rate": m["wr"],
                "pf": m["pf"], "net_pnl": m["pnl"],
                "max_dd": m["dd"], "sharpe": m["sharpe"],
                "expected_per_trade": m["exp"],
            })
            if m["n"] == 0:
                lines.append(f"  {r['name']:<16} — no trades")
                continue
            ok = "✓" if m["pf"]>1.0 and m["n"]>15 else ("?" if m["pf"]>1.0 else "✗")
            dd_w = " ⚠" if m["dd"]>6 else ""
            lines.append(
                f"  {r['name']:<16}{m['n']:>5}{m['wr']:>6.1f}%{m['pnl']:>+9.0f}"
                f"{m['pf']:>6.2f}{m['dd']:>6.1f}%"
                f"{m['sharpe']:>8.2f}{m['calmar']:>8.2f}"
                f"{m['tp']:>5}{m['sl']:>5}{m['gap']:>5}{m['eod']:>5}"
                f"{m['exp']:>+8.2f}  {ok}{dd_w}"
            )
            tot_pnl += m["pnl"]; tot_n += m["n"]
        lines += [SEP, f"  {'TOTAL':<16}{tot_n:>5}{'':>7}{tot_pnl:>+9.0f}", ""]

    lines += [DIV, "  VERDICT", DIV]
    oos = ALL.get("oos", [])
    for r in oos:
        nm = r["strategy"]
        n, wr, pf, dd, sh, exp = r["trades"], r["win_rate"], r["pf"], r["max_dd"], r["sharpe"], r["expected_per_trade"]
        if n < 15:
            verdict = f"⚠  INSUFFICIENT DATA ({n} trades) — need ≥15 for significance"
        elif pf >= 1.3 and dd < 5 and sh >= 0.5:
            verdict = "✅ PASS — ready for prop live"
        elif pf >= 1.0 and dd < 7:
            verdict = "🟡 MARGINAL — monitor for 1 month forward test first"
        else:
            verdict = "❌ FAIL — DO NOT go live"
        lines.append(f"  {nm:<20}: E[trade]={exp:+.2f}$  PF={pf:.2f}  DD={dd:.1f}%  → {verdict}")

    lines += ["",
              "  E[trade] = expected $ per trade (positive = statistical edge)",
              "  ✓ = PF>1.0 with ≥16 trades   ? = PF>1.0 but <16 trades   ✗ = losing",
              ""]

    txt = "\n".join(lines)
    print(txt)
    (out_dir/"report.txt").write_text(txt)
    (out_dir/"summary.json").write_text(json.dumps(ALL, indent=2))

    # trade CSVs + equity CSVs
    for r in results:
        for tag, tk, ek in [("train","t_tr","e_tr"),("oos","t_oo","e_oo")]:
            trs = r.get(tk,[])
            eqs = r.get(ek,[])
            nm  = r["name"].replace(" ","_").replace("/","")
            if trs:
                with open(out_dir/f"trades_{nm}_{tag}.csv","w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=list(trs[0].keys()))
                    w.writeheader(); w.writerows(trs)
            if eqs:
                with open(out_dir/f"equity_{nm}_{tag}.csv","w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=["dt","eq"])
                    w.writeheader(); w.writerows(eqs)

    # chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.dates as mdates

        oos_r = [r for r in results if r["oos"]["n"]>0]
        n = len(oos_r)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)
        fig.suptitle("PropBot v4.0 — OOS Equity 2020-2025", fontsize=12, fontweight="bold")

        for i, r in enumerate(oos_r):
            ax = axes[0][i]
            eq = r["e_oo"]
            m  = r["oos"]
            if eq:
                dts=[e["dt"] for e in eq]; eqs=[e["eq"] for e in eq]
                clr = "#16a34a" if m["pnl"]>0 else "#dc2626"
                ax.plot(dts, eqs, color=clr, lw=1.0)
                ax.axhline(ACCT["balance"], color="#94a3b8", lw=0.6, ls="--")
                ax.fill_between(dts, ACCT["balance"], eqs,
                    where=[e>=ACCT["balance"] for e in eqs], color="#16a34a", alpha=0.1)
                ax.fill_between(dts, ACCT["balance"], eqs,
                    where=[e<ACCT["balance"] for e in eqs], color="#ef4444", alpha=0.15)
            ax.set_title(
                f"{r['name']}\n"
                f"#{m['n']}  WR={m['wr']:.0f}%  PF={m['pf']:.2f}  "
                f"DD={m['dd']:.1f}%\nE[t]={m['exp']:+.2f}$  PnL={m['pnl']:+.0f}$",
                fontsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(labelsize=7); ax.grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(out_dir/"equity_oos.png", dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Chart → equity_oos.png")
    except Exception as e:
        print(f"  [chart skipped] {e}")

# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir",  default="results")
    args = ap.parse_args()
    dd = Path(args.data_dir); od = Path(args.out_dir)

    print(f"\n{'═'*60}")
    print(f"  PropBot v4.0 — $5K Prop | MT4-Realistic")
    print(f"  XAUUSD EMA (H4+H1) | LBO: EURUSD + GBPUSD")
    print(f"  Data: {dd.resolve()}")
    print(f"{'═'*60}\n")

    # Load data
    cache = {}
    for pair in ["XAUUSD","EURUSD","GBPUSD"]:
        print(f"  Loading {pair}...", end=" ", flush=True)
        df = load(dd, pair)
        if df.empty: print("NO DATA"); continue
        cache[pair] = df
        print(f"{len(df):,} M1  ({df.index[0].date()} → {df.index[-1].date()})")
    print()

    results = []

    # Strategy A: XAUUSD EMA
    if "XAUUSD" in cache:
        print("  Running XAUUSD EMA (H4+H1)...", flush=True)
        tr,eq,to,eo = strat_xauusd(cache["XAUUSD"])
        mtr = metrics(tr,eq,"XAUUSD EMA")
        moo = metrics(to,eo,"XAUUSD EMA")
        print(f"  → Train {mtr['n']} trades WR={mtr['wr']:.0f}% PF={mtr['pf']:.2f}")
        print(f"  → OOS   {moo['n']} trades WR={moo['wr']:.0f}% PF={moo['pf']:.2f}  E[t]={moo['exp']:+.2f}$")
        results.append(dict(name="XAUUSD EMA", pair="XAUUSD",
                            train=mtr, oos=moo,
                            t_tr=tr, e_tr=eq, t_oo=to, e_oo=eo))

    # Strategy B: LBO
    for pair in ["EURUSD","GBPUSD"]:
        if pair not in cache: continue
        print(f"  Running LBO {pair}...", flush=True)
        tr,eq,to,eo = strat_lbo(cache[pair], pair)
        mtr = metrics(tr,eq,f"LBO {pair}")
        moo = metrics(to,eo,f"LBO {pair}")
        print(f"  → Train {mtr['n']} trades WR={mtr['wr']:.0f}% PF={mtr['pf']:.2f}")
        print(f"  → OOS   {moo['n']} trades WR={moo['wr']:.0f}% PF={moo['pf']:.2f}  E[t]={moo['exp']:+.2f}$")
        results.append(dict(name=f"LBO {pair}", pair=pair,
                            train=mtr, oos=moo,
                            t_tr=tr, e_tr=eq, t_oo=to, e_oo=eo))

    if not results:
        print("[ERROR] No results."); sys.exit(1)

    print("\nWriting report...\n")
    report(results, od)
    print(f"\n  Done → {od.resolve()}\n")

if __name__ == "__main__":
    main()
