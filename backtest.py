#!/usr/bin/env python3
"""
PropBot Backtester v3.0
═══════════════════════════════════════════════════════════════════════
MT4-Realistic Simulation:
  - Stop/limit orders filled at exact price (not next-bar open)
  - Same-bar SL/TP: bar direction decides order (bearish bar → Low before High)
  - Gap fill: if bar opens past SL, position closed at open (not SL)
  - Overnight swap cost applied daily
  - Spread applied at execution (buy at ask, sell at bid)
  - Minimum SL distance enforced (broker minimum)

Strategy A — EMA Trend (H1)
  Pairs  : EURUSD, GBPUSD, XAUUSD
  Signal : EMA20 > EMA50 (trend) + RSI(14) < 70 for long (not extreme)
           Price touches or dips below EMA20 (pullback entry)
  SL/TP  : 1.5×ATR / 2.5×ATR  (RR ≈ 1.67)
  No forced EOD close — let SL/TP work naturally

Strategy B — London Session Breakout (LBO)
  Pairs  : EURUSD, GBPUSD
  Logic  : Asian range 00:00-07:00 UTC → break at London open
           Entry: stop order at range boundary
           Force close: 13:00 UTC
  SL/TP  : Inside range / range × 1.8

Account : $5,000 prop firm
Split   : Train 2010-2019 | OOS 2020-2025
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
ACCOUNT = dict(
    initial_bal  = 5_000.0,
    risk_pct     = 0.01,      # 1% = $50 per trade
    max_open     = 2,
    daily_dd_lim = 0.03,      # $150 → daily freeze
    max_dd_kill  = 0.07,      # $350 → kill bot
)

# Spread in pips (bid-ask half-spread added to each side)
SPREAD = dict(EURUSD=1.2, GBPUSD=1.5, XAUUSD=30)
PIP    = dict(EURUSD=1e-4, GBPUSD=1e-4, XAUUSD=0.01)

# Overnight swap (pips per night, approximate mid-range values)
# Negative = cost, positive = gain
SWAP_L = dict(EURUSD=-0.5, GBPUSD=-0.7, XAUUSD=-2.5)   # long swap per night
SWAP_S = dict(EURUSD= 0.1, GBPUSD=-0.2, XAUUSD=-1.5)   # short swap per night

MIN_SL_PIPS = 5    # broker minimum SL distance
SLIPPAGE    = 0.5  # extra pips on market fills

TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ──────────────────────────────────────────────────────────────
#  STRATEGY PARAMS  — never tuned on OOS data
# ──────────────────────────────────────────────────────────────
EMA_P = dict(
    pairs        = ["EURUSD", "GBPUSD", "XAUUSD"],
    ema_fast     = 20,
    ema_slow     = 50,
    rsi_period   = 14,
    rsi_max_long = 65,   # don't buy when overbought
    rsi_min_shrt = 35,   # don't sell when oversold
    atr_period   = 14,
    pullback_pct = 1.0,  # price within 1.0×ATR below EMA20 (long)
    sl_atr       = 1.5,
    rr           = 1.67,
    sess_open    = 7,
    sess_close   = 21,
)

LBO_P = dict(
    pairs        = ["EURUSD", "GBPUSD"],
    asian_h_beg  = 0,
    asian_h_end  = 7,    # Asian range: 00:00–06:59 UTC
    entry_h_beg  = 7,
    entry_h_end  = 10,   # enter only in 07:00–10:59 UTC
    force_close  = 13,   # force close at 13:00 UTC (before NY)
    min_range    = 8,    # pips
    max_range    = 50,
    buf_pips     = 1.5,  # stop entry buffer beyond range
    sl_pips_in   = 4,    # SL inside range from break level
    rr           = 1.8,
)

# ──────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────
def _sep(raw): return ";" if raw[:300].decode("utf-8","replace").count(";") > 2 else ","

def _parse(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(
        io.BytesIO(raw), sep=_sep(raw), header=None,
        names=["dt","O","H","L","C","V"],
        dtype={"O":float,"H":float,"L":float,"C":float,"V":float},
        on_bad_lines="skip",
    )
    df["T"] = pd.to_datetime(df["dt"].astype(str).str.strip(),
                             format="%Y%m%d %H%M%S", utc=True, errors="coerce")
    return (df.dropna(subset=["T"])
              .rename(columns={"O":"open","H":"high","L":"low","C":"close"})
              .drop(columns=["dt","V"])
              .set_index("T").sort_index()
              .loc[lambda d: (d>0).all(axis=1)])

def load(data_dir: Path, pair: str) -> pd.DataFrame:
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
    if not frames: return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]

def rs(df, tf):
    return df.resample(tf).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()

# ──────────────────────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────────────────────
def ema_s(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi_s(c,n):
    d=c.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(com=n-1,adjust=False).mean()
    al=l.ewm(com=n-1,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))

def atr_s(h,l,c,n):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(com=n-1,adjust=False).mean()

# ──────────────────────────────────────────────────────────────
#  MT4-REALISTIC BAR CHECKER
# ──────────────────────────────────────────────────────────────
def bar_result(pos_dir, entry, sl, tp, bar_open, bar_high, bar_low, bar_close):
    """
    Given an open position, determine what happened this bar.

    MT4 rules simulated:
    1. Gap fill: if bar opens past SL → close at bar_open (gap fill)
    2. Same-bar SL/TP ambiguity: use bar direction to guess price path
       - Bullish bar (close > open): path Low → High  (check Low/SL first)
       - Bearish bar (close ≤ open): path High → Low  (check High/SL first for short)
    3. If only one level is breached, that's the result.

    Returns (exit_price, result_tag) or (None, None)
    """
    pip_dir = pos_dir  # +1 long, -1 short

    # ── 1. Gap past SL ──
    if pip_dir == 1 and bar_open <= sl:
        return bar_open, "sl_gap"
    if pip_dir == -1 and bar_open >= sl:
        return bar_open, "sl_gap"

    sl_hit = (pip_dir == 1 and bar_low  <= sl) or (pip_dir == -1 and bar_high >= sl)
    tp_hit = (pip_dir == 1 and bar_high >= tp) or (pip_dir == -1 and bar_low  <= tp)

    if not sl_hit and not tp_hit:
        return None, None
    if sl_hit and not tp_hit:
        return sl, "sl"
    if tp_hit and not sl_hit:
        return tp, "tp"

    # ── Both hit same bar: use bar direction ──
    bull = bar_close > bar_open
    if pip_dir == 1:
        # Long: bullish → Low before High → SL checked first (worst case)
        #       bearish → High before Low → TP checked first (favorable)
        if bull:
            return sl, "sl"   # path: open → low(sl) → high(tp) → SL wins
        else:
            return tp, "tp"   # path: open → high → low(sl) → TP wins
    else:
        # Short: bearish → High before Low → SL first (worst case)
        #        bullish → Low before High → TP first
        if not bull:
            return sl, "sl"
        else:
            return tp, "tp"

# ──────────────────────────────────────────────────────────────
#  POSITION CLASS
# ──────────────────────────────────────────────────────────────
class Pos:
    __slots__ = ["d","entry","sl","tp","t0","risk_usd","pip","last_swap_date"]
    def __init__(self, d, entry, sl, tp, t0, risk_usd, pip):
        self.d=d; self.entry=entry; self.sl=sl; self.tp=tp
        self.t0=t0; self.risk_usd=risk_usd; self.pip=pip
        self.last_swap_date = t0.date()

# ──────────────────────────────────────────────────────────────
#  P&L CALCULATION
# ──────────────────────────────────────────────────────────────
def calc_pnl(pos: Pos, exit_price: float, pair: str) -> float:
    pip     = PIP[pair]
    sl_pips = abs(pos.entry - pos.sl) / pip
    if sl_pips < 1: return 0.0
    move_pips = (exit_price - pos.entry) * pos.d / pip
    pnl = move_pips / sl_pips * pos.risk_usd
    # round-trip spread + slippage cost (in pips * risk per pip)
    cost_pips = SPREAD.get(pair, 2) + SLIPPAGE * 2
    pnl -= cost_pips / sl_pips * pos.risk_usd
    return pnl

def apply_swap(pos: Pos, current_date, pair: str) -> float:
    """Apply overnight swap if position held past midnight."""
    if current_date <= pos.last_swap_date:
        return 0.0
    nights = (current_date - pos.last_swap_date).days
    pos.last_swap_date = current_date
    swap_table = SWAP_L if pos.d == 1 else SWAP_S
    swap_pip = swap_table.get(pair, -0.5) * nights
    pip     = PIP[pair]
    sl_pips = abs(pos.entry - pos.sl) / pip
    if sl_pips < 1: return 0.0
    return swap_pip / sl_pips * pos.risk_usd

# ──────────────────────────────────────────────────────────────
#  CORE SIMULATOR
# ──────────────────────────────────────────────────────────────
def simulate(
    bars       : pd.DataFrame,   # OHLC indexed by UTC datetime
    pair       : str,
    signals    : pd.DataFrame,   # must have: sig_long, sig_short (bool)
                                 # optional:  stop_entry_l, stop_sl_l, stop_tp_l
                                 #            stop_entry_s, stop_sl_s, stop_tp_s
    force_close_hour: int = 99,  # 99 = never force close
) -> tuple:

    pip       = PIP[pair]
    positions : list[Pos] = []
    trades    : list[dict] = []
    eq_curve  : list[dict] = []

    equity    = ACCOUNT["initial_bal"]
    peak      = equity
    day_eq    = equity
    last_day  = None
    frozen    = False
    killed    = False

    # pre-check columns
    has_stop = "stop_entry_l" in signals.columns

    for ts, bar in bars.iterrows():
        if killed: break

        d = ts.date()
        if d != last_day:
            day_eq, last_day, frozen = equity, d, False

        O, H, L, C = bar["open"], bar["high"], bar["low"], bar["close"]

        # ── apply swap for overnight holds ──
        for pos in positions:
            swap = apply_swap(pos, d, pair)
            equity += swap

        # ── check existing positions ──
        closed = []
        for pos in positions:
            ep, res = bar_result(pos.d, pos.entry, pos.sl, pos.tp, O, H, L, C)

            # force close (LBO strategy: close before NY open)
            if res is None and ts.hour >= force_close_hour < 23:
                ep, res = O, "eod"

            if ep is not None:
                pnl = calc_pnl(pos, ep, pair)
                equity += pnl
                peak    = max(peak, equity)
                trades.append(dict(
                    pair=pair, dir="L" if pos.d==1 else "S",
                    t_open=str(pos.t0)[:16], t_close=str(ts)[:16],
                    entry=round(pos.entry,5), exit=round(ep,5),
                    sl=round(pos.sl,5), tp=round(pos.tp,5),
                    pnl=round(pnl,2), result=res, eq=round(equity,2),
                ))
                closed.append(pos)

        for p in closed: positions.remove(p)

        eq_curve.append({"dt": ts, "equity": equity})

        # ── safety checks ──
        if (peak - equity) / max(peak, 1) >= ACCOUNT["max_dd_kill"]:
            killed = True
            print(f"  [KILL] {pair} @ {ts.date()} DD={(peak-equity)/peak*100:.1f}%")
            break

        if not frozen and (day_eq - equity) / max(day_eq, 1) >= ACCOUNT["daily_dd_lim"]:
            frozen = True

        if frozen: continue

        # ── new entries ──
        if ts not in signals.index: continue
        sig = signals.loc[ts]

        if len(positions) >= ACCOUNT["max_open"]: continue

        has_l = any(p.d == 1 for p in positions)
        has_s = any(p.d ==-1 for p in positions)
        risk  = equity * ACCOUNT["risk_pct"]

        def open_pos(d, entry, sl, tp):
            sl_p = abs(entry - sl) / pip
            if sl_p < MIN_SL_PIPS: return False
            positions.append(Pos(d, entry, sl, tp, ts, risk, pip))
            return True

        if sig["sig_long"] and not has_l:
            if has_stop and not pd.isna(sig.get("stop_entry_l")):
                # Stop entry: only if bar's High crossed the stop level
                lvl = sig["stop_entry_l"]
                if H > lvl:
                    entry = lvl + SPREAD.get(pair,1.2)*pip/2
                    open_pos(1, entry, sig["stop_sl_l"], sig["stop_tp_l"])
            else:
                # Market entry at next bar open (already shifted)
                entry = O + SPREAD.get(pair,1.2)*pip/2 + SLIPPAGE*pip
                atr_v = bar.get("atr", 0)
                if atr_v < SPREAD.get(pair,1.2)*pip*2: continue
                sl = entry - atr_v * EMA_P["sl_atr"]
                tp = entry + atr_v * EMA_P["sl_atr"] * EMA_P["rr"]
                open_pos(1, entry, sl, tp)

        elif sig["sig_short"] and not has_s:
            if has_stop and not pd.isna(sig.get("stop_entry_s")):
                lvl = sig["stop_entry_s"]
                if L < lvl:
                    entry = lvl - SPREAD.get(pair,1.2)*pip/2
                    open_pos(-1, entry, sig["stop_sl_s"], sig["stop_tp_s"])
            else:
                entry = O - SPREAD.get(pair,1.2)*pip/2 - SLIPPAGE*pip
                atr_v = bar.get("atr", 0)
                if atr_v < SPREAD.get(pair,1.2)*pip*2: continue
                sl = entry + atr_v * EMA_P["sl_atr"]
                tp = entry - atr_v * EMA_P["sl_atr"] * EMA_P["rr"]
                open_pos(-1, entry, sl, tp)

    return trades, eq_curve

# ──────────────────────────────────────────────────────────────
#  STRATEGY A — EMA TREND (H1)
# ──────────────────────────────────────────────────────────────
def strat_ema(m1: pd.DataFrame, pair: str):
    p  = EMA_P
    h1 = rs(m1, "1h")
    h1["ema_f"] = ema_s(h1["close"], p["ema_fast"])
    h1["ema_s"] = ema_s(h1["close"], p["ema_slow"])
    h1["rsi"]   = rsi_s(h1["close"], p["rsi_period"])
    h1["atr"]   = atr_s(h1["high"], h1["low"], h1["close"], p["atr_period"])

    # Trend: uptrend when fast > slow
    bull = h1["ema_f"] > h1["ema_s"]
    bear = h1["ema_f"] < h1["ema_s"]

    # Pullback into EMA20 zone: low touched EMA20 area (within pullback_pct × ATR)
    near_l = (h1["low"]  <= h1["ema_f"] + h1["atr"] * p["pullback_pct"])
    near_s = (h1["high"] >= h1["ema_f"] - h1["atr"] * p["pullback_pct"])

    # Session filter
    sess = (h1.index.hour >= p["sess_open"]) & (h1.index.hour < p["sess_close"])

    # Entry condition (current bar)
    raw_l = bull & near_l & (h1["rsi"] < p["rsi_max_long"])  & sess
    raw_s = bear & near_s & (h1["rsi"] > p["rsi_min_shrt"]) & sess

    # Shift 1: signal fires next bar (no lookahead)
    sigs = pd.DataFrame(index=h1.index)
    sigs["sig_long"]  = raw_l.shift(1).fillna(False)
    sigs["sig_short"] = raw_s.shift(1).fillna(False)

    tr, eq = (lambda s, sg: simulate(s, pair, sg))(
        h1[h1.index <= TRAIN_END], sigs[sigs.index <= TRAIN_END])
    to, eo = (lambda s, sg: simulate(s, pair, sg))(
        h1[h1.index >= TEST_START], sigs[sigs.index >= TEST_START])
    return tr, eq, to, eo

# ──────────────────────────────────────────────────────────────
#  STRATEGY B — LONDON BREAKOUT
# ──────────────────────────────────────────────────────────────
def strat_lbo(m1: pd.DataFrame, pair: str):
    p   = LBO_P
    pip = PIP[pair]
    h1  = rs(m1, "1h")
    h1["date"] = h1.index.date

    # Asian range (00:00–06:59 UTC)
    asian = (h1[h1.index.hour < p["asian_h_end"]]
             .groupby("date")
             .agg(a_high=("high","max"), a_low=("low","min")))
    asian["range_pip"] = (asian["a_high"] - asian["a_low"]) / pip
    asian = asian[asian["range_pip"].between(p["min_range"], p["max_range"])]

    # London bars (07–10 UTC)
    lon = h1[h1.index.hour.isin(range(p["entry_h_beg"], p["entry_h_end"]+1))].copy()
    lon["date"] = lon.index.date
    lon = lon.join(asian, on="date").dropna(subset=["a_high","a_low"])

    buf   = p["buf_pips"] * pip
    sl_in = p["sl_pips_in"] * pip

    # Stop entry levels
    lon["stop_entry_l"] = lon["a_high"] + buf
    lon["stop_sl_l"]    = lon["a_high"] - sl_in
    lon["stop_tp_l"]    = lon["a_high"] + buf + (lon["a_high"] - lon["a_low"]) * p["rr"]

    lon["stop_entry_s"] = lon["a_low"] - buf
    lon["stop_sl_s"]    = lon["a_low"]  + sl_in
    lon["stop_tp_s"]    = lon["a_low"]  - buf - (lon["a_high"] - lon["a_low"]) * p["rr"]

    # Signal: arm the stop orders at London open bars
    # sig_long = we WANT a long breakout (armed); engine checks if bar breaks level
    lon["sig_long"]  = True
    lon["sig_short"] = True

    # One entry per day: keep only first London bar per date (07:00)
    lon = lon[~lon["date"].duplicated(keep="first")]

    sigs_full = lon[["sig_long","sig_short",
                     "stop_entry_l","stop_sl_l","stop_tp_l",
                     "stop_entry_s","stop_sl_s","stop_tp_s"]].reindex(h1.index)
    sigs_full[["sig_long","sig_short"]] = sigs_full[["sig_long","sig_short"]].fillna(False)

    tr, eq = simulate(h1[h1.index <= TRAIN_END],
                      pair, sigs_full[sigs_full.index <= TRAIN_END],
                      force_close_hour=p["force_close"])
    to, eo = simulate(h1[h1.index >= TEST_START],
                      pair, sigs_full[sigs_full.index >= TEST_START],
                      force_close_hour=p["force_close"])
    return tr, eq, to, eo

# ──────────────────────────────────────────────────────────────
#  METRICS
# ──────────────────────────────────────────────────────────────
def metrics(trades, eq_curve, label):
    base = dict(label=label, trades=0, win_rate=0, net_pnl=0,
                pf=0, max_dd=0, sharpe=0, calmar=0,
                tp=0, sl=0, sl_gap=0, eod=0, consec_l=0,
                avg_win=0, avg_loss=0, ret_pct=0)
    if not trades: return base

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    eqs  = pd.Series([e["equity"] for e in eq_curve])
    dd   = (eqs.cummax() - eqs) / eqs.cummax() * 100

    net  = sum(pnls)
    gp, gl = sum(wins), abs(sum(loss))

    eq_s    = pd.DataFrame(eq_curve).rename(columns={"dt":"datetime"}).set_index("datetime")["equity"]
    monthly = eq_s.resample("ME").last().pct_change().dropna()
    sharpe  = (monthly.mean()/monthly.std()*12**0.5
               if len(monthly) >= 3 and monthly.std() > 0 else 0.0)
    ret_pct = net / ACCOUNT["initial_bal"] * 100
    calmar  = ret_pct / dd.max() if dd.max() > 0 else 0.0

    best = cur = 0
    for p in pnls:
        cur = cur+1 if p<=0 else 0; best=max(best,cur)

    return dict(
        label=label, trades=len(trades),
        win_rate=round(len(wins)/len(trades)*100, 1),
        net_pnl=round(net,2), ret_pct=round(ret_pct,2),
        pf=round(gp/gl if gl>0 else float("inf"), 2),
        max_dd=round(dd.max(),2), sharpe=round(sharpe,2), calmar=round(calmar,2),
        avg_win=round(np.mean(wins),2) if wins else 0,
        avg_loss=round(np.mean(loss),2) if loss else 0,
        tp=sum(1 for t in trades if t["result"]=="tp"),
        sl=sum(1 for t in trades if t["result"] in ("sl","sl_gap")),
        sl_gap=sum(1 for t in trades if t["result"]=="sl_gap"),
        eod=sum(1 for t in trades if t["result"]=="eod"),
        consec_l=best,
    )

# ──────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────
def report(ema_res, lbo_res, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    DIV = "═"*74
    HDR = (f"  {'Pair':<8}{'#':>5}{'WinR':>7}{'NetPnL$':>10}"
           f"{'PF':>6}{'MaxDD':>7}{'Sharpe':>8}{'Calmar':>8}"
           f"{'TP':>5}{'SL':>5}{'GAP':>5}{'EOD':>5}")
    SEP = "  " + "─"*72

    lines = [
        DIV,
        f"  PropBot v3.0  —  $5,000 Prop Account  |  MT4-Realistic Simulation",
        f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Train 2010–2019  |  OOS 2020–2025  (OOS is what matters)",
        f"  Costs: spread + {SLIPPAGE}pip slippage + overnight swap",
        f"  Bar logic: gap-fill at open | same-bar direction for SL/TP order",
        DIV, "",
    ]

    all_summary = {}

    for strat_nm, results in [("A — EMA Trend H1", ema_res), ("B — London Breakout", lbo_res)]:
        lines += [f"  ╔{'═'*70}╗",
                  f"  ║  Strategy {strat_nm:<60}║",
                  f"  ╚{'═'*70}╝"]
        all_summary[strat_nm.split(" ")[0]] = []

        for period, key in [("TRAIN 2010-2019", "train"), ("OOS  2020-2025 ★", "oos")]:
            lines += [f"\n  ┌── {period} {'─'*(60-len(period))}┐", HDR, SEP]
            tot_pnl = tot_trades = 0

            for r in results:
                m = r[key]
                all_summary[strat_nm.split(" ")[0]].append(
                    {"pair": r["pair"], "period": key,
                     "trades": m["trades"], "win_rate": m["win_rate"],
                     "pf": m["pf"], "net_pnl": m["net_pnl"],
                     "max_dd": m["max_dd"], "sharpe": m["sharpe"]})

                if m["trades"] == 0:
                    lines.append(f"  {r['pair']:<8} — no trades generated")
                    continue

                flag = " ⚠ DD>" if m["max_dd"] > 6 else ""
                profitable = "✓" if m["pf"] > 1.0 else "✗"
                lines.append(
                    f"  {m['label']:<8}{m['trades']:>5}"
                    f"{m['win_rate']:>6.1f}%{m['net_pnl']:>+10.0f}"
                    f"{m['pf']:>6.2f}{m['max_dd']:>6.1f}%"
                    f"{m['sharpe']:>8.2f}{m['calmar']:>8.2f}"
                    f"{m['tp']:>5}{m['sl']:>5}{m['sl_gap']:>5}{m['eod']:>5}"
                    f"  {profitable}{flag}"
                )
                tot_pnl += m["net_pnl"]; tot_trades += m["trades"]

            lines += [SEP,
                      f"  {'TOTAL':<8}{tot_trades:>5}{'':>7}{tot_pnl:>+10.0f}",
                      f"  └{'─'*72}┘"]

    # ── verdict ──
    def oos_pnl(res): return sum(r["oos"]["net_pnl"] for r in res)
    def oos_pf(res):
        tp = sum(r["oos"]["tp"] for r in res)
        sl = sum(r["oos"]["sl"] for r in res)
        win_pnl = sum(r["oos"]["avg_win"]*r["oos"]["tp"] for r in res)
        los_pnl = abs(sum(r["oos"]["avg_loss"]*r["oos"]["sl"] for r in res))
        return win_pnl/los_pnl if los_pnl>0 else 0

    ea, la = oos_pnl(ema_res), oos_pnl(lbo_res)
    winner = "Strategy A (EMA)" if ea > la else "Strategy B (London Breakout)"

    lines += [
        "", DIV, "  VERDICT", DIV,
        f"  OOS P&L  →  Str-A: {ea:+.0f}$    Str-B: {la:+.0f}$",
        f"  Winner   →  {winner}",
        "",
        "  Go/No-Go criteria for PROP LIVE:",
        "    ✓ OOS Profit Factor > 1.3",
        "    ✓ OOS Max DD < 5%",
        "    ✓ OOS Sharpe > 0.5",
        "    ✓ OOS Win Rate × RR > 1.0  (edge confirmed)",
        "    ✗ Any of above fails → DO NOT go live",
        "",
        f"  Avg Win / Avg Loss analysis:",
    ]
    for nm, res in [("Str-A", ema_res), ("Str-B", lbo_res)]:
        for r in res:
            m = r["oos"]
            if m["trades"] < 5: continue
            edge = m["win_rate"]/100 * m["avg_win"] + (1-m["win_rate"]/100)*m["avg_loss"]
            lines.append(
                f"    {nm} {r['pair']}: E[trade]={edge:+.2f}$  "
                f"WR={m['win_rate']:.0f}% × avgW={m['avg_win']:.0f}$  "
                f"LR={(1-m['win_rate']/100)*100:.0f}% × avgL={m['avg_loss']:.0f}$"
            )
    lines += ["", "  NOTE: sl_gap = SL triggered on gap open (worst-case fill)", ""]

    txt = "\n".join(lines)
    print(txt)
    (out_dir/"report.txt").write_text(txt)
    (out_dir/"summary.json").write_text(json.dumps(all_summary, indent=2))

    # trade CSVs
    for nm, results in [("ema", ema_res), ("lbo", lbo_res)]:
        for r in results:
            for tag, key in [("train","trades_tr"), ("oos","trades_oo")]:
                trs = r.get(key, [])
                if not trs: continue
                with open(out_dir/f"trades_{nm}_{r['pair']}_{tag}.csv","w",newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(trs[0].keys()))
                    w.writeheader(); w.writerows(trs)

    # equity CSVs
    for nm, results in [("ema", ema_res), ("lbo", lbo_res)]:
        for r in results:
            for tag, key in [("train","eq_tr"), ("oos","eq_oo")]:
                eq = r.get(key, [])
                if not eq: continue
                with open(out_dir/f"equity_{nm}_{r['pair']}_{tag}.csv","w",newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["dt","equity"])
                    w.writeheader(); w.writerows(eq)

    # chart
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.dates as mdates

        all_pairs_ema = [r["pair"] for r in ema_res]
        all_pairs_lbo = [r["pair"] for r in lbo_res]
        n = max(len(all_pairs_ema), len(all_pairs_lbo))
        fig, axes = plt.subplots(2, max(n,1), figsize=(6*max(n,1), 8), squeeze=False)
        fig.suptitle("PropBot v3.0 — OOS Equity (2020-2025)", fontsize=12, fontweight="bold")

        for row_i, (nm, results, clr) in enumerate([
            ("Str-A EMA", ema_res, "#1d4ed8"),
            ("Str-B LBO", lbo_res, "#15803d"),
        ]):
            for col_i, r in enumerate(results):
                ax = axes[row_i][col_i]
                eq = r.get("eq_oo", [])
                if eq:
                    dts = [e["dt"]     for e in eq]
                    eqs = [e["equity"] for e in eq]
                    ax.plot(dts, eqs, color=clr, lw=0.9)
                    ax.axhline(ACCOUNT["initial_bal"], color="#94a3b8", lw=0.5, ls="--")
                    ax.fill_between(dts, ACCOUNT["initial_bal"], eqs,
                                    where=[e < ACCOUNT["initial_bal"] for e in eqs],
                                    color="#ef4444", alpha=0.2)
                    ax.fill_between(dts, ACCOUNT["initial_bal"], eqs,
                                    where=[e >= ACCOUNT["initial_bal"] for e in eqs],
                                    color=clr, alpha=0.1)
                m = r["oos"]
                ax.set_title(
                    f"{nm} | {r['pair']}\n"
                    f"WR={m['win_rate']:.0f}%  PF={m['pf']:.2f}  "
                    f"DD={m['max_dd']:.1f}%  PnL={m['net_pnl']:+.0f}$",
                    fontsize=8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.tick_params(labelsize=7); ax.grid(alpha=0.2)

            for col_i in range(len(results), n):
                axes[row_i][col_i].set_visible(False)

        plt.tight_layout()
        plt.savefig(out_dir/"equity_oos.png", dpi=140, bbox_inches="tight")
        plt.close()
        print(f"\n  [OK] Chart → equity_oos.png")
    except Exception as e:
        print(f"  [chart] {e}")

# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir",  default="results")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    print(f"\n{'═'*60}")
    print(f"  PropBot Backtester v3.0  —  $5K Prop Firm")
    print(f"  MT4-realistic: gap-fill | bar-direction SL/TP | swap")
    print(f"  Str-A: EMA Trend H1    Str-B: London Breakout")
    print(f"  Data: {data_dir.resolve()}")
    print(f"{'═'*60}\n")

    # load M1 once, share across strategies
    all_pairs = list(dict.fromkeys(EMA_P["pairs"] + LBO_P["pairs"]))
    cache = {}
    for pair in all_pairs:
        if pair not in PIP: continue
        print(f"  Loading {pair}...", end=" ", flush=True)
        df = load(data_dir, pair)
        if df.empty: print("NO DATA"); continue
        cache[pair] = df
        print(f"{len(df):,} M1  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
    print()

    # Strategy A
    print("  ── Strategy A: EMA Trend ──────────────────────────────")
    ema_res = []
    for pair in EMA_P["pairs"]:
        if pair not in cache: continue
        print(f"  {pair}...", end=" ", flush=True)
        tr, eq, to, eo = strat_ema(cache[pair], pair)
        mtr = metrics(tr, eq, pair); moo = metrics(to, eo, pair)
        print(f"  Train {mtr['trades']} tr WR={mtr['win_rate']:.0f}%  "
              f"OOS {moo['trades']} tr WR={moo['win_rate']:.0f}% PF={moo['pf']:.2f}")
        ema_res.append(dict(pair=pair, train=mtr, oos=moo,
                            trades_tr=tr, trades_oo=to, eq_tr=eq, eq_oo=eo))
    print()

    # Strategy B
    print("  ── Strategy B: London Breakout ────────────────────────")
    lbo_res = []
    for pair in LBO_P["pairs"]:
        if pair not in cache: continue
        print(f"  {pair}...", end=" ", flush=True)
        tr, eq, to, eo = strat_lbo(cache[pair], pair)
        mtr = metrics(tr, eq, pair); moo = metrics(to, eo, pair)
        print(f"  Train {mtr['trades']} tr WR={mtr['win_rate']:.0f}%  "
              f"OOS {moo['trades']} tr WR={moo['win_rate']:.0f}% PF={moo['pf']:.2f}")
        lbo_res.append(dict(pair=pair, train=mtr, oos=moo,
                            trades_tr=tr, trades_oo=to, eq_tr=eq, eq_oo=eo))
    print()

    if not ema_res and not lbo_res:
        print("[ERROR] No results."); sys.exit(1)

    print("Writing report...\n")
    report(ema_res, lbo_res, out_dir)
    print(f"\n  Done → {out_dir.resolve()}\n")

if __name__ == "__main__":
    main()
