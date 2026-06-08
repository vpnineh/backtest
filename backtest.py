#!/usr/bin/env python3
"""
PropBot Backtester v2.1  —  Artificial Pessimism Edition
══════════════════════════════════════════════════════════════════
STR-A : EMA Multi-Timeframe  (H4 trend + H1 entry + RSI fixed)
STR-B : London Breakout      (Asian range break at London open)
Account: $5,000 prop firm
Split  : Train 2010-2019  |  OOS 2020-2025
Costs  : Dynamic Spread (Wider at London Open) + Breakout Slippage
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
    risk_pct     = 0.01,      # 1% per trade = $50
    max_open     = 2,
    daily_dd_lim = 0.03,      # 3% daily freeze  ($150)
    max_dd_kill  = 0.07,      # 7% kill switch    ($350)
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

SLIPPAGE_PIP_STD = 0.5        # اسلیپیج استاندارد برای ورودهای پولبک (STR-A)
SLIPPAGE_PIP_LBO = 1.5        # اسلیپیج سنگین‌تر برای سفارش‌های استاپ در بریک‌اوت (STR-B)
LONDON_SPREAD_MULTIPLIER = 2.5 # ضریب واید شدن اسپرد در ساعت باز شدن لندن (07:00 UTC)

TRAIN_END  = pd.Timestamp("2019-12-31", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")

# ─────────────────────────────────────────────────────────────
#  STRATEGY PARAMS  —  fixed, not optimised on test data
# ─────────────────────────────────────────────────────────────
EMA_CFG = dict(
    pairs         = ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD"],
    h4_ema_fast   = 20,
    h4_ema_slow   = 50,
    h1_rsi        = 14,
    h1_rsi_bull   = 45,
    h1_rsi_bear   = 55,
    h1_atr        = 14,
    pullback_band = 0.8,  
    sl_atr        = 1.5,
    rr            = 1.6,
    session_open  = 7,
    session_close = 20,
)

LBO_CFG = dict(
    pairs         = ["EURUSD", "GBPUSD"],
    asian_start   = 0,    
    asian_end     = 7,    
    entry_open    = 7,    
    entry_close   = 10,   
    force_close   = 13,   
    min_range_pip = 8,    
    max_range_pip = 50,   
    buffer_pip    = 2,    
    sl_inside_pip = 3,    
    rr            = 1.5,
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
#  INDICATORS
# ─────────────────────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(c, n):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(com=n-1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def atr(h, l, c, n):
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(com=n-1, adjust=False).mean()

# ─────────────────────────────────────────────────────────────
#  TRADE ENGINE  (Pessimistic Upgrades)
# ─────────────────────────────────────────────────────────────
def get_trade_cost(pair: str, ts: pd.Timestamp, is_breakout: bool) -> float:
    """محاسبه هزینه داینامیک بر اساس استراتژی و زمان"""
    pip = PIP_SIZE[pair]
    base_spread = SPREAD_PIPS.get(pair, 2.0)
    
    # واید شدن اسپرد در سشن لندن
    if ts.hour == 7:
        spread = base_spread * LONDON_SPREAD_MULTIPLIER
    else:
        spread = base_spread
        
    # جریمه اسلیپیج برای بریک‌اوت
    slippage = SLIPPAGE_PIP_LBO if is_breakout else SLIPPAGE_PIP_STD
    
    return (spread + slippage * 2) * pip

class Pos:
    __slots__ = ["d","entry","sl","tp","t0","risk_usd","pip","c_cost"]
    def __init__(self, d, entry, sl, tp, t0, risk_usd, pip, c_cost):
        self.d, self.entry, self.sl, self.tp = d, entry, sl, tp
        self.t0, self.risk_usd, self.pip = t0, risk_usd, pip
        self.c_cost = c_cost # هزینه ذخیره شده در زمان باز شدن معامله

def _close_pos(pos, ep, ts, pair):
    move    = (ep - pos.entry) * pos.d
    pnl_pip = move / pos.pip
    sl_pip  = abs(pos.entry - pos.sl) / pos.pip
    if sl_pip < 1e-5: return 0.0
    pnl_usd = pnl_pip / sl_pip * pos.risk_usd
    pnl_usd -= (pos.c_cost / pos.pip) * (pos.risk_usd / sl_pip)
    return pnl_usd

def run_sim(bars: pd.DataFrame, pair: str, signals: pd.DataFrame,
            force_close_hour: int = 20, is_breakout: bool = False) -> tuple:
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

    has_per_trade_sl = "force_sl" in signals.columns

    for ts in bars.index:
        if killed: break
        row = bars.loc[ts]
        sig = signals.loc[ts] if ts in signals.index else None

        d = ts.date()
        if d != last_day:
            day_eq, last_day, frozen = equity, d, False

        o, h, l = row["open"], row["high"], row["low"]

        # ── close positions (Pessimistic Check Maintained) ──
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
                trades.append(dict(
                    pair=pair, dir="long" if pos.d==1 else "short",
                    open_time=str(pos.t0), close_time=str(ts),
                    open_px=round(pos.entry,5), close_px=round(ep,5),
                    sl=round(pos.sl,5), tp=round(pos.tp,5),
                    cost_pips=round(pos.c_cost/pip, 1), # اضافه شدن گزارش هزینه به رکوردها
                    pnl=round(pnl,2), result=res, equity=round(equity,2),
                ))
                closed.append(pos)
        for p in closed: positions.remove(p)

        eq_curve.append({"datetime": ts, "equity": equity})

        # ── kill / freeze ──
        if (peak - equity) / peak >= acct["max_dd_kill"]:
            killed = True
            print(f"  [KILL] {pair} DD={((peak-equity)/peak*100):.1f}% at {ts.date()}")
            break
        if not frozen and (day_eq - equity) / max(day_eq,1) >= acct["daily_dd_lim"]:
            frozen = True
        if frozen or sig is None: continue

        # ── open positions ──
        if len(positions) >= acct["max_open"]: continue

        has_long  = any(p.d== 1 for p in positions)
        has_short = any(p.d==-1 for p in positions)
        risk_usd  = equity * acct["risk_pct"]

        if sig.get("sig_long") and not has_long:
            c_cost = get_trade_cost(pair, ts, is_breakout)
            entry = o + c_cost / 2
            if has_per_trade_sl:
                sl, tp = sig["force_sl"], sig["force_tp_l"]
            else:
                atr_v = row.get("atr", 0)
                if atr_v < c_cost * 2: continue
                sl = entry - atr_v * EMA_CFG["sl_atr"]
                tp = entry + atr_v * EMA_CFG["sl_atr"] * EMA_CFG["rr"]
            if abs(entry - sl) / pip < 5: continue
            positions.append(Pos(1, entry, sl, tp, ts, risk_usd, pip, c_cost))

        elif sig.get("sig_short") and not has_short:
            c_cost = get_trade_cost(pair, ts, is_breakout)
            entry = o - c_cost / 2
            if has_per_trade_sl:
                sl, tp = sig["force_sl"], sig["force_tp_s"]
            else:
                atr_v = row.get("atr", 0)
                if atr_v < c_cost * 2: continue
                sl = entry + atr_v * EMA_CFG["sl_atr"]
                tp = entry - atr_v * EMA_CFG["sl_atr"] * EMA_CFG["rr"]
            if abs(entry - sl) / pip < 5: continue
            positions.append(Pos(-1, entry, sl, tp, ts, risk_usd, pip, c_cost))

    return trades, eq_curve

# ─────────────────────────────────────────────────────────────
#  STRATEGY A  —  EMA Multi-TF
# ─────────────────────────────────────────────────────────────
def strategy_ema(m1: pd.DataFrame, pair: str) -> tuple:
    cfg = EMA_CFG

    h4 = resample(m1, "4h")
    h4["ef"] = ema(h4["close"], cfg["h4_ema_fast"])
    h4["es"] = ema(h4["close"], cfg["h4_ema_slow"])
    h4["h4_bull"] = (h4["ef"] > h4["es"]).astype(int)
    h4_trend = h4["h4_bull"].reindex(m1.index, method="ffill")

    h1 = resample(m1, "1h")
    h1["rsi_v"] = rsi(h1["close"], cfg["h1_rsi"])
    h1["atr_v"] = atr(h1["high"], h1["low"], h1["close"], cfg["h1_atr"])
    h1["ema20"]  = ema(h1["close"], cfg["h4_ema_fast"])
    h1["h4_bull"] = h4_trend.reindex(h1.index, method="ffill").fillna(0).astype(int)

    near = (h1["close"] - h1["ema20"]).abs() <= cfg["pullback_band"] * h1["atr_v"]
    rsi_bull = h1["rsi_v"] > cfg["h1_rsi_bull"]
    rsi_bear = h1["rsi_v"] < cfg["h1_rsi_bear"]
    in_sess  = (h1.index.hour >= cfg["session_open"]) & \
               (h1.index.hour <  cfg["session_close"])

    raw_long  = (h1["h4_bull"] == 1) & near & rsi_bull & in_sess
    raw_short = (h1["h4_bull"] == 0) & near & rsi_bear & in_sess

    sigs = pd.DataFrame(index=h1.index)
    sigs["sig_long"]  = raw_long.shift(1).fillna(False)
    sigs["sig_short"] = raw_short.shift(1).fillna(False)

    h1["atr"] = h1["atr_v"]

    h1_train = h1[h1.index <= TRAIN_END]
    h1_oos   = h1[h1.index >= TEST_START]
    s_train  = sigs[sigs.index <= TRAIN_END]
    s_oos    = sigs[sigs.index >= TEST_START]

    # پارامتر is_breakout=False
    t_tr, eq_tr = run_sim(h1_train, pair, s_train, cfg["session_close"], False)
    t_oo, eq_oo = run_sim(h1_oos,   pair, s_oos,   cfg["session_close"], False)
    return t_tr, eq_tr, t_oo, eq_oo

# ─────────────────────────────────────────────────────────────
#  STRATEGY B  —  London Breakout
# ─────────────────────────────────────────────────────────────
def strategy_lbo(m1: pd.DataFrame, pair: str) -> tuple:
    cfg = LBO_CFG
    pip = PIP_SIZE[pair]

    h1 = resample(m1, "1h")
    h1["date"] = h1.index.date

    asian_mask = (h1.index.hour >= cfg["asian_start"]) & \
                 (h1.index.hour <  cfg["asian_end"])
    asian = h1[asian_mask].groupby("date").agg(
        asian_high=("high","max"),
        asian_low =("low","min"),
    )
    asian["range_pip"] = (asian["asian_high"] - asian["asian_low"]) / pip

    asian = asian[asian["range_pip"].between(cfg["min_range_pip"], cfg["max_range_pip"])]

    london_mask = (h1.index.hour >= cfg["entry_open"]) & \
                  (h1.index.hour <= cfg["entry_close"])
    london = h1[london_mask].copy()
    london["date"] = london.index.date
    london = london.join(asian, on="date", how="inner")
    london = london.dropna(subset=["asian_high","asian_low"])

    buf      = cfg["buffer_pip"] * pip
    sl_buf   = cfg["sl_inside_pip"] * pip

    london["raw_long"]  = london["open"] > (london["asian_high"] + buf)
    london["raw_short"] = london["open"] < (london["asian_low"]  - buf)

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

    sigs_full = sigs.reindex(h1.index)
    sigs_full[["sig_long","sig_short"]] = sigs_full[["sig_long","sig_short"]].fillna(False)

    h1_train = h1[h1.index <= TRAIN_END]
    h1_oos   = h1[h1.index >= TEST_START]
    s_train  = sigs_full[sigs_full.index <= TRAIN_END]
    s_oos    = sigs_full[sigs_full.index >= TEST_START]

    # پارامتر is_breakout=True
    t_tr, eq_tr = run_sim(h1_train, pair, s_train, cfg["force_close"], True)
    t_oo, eq_oo = run_sim(h1_oos,   pair, s_oos,   cfg["force_close"], True)
    return t_tr, eq_tr, t_oo, eq_oo

# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────
def metrics(trades, eq_curve, label):
    if not trades:
        return dict(label=label, trades=0, net_pnl=0, win_rate=0,
                    pf=0, max_dd=0, sharpe=0, calmar=0,
                    tp_count=0, sl_count=0, eod_count=0, max_consec_l=0)
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
        eod_count=sum(1 for t in trades if t["result"]=="eod"),
        max_consec_l=best,
        avg_win=round(np.mean(wins),2) if wins else 0,
        avg_loss=round(np.mean(loss),2) if loss else 0,
    )

# ─────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────
def write_report(ema_res, lbo_res, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    div = "═"*72
    lines = [
        div,
        f"  PropBot Backtester v2.1  —  Account: $5,000",
        f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Split : Train 2010-2019  |  OOS 2020-2025",
        f"  Costs : Dynamic Spread & Breakout Penalty Included",
        div,"",
    ]

    HDR = (f"  {'Pair':<8} {'#':>5} {'WinR':>6} {'NetPnL':>9}"
           f" {'PF':>5} {'MaxDD':>7} {'Sharpe':>7} {'Calmar':>7}"
           f" {'TP':>4} {'SL':>4} {'EOD':>4}")
    SEP = "  " + "─"*68

    for strat_name, results in [
        ("STRATEGY A — EMA Multi-Timeframe (H4 trend + H1 entry + RSI)", ema_res),
        ("STRATEGY B — London Breakout    (Asian range → London open)",   lbo_res),
    ]:
        lines += ["", f"  ╔{'═'*70}╗",
                  f"  ║  {strat_name:<68}║",
                  f"  ╚{'═'*70}╝"]

        for period, key in [("TRAIN 2010-2019","train"),("OOS  2020-2025","oos")]:
            lines += [f"\n  ┌── {period} {'─'*(58-len(period))}┐", HDR, SEP]
            tot_pnl = tot_trades = 0
            for r in results:
                m = r[key]
                if m["trades"]==0:
                    lines.append(f"  {r['pair']:<8} {'—no trades—':>60}")
                    continue
                flag = " ⚠" if m["max_dd"] > 6 else ""
                lines.append(
                    f"  {m['label']:<8} {m['trades']:>5} {m['win_rate']:>5.1f}%"
                    f" {m['net_pnl']:>+9.0f} {m['pf']:>5.2f}"
                    f" {m['max_dd']:>6.1f}% {m['sharpe']:>7.2f} {m['calmar']:>7.2f}"
                    f" {m['tp_count']:>4} {m['sl_count']:>4} {m['eod_count']:>4}{flag}"
                )
                tot_pnl    += m["net_pnl"]
                tot_trades += m["trades"]
            lines += [SEP,
                      f"  {'TOTAL':<8} {tot_trades:>5}{'':>7} {tot_pnl:>+9.0f}",
                      f"  └{'─'*68}┘"]

    # ── verdict ──
    lines += ["", div, "  VERDICT", div]
    def oos_pnl(res): return sum(r["oos"]["net_pnl"] for r in res)
    
    ea_pnl = oos_pnl(ema_res); la_pnl = oos_pnl(lbo_res)
    winner = "STR-A (EMA)" if ea_pnl > la_pnl else "STR-B (London Breakout)"
    lines += [
        f"  OOS net P&L  →  STR-A: {ea_pnl:+.0f}$   STR-B: {la_pnl:+.0f}$",
        f"  Winner (OOS) →  {winner}",
        "",
    ]

    txt = "\n".join(lines)
    print(txt)
    (out_dir / "report.txt").write_text(txt)

    # save trades & equity
    for name, results in [("ema", ema_res), ("lbo", lbo_res)]:
        for r in results:
            all_t = r["trades_train"] + r["trades_oos"]
            if all_t:
                with open(out_dir/f"trades_{name}_{r['pair']}.csv","w",newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(all_t[0].keys()))
                    w.writeheader(); w.writerows(all_t)
            for tag in ["train","oos"]:
                eq = r.get(f"eq_{tag}",[])
                if eq:
                    with open(out_dir/f"equity_{name}_{r['pair']}_{tag}.csv","w",newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["datetime","equity"])
                        w.writeheader(); w.writerows(eq)

    # summary json
    summary = {"ema": [{
        "pair":r["pair"],"train":r["train"],"oos":r["oos"]
    } for r in ema_res], "lbo": [{
        "pair":r["pair"],"train":r["train"],"oos":r["oos"]
    } for r in lbo_res]}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))

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
    print(f"  PropBot v2.1  —  Artificial Pessimism Edition")
    print(f"  Costs: Dynamic Spread & Breakout Penalty Included")
    print(f"{'═'*60}\n")

    m1_cache = {}
    all_pairs = list(set(EMA_CFG["pairs"] + LBO_CFG["pairs"]))
    for pair in all_pairs:
        print(f"  Loading {pair}...", end=" ", flush=True)
        df = load_pair(data_dir, pair)
        if df.empty:
            print("NO DATA"); continue
        m1_cache[pair] = df
        print(f"{len(df):,} M1  ({df.index[0].date()} → {df.index[-1].date()})")

    print()

    print("  ── Strategy A: EMA Multi-TF ──────────────────────────")
    ema_results = []
    for pair in EMA_CFG["pairs"]:
        if pair not in m1_cache: continue
        print(f"  {pair}...", end=" ", flush=True)
        t_tr,eq_tr,t_oo,eq_oo = strategy_ema(m1_cache[pair], pair)
        m_tr = metrics(t_tr, eq_tr, pair)
        m_oo = metrics(t_oo, eq_oo, pair)
        print(f"Train {m_tr['trades']} trades WR={m_tr['win_rate']:.0f}%  "
              f"OOS {m_oo['trades']} trades WR={m_oo['win_rate']:.0f}%")
        ema_results.append(dict(pair=pair, train=m_tr, oos=m_oo,
                                trades_train=t_tr, trades_oos=t_oo,
                                eq_train=eq_tr, eq_oos=eq_oo))
    print()

    print("  ── Strategy B: London Breakout ────────────────────────")
    lbo_results = []
    for pair in LBO_CFG["pairs"]:
        if pair not in m1_cache: continue
        print(f"  {pair}...", end=" ", flush=True)
        t_tr,eq_tr,t_oo,eq_oo = strategy_lbo(m1_cache[pair], pair)
        m_tr = metrics(t_tr, eq_tr, pair)
        m_oo = metrics(t_oo, eq_oo, pair)
        print(f"Train {m_tr['trades']} trades WR={m_tr['win_rate']:.0f}%  "
              f"OOS {m_oo['trades']} trades WR={m_oo['win_rate']:.0f}%")
        lbo_results.append(dict(pair=pair, train=m_tr, oos=m_oo,
                                trades_train=t_tr, trades_oos=t_oo,
                                eq_train=eq_tr, eq_oos=eq_oo))
    print()

    if not ema_results and not lbo_results:
        print("[ERROR] No results."); sys.exit(1)

    print("Writing report...\n")
    write_report(ema_results, lbo_results, out_dir)
    print(f"\n  Done  →  {out_dir.resolve()}\n")

if __name__ == "__main__":
    main()
