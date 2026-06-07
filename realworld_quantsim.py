"""
CorrArb Prop Simulator — v9.5 DIRECT CROSS DATA (Realistic Execution)
==============================================================================
استراتژی:
 - ترید مستقیم روی جفت‌ارزهای کراس (بدون لگ‌های سینتتیک).
 - کاهش هزینه‌های تراکنش (یک بار کمیسیون، یک بار اسپرد).
 - حفظ منطق واقع‌گرایانه (محاسبه سایه‌ها برای DD و اسلیپیج در خروج‌های مارکت).
"""

import pandas as pd
import numpy as np
import glob
import zipfile
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ ──
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05        
    max_daily_loss_pct = 0.05        
    max_total_dd_pct   = 0.10        

    # ── مدیریت ریسک ──
    risk_base_pct      = 0.015       
    risk_min_pct       = 0.0075      
    consec_loss_n      = 2
    risk_reduce        = 0.5

    # 🔴 هزینه‌های واقعی (Direct Trading)
    spread_pips        = 2.0         # میانگین اسپرد برای کراس‌ها در پراپ
    commission_per_lot = 7.0         # فقط ۷ دلار (یک لگ)
    slippage_pips      = 0.5         

    # ── مشخصات ──
    pip      = 0.0001
    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── پارامترهای z-score ──
    z_fast_period      = 96          
    z_entry            = 2.1
    z_exit_partial     = 0.5         
    z_exit_full        = 0.0         
    z_stop_margin      = 4.0         
    min_net_profit_usd = 20.0        

    # ── فیلترها ──
    # نکته: در ترید مستقیم، دیگر corr_min روی لگ‌ها نداریم، فقط رفتار Mean Reversion خود جفت‌ارز را بررسی می‌کنیم.
    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2           

    # ── خروج اضطراری ──
    sl_pips            = 30.0
    tp_pips            = 90.0
    time_stop_bars     = 36          

    # ── فیلتر ATR ──
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    # ── فیلتر رژیم: Variance Ratio ──
    vr_period          = 200         
    vr_k               = 4           
    vr_max             = 0.90        

    # ── Partial Exit ──
    partial_ratio      = 0.50        

    # 🔴 نرخ تبدیل تقریبی برای محاسبه PnL دلاری
    # در واقعیت این نرخ لحظه‌ای است، اما برای تست کراس‌ها از یک میانگین استفاده می‌کنیم.
    approx_quote_rate = {
        'EURGBP': 1.25,  # 1 GBP = ~1.25 USD
        'AUDNZD': 0.60   # 1 NZD = ~0.60 USD
    }

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING (فقط ZIP‌های مستقیم)
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_zip(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths: raise FileNotFoundError(f"No files found: {pattern}")
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p, 'r') as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if csv_name is None: continue
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=';', header=None, names=['ts', 'o', 'h', 'l', 'c', 'v'])
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Skip {os.path.basename(p)}: {e}")

    if not frames: raise ValueError(f"No valid data loaded from: {pattern}")
    raw = pd.concat(frames, ignore_index=True).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw

def to_15min(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        'o': raw['o'].resample('15min').first(),
        'h': raw['h'].resample('15min').max(),
        'l': raw['l'].resample('15min').min(),
        'c': raw['c'].resample('15min').last(),
    }).dropna()
    # حذف دیتای آخر هفته
    return df[df.index.weekday < 5].copy()

def load_all_direct_pairs() -> dict:
    print("\n  Loading Direct Cross Pair Datasets...")
    pairs = {}
    
    # EURGBP
    try:
        raw_eg = load_raw_zip('data/*EURGBP*.zip')
        df_eg = to_15min(raw_eg)
        pairs['EURGBP'] = {'df': df_eg}
        print(f"  ✅ EURGBP : {len(df_eg):>7,} candles | {df_eg.index[0].date()} → {df_eg.index[-1].date()}")
    except Exception as e: print(f"  ❌ EURGBP : {e}")

    # AUDNZD
    try:
        raw_an = load_raw_zip('data/*AUDNZD*.zip')
        df_an = to_15min(raw_an)
        pairs['AUDNZD'] = {'df': df_an}
        print(f"  ✅ AUDNZD : {len(df_an):>7,} candles | {df_an.index[0].date()} → {df_an.index[-1].date()}")
    except Exception as e: print(f"  ❌ AUDNZD : {e}")
    
    if not pairs: raise RuntimeError("No direct pairs loaded. Check data/ directory names.")
    return pairs

# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION (Direct Mean Reversion)
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_variance_ratio(series: pd.Series, k: int, window: int) -> pd.Series:
    r1 = series.diff(1)
    rk = series.diff(k)
    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()
    return vark / (k * var1.replace(0, np.nan))

def compute_signals(pair_name: str, pair_info: dict) -> tuple:
    C  = Config
    df = pair_info['df']

    # در ترید مستقیم، Z-Score مستقیماً روی قیمت क्लوز اعمال می‌شود (یا لگاریتم آن)
    log_price = np.log(df['c'])
    z_mean    = log_price.rolling(C.z_fast_period).mean()
    z_std     = log_price.rolling(C.z_fast_period).std()
    z_score   = (log_price - z_mean) / z_std.replace(0, np.nan)

    vr        = calc_variance_ratio(log_price, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

    atr    = calc_atr(df['h'], df['l'], df['c'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    hour    = pd.Series(df.index.hour,       index=df.index)
    dow     = pd.Series(df.index.dayofweek,  index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    long_cond  = (z_score < -C.z_entry) & vol_ok & time_ok & regime_ok
    short_cond = (z_score >  C.z_entry) & vol_ok & time_ok & regime_ok

    sig = pd.Series(0, index=df.index)
    sig[long_cond]  =  1
    sig[short_cond] = -1
    sig = sig.where(sig != sig.shift(), 0) 

    n = int((sig != 0).sum())
    l = int((sig ==  1).sum())
    s = int((sig == -1).sum())
    r = int(regime_ok.sum())
    print(f"    {pair_name}: {n:,} signals (L:{l} | S:{s}) | Regime OK: {r:,} bars")
    return sig, z_score

# ═══════════════════════════════════════════════════════════════════════════
#  FINANCIAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl(direction: int, entry_px: float, exit_px: float, lot: float, pair_name: str, apply_slippage: bool = False) -> float:
    C = Config
    quote_rate = C.approx_quote_rate.get(pair_name, 1.0)
    
    if apply_slippage:
        exit_px = exit_px - (direction * C.slippage_pips * C.pip)

    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    gross_usd   = gross_quote * quote_rate
    commission  = C.commission_per_lot * lot
    return gross_usd - commission

def calc_lot(equity: float, sl_pips: float, consec_loss: int, pair_name: str) -> float:
    C = Config
    quote_rate = C.approx_quote_rate.get(pair_name, 1.0)
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
    pip_value_usd = C.pip * C.lot_size * quote_rate
    risk_usd = equity * risk
    raw = risk_usd / (sl_pips * pip_value_usd)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)

def new_acc(ts) -> dict:
    C = Config
    return {'equity': C.initial_balance, 'start_ts': ts, 'trades': [], 'blown': False, 'blown_rsn': '', 'peak': C.initial_balance, 'consec_loss': 0}

# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs: dict, pair_signals: dict) -> dict:
    C          = Config
    pip        = C.pip
    pair_names = list(pairs.keys())

    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)

    pa = {}
    for name in pair_names:
        df_p  = pairs[name]['df'].reindex(common_idx).ffill()
        sig_s, z_s = pair_signals[name]
        pa[name] = {
            'o':   df_p['o'].values.astype(float),
            'c':   df_p['c'].values.astype(float),
            'h':   df_p['h'].values.astype(float),
            'l':   df_p['l'].values.astype(float),
            'sig': sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z':   z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
        }

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    acc             = new_acc(common_idx[C.warmup])
    total_withdrawn = 0.0
    acc_num         = 1
    day_start_eq    = C.initial_balance
    all_trades      = []
    acc_logs        = []
    eq_curve        = []

    positions    = {name: None for name in pair_names}
    trades_today = {name: 0    for name in pair_names}
    pending_sig  = {name: 0    for name in pair_names}

    print(f"\n  ▶ Running Direct Cross Prop Simulator v9.5...")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq = acc['equity']
        eq_curve.append((ts, round(eq, 4)))

        if eq > acc['peak']: acc['peak'] = eq

        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0

        if (bar - C.warmup) % 100_000 == 0 and bar > C.warmup:
            pct = (bar - C.warmup) / (n_bars - C.warmup) * 100
            print(f"    Progress: {pct:5.1f}% | Eq: ${acc['equity']:,.2f} | Bank: ${total_withdrawn:,.2f}", end='\r')

        if acc['blown']:
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': acc['blown_rsn'], 'pnl': acc['equity'] - C.initial_balance})
            print(f"\n    💥 #{acc_num:>3} | {ts.date()} | Eq: ${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names:
                trades_today[name] = 0; pending_sig[name] = 0; positions[name] = None
            continue

        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0 and positions[name] is None and trades_today[name] < C.max_trades_day):
                sv  = pending_sig[name]
                lot = calc_lot(acc['equity'], C.sl_pips, acc['consec_loss'], name)
                # در ورود با مارکت معمولاً اسپرد + اسلیپیج محاسبه می‌شود
                ep  = a['o'][bar] + sv * (C.slippage_pips + C.spread_pips / 2) * pip
                sl  = ep - sv * C.sl_pips * pip
                tp  = ep + sv * C.tp_pips * pip
                positions[name] = {
                    'pair': name, 'dir': sv, 'lot': lot, 'lot_remaining': lot, 
                    'partial_done': False, 'entry': ep, 'sl': sl, 'tp': tp, 
                    'entry_ts': ts, 'entry_bar': bar
                }
                trades_today[name] += 1
            pending_sig[name] = 0

        # محاسبه DD با بدترین قیمت کندل
        total_float = 0.0
        for name in pair_names:
            pos = positions[name]
            if pos is not None:
                a = pa[name]
                worst_px = a['l'][bar] if pos['dir'] == 1 else a['h'][bar]
                total_float += calc_pnl(pos['dir'], pos['entry'], worst_px, pos['lot_remaining'], name)

        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            acc['blown']     = True
            acc['blown_rsn'] = "Wick/DailyDD" if current_eq <= daily_limit else "TotalDD"
            for name in pair_names:
                pos = positions[name]
                if pos is None: continue
                a   = pa[name]
                pnl = calc_pnl(pos['dir'], pos['entry'], a['c'][bar], pos['lot_remaining'], name)
                acc['equity'] += pnl
                rec = _make_rec(pos, a['c'][bar], ts, pnl, 'BLOWN', pos['lot_remaining'])
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None
            continue

        for name in pair_names:
            pos = positions[name]
            if pos is None: continue

            a       = pa[name]
            cp      = a['c'][bar]
            d       = pos['dir']
            ep      = pos['entry']
            zn      = a['z'][bar]
            lot_rem = pos['lot_remaining']

            if not pos['partial_done'] and not np.isnan(zn):
                hit_p = ((d ==  1 and zn >= -C.z_exit_partial) or (d == -1 and zn <=  C.z_exit_partial))
                if hit_p:
                    p_lot = round(lot_rem * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl(d, ep, cp, p_lot, name, apply_slippage=True)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            rec = _make_rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(rec)
                            acc['trades'].append(rec)
                            pos['lot_remaining'] = round(lot_rem - p_lot, 2)
                            pos['partial_done']  = True
                            pos['sl'] = pos['entry']
                            lot_rem = pos['lot_remaining']
                            if lot_rem < C.min_lot:
                                positions[name] = None
                                continue

            hit_z_stop = (not np.isnan(zn)) and ((d == 1 and zn <= -C.z_stop_margin) or (d == -1 and zn >= C.z_stop_margin))

            hit_z_exit = False
            if not np.isnan(zn):
                z_crossed = ((d == 1 and zn >= -C.z_exit_full) or (d == -1 and zn <= C.z_exit_full))
                if z_crossed:
                    pnl_check = calc_pnl(d, ep, cp, lot_rem, name)
                    if pnl_check >= C.min_net_profit_usd or pos['partial_done']:
                        hit_z_exit = True

            hit_sl = (d == 1 and a['l'][bar] <= pos['sl']) or (d == -1 and a['h'][bar] >= pos['sl'])
            hit_tp = (d == 1 and a['h'][bar] >= pos['tp']) or (d == -1 and a['l'][bar] <= pos['tp'])

            bars_open = bar - pos['entry_bar']
            current_pos_pnl = calc_pnl(d, ep, cp, lot_rem, name)
            time_stop = ((bars_open >= C.time_stop_bars and current_pos_pnl < 0) or (bars_open >= C.time_stop_bars * 2))

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                has_slippage = True if (hit_sl or time_stop) else False
                if hit_sl:       exit_px, st = pos['sl'], 'SL'
                elif hit_tp:     exit_px, st = pos['tp'], 'TP'
                elif hit_z_stop: exit_px, st = cp, 'Z-Stop'
                elif time_stop:  exit_px, st = cp, 'TimeStop'
                else:            exit_px, st = cp, 'Z-Exit'

                final_pnl = calc_pnl(d, ep, exit_px, lot_rem, name, apply_slippage=has_slippage)
                acc['equity'] += final_pnl

                rec = _make_rec(pos, exit_px, ts, final_pnl, st, lot_rem)
                all_trades.append(rec)
                acc['trades'].append(rec)
                positions[name] = None

                if final_pnl > 0: acc['consec_loss'] = 0
                else:             acc['consec_loss'] += 1

        all_closed = all(positions[name] is None for name in pair_names)
        if acc['equity'] >= PROFIT_LEVEL and all_closed and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            acc_logs.append({'account': acc_num, 'start_ts': acc['start_ts'], 'end_ts': ts, 'reason': 'TARGET_HIT', 'pnl': w})
            print(f"\n    💰 #{acc_num:>3} | {ts.date()} | Target Hit: ${w:>7.2f} | Total Bank: ${total_withdrawn:>9.2f}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq = acc['equity']
            for name in pair_names: trades_today[name] = 0; pending_sig[name] = 0
            continue

        for name in pair_names:
            a = pa[name]
            if (positions[name] is None and not acc['blown'] and trades_today[name] < C.max_trades_day and a['sig'][bar] != 0):
                pending_sig[name] = int(a['sig'][bar])

    print() 
    return {
        'all_trades':      all_trades,
        'account_logs':    acc_logs,
        'eq_curve':        eq_curve,
        'total_withdrawn': total_withdrawn,
        'final_equity':    acc['equity'],
        'total_accounts':  acc_num,
        'pair_names':      pair_names,
    }

def _make_rec(pos: dict, exit_px: float, exit_ts, pnl: float, status: str, lot: float) -> dict:
    return {'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot, 'entry': pos['entry'], 'exit': exit_px, 'entry_ts': pos['entry_ts'], 'exit_ts': exit_ts, 'pnl': pnl, 'status': status, 'entry_bar': pos['entry_bar']}

# ═══════════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results: dict):
    trades = results['all_trades']
    pair_names = results.get('pair_names', [])
    if not trades: return print("\n❌ No trades executed.")
    df_t = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month']   = df_t['exit_ts'].dt.to_period('M')

    wins, losses = df_t[df_t['pnl'] > 0], df_t[df_t['pnl'] < 0]
    wr = len(wins) / len(df_t) * 100 if len(df_t) else 0
    pf = (wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf'))

    print("\n" + "═" * 70)
    print(f" ▌  CorrArb Prop Simulator v9.5 (Direct Data) — {'+'.join(pair_names)}  ▐")
    print("═" * 70)
    print(f" Total Trades:    {len(df_t):,}\n Win Rate:        {wr:.2f}%\n Profit Factor:   {pf:.2f}")
    print(f" Total Banked:    ${results['total_withdrawn']:,.2f}\n Active Equity:   ${results['final_equity']:,.2f}")
    
    if 'pair' in df_t.columns and len(pair_names) > 1:
        print("-" * 70 + "\n عملکرد هر جفت ارز:")
        for pair in pair_names:
            pt = df_t[df_t['pair'] == pair]
            if len(pt) == 0: continue
            pw, pl = pt[pt['pnl'] > 0], pt[pt['pnl'] < 0]
            p_wr = len(pw) / len(pt) * 100 if len(pt) else 0
            p_pf = (pw['pnl'].sum() / abs(pl['pnl'].sum()) if len(pl) > 0 else float('inf'))
            print(f"   {pair}: {len(pt):>4} trades | WR: {p_wr:5.1f}% | PF: {p_pf:.2f} | Net PnL: ${pt['pnl'].sum():>8,.2f}")

    print("-" * 70 + "\n خروج‌ها بر اساس نوع:")
    for st, cnt in df_t['status'].value_counts().items(): print(f"   {st:<12}: {cnt:>4} معامله")
    
    logs = results['account_logs']
    targets = sum(1 for l in logs if l['reason'] == 'TARGET_HIT')
    blown = sum(1 for l in logs if l['reason'] != 'TARGET_HIT')
    print("-" * 70 + f"\n حساب‌ها: {results['total_accounts']} کل | ✅ Target Hit: {targets} | 💥 Blown: {blown}")
    print("═" * 70)

if __name__ == "__main__":
    t0 = datetime.now()
    pairs = load_all_direct_pairs()
    print("\n  Computing Statistical Signals...")
    pair_signals = {name: compute_signals(name, info) for name, info in pairs.items()}
    results = run_backtest(pairs, pair_signals)
    print_report(results)
    print(f"  ✅ Executed in: {(datetime.now() - t0).total_seconds():.2f}s")
