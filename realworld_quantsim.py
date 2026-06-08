"""
CorrArb v14d — Dynamic Timeframe + Strict Realistic Exits (High/Low)
====================================================================
"""

import pandas as pd
import numpy as np
import glob, zipfile, warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class GlobalConfig:
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10
    commission_per_lot = 7.0
    slippage_pips      = 0.5
    lot_size           = 100_000
    max_lot            = 3.0
    min_lot            = 0.01
    consec_loss_n      = 2
    risk_reduce        = 0.5
    cooldown_days      = 10
    monthly_loss_threshold = -250.0
    hour_start         = 2
    hour_end           = 19
    bad_hours          = {4, 5, 7, 9, 13, 18, 20}
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2
    
    # ── پارامترهای مرجع (بر اساس 15 دقیقه) ──
    warmup_base        = 500
    z_fast_period_base = 96
    atr_period_base    = 14
    atr_ma_period_base = 96
    vr_period_base     = 200
    corr_period_base   = 96
    
    z_entry            = 2.1
    z_exit_partial     = 0.50
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0
    partial_ratio      = 0.75
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5
    vr_k               = 4

    # ── Risk Control ──
    dd_levels = [
        (0.04, 0.75),   
        (0.07, 0.50),   
        (0.09, 0.30),   
    ]
    rolling_pf_n   = 30
    rolling_pf_bad = 0.80
    rolling_pf_mult= 0.80  


PAIR_CFG = {
    'AUDNZD': {
        'leg1': 'AUDUSD', 'leg2': 'NZDUSD',
        'formula': 'div', 'quote': 'leg2',
        'spread_pip': 2.5, 'pip_size': 0.0001,
        'vr_max': 0.75, 'corr_min': 0.80,
        'risk_pct': 0.015, 'risk_min': 0.005,
        'sl_pips': 30.0, 'tp_pips': 90.0,
    },
    'AUDCAD': {
        'leg1': 'AUDUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote': 'inv_leg2',
        'spread_pip': 2.5, 'pip_size': 0.0001,
        'vr_max': 0.85, 'corr_min': 0.55,
        'risk_pct': 0.010, 'risk_min': 0.004,
        'sl_pips': 25.0, 'tp_pips': 75.0,
    },
    'GBPCAD': {
        'leg1': 'GBPUSD', 'leg2': 'USDCAD',
        'formula': 'mul', 'quote': 'inv_leg2',
        'spread_pip': 4.0, 'pip_size': 0.0001,
        'vr_max': 0.88, 'corr_min': 0.45,
        'risk_pct': 0.008, 'risk_min': 0.003,
        'sl_pips': 28.0, 'tp_pips': 84.0,
    },
    'GBPCHF': {
        'leg1': 'GBPUSD', 'leg2': 'USDCHF',
        'formula': 'div', 'quote': 'inv_leg2',
        'spread_pip': 3.0, 'pip_size': 0.0001,
        'vr_max': 0.90, 'corr_min': 0.40,
        'risk_pct': 0.007, 'risk_min': 0.003,
        'sl_pips': 35.0, 'tp_pips': 105.0,
    },
}

# ═══════════════════════════════════════════════════════
# DYNAMIC CONFIGURATOR
# ═══════════════════════════════════════════════════════
def get_dynamic_config(tf_str):
    tf_map = {'1min': 1, '5min': 5, '15min': 15, '1h': 60}
    mins = tf_map.get(tf_str, 15)
    ratio = 15.0 / mins  

    G = GlobalConfig
    return {
        'warmup': max(50, int(G.warmup_base * ratio)),
        'z_fast_period': max(2, int(G.z_fast_period_base * ratio)),
        'atr_period': max(2, int(G.atr_period_base * ratio)),
        'atr_ma_period': max(2, int(G.atr_ma_period_base * ratio)),
        'vr_period': max(2, int(G.vr_period_base * ratio)),
        'corr_period': max(2, int(G.corr_period_base * ratio)),
    }

# ═══════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════
def load_raw_zip(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths: return None
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p) as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if not csv_name: continue
                with z.open(csv_name) as f:
                    frames.append(pd.read_csv(
                        f, sep=';', header=None,
                        names=['ts', 'o', 'h', 'l', 'c', 'v']))
        except Exception:
            continue
    if not frames: return None
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw

def load_raw_csv(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths: return None
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(
                p, sep=';', header=None,
                names=['ts', 'o', 'h', 'l', 'c', 'v']))
        except Exception:
            continue
    if not frames: return None
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o', 'h', 'l', 'c']] = raw[['o', 'h', 'l', 'c']].astype(float)
    return raw

def load_instrument(name):
    for pat in [
        f'data/HISTDATA*{name}*.zip', f'data/*{name}*.zip',
        f'data/HISTDATA*{name}*.csv', f'data/*{name}*.csv',
    ]:
        raw = (load_raw_zip(pat) if '.zip' in pat else load_raw_csv(pat))
        if raw is not None and len(raw) > 1000:
            return raw
    return None

def to_tf(raw, sfx, tf):
    tf_str = '60min' if tf == '1h' else tf
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample(tf_str).first(),
        f'h_{sfx}': raw['h'].resample(tf_str).max(),
        f'l_{sfx}': raw['l'].resample(tf_str).min(),
        f'c_{sfx}': raw['c'].resample(tf_str).last(),
    }).dropna()

def build_pair(pcfg, tf):
    r1 = load_instrument(pcfg['leg1'])
    r2 = load_instrument(pcfg['leg2'])
    if r1 is None or r2 is None: return None
    d1 = to_tf(r1, 'leg1', tf)
    d2 = to_tf(r2, 'leg2', tf)
    m  = d1.join(d2, how='inner').dropna()
    
    if pcfg['formula'] == 'div':
        m['c_spread'] = m['c_leg1'] / m['c_leg2']
        m['o_spread'] = m['o_leg1'] / m['o_leg2']
        m['h_spread'] = m['h_leg1'] / m['l_leg2']
        m['l_spread'] = m['l_leg1'] / m['h_leg2']
    else:
        m['c_spread'] = m['c_leg1'] * m['c_leg2']
        m['o_spread'] = m['o_leg1'] * m['o_leg2']
        m['h_spread'] = m['h_leg1'] * m['h_leg2']
        m['l_spread'] = m['l_leg1'] * m['l_leg2']
        
    if pcfg['quote'] == 'leg2':
        m['quote_rate'] = m['c_leg2']
    elif pcfg['quote'] == 'inv_leg2':
        m['quote_rate'] = 1.0 / m['c_leg2'].replace(0, np.nan)
    else:
        m['quote_rate'] = 1.0
        
    return m[m.index.weekday < 5].dropna().copy()


# ═══════════════════════════════════════════════════════
# SIGNALS & RISK MULTIPLIER
# ═══════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_vr(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    v1 = r1.rolling(window).var()
    vk = rk.rolling(window).var()
    return vk / (k * v1.replace(0, np.nan))

def compute_signals(df, pcfg, dyn_cfg):
    G = GlobalConfig
    log_r  = np.log(df['c_spread'].replace(0, np.nan))
    
    z_mean = log_r.rolling(dyn_cfg['z_fast_period']).mean()
    z_std  = log_r.rolling(dyn_cfg['z_fast_period']).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    corr    = (df['c_leg1'].pct_change().rolling(dyn_cfg['corr_period']).corr(df['c_leg2'].pct_change()))
    corr_ok = corr.abs() > pcfg['corr_min']

    vr        = calc_vr(log_r, G.vr_k, dyn_cfg['vr_period'])
    regime_ok = vr < pcfg['vr_max']

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], dyn_cfg['atr_period'])
    atr_ma = atr.rolling(dyn_cfg['atr_ma_period']).mean()
    vol_ok = ((atr > atr_ma * G.atr_min_mult) & (atr < atr_ma * G.atr_max_mult))

    hour    = pd.Series(df.index.hour, index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = (hour.between(G.hour_start, G.hour_end) & (~hour.isin(G.bad_hours)) & dow.isin(G.trade_days))

    sig  = pd.Series(0, index=df.index)
    cond = vol_ok & time_ok & corr_ok & regime_ok
    sig[(z < -G.z_entry) & cond] =  1
    sig[(z >  G.z_entry) & cond] = -1
    sig = sig.where(sig != sig.shift(), 0)
    return sig, z

def get_risk_mult(equity, peak, pnl_hist, month_pnl, month_threshold):
    G = GlobalConfig
    mult = 1.0
    if peak > 0:
        dd = (peak - equity) / peak
        for dd_thresh, dd_mult in G.dd_levels:
            if dd >= dd_thresh: mult = min(mult, dd_mult)
    if len(pnl_hist) >= G.rolling_pf_n // 2:
        recent = pnl_hist[-G.rolling_pf_n:]
        wins   = sum(p for p in recent if p > 0)
        losses = abs(sum(p for p in recent if p < 0))
        rpf    = wins / losses if losses > 0 else 1.5
        if rpf < G.rolling_pf_bad: mult *= G.rolling_pf_mult
    if month_pnl < month_threshold: mult *= 0.60
    return max(mult, 0.20)


# ═══════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════
def pnl_calc(d, entry, xp, lot, qr, pip):
    gross = d * (xp - entry) * lot * GlobalConfig.lot_size * qr
    return gross - GlobalConfig.commission_per_lot * lot

def new_acc(ts):
    G = GlobalConfig
    return {
        'equity': G.initial_balance, 'start_ts': ts, 'trades': [],
        'blown': False, 'blown_rsn': '', 'peak': G.initial_balance,
        'consec_loss': 0,
    }

def run_portfolio(pair_data, tf_name, dyn_cfg):
    G = GlobalConfig
    cidx = None
    for name, (df, sig, z, pcfg) in pair_data.items():
        cidx = df.index if cidx is None else cidx.intersection(df.index)
    cidx = cidx.sort_values()
    N = len(cidx)

    pa = {}
    for name, (df, sig, z, pcfg) in pair_data.items():
        df_r = df.reindex(cidx).ffill()
        pa[name] = {
            'o':   df_r['o_spread'].values.astype(float),
            'h':   df_r['h_spread'].values.astype(float),  # اضافه شده برای SL/TP واقعی
            'l':   df_r['l_spread'].values.astype(float),  # اضافه شده برای SL/TP واقعی
            'c':   df_r['c_spread'].values.astype(float),
            'qr':  df_r['quote_rate'].values.astype(float),
            'sig': sig.reindex(cidx).fillna(0).values.astype(int),
            'z':   z.reindex(cidx).fillna(np.nan).values.astype(float),
            'cfg': pcfg,
        }

    FLOOR  = G.initial_balance * (1 - G.max_total_dd_pct)
    TARGET = G.initial_balance * (1 + G.profit_target_pct)
    
    warmup = dyn_cfg['warmup']
    acc          = new_acc(cidx[warmup])
    withdrawn    = 0.0
    acc_num      = 1
    day_eq       = G.initial_balance
    month_eq     = G.initial_balance
    cooldown_til = None
    all_trades   = []
    acc_logs     = []
    eq_curve     = []
    pnl_hist     = []

    positions  = {n: None for n in pair_data}
    day_trades = {n: 0    for n in pair_data}
    pending    = {n: 0    for n in pair_data}
    prev_date  = None
    prev_month = None

    for bar in range(warmup, N):
        ts        = cidx[bar]
        cur_date  = ts.date()
        cur_month = (ts.year, ts.month)

        if cur_date != prev_date:
            day_eq = acc['equity']
            for n in pair_data: day_trades[n] = 0
            eq_curve.append({'date': str(cur_date), 'equity': acc['equity'] + withdrawn, 'account_eq': acc['equity']})
            prev_date = cur_date

        if cur_month != prev_month:
            month_eq   = acc['equity']
            prev_month = cur_month

        if acc['equity'] > acc['peak']:
            acc['peak'] = acc['equity']

        in_cd = cooldown_til is not None and ts < cooldown_til

        if acc['blown']:
            acc_logs.append({
                'reason': acc['blown_rsn'], 'pnl': acc['equity'] - G.initial_balance,
                'days': (ts - acc['start_ts']).days, 'n_trades': len(acc['trades']), 'end_ts': ts,
            })
            cooldown_til = ts + pd.Timedelta(days=G.cooldown_days)
            acc_num += 1
            acc = new_acc(ts)
            day_eq = month_eq = acc['equity']
            for n in pair_data:
                day_trades[n] = 0; pending[n] = 0; positions[n] = None
            pnl_hist = []
            prev_date = cur_date; prev_month = cur_month
            continue

        if in_cd: continue

        month_pnl = acc['equity'] - month_eq
        risk_mult = get_risk_mult(acc['equity'], acc['peak'], pnl_hist, month_pnl, G.monthly_loss_threshold)

        # OPEN POSITIONS
        for name in pair_data:
            a = pa[name]; pcfg = a['cfg']
            if pending[name] != 0 and positions[name] is None and day_trades[name] < G.max_trades_day:
                sv = pending[name]; pip = pcfg['pip_size']; sp = pcfg['spread_pip']; qr = a['qr'][bar]
                risk = pcfg['risk_pct'] * risk_mult
                if acc['consec_loss'] >= G.consec_loss_n:
                    risk = max(risk * G.risk_reduce, pcfg['risk_min'])
                pv  = pip * G.lot_size * qr
                if pv <= 0: pv = 10.0
                lot = round(float(np.clip(acc['equity'] * risk / (pcfg['sl_pips'] * pv), G.min_lot, G.max_lot)), 2)
                ep = a['o'][bar] + sv * (G.slippage_pips + sp / 2) * pip
                positions[name] = {
                    'dir': sv, 'lot': lot, 'lot_rem': lot, 'partial_done': False, 'entry': ep,
                    'sl': ep - sv * pcfg['sl_pips'] * pip, 'tp': ep + sv * pcfg['tp_pips'] * pip,
                    'entry_ts': ts, 'entry_bar': bar, 'pip': pip,
                }
                day_trades[name] += 1
            pending[name] = 0

        # DAILY DD CHECK
        total_float = sum(
            pnl_calc(p['dir'], p['entry'], pa[n]['c'][bar], p['lot_rem'], pa[n]['qr'][bar], p['pip'])
            for n in pair_data if (p := positions[n]) is not None
        )
        cur_eq = acc['equity'] + total_float
        daily_lim = day_eq * (1 - G.max_daily_loss_pct)

        if cur_eq <= daily_lim or cur_eq <= FLOOR:
            rsn = "DailyDD" if cur_eq <= daily_lim else "TotalDD"
            acc['blown'] = True; acc['blown_rsn'] = rsn
            for name in pair_data:
                pos = positions[name]
                if pos is None: continue
                pnl = pnl_calc(pos['dir'], pos['entry'], pa[name]['c'][bar], pos['lot_rem'], pa[name]['qr'][bar], pos['pip'])
                acc['equity'] += pnl
                all_trades.append({'pair': name, 'pnl': pnl, 'status': 'BLOWN', 'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                positions[name] = None
            continue

        # EXIT LOGIC (Strict High/Low Check)
        for name in pair_data:
            pos = positions[name]
            if pos is None: continue
            
            a = pa[name]; pcfg = a['cfg']; d = pos['dir']
            hp = a['h'][bar]; lp = a['l'][bar]; cp = a['c'][bar]
            ep = pos['entry']; zn = a['z'][bar]; lr = pos['lot_rem']; qr = a['qr'][bar]; pip = pos['pip']

            # 1. Hard SL/TP Check (Pessimistic Execution)
            hit_sl = (d == 1 and lp <= pos['sl']) or (d == -1 and hp >= pos['sl'])
            # Only check TP if SL wasn't hit first (Worst-case scenario)
            hit_tp = not hit_sl and ((d == 1 and hp >= pos['tp']) or (d == -1 and lp <= pos['tp']))

            st = None
            xp = None

            if hit_sl:
                st = 'SL'
                xp = pos['sl']
            elif hit_tp:
                st = 'TP'
                xp = pos['tp']
            else:
                # 2. Z-Score Exits (Evaluated at Close)
                hit_zs = (not np.isnan(zn) and ((d==1 and zn<=-G.z_stop_margin) or (d==-1 and zn>=G.z_stop_margin)))
                hit_ze = (not np.isnan(zn) and ((d==1 and zn>=-G.z_exit_full) or (d==-1 and zn<=G.z_exit_full)))
                pnl_now = pnl_calc(d, ep, cp, lr, qr, pip)

                if hit_ze and pnl_now < G.min_net_profit_usd and not pos['partial_done']: hit_ze = False

                if hit_zs:
                    st = 'Z-Stop'
                    xp = cp
                elif hit_ze:
                    st = 'Z-Exit'
                    xp = cp
                else:
                    # 3. Partial Exit Check
                    if not pos['partial_done'] and not np.isnan(zn):
                        if ((d==1 and zn>=-G.z_exit_partial) or (d==-1 and zn<=G.z_exit_partial)):
                            p_lot = round(lr * G.partial_ratio, 2)
                            if p_lot >= G.min_lot:
                                p_pnl = pnl_calc(d, ep, cp, p_lot, qr, pip)
                                if p_pnl > 0:
                                    acc['equity'] += p_pnl; pnl_hist.append(p_pnl)
                                    all_trades.append({'pair': name, 'pnl': p_pnl, 'status': 'Partial', 'exit_ts': ts})
                                    acc['trades'].append(all_trades[-1])
                                    pos['lot_rem'] = round(lr - p_lot, 2)
                                    pos['partial_done'] = True
                                    pos['sl'] = pos['entry'] # Break-even shift
                                    lr = pos['lot_rem']
                                    if lr < G.min_lot: positions[name] = None
                    continue # No full exit this bar, proceed to next pair
            
            # Execute Full Exit
            if st is not None:
                fpnl = pnl_calc(d, ep, xp, lr, qr, pip)
                acc['equity'] += fpnl; pnl_hist.append(fpnl)
                all_trades.append({'pair': name, 'pnl': fpnl, 'status': st, 'exit_ts': ts})
                acc['trades'].append(all_trades[-1])
                positions[name] = None
                if fpnl > 0: acc['consec_loss'] = 0
                else:        acc['consec_loss'] += 1

        # TARGET CHECK
        if acc['equity'] >= TARGET and all(positions[n] is None for n in pair_data):
            w = acc['equity'] - G.initial_balance
            withdrawn += w
            dt = (ts - acc['start_ts']).days; nt = len(acc['trades'])
            acc_logs.append({'reason': 'TARGET_HIT', 'pnl': w, 'days': dt, 'n_trades': nt, 'end_ts': ts})
            acc_num += 1
            acc = new_acc(ts); day_eq = month_eq = acc['equity']; pnl_hist = []
            for n in pair_data: day_trades[n] = 0; pending[n] = 0
            prev_date = cur_date; prev_month = cur_month
            continue

        # REGISTER SIGNALS
        for name in pair_data:
            a = pa[name]
            if (positions[name] is None and not acc['blown'] and not in_cd
                    and day_trades[name] < G.max_trades_day and a['sig'][bar] != 0):
                pending[name] = int(a['sig'][bar])

    return {'all_trades': all_trades, 'account_logs': acc_logs,
            'withdrawn': withdrawn, 'final_equity': acc['equity'],
            'common_idx': cidx, 'eq_curve': eq_curve, 'warmup': warmup}


# ═══════════════════════════════════════════════════════
# REPORT & METRICS EXTRACTION
# ═══════════════════════════════════════════════════════
def print_report(res, title, tf_str):
    if not res['all_trades']:
        print("❌ No trades generated.")
        return None

    df = pd.DataFrame(res['all_trades'])
    df['exit_ts'] = pd.to_datetime(df['exit_ts'])
    df['month']   = df['exit_ts'].dt.to_period('M')
    df['year']    = df['exit_ts'].dt.year

    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    wr     = len(wins) / len(df) * 100
    pf     = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else 99.0

    ci = res['common_idx']
    warmup = res['warmup']
    all_months = pd.period_range(
        start=ci[warmup].to_period('M'),
        end=ci[-1].to_period('M'), freq='M')
    monthly = df.groupby('month')['pnl'].sum().reindex(all_months, fill_value=0.0)

    logs   = pd.DataFrame(res['account_logs']) if res['account_logs'] else pd.DataFrame()
    n_blow = int((logs['reason'] != 'TARGET_HIT').sum()) if len(logs) else 0

    m_std  = monthly.std()
    sharpe = (monthly.mean() / m_std * np.sqrt(12)) if m_std > 0 else 0

    print(f"\n  [ {title} Summary ]")
    print(f"  Trades:{len(df):,}  WR:{wr:.1f}%  PF:{pf:.3f}")
    print(f"  Net:${df['pnl'].sum():,.2f}  MonthAvg:${monthly.mean():.2f}  Sharpe:{sharpe:.2f}  Blows:{n_blow}")
    
    monthly.to_csv(f'monthly_v14d_{tf_str}.csv', header=['pnl'])
    pd.DataFrame(res['eq_curve']).to_csv(f'equity_v14d_{tf_str}.csv', index=False)

    return {
        'trades': len(df),
        'wr': wr,
        'pf': pf,
        'net_pnl': df['pnl'].sum(),
        'monthly_avg': monthly.mean(),
        'sharpe': sharpe,
        'blows': n_blow
    }


# ═══════════════════════════════════════════════════════
# MAIN MULTI-TIMEFRAME EXECUTION
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  CorrArb v14d — Strict Realistic Exits (High/Low logic)    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    timeframes_to_test = ['1min', '5min', '15min', '1h']
    comparison_results = {}

    for tf in timeframes_to_test:
        print("\n" + "═"*70)
        print(f"  ⏳ RUNNING BACKTEST FOR TIMEFRAME: {tf}")
        dyn_cfg = get_dynamic_config(tf)
        print(f"  ⚙️ Dynamic Params: Z-Period={dyn_cfg['z_fast_period']} | ATR={dyn_cfg['atr_period']} | VR={dyn_cfg['vr_period']} | Warmup={dyn_cfg['warmup']}")
        print("═"*70)

        pair_data = {}
        for name, pcfg in PAIR_CFG.items():
            df = build_pair(pcfg, tf)
            if df is None:
                print(f"  ❌ {name}: not found or insufficient data"); continue
            sig, z = compute_signals(df, pcfg, dyn_cfg)
            n = int((sig != 0).sum())
            print(f"  ✅ {name} ({tf}): {len(df):,} bars | {n:,} signals")
            pair_data[name] = (df, sig, z, pcfg)

        if not pair_data:
            print(f"  ⚠️ No valid data for {tf}. Skipping...")
            continue

        res = run_portfolio(pair_data, tf, dyn_cfg)
        metrics = print_report(res, f"Timeframe {tf}", tf)
        
        if metrics:
            comparison_results[tf] = metrics

    print("\n" + "█"*85)
    print("  🏆 FINAL TIMEFRAME COMPARISON (STRICT REALITY) 🏆")
    print("█"*85)
    print(f"  | {'Timeframe':<10} | {'Trades':<8} | {'WR%':<6} | {'PF':<5} | {'Net PnL ($)':<12} | {'Mo Avg ($)':<10} | {'Sharpe':<6} | {'Blows':<5} |")
    print("  " + "-" * 81)
    
    for tf in timeframes_to_test:
        if tf in comparison_results:
            m = comparison_results[tf]
            print(f"  | {tf:<10} | {m['trades']:<8} | {m['wr']:<6.1f} | {m['pf']:<5.2f} | {m['net_pnl']:<12,.2f} | {m['monthly_avg']:<10.2f} | {m['sharpe']:<6.2f} | {m['blows']:<5} |")
        else:
            print(f"  | {tf:<10} | {'N/A':<8} | {'N/A':<6} | {'N/A':<5} | {'N/A':<12} | {'N/A':<10} | {'N/A':<6} | {'N/A':<5} |")
    
    print("█"*85)
    print(f"\n  ✅ All tests completed in {(datetime.now()-t0).total_seconds():.1f} seconds.")
