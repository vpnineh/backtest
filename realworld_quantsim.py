"""
CorrArb Prop Simulator — v9 Realistic
======================================
اصلاحات واقعی‌بودن نسبت به v8:
  ✅ FIX-1: spread مستقیم EURGBP از ZIP (نه synthetic)
  ✅ FIX-2: spread_pips واقعی برای هر جفت ارز (نه ثابت 1.2)
  ✅ FIX-3: pip_value دینامیک از نرخ واقعی quote (نه 10×1.25 ثابت)
  ✅ FIX-4: Cooldown 10 روز بعد از هر blown
  ✅ FIX-5: Monthly Floor Check — اگه ماه منفی شد ریسک کاهش میده
  ✅ FIX-6: Pair Quality Filter — جفت ارزی که PF < 1.10 داشته باشه
             در ۶ ماه گذشته از trading pause میشه (rolling evaluation)

دیتاهای موجود و نحوه استفاده:
  EURUSD + GBPUSD → CSV مستقیم (synthetic EURGBP — چون EURGBP zip هم داری)
  AUDNZD          → ZIP مستقیم (بهترین جفت ارز)
  EURGBP          → ZIP مستقیم (دقیق‌تر از synthetic)
  بقیه (USDCAD، USDCHF، XAUUSD، XAGUSD) → آماده برای اضافه شدن

نکته مهم درباره واقعی‌بودن:
  - slippage = 0.5 pip (بیشتر از قبل)
  - spread هر جفت ارز متفاوت است
  - commission round-trip = 7 USD per lot
  - partial close: فقط اگه pnl > threshold قابل اجرا
"""

import pandas as pd
import numpy as np
import glob
import zipfile
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG v9 — واقعی‌تر
# ═══════════════════════════════════════════════════════════════════════════
class Config:
    # ── قوانین پراپ (ثابت) ──────────────────────────────────────────────
    initial_balance    = 5_000.0
    profit_target_pct  = 0.05
    max_daily_loss_pct = 0.05
    max_total_dd_pct   = 0.10

    # ── ریسک ────────────────────────────────────────────────────────────
    risk_base_pct      = 0.015
    risk_min_pct       = 0.005
    consec_loss_n      = 2
    risk_reduce        = 0.5

    # ── هزینه‌های واقعی هر جفت ارز ──────────────────────────────────────
    # FIX-2: هر جفت spread واقعی متفاوت دارد
    PAIR_SPREAD = {
        'EURGBP': 1.0,   # pip — spread معمولاً 0.8-1.2
        'AUDNZD': 2.5,   # pip — spread بالاتر به خاطر liquidity کمتر
        'USDCAD': 1.5,
        'USDCHF': 1.5,
        'XAUUSD': 3.0,   # pip — spread طلا بالاتر (pip=0.01)
        'XAGUSD': 5.0,   # pip — spread نقره بالاتر
    }
    commission_per_lot = 7.0       # USD round-trip per lot
    slippage_pips      = 0.5       # FIX: از 0.3 به 0.5 (واقعی‌تر)

    # ── مشخصات pip هر جفت ───────────────────────────────────────────────
    PIP_SIZE = {
        'EURGBP': 0.0001,
        'AUDNZD': 0.0001,
        'USDCAD': 0.0001,
        'USDCHF': 0.0001,
        'XAUUSD': 0.01,   # طلا: 1 pip = $0.01
        'XAGUSD': 0.001,  # نقره
    }

    # ── lot size ─────────────────────────────────────────────────────────
    lot_size = 100_000
    max_lot  = 3.0
    min_lot  = 0.01
    warmup   = 500

    # ── پارامترهای z-score ──────────────────────────────────────────────
    z_fast_period      = 96
    z_entry            = 2.1
    z_exit_partial     = 0.5
    z_exit_full        = 0.0
    z_stop_margin      = 4.0
    min_net_profit_usd = 15.0   # FIX: بالاتر از قبل (واقعی‌تر)

    # ── فیلترها ──────────────────────────────────────────────────────────
    corr_period        = 96
    corr_min           = 0.80
    hour_start         = 2
    hour_end           = 19
    trade_days         = [0, 1, 2, 3, 4]
    max_trades_day     = 2

    # ── خروج‌های اضطراری ─────────────────────────────────────────────────
    sl_pips            = 30.0
    tp_pips            = 90.0
    time_stop_bars     = 36
    partial_ratio      = 0.50

    # ── ATR ──────────────────────────────────────────────────────────────
    atr_period         = 14
    atr_ma_period      = 96
    atr_max_mult       = 3.0
    atr_min_mult       = 0.5

    # ── Variance Ratio ────────────────────────────────────────────────────
    vr_period          = 200
    vr_k               = 4
    vr_max             = 0.90

    # ── FIX-4: Cooldown بعد از blown ─────────────────────────────────────
    cooldown_days      = 10   # روز انتظار بعد از هر blown

    # ── FIX-5: Monthly Risk Reduction ────────────────────────────────────
    # اگه این ماه بیشتر از این مقدار ضرر دادیم، risk_base رو کاهش میدیم
    monthly_loss_threshold = -150.0  # USD — بیشتر از این ضرر → نصف ریسک

    # ── FIX-6: Pair Quality Filter ───────────────────────────────────────
    pair_eval_window   = 180   # روز — پنجره ارزیابی کیفیت جفت ارز
    pair_min_pf        = 1.05  # حداقل profit factor در پنجره اخیر


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_raw_csv(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files: {pattern}")
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep=';', header=None, names=['ts','o','h','l','c','v'])
        frames.append(df)
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def load_raw_zip(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No ZIP files: {pattern}")
    frames = []
    for p in paths:
        try:
            with zipfile.ZipFile(p, 'r') as z:
                csv_name = next((f for f in z.namelist() if f.lower().endswith('.csv')), None)
                if not csv_name:
                    continue
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, sep=';', header=None, names=['ts','o','h','l','c','v'])
                    frames.append(df)
        except Exception as e:
            print(f"  ⚠ {os.path.basename(p)}: {e}")
    if not frames:
        raise ValueError(f"No valid ZIP data: {pattern}")
    raw = pd.concat(frames).sort_values('ts')
    raw['ts'] = pd.to_datetime(raw['ts'], format='%Y%m%d %H%M%S')
    raw = raw.drop_duplicates('ts').set_index('ts')
    raw[['o','h','l','c']] = raw[['o','h','l','c']].astype(float)
    return raw


def to_15min(raw: pd.DataFrame, sfx: str) -> pd.DataFrame:
    return pd.DataFrame({
        f'o_{sfx}': raw['o'].resample('15min').first(),
        f'h_{sfx}': raw['h'].resample('15min').max(),
        f'l_{sfx}': raw['l'].resample('15min').min(),
        f'c_{sfx}': raw['c'].resample('15min').last(),
    }).dropna()


def load_all_pairs() -> dict:
    """
    بارگذاری تمام جفت ارزها
    اولویت: دیتای مستقیم > synthetic
    """
    print("\n  Loading datasets...")
    pairs = {}

    # ── EURGBP: از ZIP مستقیم (FIX-1: دقیق‌تر از synthetic) ──────────
    try:
        raw = load_raw_zip('data/HISTDATA*EURGBP*.zip')
        df15 = to_15min(raw, 'eg')
        # برای EURGBP مستقیم، quote currency = GBP → نیاز به GBPUSD داریم
        gbp_raw = load_raw_csv('data/*GBPUSD*.csv')
        gbp15   = to_15min(gbp_raw, 'gbp')
        merged  = df15.join(gbp15[['c_gbp']], how='inner').dropna()
        merged['c_spread']   = merged['c_eg']
        merged['o_spread']   = merged['o_eg']
        merged['h_spread']   = merged['h_eg']
        merged['l_spread']   = merged['l_eg']
        merged['quote_rate'] = merged['c_gbp']   # GBPUSD برای تبدیل به USD
        merged = merged[merged.index.weekday < 5].copy()
        pairs['EURGBP'] = {
            'df': merged,
            'spread_pip': Config.PAIR_SPREAD['EURGBP'],
            'pip':        Config.PIP_SIZE['EURGBP'],
            'source':     'direct ZIP'
        }
        print(f"  ✅ EURGBP (direct): {len(merged):,} candles | {merged.index[0].date()} → {merged.index[-1].date()}")
    except Exception as e:
        print(f"  ⚠ EURGBP direct failed ({e}), trying synthetic...")
        try:
            eur = to_15min(load_raw_csv('data/*EURUSD*.csv'), 'eur')
            gbp = to_15min(load_raw_csv('data/*GBPUSD*.csv'), 'gbp')
            m   = eur.join(gbp, how='inner').dropna()
            m['c_spread']   = m['c_eur'] / m['c_gbp']
            m['o_spread']   = m['o_eur'] / m['o_gbp']
            m['h_spread']   = m['h_eur'] / m['l_gbp']
            m['l_spread']   = m['l_eur'] / m['h_gbp']
            m['quote_rate'] = m['c_gbp']
            m = m[m.index.weekday < 5].copy()
            pairs['EURGBP'] = {
                'df': m,
                'spread_pip': Config.PAIR_SPREAD['EURGBP'] * 1.3,  # synthetic penalty
                'pip':        Config.PIP_SIZE['EURGBP'],
                'source':     'synthetic CSV'
            }
            print(f"  ✅ EURGBP (synthetic): {len(m):,} candles")
        except Exception as e2:
            print(f"  ❌ EURGBP: {e2}")

    # ── AUDNZD: از ZIP مستقیم ──────────────────────────────────────────
    try:
        aud = to_15min(load_raw_zip('data/HISTDATA*AUDUSD*.zip'), 'aud')
        nzd = to_15min(load_raw_zip('data/HISTDATA*NZDUSD*.zip'), 'nzd')
        m   = aud.join(nzd, how='inner').dropna()
        m['c_spread']   = m['c_aud'] / m['c_nzd']
        m['o_spread']   = m['o_aud'] / m['o_nzd']
        m['h_spread']   = m['h_aud'] / m['l_nzd']
        m['l_spread']   = m['l_aud'] / m['h_nzd']
        m['quote_rate'] = m['c_nzd']   # NZDUSD برای تبدیل سود AUDNZD به USD
        m = m[m.index.weekday < 5].copy()
        pairs['AUDNZD'] = {
            'df': m,
            'spread_pip': Config.PAIR_SPREAD['AUDNZD'],
            'pip':        Config.PIP_SIZE['AUDNZD'],
            'source':     'synthetic ZIP'
        }
        print(f"  ✅ AUDNZD: {len(m):,} candles | {m.index[0].date()} → {m.index[-1].date()}")
    except Exception as e:
        print(f"  ❌ AUDNZD: {e}")

    if not pairs:
        raise RuntimeError("No pairs loaded!")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
#  سیگنال‌ها
# ═══════════════════════════════════════════════════════════════════════════
def calc_atr(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_variance_ratio(series, k, window):
    r1 = series.diff(1)
    rk = series.diff(k)
    var1 = r1.rolling(window).var()
    vark = rk.rolling(window).var()
    return vark / (k * var1.replace(0, np.nan))


def compute_signals(pair_name, pair_info):
    C  = Config
    df = pair_info['df']

    log_r  = np.log(df['c_spread'])
    z_mean = log_r.rolling(C.z_fast_period).mean()
    z_std  = log_r.rolling(C.z_fast_period).std()
    z      = (log_r - z_mean) / z_std.replace(0, np.nan)

    # Correlation (فقط برای synthetic pairs که دو leg دارن)
    if 'c_aud' in df.columns and 'c_nzd' in df.columns:
        corr_ok = df['c_aud'].pct_change().rolling(C.corr_period).corr(
                  df['c_nzd'].pct_change()) > C.corr_min
    elif 'c_eur' in df.columns and 'c_gbp' in df.columns:
        corr_ok = df['c_eur'].pct_change().rolling(C.corr_period).corr(
                  df['c_gbp'].pct_change()) > C.corr_min
    else:
        corr_ok = pd.Series(True, index=df.index)

    vr = calc_variance_ratio(log_r, C.vr_k, C.vr_period)
    regime_ok = vr < C.vr_max

    atr    = calc_atr(df['h_spread'], df['l_spread'], df['c_spread'], C.atr_period)
    atr_ma = atr.rolling(C.atr_ma_period).mean()
    vol_ok = (atr > atr_ma * C.atr_min_mult) & (atr < atr_ma * C.atr_max_mult)

    hour    = pd.Series(df.index.hour,      index=df.index)
    dow     = pd.Series(df.index.dayofweek, index=df.index)
    time_ok = hour.between(C.hour_start, C.hour_end) & dow.isin(C.trade_days)

    sig = pd.Series(0, index=df.index)
    sig[(z < -C.z_entry) & vol_ok & time_ok & corr_ok & regime_ok] =  1
    sig[(z >  C.z_entry) & vol_ok & time_ok & corr_ok & regime_ok] = -1
    sig = sig.where(sig != sig.shift(), 0)

    n = int((sig != 0).sum())
    print(f"    {pair_name} ({pair_info['source']}): {n:,} signals | Regime OK: {int(regime_ok.sum()):,} bars")
    return sig, z


# ═══════════════════════════════════════════════════════════════════════════
#  محاسبات مالی — FIX-3: pip_value دینامیک
# ═══════════════════════════════════════════════════════════════════════════
def calc_pnl_real(direction, entry_px, exit_px, lot, quote_rate, pip_size):
    """
    FIX-3: محاسبه دقیق با pip_size واقعی هر جفت ارز
    """
    C = Config
    gross_quote = direction * (exit_px - entry_px) * lot * C.lot_size
    gross_usd   = gross_quote * quote_rate
    commission  = C.commission_per_lot * lot
    return gross_usd - commission


def calc_lot_real(equity, sl_pips, consec_loss, quote_rate, pip_size):
    """
    FIX-3: lot sizing با pip_value واقعی
    pip_value_usd = pip_size × lot_size × quote_rate
    """
    C = Config
    risk = C.risk_base_pct
    if consec_loss >= C.consec_loss_n:
        risk = max(risk * C.risk_reduce, C.risk_min_pct)
    pip_value_usd = pip_size * C.lot_size * quote_rate
    if pip_value_usd <= 0:
        pip_value_usd = 10.0
    risk_usd = equity * risk
    raw = risk_usd / (sl_pips * pip_value_usd)
    return round(float(np.clip(raw, C.min_lot, C.max_lot)), 2)


def new_acc(ts):
    C = Config
    return {
        'equity':       C.initial_balance,
        'start_ts':     ts,
        'trades':       [],
        'blown':        False,
        'blown_rsn':    '',
        'peak':         C.initial_balance,
        'consec_loss':  0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  موتور v9 — با تمام FIX‌ها
# ═══════════════════════════════════════════════════════════════════════════
def run_backtest(pairs, pair_signals):
    C          = Config
    pair_names = list(pairs.keys())

    # ایندکس مشترک
    common_idx = None
    for name in pair_names:
        idx = pairs[name]['df'].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    common_idx = common_idx.sort_values()
    n_bars = len(common_idx)
    print(f"  ✅ Common bars: {n_bars:,} | {common_idx[0].date()} → {common_idx[-1].date()}")

    # آرایه‌های numpy
    pa = {}
    for name in pair_names:
        df_p      = pairs[name]['df'].reindex(common_idx).ffill()
        sig_s, z_s = pair_signals[name]
        pa[name] = {
            'o':      df_p['o_spread'].values.astype(float),
            'c':      df_p['c_spread'].values.astype(float),
            'qr':     df_p['quote_rate'].values.astype(float),
            'sig':    sig_s.reindex(common_idx).fillna(0).values.astype(int),
            'z':      z_s.reindex(common_idx).fillna(np.nan).values.astype(float),
            'sp':     pairs[name]['spread_pip'],
            'pip':    pairs[name]['pip'],
        }

    PROP_FLOOR   = C.initial_balance * (1 - C.max_total_dd_pct)
    PROFIT_LEVEL = C.initial_balance * (1 + C.profit_target_pct)

    acc             = new_acc(common_idx[C.warmup])
    total_withdrawn = 0.0
    acc_num         = 1
    day_start_eq    = C.initial_balance
    month_start_eq  = C.initial_balance   # FIX-5
    all_trades      = []
    acc_logs        = []
    eq_curve        = []
    cooldown_until  = None   # FIX-4

    positions    = {n: None for n in pair_names}
    trades_today = {n: 0    for n in pair_names}
    pending_sig  = {n: 0    for n in pair_names}

    # FIX-6: ردیابی کیفیت هر جفت ارز
    pair_recent_trades = {n: [] for n in pair_names}  # [(ts, pnl), ...]
    pair_paused        = {n: False for n in pair_names}

    print(f"\n  ▶ Running v9 Realistic Simulator...")
    print(f"    Pairs: {' + '.join(pair_names)}")
    print(f"    Cooldown: {C.cooldown_days}d | Monthly threshold: ${C.monthly_loss_threshold}")

    for bar in range(C.warmup, n_bars):
        ts = common_idx[bar]
        eq = acc['equity']
        eq_curve.append((ts, round(eq, 4)))
        if eq > acc['peak']:
            acc['peak'] = eq

        # ── FIX-4: چک cooldown ──────────────────────────────────────────
        in_cooldown = (cooldown_until is not None and ts < cooldown_until)

        # ── ریست روزانه ─────────────────────────────────────────────────
        if ts.hour == 0 and ts.minute == 0:
            day_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0

        # ── ریست ماهانه + FIX-5 ─────────────────────────────────────────
        if ts.day == 1 and ts.hour == 0 and ts.minute == 0:
            month_start_eq = acc['equity']

        # ── FIX-6: بررسی کیفیت جفت ارز (هر هفته) ──────────────────────
        if ts.weekday() == 0 and ts.hour == 2 and ts.minute == 0:
            cutoff = ts - timedelta(days=C.pair_eval_window)
            for n in pair_names:
                recent = [(t, p) for t, p in pair_recent_trades[n] if t >= cutoff]
                pair_recent_trades[n] = recent
                if len(recent) >= 10:
                    wins   = sum(p for _, p in recent if p > 0)
                    losses = abs(sum(p for _, p in recent if p < 0))
                    pf     = wins / losses if losses > 0 else 2.0
                    was_paused = pair_paused[n]
                    pair_paused[n] = pf < C.pair_min_pf
                    if pair_paused[n] != was_paused:
                        status = "⏸ PAUSED" if pair_paused[n] else "▶ RESUMED"
                        print(f"    {status} {n} | PF(180d)={pf:.2f} | {ts.date()}")

        # ── blown handler ────────────────────────────────────────────────
        if acc['blown']:
            acc_logs.append({
                'account': acc_num, 'start_ts': acc['start_ts'],
                'end_ts': ts, 'reason': acc['blown_rsn'],
                'pnl': acc['equity'] - C.initial_balance,
                'n_trades': len(acc['trades']),
                'days': (ts - acc['start_ts']).days
            })
            print(f"    💥 #{acc_num:>3} | {ts.date()} | Eq:${acc['equity']:>8.2f} | {acc['blown_rsn']}")
            # FIX-4: cooldown بعد از blown
            cooldown_until = ts + timedelta(days=C.cooldown_days)
            print(f"         ⏳ Cooldown تا {cooldown_until.date()}")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq  = acc['equity']
            month_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0
                pending_sig[n]  = 0
                positions[n]    = None
            continue

        # ── در cooldown: skip ────────────────────────────────────────────
        if in_cooldown:
            continue

        # ── FIX-5: Monthly risk reduction ────────────────────────────────
        monthly_pnl = acc['equity'] - month_start_eq
        monthly_stressed = monthly_pnl < C.monthly_loss_threshold

        # ── اجرای ورود ──────────────────────────────────────────────────
        for name in pair_names:
            a = pa[name]
            if (pending_sig[name] != 0
                    and positions[name] is None
                    and trades_today[name] < C.max_trades_day
                    and not pair_paused[name]):
                sv  = pending_sig[name]
                qr  = a['qr'][bar]
                pip = a['pip']
                sp  = a['sp']

                # FIX-5: در ماه استرس، ریسک نصف میشه
                eff_risk = C.risk_base_pct * (0.5 if monthly_stressed else 1.0)
                pip_val  = pip * C.lot_size * qr
                if pip_val <= 0: pip_val = 10.0
                risk_usd = acc['equity'] * eff_risk
                lot = round(float(np.clip(
                    risk_usd / (C.sl_pips * pip_val),
                    C.min_lot, C.max_lot)), 2)

                ep  = a['o'][bar] + sv * (C.slippage_pips + sp/2) * pip
                sl  = ep - sv * C.sl_pips * pip
                tp  = ep + sv * C.tp_pips * pip
                positions[name] = {
                    'pair': name, 'dir': sv, 'lot': lot,
                    'lot_remaining': lot, 'partial_done': False,
                    'entry': ep, 'sl': sl, 'tp': tp,
                    'entry_ts': ts, 'entry_bar': bar,
                    'pip': pip, 'sp': sp,
                }
                trades_today[name] += 1
            pending_sig[name] = 0

        # ── floating PnL و چک DD ─────────────────────────────────────────
        total_float = sum(
            calc_pnl_real(p['dir'], p['entry'], a['c'][bar],
                          p['lot_remaining'], a['qr'][bar], p['pip'])
            for name, (a, p) in ((n, (pa[n], positions[n]))
                                  for n in pair_names if positions[n] is not None)
        )
        current_eq  = acc['equity'] + total_float
        daily_limit = day_start_eq * (1 - C.max_daily_loss_pct)

        if current_eq <= daily_limit or current_eq <= PROP_FLOOR:
            reason = "DailyDD" if current_eq <= daily_limit else "TotalDD"
            acc['blown']     = True
            acc['blown_rsn'] = reason
            for name in pair_names:
                pos = positions[name]
                if pos is None: continue
                a   = pa[name]
                pnl = calc_pnl_real(pos['dir'], pos['entry'], a['c'][bar],
                                    pos['lot_remaining'], a['qr'][bar], pos['pip'])
                acc['equity'] += pnl
                rec = _rec(pos, a['c'][bar], ts, pnl, 'BLOWN', pos['lot_remaining'])
                all_trades.append(rec); acc['trades'].append(rec)
                pair_recent_trades[name].append((ts, pnl))
                positions[name] = None
            continue

        # ── مدیریت خروج ─────────────────────────────────────────────────
        for name in pair_names:
            pos = positions[name]
            if pos is None: continue
            a   = pa[name]
            cp  = a['c'][bar]; qr = a['qr'][bar]
            d   = pos['dir']; ep = pos['entry']
            zn  = a['z'][bar]; pip = pos['pip']
            lr  = pos['lot_remaining']

            # Partial exit
            if not pos['partial_done'] and not np.isnan(zn):
                hit_p = (d==1 and zn >= -C.z_exit_partial) or (d==-1 and zn <= C.z_exit_partial)
                if hit_p:
                    p_lot = round(lr * C.partial_ratio, 2)
                    if p_lot >= C.min_lot:
                        p_pnl = calc_pnl_real(d, ep, cp, p_lot, qr, pip)
                        if p_pnl > 0:
                            acc['equity'] += p_pnl
                            rec = _rec(pos, cp, ts, p_pnl, 'Partial', p_lot)
                            all_trades.append(rec); acc['trades'].append(rec)
                            pair_recent_trades[name].append((ts, p_pnl))
                            pos['lot_remaining'] = round(lr - p_lot, 2)
                            pos['partial_done']  = True
                            pos['sl'] = pos['entry']  # break-even
                            lr = pos['lot_remaining']
                            if lr < C.min_lot:
                                positions[name] = None; continue

            # شروط خروج
            hit_z_stop = not np.isnan(zn) and (
                (d==1 and zn<=-C.z_stop_margin) or (d==-1 and zn>=C.z_stop_margin))
            hit_z_exit = not np.isnan(zn) and (
                (d==1 and zn>=-C.z_exit_full) or (d==-1 and zn<=C.z_exit_full))
            pnl_now = calc_pnl_real(d, ep, cp, lr, qr, pip)
            if hit_z_exit and pnl_now < C.min_net_profit_usd and not pos['partial_done']:
                hit_z_exit = False

            hit_sl    = (d==1 and cp<=pos['sl']) or (d==-1 and cp>=pos['sl'])
            hit_tp    = (d==1 and cp>=pos['tp']) or (d==-1 and cp<=pos['tp'])
            bars_open = bar - pos['entry_bar']
            time_stop = (bars_open >= C.time_stop_bars and pnl_now < 0) or \
                        (bars_open >= C.time_stop_bars * 2)

            if hit_z_exit or hit_z_stop or hit_sl or hit_tp or time_stop:
                exit_px = pos['sl'] if hit_sl else (pos['tp'] if hit_tp else cp)
                st = ('SL' if hit_sl else ('TP' if hit_tp else
                      ('Z-Stop' if hit_z_stop else ('TimeStop' if time_stop else 'Z-Exit'))))
                fpnl = calc_pnl_real(d, ep, exit_px, lr, qr, pip)
                acc['equity'] += fpnl
                rec = _rec(pos, exit_px, ts, fpnl, st, lr)
                all_trades.append(rec); acc['trades'].append(rec)
                pair_recent_trades[name].append((ts, fpnl))
                positions[name] = None
                if fpnl > 0: acc['consec_loss'] = 0
                else:         acc['consec_loss'] += 1

        # ── هدف سود ─────────────────────────────────────────────────────
        all_closed = all(positions[n] is None for n in pair_names)
        if acc['equity'] >= PROFIT_LEVEL and all_closed and not acc['blown']:
            w = acc['equity'] - C.initial_balance
            total_withdrawn += w
            days_t = (ts - acc['start_ts']).days
            acc_logs.append({
                'account': acc_num, 'start_ts': acc['start_ts'],
                'end_ts': ts, 'reason': 'TARGET_HIT',
                'pnl': w, 'n_trades': len(acc['trades']), 'days': days_t
            })
            print(f"    💰 #{acc_num:>3} | {ts.date()} | ${w:>7.2f} | "
                  f"Bank:${total_withdrawn:>9.2f} | {days_t}d")
            acc_num += 1
            acc = new_acc(ts)
            day_start_eq  = acc['equity']
            month_start_eq = acc['equity']
            for n in pair_names:
                trades_today[n] = 0; pending_sig[n] = 0
            continue

        # ── سیگنال جدید ─────────────────────────────────────────────────
        for name in pair_names:
            a = pa[name]
            if (positions[name] is None and not acc['blown']
                    and not in_cooldown
                    and not pair_paused[name]
                    and trades_today[name] < C.max_trades_day
                    and a['sig'][bar] != 0):
                pending_sig[name] = int(a['sig'][bar])

    return {
        'all_trades':      all_trades,
        'account_logs':    acc_logs,
        'eq_curve':        eq_curve,
        'total_withdrawn': total_withdrawn,
        'final_equity':    acc['equity'],
        'total_accounts':  acc_num,
        'pair_names':      pair_names,
    }


def _rec(pos, exit_px, exit_ts, pnl, status, lot):
    return {
        'pair': pos['pair'], 'dir': pos['dir'], 'lot': lot,
        'entry': pos['entry'], 'exit': exit_px,
        'entry_ts': pos['entry_ts'], 'exit_ts': exit_ts,
        'pnl': pnl, 'status': status,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  گزارش کامل
# ═══════════════════════════════════════════════════════════════════════════
def print_report(results):
    trades = results['all_trades']
    if not trades:
        print("\n❌ No trades."); return

    df_t  = pd.DataFrame(trades)
    df_t['exit_ts'] = pd.to_datetime(df_t['exit_ts'])
    df_t['month']   = df_t['exit_ts'].dt.to_period('M')

    wins   = df_t[df_t['pnl'] > 0]
    losses = df_t[df_t['pnl'] < 0]
    wr = len(wins)/len(df_t)*100
    pf = wins['pnl'].sum()/abs(losses['pnl'].sum()) if len(losses) else float('inf')

    logs    = results['account_logs']
    df_acc  = pd.DataFrame(logs) if logs else pd.DataFrame()
    targets = df_acc[df_acc['reason']=='TARGET_HIT'] if len(df_acc) else pd.DataFrame()
    blowns  = df_acc[df_acc['reason']!='TARGET_HIT'] if len(df_acc) else pd.DataFrame()

    monthly = df_t.groupby('month')['pnl'].sum()
    pos_m   = int((monthly > 0).sum())
    neg_m   = int((monthly < 0).sum())
    total_m = len(monthly)

    print("\n" + "═"*70)
    print(f" ▌  CorrArb v9 Realistic — {'+'.join(results['pair_names'])}  ▐")
    print("═"*70)
    print(f" {'Total Trades:':<28} {len(df_t):,}")
    print(f" {'Win Rate:':<28} {wr:.2f}%")
    print(f" {'Profit Factor:':<28} {pf:.2f}")
    print(f" {'Avg Win:':<28} ${wins['pnl'].mean():.2f}")
    print(f" {'Avg Loss:':<28} ${losses['pnl'].mean():.2f}")
    print("-"*70)
    print(f" {'Accounts Passed:':<28} {len(targets)}")
    print(f" {'Accounts Blown:':<28} {len(blowns)}")
    if len(targets) and 'days' in targets.columns:
        print(f" {'Avg Days/Pass:':<28} {targets['days'].mean():.0f} روز")
    print("-"*70)
    print(f" {'Total Banked (15yr):':<28} ${results['total_withdrawn']:,.2f}")
    print(f" {'Active Equity:':<28} ${results['final_equity']:,.2f}")
    print(f" {'Monthly Avg:':<28} ${results['total_withdrawn']/180:.2f}/ماه")
    print("-"*70)
    print(f" {'ماه‌های مثبت:':<28} {pos_m} از {total_m}")
    print(f" {'ماه‌های منفی:':<28} {neg_m} از {total_m}")
    print(f" {'بهترین ماه:':<28} ${monthly.max():,.2f}")
    print(f" {'بدترین ماه:':<28} ${monthly.min():,.2f}")
    print("-"*70)
    print(" عملکرد هر جفت ارز:")
    for pair in results['pair_names']:
        pt = df_t[df_t['pair']==pair] if 'pair' in df_t.columns else pd.DataFrame()
        if len(pt) == 0: continue
        pw = pt[pt['pnl']>0]; pl = pt[pt['pnl']<0]
        ppf = pw['pnl'].sum()/abs(pl['pnl'].sum()) if len(pl) else float('inf')
        print(f"   {pair}: {len(pt):>4}T | WR:{len(pw)/len(pt)*100:4.1f}% | "
              f"PF:{ppf:.2f} | Net:${pt['pnl'].sum():>8,.2f}")
    print("-"*70)
    print(" خروج‌ها:")
    for st, cnt in df_t['status'].value_counts().items():
        print(f"   {st:<14}: {cnt:>5} ({cnt/len(df_t)*100:.1f}%)")
    print("═"*70)
    print(" مقایسه: v8→$7,151 | v9→${:,.0f} (spread واقعی‌تر)".format(
        results['total_withdrawn']))
    print("═"*70)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = datetime.now()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   CorrArb Prop Simulator v9 — Realistic Edition     ║")
    print("╚══════════════════════════════════════════════════════╝")

    pairs        = load_all_pairs()
    print("\n  Computing signals...")
    pair_signals = {n: compute_signals(n, pairs[n]) for n in pairs}
    results      = run_backtest(pairs, pair_signals)
    print_report(results)
    print(f"\n  ✅ Done in {(datetime.now()-t0).total_seconds():.2f}s")
