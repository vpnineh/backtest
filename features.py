"""
features.py — Feature Engineering برای ML CorrArb
═══════════════════════════════════════════════════
تمام featureها با rolling/expanding محاسبه میشن
هیچ اطلاعات آینده‌ای leak نمیشه (strict point-in-time)

Feature Groups:
  1. Z-score family (multi-window)
  2. Momentum & divergence
  3. Volatility regime
  4. Correlation structure
  5. Market microstructure
  6. Calendar / session
  7. Trend / mean-reversion regime
  8. Target labels (forward-looking — فقط برای train)
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════
#  اندیکاتورهای پایه
# ═══════════════════════════════════════════════════════════════════════════

def _atr(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean()

def _rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    ls = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1 + g/ls.replace(0, np.nan))

def _adx(h, l, c, p=14):
    up = h.diff(); dn = -l.diff()
    dmp = up.where((up>dn)&(up>0), 0.0)
    dmn = dn.where((dn>up)&(dn>0), 0.0)
    atr1 = _atr(h, l, c, 1)
    s = atr1.rolling(p).sum().replace(0, np.nan)
    dip = 100*dmp.rolling(p).sum()/s
    din = 100*dmn.rolling(p).sum()/s
    dx = (abs(dip-din)/(dip+din).replace(0, np.nan))*100
    return dx.rolling(p).mean()

def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _sma(s, p): return s.rolling(p).mean()
def _std(s, p): return s.rolling(p).std()

def _zscore(s, p):
    m = _sma(s, p); st = _std(s, p)
    return (s - m) / st.replace(0, np.nan)

def _hurst(s, window=100):
    """Hurst exponent تقریبی — H<0.5 mean-reversion، H>0.5 trend"""
    def hurst_single(x):
        if len(x) < 20 or x.std() == 0:
            return 0.5
        lags = range(2, min(20, len(x)//2))
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    return s.rolling(window).apply(hurst_single, raw=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Builder اصلی
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, label_horizon: int = 40) -> pd.DataFrame:
    """
    ورودی: df با ستونهای c_eur, h_eur, l_eur, o_eur, v_eur,
                              c_gbp, h_gbp, l_gbp, o_gbp, v_gbp
    خروجی: DataFrame با همه featureها + target labels
    
    label_horizon: چند کندل بعد رو برای label در نظر بگیریم (۴۰×۱۵min ≈ ۱۰ساعت)
    """
    print("  ساخت feature matrix...", end="", flush=True)
    
    ce = df['c_eur']; he = df['h_eur']; le = df['l_eur']; oe = df['o_eur']; ve = df['v_eur']
    cg = df['c_gbp']; hg = df['h_gbp']; lg = df['l_gbp']; og = df['o_gbp']; vg = df['v_gbp']
    
    ft = pd.DataFrame(index=df.index)
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 1: Z-score family — چندین window
    # ─────────────────────────────────────────────────────────────────────
    ratio = ce / cg
    for w in [24, 48, 96, 192, 384, 768]:
        ft[f'z_ratio_{w}'] = _zscore(ratio, w)
    
    # spread بین z-scoreهای مختلف (momentum of mean-reversion signal)
    ft['z_diff_fast_slow']   = ft['z_ratio_48']  - ft['z_ratio_384']
    ft['z_diff_mid_slow']    = ft['z_ratio_96']  - ft['z_ratio_768']
    ft['z_ratio_accel']      = ft['z_ratio_48']  - ft['z_ratio_48'].shift(12)
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 2: Momentum & Divergence
    # ─────────────────────────────────────────────────────────────────────
    for lag in [4, 12, 24, 48, 96, 192]:
        ft[f'ret_e_{lag}']   = ce.pct_change(lag)
        ft[f'ret_g_{lag}']   = cg.pct_change(lag)
        ft[f'div_{lag}']     = ft[f'ret_e_{lag}'] - ft[f'ret_g_{lag}']
    
    # momentum of divergence
    ft['div_momentum'] = ft['div_12'] - ft['div_12'].shift(12)
    ft['div_accel']    = ft['div_24'] - ft['div_24'].shift(24)
    
    # RSI divergence
    rsi_e = _rsi(ce, 14); rsi_g = _rsi(cg, 14)
    ft['rsi_eur']     = rsi_e
    ft['rsi_gbp']     = rsi_g
    ft['rsi_div']     = rsi_e - rsi_g
    ft['rsi_e_z']     = _zscore(rsi_e, 96)
    ft['rsi_g_z']     = _zscore(rsi_g, 96)
    
    # ADX
    ft['adx_eur']     = _adx(he, le, ce, 14)
    ft['adx_gbp']     = _adx(hg, lg, cg, 14)
    ft['adx_diff']    = ft['adx_eur'] - ft['adx_gbp']
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 3: Volatility Regime
    # ─────────────────────────────────────────────────────────────────────
    atr_e = _atr(he, le, ce, 14)
    atr_g = _atr(hg, lg, cg, 14)
    
    ft['atr_eur']         = atr_e
    ft['atr_gbp']         = atr_g
    ft['atr_ratio']       = atr_e / atr_g.replace(0, np.nan)
    
    # normalize ATR به میانگین بلندمدت
    for p in [48, 96, 192]:
        ft[f'atr_e_norm_{p}'] = atr_e / atr_e.rolling(p).mean().replace(0, np.nan)
        ft[f'atr_g_norm_{p}'] = atr_g / atr_g.rolling(p).mean().replace(0, np.nan)
    
    # Volatility regime: کم، نرمال، زیاد
    atr_pct = atr_e.rolling(480).rank(pct=True)  # percentile rank در ۵ روز
    ft['vol_regime'] = pd.cut(atr_pct, bins=[0, 0.25, 0.75, 1.0],
                               labels=[0, 1, 2]).astype(float)
    
    # Historical volatility (realized)
    log_ret_e = np.log(ce/ce.shift(1))
    log_ret_g = np.log(cg/cg.shift(1))
    for p in [24, 96, 384]:
        ft[f'hvol_e_{p}'] = log_ret_e.rolling(p).std() * np.sqrt(p)
        ft[f'hvol_g_{p}'] = log_ret_g.rolling(p).std() * np.sqrt(p)
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 4: Correlation Structure
    # ─────────────────────────────────────────────────────────────────────
    for w in [24, 48, 96, 192]:
        ft[f'corr_{w}'] = log_ret_e.rolling(w).corr(log_ret_g)
    
    # correlation momentum (در حال افت یا افزایش؟)
    ft['corr_momentum']  = ft['corr_48'] - ft['corr_48'].shift(24)
    ft['corr_breakdown'] = (ft['corr_96'] - ft['corr_192']).abs()  # ناپایداری correlation
    
    # Beta (حساسیت EUR به GBP)
    cov = log_ret_e.rolling(96).cov(log_ret_g)
    var_g = log_ret_g.rolling(96).var().replace(0, np.nan)
    ft['beta_eur_gbp'] = cov / var_g
    ft['beta_z']       = _zscore(ft['beta_eur_gbp'], 192)
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 5: Market Microstructure
    # ─────────────────────────────────────────────────────────────────────
    # Candle body ratio
    body_e = (ce - oe).abs() / (he - le + 1e-10)
    body_g = (cg - og).abs() / (hg - lg + 1e-10)
    ft['body_e']     = body_e
    ft['body_g']     = body_g
    ft['body_diff']  = body_e - body_g
    
    # Upper/lower wick
    upper_e = (he - ce.clip(lower=oe)) / (he - le + 1e-10)
    lower_e = (ce.clip(upper=oe) - le) / (he - le + 1e-10)
    ft['wick_upper_e'] = upper_e.rolling(12).mean()
    ft['wick_lower_e'] = lower_e.rolling(12).mean()
    
    # Volume relative
    if ve.sum() > 0:
        ft['vol_e_rel'] = ve / ve.rolling(96).mean().replace(0, np.nan)
        ft['vol_g_rel'] = vg / vg.rolling(96).mean().replace(0, np.nan)
        ft['vol_div']   = ft['vol_e_rel'] - ft['vol_g_rel']
    else:
        ft['vol_e_rel'] = ft['vol_g_rel'] = ft['vol_div'] = 0.0
    
    # Price distance from EMA
    for p in [24, 96, 192]:
        ema_e = _ema(ce, p); ema_g = _ema(cg, p)
        ft[f'ema_dist_e_{p}'] = (ce - ema_e) / ema_e.replace(0, np.nan)
        ft[f'ema_dist_g_{p}'] = (cg - ema_g) / ema_g.replace(0, np.nan)
    
    # Bollinger Band position
    for p in [48, 96]:
        m = _sma(ratio, p); s = _std(ratio, p)
        ft[f'bb_pos_{p}'] = (ratio - m) / (2*s.replace(0, np.nan))  # -1 تا +1
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 6: Calendar / Session
    # ─────────────────────────────────────────────────────────────────────
    ft['hour']         = df.index.hour.astype(float)
    ft['dow']          = df.index.dayofweek.astype(float)
    ft['month']        = df.index.month.astype(float)
    
    # Session encoding (sine/cosine برای cyclical features)
    ft['hour_sin']     = np.sin(2*np.pi*df.index.hour/24)
    ft['hour_cos']     = np.cos(2*np.pi*df.index.hour/24)
    ft['dow_sin']      = np.sin(2*np.pi*df.index.dayofweek/5)
    ft['dow_cos']      = np.cos(2*np.pi*df.index.dayofweek/5)
    
    # Session flags
    hour = df.index.hour
    ft['sess_asia']    = ((hour >= 0) & (hour < 7)).astype(float)
    ft['sess_london']  = ((hour >= 7) & (hour < 12)).astype(float)
    ft['sess_overlap'] = ((hour >= 12) & (hour < 16)).astype(float)
    ft['sess_ny']      = ((hour >= 16) & (hour < 21)).astype(float)
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 7: Regime Detection
    # ─────────────────────────────────────────────────────────────────────
    # Hurst exponent (mean-reversion vs trend)
    ft['hurst_ratio']  = _hurst(ratio, 100)
    ft['hurst_eur']    = _hurst(ce, 100)
    
    # EMA crossover regime
    ema24 = _ema(ratio, 24); ema96 = _ema(ratio, 96)
    ft['ema_cross']    = (ema24 - ema96) / ema96.replace(0, np.nan)
    ft['ema_cross_z']  = _zscore(ft['ema_cross'], 192)
    
    # چند روز متوالی در یه طرف Z بودیم؟
    z_sign = np.sign(ft['z_ratio_96'].fillna(0))
    ft['z_streak'] = z_sign.groupby((z_sign != z_sign.shift()).cumsum()).cumcount() + 1
    ft['z_streak']  *= z_sign
    
    # ─────────────────────────────────────────────────────────────────────
    # Group 8: Target Labels (FORWARD-LOOKING — فقط برای train)
    # ─────────────────────────────────────────────────────────────────────
    # آیا ratio در horizon کندل بعدی به mean برمیگرده؟
    ratio_future = ratio.shift(-label_horizon)
    
    # label برای Long (ratio پایین → انتظار افزایش)
    ft['target_long']  = (
        (ratio_future > ratio) &                    # ratio افزایش یافت
        (ft['z_ratio_96'] < -1.0)                  # الان در منطقه oversold
    ).astype(float)
    
    # label برای Short
    ft['target_short'] = (
        (ratio_future < ratio) &
        (ft['z_ratio_96'] > 1.0)
    ).astype(float)
    
    # label ترکیبی (برای classification)
    # 0 = no trade, 1 = long, 2 = short
    ft['target'] = 0.0
    ft.loc[ft['target_long']  == 1, 'target'] = 1.0
    ft.loc[ft['target_short'] == 1, 'target'] = 2.0
    
    # target کیفی‌تر: میزان بازگشت (برای regression)
    ft['target_ret'] = (ratio_future - ratio) / ratio.replace(0, np.nan)

    # ─────────────────────────────────────────────────────────────────────
    # پاک‌سازی نشت لیبل: shift(-label_horizon) در انتهای دیتاست NaN می‌سازد.
    # چون target_long/short با astype(float) به ۰ کلمپ می‌شوند، این ردیف‌ها
    # به‌صورت کاذب وارد کلاس no-trade می‌شوند. آن‌ها را بر اساس target_ret حذف می‌کنیم.
    n_before = len(ft)
    ft = ft.dropna(subset=['target_ret'])
    n_dropped = n_before - len(ft)
    if n_dropped:
        print(f"  ⓘ {n_dropped} ردیف انتهایی با لیبل ناقص حذف شد (label_horizon={label_horizon})")

    print(f"  ✓ — {len(ft.columns)} feature")
    return ft



# ═══════════════════════════════════════════════════════════════════════════
#  Walk-Forward Data Splitter
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_splits(
    df: pd.DataFrame,
    train_years: float = 4.0,
    val_years:   float = 1.0,
    test_years:  float = 1.0,
    step_months: int   = 6,
    embargo_bars: int  = 40,
) -> list:
    """
    Walk-forward splits با embargo برای حذف data leakage مرزی.

    embargo_bars باید برابر label_horizon باشد (پیش‌فرض ۴۰ کندل) تا
    ردیف‌هایی که لیبلشان به آینده‌ی بازه‌ی بعدی نگاه می‌کند purge شوند.

    مثال با ۱۰ سال داده:
        Split 0: train=[2000-2004], val=[2004-2005], test=[2005-2006]
        Split 1: train=[2000.5-2004.5], val=[2004.5-2005.5], test=[2005.5-2006.5]
        ...

    خروجی: list از dict {train, val, test, ...}
    """
    import pandas as pd
    from dateutil.relativedelta import relativedelta

    start = df.index[0]
    end   = df.index[-1]

    train_d = relativedelta(years=int(train_years),
                            months=int((train_years % 1) * 12))
    val_d   = relativedelta(years=int(val_years),
                            months=int((val_years % 1) * 12))
    test_d  = relativedelta(years=int(test_years),
                            months=int((test_years % 1) * 12))
    step_d  = relativedelta(months=step_months)

    splits = []
    train_start = start

    while True:
        val_start  = train_start + train_d
        test_start = val_start   + val_d
        test_end   = test_start  + test_d

        if test_end > end:
            break

        train_mask = (df.index >= train_start) & (df.index < val_start)
        val_mask   = (df.index >= val_start)   & (df.index < test_start)
        test_mask  = (df.index >= test_start)  & (df.index < test_end)

        # ── Embargo / Purge ──────────────────────────────────────────────
        # آخرین embargo_bars ردیفِ هر بازه‌ی train و val حذف می‌شوند چون
        # لیبلشان (shift(-label_horizon)) به ردیف‌های بازه‌ی بعدی نشت می‌کند.
        if embargo_bars > 0:
            for mask in (train_mask, val_mask):
                idx = np.flatnonzero(mask)
                if idx.size > embargo_bars:
                    mask[idx[-embargo_bars:]] = False

        if train_mask.sum() > 100 and val_mask.sum() > 100 and test_mask.sum() > 100:
            splits.append({
                'train': df.index[train_mask],
                'val':   df.index[val_mask],
                'test':  df.index[test_mask],
                'train_start': train_start,
                'val_start':   val_start,
                'test_start':  test_start,
                'test_end':    test_end,
            })

        train_start += step_d

    print(f"  Walk-forward splits: {len(splits)}  (embargo={embargo_bars} bars)")
    for i, s in enumerate(splits):
        print(f"    [{i}] train={s['train_start'].date()}→{s['val_start'].date()} "
              f"| val={s['val_start'].date()}→{s['test_start'].date()} "
              f"| test={s['test_start'].date()}→{s['test_end'].date()}")

    return splits


def get_feature_cols(ft: pd.DataFrame) -> list:
    """برگرداندن لیست feature columnها (بدون targetها)"""
    exclude = {'target', 'target_long', 'target_short', 'target_ret'}
    return [c for c in ft.columns if c not in exclude]


if __name__ == '__main__':
    # تست سریع
    import glob, sys
    sys.path.insert(0, '.')
    
    files_e = sorted(glob.glob('data/*EURUSD*.csv'))
    files_g = sorted(glob.glob('data/*GBPUSD*.csv'))
    
    if not files_e:
        print("❌ فایل داده پیدا نشد. مسیر data/ رو چک کن.")
    else:
        def _read(paths, suf):
            frames = []
            for p in paths:
                d = pd.read_csv(p, sep=';', header=None,
                                names=['ts','o','h','l','c','v'])
                d['ts'] = pd.to_datetime(d['ts'], format='%Y%m%d %H%M%S')
                d = d.set_index('ts')
                d.columns = [f'{c}_{suf}' for c in d.columns]
                frames.append(d)
            return pd.concat(frames).sort_index()
        
        eur = _read(files_e, 'eur')
        gbp = _read(files_g, 'gbp')
        raw = eur.join(gbp, how='inner').dropna()
        df  = pd.DataFrame({
            'o_eur': raw['o_eur'].resample('15min').first(),
            'h_eur': raw['h_eur'].resample('15min').max(),
            'l_eur': raw['l_eur'].resample('15min').min(),
            'c_eur': raw['c_eur'].resample('15min').last(),
            'v_eur': raw['v_eur'].resample('15min').sum(),
            'o_gbp': raw['o_gbp'].resample('15min').first(),
            'h_gbp': raw['h_gbp'].resample('15min').max(),
            'l_gbp': raw['l_gbp'].resample('15min').min(),
            'c_gbp': raw['c_gbp'].resample('15min').last(),
            'v_gbp': raw['v_gbp'].resample('15min').sum(),
        }).dropna()
        df = df[df.index.weekday < 5]
        
        ft = build_features(df)
        print(f"\n✅ Feature matrix: {ft.shape}")
        print(f"  Feature cols: {len(get_feature_cols(ft))}")
        print(f"  Target dist:\n{ft['target'].value_counts()}")
        
        splits = walk_forward_splits(df)
        print(f"\n✅ {len(splits)} splits آماده")
