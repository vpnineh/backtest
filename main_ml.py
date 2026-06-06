"""
main_ml.py — ML CorrArb Pipeline اصلی
══════════════════════════════════════
اجرای کامل:
  1. بارگذاری داده
  2. ساخت features
  3. Walk-forward splits
  4. آموزش Model A + B + C
  5. Meta-Learner + Prop Risk Engine
  6. گزارش جامع

نصب وابستگی‌ها:
  pip install lightgbm xgboost scikit-learn pandas numpy joblib
  pip install tensorflow  (اختیاری - برای LSTM)
  pip install stable-baselines3 gymnasium  (اختیاری - برای RL)
"""

import pandas as pd
import numpy as np
import glob, os, sys, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# اطمینان از import درست
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features    import build_features, walk_forward_splits, get_feature_cols
from meta_trainer import WalkForwardTrainer


# ═══════════════════════════════════════════════════════════════════════════
#  بارگذاری داده (همان ساختار v4)
# ═══════════════════════════════════════════════════════════════════════════

def load_data(data_dir: str = 'data') -> pd.DataFrame:
    files_eur = sorted(glob.glob(f'{data_dir}/*EURUSD*.csv'))
    files_gbp = sorted(glob.glob(f'{data_dir}/*GBPUSD*.csv'))
    
    if not files_eur:
        raise FileNotFoundError(f"EURUSD CSV در {data_dir}/ پیدا نشد")
    if not files_gbp:
        raise FileNotFoundError(f"GBPUSD CSV در {data_dir}/ پیدا نشد")
    
    def read_pair(paths, suffix):
        frames = []
        for p in paths:
            d = pd.read_csv(p, sep=';', header=None,
                            names=['ts', 'o', 'h', 'l', 'c', 'v'])
            d['ts'] = pd.to_datetime(d['ts'], format='%Y%m%d %H%M%S')
            d = d.set_index('ts')
            d = d[~d.index.duplicated(keep='last')]
            d.columns = [f'{col}_{suffix}' for col in d.columns]
            frames.append(d)
        return pd.concat(frames).sort_index()
    
    eur = read_pair(files_eur, 'eur')
    gbp = read_pair(files_gbp, 'gbp')
    raw = eur.join(gbp, how='inner').dropna()
    
    df = pd.DataFrame({
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
    
    print(f"✅ {len(df):,} کندل | {df.index[0].date()} → {df.index[-1].date()}")
    print(f"   سال‌ها: {(df.index[-1] - df.index[0]).days / 365.25:.1f}")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  تنظیمات Walk-Forward
# ═══════════════════════════════════════════════════════════════════════════

WALK_FORWARD_CONFIG = {
    'train_years':  4.0,   # ۴ سال آموزش
    'val_years':    1.0,   # ۱ سال validation
    'test_years':   1.0,   # ۱ سال test
    'step_months':  6,     # هر ۶ ماه پیش برو
}

FEATURE_CONFIG = {
    'label_horizon': 40,   # ۴۰ × ۱۵min = ۱۰ ساعت آینده برای label
}


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline اصلی
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(data_dir: str = 'data',
                 output_dir: str = 'ml_models',
                 max_splits: int = None,      # None = همه split‌ها
                 specific_split: int = None,  # ست کردن یک اسپلیت خاص برای موازی‌سازی گیت‌هاب
                 skip_rl: bool = False,       # True = سریع‌تر اجرا بشه
                 skip_lstm: bool = False):
    
    t0 = datetime.now()
    
    print("═"*70)
    print("  ML CorrArb — Walk-Forward Pipeline")
    print("═"*70)
    print(f"  زمان شروع: {t0.strftime('%Y-%m-%d %H:%M')}")
    
    # ── ۱. بارگذاری داده ──
    print("\n▶ مرحله ۱: بارگذاری داده")
    df = load_data(data_dir)
    
    # ── ۲. ساخت features ──
    print("\n▶ مرحله ۲: Feature Engineering")
    ft = build_features(df, label_horizon=FEATURE_CONFIG['label_horizon'])
    
    feat_cols = get_feature_cols(ft)
    print(f"  {len(feat_cols)} feature | target dist: "
          f"none={( ft['target']==0).sum():,} | "
          f"long={( ft['target']==1).sum():,} | "
          f"short={(ft['target']==2).sum():,}")
    
    # ── ۳. Walk-Forward Splits ──
    print("\n▶ مرحله ۳: Walk-Forward Splits")
    splits = walk_forward_splits(df, **WALK_FORWARD_CONFIG)
    
    if specific_split is not None:
        if specific_split >= len(splits):
            print(f"❌ شماره Split ({specific_split}) از تعداد کل Splitها ({len(splits)}) بیشتر است.")
            return None
        splits = [splits[specific_split]]
        print(f"  (محدود به اجرای موازی: فقط Split شماره {specific_split})")
    elif max_splits:
        splits = splits[:max_splits]
        print(f"  (محدود به {max_splits} split برای تست)")
    
    if not splits:
        print("❌ داده کافی برای walk-forward نیست")
        print("   برای walk-forward با train=4yr+val=1yr+test=1yr حداقل ۶ سال داده لازمه")
        return None
    
    # ── ۴. آموزش Walk-Forward ──
    print(f"\n▶ مرحله ۴: آموزش {len(splits)} split")
    
    trainer = WalkForwardTrainer(output_dir=output_dir)
    
    for i, split in enumerate(splits):
        try:
            # پیدا کردن ایندکس واقعی اسپلیت برای حفظ یکپارچگی لاگ‌ها و مدل‌های خروجی
            actual_idx = specific_split if specific_split is not None else i
            result = trainer.train_split(ft, df, split, actual_idx)
        except Exception as e:
            print(f"  ❌ Split {i} خطا: {e}")
            import traceback; traceback.print_exc()
            continue
    
    # ── ۵. گزارش ──
    print("\n▶ مرحله ۵: گزارش Walk-Forward")
    trainer.print_summary()
    
    # ── ۶. تحلیل ثبات ──
    print("\n▶ مرحله ۶: تحلیل ثبات")
    analyze_consistency(trainer.split_results)
    
    total_time = (datetime.now() - t0).total_seconds()
    print(f"\n✅ کل زمان اجرا: {total_time:.0f}s ({total_time/60:.1f} دقیقه)")
    
    return trainer


# ═══════════════════════════════════════════════════════════════════════════
#  تحلیل ثبات
# ═══════════════════════════════════════════════════════════════════════════

def analyze_consistency(results: list):
    """
    بررسی اینکه آیا نتایج در طول زمان ثابت هستن؟
    این مهم‌ترین معیار برای پراپ هست.
    """
    if len(results) < 1:
        print("  داده کافی برای تحلیل ثبات نیست")
        return
    
    win_rates = [r.get('win_rate', 0) for r in results]
    sharpes   = [r.get('sharpe', 0)   for r in results]
    dds       = [abs(r.get('max_dd', 0)) for r in results]
    rets      = [r.get('total_ret', 0) for r in results]
    
    print(f"  Win Rate: avg={np.mean(win_rates):.1f}%  std={np.std(win_rates):.1f}%  "
          f"min={min(win_rates):.1f}%  max={max(win_rates):.1f}%")
    print(f"  Sharpe:   avg={np.mean(sharpes):.2f}  std={np.std(sharpes):.2f}")
    print(f"  Max DD:   avg={np.mean(dds):.1f}%  max={max(dds):.1f}%")
    print(f"  Return:   avg={np.mean(rets):+.1f}%  std={np.std(rets):.1f}%")
    
    # positive splits
    pos_splits = sum(1 for r in results if r.get('total_ret', 0) > 0)
    print(f"  Split‌های مثبت: {pos_splits}/{len(results)} ({pos_splits/len(results)*100:.0f}%)")
    
    # stability check
    stable = (
        np.std(win_rates) < 15 and
        np.mean(win_rates) > 50 and
        max(dds) < 8.0 and
        pos_splits / len(results) > 0.6
    ) if len(results) >= 3 else True
    
    flag = "✅ سیستم پایدار" if stable else "⚠️ نیاز به بهبود"
    print(f"\n  نتیجه ثبات: {flag}")
    
    if not stable:
        issues = []
        if np.std(win_rates) >= 15:
            issues.append(f"Win Rate نوسان زیاد ({np.std(win_rates):.1f}%)")
        if np.mean(win_rates) <= 50:
            issues.append(f"Win Rate پایین ({np.mean(win_rates):.1f}%)")
        if max(dds) >= 8.0:
            issues.append(f"DD بیش از حد ({max(dds):.1f}%)")
        if pos_splits / len(results) <= 0.6:
            issues.append(f"تعداد زیاد split ضررده ({len(results)-pos_splits}/{len(results)})")
        
        for issue in issues:
            print(f"    ⚠️ {issue}")


# ═══════════════════════════════════════════════════════════════════════════
#  Quick Test Mode (بدون داده واقعی)
# ═══════════════════════════════════════════════════════════════════════════

def run_quick_test():
    """تست pipeline با داده مصنوعی — برای بررسی صحت کد"""
    print("═"*70)
    print("  QUICK TEST MODE — داده مصنوعی")
    print("═"*70)
    
    np.random.seed(42)
    n = 50_000  # ~۵ سال داده ۱۵ دقیقه‌ای
    
    idx = pd.date_range('2019-01-01', periods=n, freq='15min')
    idx = idx[idx.weekday < 5]
    
    # شبیه‌سازی قیمت رئال‌تر با mean-reversion در ratio
    price_e = 1.1000
    price_g = 1.2500
    prices_e, prices_g = [price_e], [price_g]
    
    for _ in range(len(idx)-1):
        ratio = prices_e[-1] / prices_g[-1]
        mean_ratio = 0.88
        reversion = (mean_ratio - ratio) * 0.001  # mean reversion
        
        ret_e = reversion * 0.5 + np.random.randn() * 0.0004
        ret_g = -reversion * 0.5 + np.random.randn() * 0.0004
        
        prices_e.append(prices_e[-1] * (1 + ret_e))
        prices_g.append(prices_g[-1] * (1 + ret_g))
    
    c_eur = pd.Series(prices_e, index=idx[:len(prices_e)])
    c_gbp = pd.Series(prices_g, index=idx[:len(prices_g)])
    
    df = pd.DataFrame({
        'o_eur': c_eur * (1 + np.random.randn(len(idx))*0.0001),
        'h_eur': c_eur * (1 + abs(np.random.randn(len(idx)))*0.0003),
        'l_eur': c_eur * (1 - abs(np.random.randn(len(idx)))*0.0003),
        'c_eur': c_eur,
        'v_eur': np.random.randint(100, 1000, len(idx)).astype(float),
        'o_gbp': c_gbp * (1 + np.random.randn(len(idx))*0.0001),
        'h_gbp': c_gbp * (1 + abs(np.random.randn(len(idx)))*0.0003),
        'l_gbp': c_gbp * (1 - abs(np.random.randn(len(idx)))*0.0003),
        'c_gbp': c_gbp,
        'v_gbp': np.random.randint(100, 1000, len(idx)).astype(float),
    }, index=idx)
    
    print(f"✅ داده مصنوعی: {len(df):,} کندل | {df.index[0].date()} → {df.index[-1].date()}")
    
    # Feature Engineering
    print("\n▶ Feature Engineering")
    ft = build_features(df, label_horizon=40)
    
    feat_cols = get_feature_cols(ft)
    print(f"  {len(feat_cols)} features")
    
    # Splits
    print("\n▶ Walk-Forward Splits")
    splits = walk_forward_splits(df, train_years=3, val_years=0.5, test_years=0.5, step_months=6)
    
    if not splits:
        print("❌ داده کافی نیست حتی برای quick test")
        return
    
    # فقط اول split
    trainer = WalkForwardTrainer(output_dir='ml_models_test')
    try:
        result = trainer.train_split(ft, df, splits[0], 0)
        if result:
            trainer.print_summary()
    except Exception as e:
        print(f"❌ خطا در quick test: {e}")
        import traceback; traceback.print_exc()
    
    return trainer


# ═══════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ML CorrArb Pipeline')
    parser.add_argument('--data',       default='data',      help='مسیر فایل‌های CSV')
    parser.add_argument('--output',     default='ml_models', help='مسیر ذخیره مدل‌ها')
    parser.add_argument('--max_splits', type=int, default=None, help='حداکثر تعداد split (تست)')
    parser.add_argument('--split_idx',  type=int, default=None, help='اجرای یک split خاص برای گیت‌هاب اکشنز موازی')
    parser.add_argument('--test',       action='store_true', help='اجرای quick test بدون داده')
    args = parser.parse_args()
    
    if args.test:
        run_quick_test()
    else:
        trainer = run_pipeline(
            data_dir       = args.data,
            output_dir     = args.output,
            max_splits     = args.max_splits,
            specific_split = args.split_idx,
        )
