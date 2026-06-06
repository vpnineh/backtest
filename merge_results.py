import json
import glob
import numpy as np

def merge_and_report():
    print("═"*70)
    print("  ML CorrArb — Final Aggregated Report (16 Years)")
    print("═"*70)

    # پیدا کردن تمام فایل‌های نتایج تولید شده در پوشه‌های مختلف
    result_files = glob.glob('ml_models_*/wf_results.json')
    
    if not result_files:
        print("❌ هیچ فایل نتیجه‌ای پیدا نشد!")
        return

    all_splits = []
    for f in result_files:
        with open(f, 'r') as file:
            data = json.load(file)
            all_splits.extend(data)

    # مرتب‌سازی بر اساس زمان (یا ایندکس اسپلیت)
    all_splits.sort(key=lambda x: x.get('split_idx', 0))

    if not all_splits:
        print("❌ داده‌ای در فایل‌های JSON یافت نشد.")
        return

    # استخراج متریک‌ها
    win_rates = [s.get('win_rate', 0) for s in all_splits]
    sharpes   = [s.get('sharpe', 0) for s in all_splits]
    dds       = [abs(s.get('max_dd', 0)) for s in all_splits]
    rets      = [s.get('total_ret', 0) for s in all_splits]
    trades    = [s.get('trades', 0) for s in all_splits]
    targets   = sum(s.get('n_target', 0) for s in all_splits)
    blown     = sum(s.get('n_blown', 0) for s in all_splits)

    print(f"  کل Split‌های بررسی شده: {len(all_splits)}")
    print(f"  مجموع معاملات:         {sum(trades):,}")
    print(f"  دفعات پاس شدن تارگت:   {targets}")
    print(f"  دفعات Blown شدن حساب:  {blown}")
    print("-" * 70)
    print(f"  میانگین Win Rate:      {np.mean(win_rates):.1f}%")
    print(f"  میانگین Sharpe:        {np.mean(sharpes):.2f}")
    print(f"  میانگین Drawdown:      {np.mean(dds):.1f}% (بدترین: {max(dds):.1f}%)")
    print(f"  میانگین بازدهی(Split): {np.mean(rets):+.1f}%")
    
    # ذخیره یک فایل واحد
    with open('FINAL_REPORT.json', 'w') as out_f:
        json.dump(all_splits, out_f, indent=2)
    
    print("-" * 70)
    print("✅ گزارش یکپارچه در فایل FINAL_REPORT.json ذخیره شد.")

if __name__ == '__main__':
    merge_results()
