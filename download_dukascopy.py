import os
import struct
import lzma
import urllib.request
import urllib.error
import time
import csv
import pandas as pd
from datetime import datetime, timedelta

# ==================== تنظیمات ====================
INSTRUMENTS = {
    'EURUSD': {'point': 100000},
    'GBPUSD': {'point': 100000},
}

START_DATE = datetime(2021, 1, 1)
END_DATE   = datetime(2026, 1, 1)

OUTPUT_DIR = './data'
MAX_RETRIES = 5
RETRY_DELAY = 8
# =================================================

BASE_URL = (
    "https://datafeed.dukascopy.com/datafeed/"
    "{symbol}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"
)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': '*/*',
    'Accept-Encoding': 'identity',
    'Connection': 'keep-alive',
}

RECORD_SIZE = 24
# فرمت دوکاسکوپی:
# uint32: milliseconds از ابتدای روز
# uint32: open  × point
# uint32: high  × point
# uint32: low   × point
# uint32: close × point
# float32: volume
RECORD_FMT = '>IIIIIf'


def download_bi5(url: str) -> bytes | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                return data if len(data) > 0 else None

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # روز تعطیل یا بدون داده - نرمال است
            print(f"    HTTP {e.code} | تلاش {attempt}/{MAX_RETRIES}")

        except urllib.error.URLError as e:
            print(f"    URLError: {e.reason} | تلاش {attempt}/{MAX_RETRIES}")

        except Exception as e:
            print(f"    خطا: {type(e).__name__} | تلاش {attempt}/{MAX_RETRIES}")

        if attempt < MAX_RETRIES:
            wait = RETRY_DELAY * attempt
            time.sleep(wait)

    return None


def parse_bi5(raw_compressed: bytes, date: datetime, point: int) -> list:
    """
    پارس فایل bi5 دوکاسکوپی
    
    مهم: ماه در URL از صفر شروع می‌شه ولی date اینجا درسته
    فرمت: uint32(ms) + uint32(O) + uint32(H) + uint32(L) + uint32(C) + float32(V)
    """
    try:
        raw = lzma.decompress(raw_compressed)
    except lzma.LZMAError as e:
        return []

    if len(raw) < RECORD_SIZE:
        return []

    total_records = len(raw) // RECORD_SIZE
    candles = []

    for i in range(total_records):
        offset = i * RECORD_SIZE
        chunk = raw[offset: offset + RECORD_SIZE]

        try:
            t_ms, o, h, l, c, v = struct.unpack(RECORD_FMT, chunk)
        except struct.error:
            continue

        # فیلتر داده نامعتبر
        if o == 0 or c == 0:
            continue
        if h < l:
            continue
        if t_ms >= 86_400_000:  # بیشتر از 24 ساعت = نامعتبر
            continue

        ts = date + timedelta(milliseconds=t_ms)

        candles.append([
            ts.strftime('%Y-%m-%d %H:%M:%S'),
            round(o / point, 6),
            round(h / point, 6),
            round(l / point, 6),
            round(c / point, 6),
            round(v, 2),
        ])

    return candles


def verify_file(path: str) -> bool:
    """چک کردن صحت فایل CSV"""
    try:
        df = pd.read_csv(path, nrows=1000)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # چک توزیع timestamp
        if df['timestamp'].nunique() < 100:
            print(f"  ❌ timestamp های یکتا خیلی کم: {df['timestamp'].nunique()}")
            return False

        # چک اینکه ساعت‌های مختلف وجود داشته باشه
        hours = df['timestamp'].dt.hour.nunique()
        if hours < 5:
            print(f"  ❌ فقط {hours} ساعت متفاوت در 1000 ردیف اول")
            return False

        print(f"  ✅ فایل سالم | ساعت‌های یکتا: {hours} | نمونه timestamp: {df['timestamp'].iloc[0]}")
        return True

    except Exception as e:
        print(f"  ❌ خطا در verify: {e}")
        return False


def download_instrument(symbol: str, point: int, start: datetime, end: datetime):
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_M1_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )

    # چک فایل موجود
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        if size > 10 * 1024 * 1024:  # بیشتر از 10MB
            if verify_file(out_path):
                print(f"✅ فایل موجود و سالم: {out_path} ({size/1024/1024:.0f} MB)")
                return
        print(f"⚠️ فایل ناقص یا خراب، دانلود مجدد...")
        os.remove(out_path)

    print(f"\n{'='*60}")
    print(f"▶ دانلود: {symbol} | {start.date()} → {end.date()}")
    total_days = (end - start).days
    print(f"  کل روز: {total_days} | تخمین زمان: {total_days*0.4/60:.0f} دقیقه")
    print(f"{'='*60}")

    fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        current = start
        downloaded_days = 0
        skipped_days = 0
        total_candles = 0

        while current < end:
            # ⚠️ دوکاسکوپی: ماه از 0 شروع میشه (Jan=00, Dec=11)
            url = BASE_URL.format(
                symbol=symbol,
                year=current.year,
                month=current.month - 1,
                day=current.day
            )

            raw_data = download_bi5(url)

            if raw_data:
                candles = parse_bi5(raw_data, current, point)
                if candles:
                    writer.writerows(candles)
                    total_candles += len(candles)
                    downloaded_days += 1
                else:
                    skipped_days += 1
            else:
                skipped_days += 1

            current += timedelta(days=1)
            time.sleep(0.4)

            # گزارش هر 60 روز
            done = downloaded_days + skipped_days
            if done % 60 == 0:
                pct = done / total_days * 100
                print(
                    f"  [{pct:5.1f}%] "
                    f"روز: {done}/{total_days} | "
                    f"دانلود: {downloaded_days} | "
                    f"خالی: {skipped_days} | "
                    f"کندل: {total_candles:,}"
                )

    # گزارش نهایی
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"✅ {symbol} کامل شد:")
    print(f"   فایل: {out_path}")
    print(f"   حجم: {size_mb:.1f} MB")
    print(f"   کندل: {total_candles:,}")
    print(f"   روز با داده: {downloaded_days} | روز خالی: {skipped_days}")

    # verify
    print(f"\n  بررسی صحت فایل...")
    verify_file(out_path)

    # نمایش نمونه
    df_sample = pd.read_csv(out_path, nrows=5)
    print(f"\n  نمونه اول:\n{df_sample.to_string()}")
    df_tail = pd.read_csv(out_path)
    print(f"\n  نمونه آخر:\n{df_tail.tail(3).to_string()}")
    print(f"{'='*60}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"دانلود داده دوکاسکوپی")
    print(f"بازه: {START_DATE.date()} → {END_DATE.date()}")
    print(f"جفت ارزها: {list(INSTRUMENTS.keys())}")
    total_est = (END_DATE - START_DATE).days * len(INSTRUMENTS) * 0.4 / 60
    print(f"تخمین کل زمان: {total_est:.0f} دقیقه")
    print(f"{'='*60}\n")

    for symbol, config in INSTRUMENTS.items():
        download_instrument(
            symbol=symbol,
            point=config['point'],
            start=START_DATE,
            end=END_DATE,
        )
        if symbol != list(INSTRUMENTS.keys())[-1]:
            print(f"\n⏳ ۱۵ ثانیه تاخیر بین جفت ارزها...")
            time.sleep(15)

    print(f"\n{'='*60}")
    print("✅ همه دانلودها کامل شد")
    print(f"{'='*60}")
    print("\nفایل‌های نهایی:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.csv'):
            path = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(path) / 1024 / 1024
            lines = sum(1 for _ in open(path)) - 1
            print(f"  {f} | {lines:,} کندل | {size:.1f} MB")


if __name__ == '__main__':
    main()
