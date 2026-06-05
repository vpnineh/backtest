import os
import struct
import lzma
import urllib.request
import urllib.error
import time
import csv
from datetime import datetime, timedelta

# ==================== تنظیمات ====================
INSTRUMENTS = {
    'EURUSD': {'point': 100000},
    'GBPUSD': {'point': 100000},
}

START_DATE = datetime(2022, 1, 1)
END_DATE   = datetime(2024, 1, 1)

OUTPUT_DIR = './data'
MAX_RETRIES = 5
RETRY_DELAY = 10
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


def download_bi5(url: str, retries: int = MAX_RETRIES) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) == 0:
                    return None
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print(f"  HTTP {e.code} | تلاش {attempt}/{retries} | {url}")
        except Exception as e:
            print(f"  خطا: {type(e).__name__}: {e} | تلاش {attempt}/{retries}")

        if attempt < retries:
            wait = RETRY_DELAY * attempt
            print(f"  صبر {wait} ثانیه...")
            time.sleep(wait)

    return None


def parse_bi5(data: bytes, date: datetime, point: int) -> list:
    """
    فرمت هر رکورد bi5: 24 بایت
    - 4 بایت: time (int32, big-endian) = میلی‌ثانیه از ابتدای روز
    - 4 بایت: open  (int32, big-endian) = قیمت × point
    - 4 بایت: high  (int32, big-endian)
    - 4 بایت: low   (int32, big-endian)
    - 4 بایت: close (int32, big-endian)
    - 4 بایت: volume (float32, big-endian)
    """
    try:
        raw = lzma.decompress(data)
    except Exception as e:
        print(f"  خطا در decompress: {e}")
        return []

    if len(raw) == 0:
        return []

    record_size = 24
    total_records = len(raw) // record_size

    if total_records == 0:
        return []

    # دیباگ اولین رکورد
    first = raw[:record_size]
    t0, o0, h0, l0, c0 = struct.unpack('>iiiii', first[:20])
    v0 = struct.unpack('>f', first[20:24])[0]
    ts0 = date + timedelta(milliseconds=t0)
    print(f"  DEBUG | {date.date()} | رکوردها: {total_records} | "
          f"اولین: {ts0.strftime('%H:%M:%S')} | "
          f"O={o0/point:.5f} C={c0/point:.5f} V={v0:.1f}")

    candles = []
    for i in range(0, total_records * record_size, record_size):
        chunk = raw[i:i + record_size]
        try:
            t, o, h, l, c = struct.unpack('>iiiii', chunk[:20])
            v = struct.unpack('>f', chunk[20:24])[0]
        except struct.error:
            continue

        # فیلتر رکوردهای نامعتبر
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            continue
        if h < l:
            continue

        ts = date + timedelta(milliseconds=t)

        candles.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'open':   round(o / point, 6),
            'high':   round(h / point, 6),
            'low':    round(l / point, 6),
            'close':  round(c / point, 6),
            'volume': round(v, 2),
        })

    return candles


def download_instrument(symbol: str, point: int, start: datetime, end: datetime):
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_M1_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        if size > 1024 * 100:  # بیشتر از 100KB
            print(f"✅ فایل موجود است: {out_path} ({size/1024/1024:.1f} MB)")
            return
        else:
            print(f"⚠️ فایل ناقص است، دانلود مجدد: {out_path}")
            os.remove(out_path)

    print(f"\n{'='*60}")
    print(f"شروع دانلود: {symbol} | {start.date()} → {end.date()}")
    print(f"{'='*60}")

    all_candles = []
    current = start
    total_days = (end - start).days
    downloaded = 0
    skipped = 0
    errors = 0

    while current < end:
        # دوکاسکوپی ماه را از 0 شروع می‌کند (ژانویه=00)
        url = BASE_URL.format(
            symbol=symbol,
            year=current.year,
            month=current.month - 1,
            day=current.day
        )

        data = download_bi5(url)

        if data and len(data) > 0:
            candles = parse_bi5(data, current, point)
            if candles:
                all_candles.extend(candles)
                downloaded += 1
            else:
                skipped += 1
        else:
            skipped += 1

        current += timedelta(days=1)

        # تاخیر برای جلوگیری از rate limit
        time.sleep(0.4)

        # گزارش پیشرفت هر 30 روز
        done = downloaded + skipped + errors
        if done % 30 == 0:
            pct = done / total_days * 100
            print(f"\n  پیشرفت: {pct:.0f}% | دانلود: {downloaded} | "
                  f"بدون داده: {skipped} | کندل جمع: {len(all_candles):,}")

    # ذخیره CSV
    print(f"\n  ذخیره {len(all_candles):,} کندل در {out_path}...")

    if all_candles:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candles)

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"✅ ذخیره شد: {out_path}")
        print(f"   کندل‌ها: {len(all_candles):,}")
        print(f"   حجم: {size_mb:.1f} MB")
        print(f"   روزهای دانلود: {downloaded} | بدون داده: {skipped}")
    else:
        print(f"❌ هیچ داده‌ای برای {symbol} دریافت نشد!")


def verify_output():
    """بررسی فایل‌های خروجی"""
    print(f"\n{'='*60}")
    print("بررسی فایل‌های خروجی:")
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]

    for fname in files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, 'r') as f:
            lines = sum(1 for _ in f) - 1  # منهای header

        size_mb = os.path.getsize(fpath) / 1024 / 1024

        # خواندن چند خط اول
        import pandas as pd
        df = pd.read_csv(fpath, nrows=3)

        print(f"\n{fname}:")
        print(f"  کندل‌ها: {lines:,}")
        print(f"  حجم: {size_mb:.1f} MB")
        print(f"  نمونه:\n{df.to_string()}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for symbol, config in INSTRUMENTS.items():
        download_instrument(
            symbol=symbol,
            point=config['point'],
            start=START_DATE,
            end=END_DATE,
        )
        print(f"\n⏳ ۱۰ ثانیه تاخیر...")
        time.sleep(10)

    verify_output()

    print(f"\n{'='*60}")
    print("✅ همه دانلودها کامل شد")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
