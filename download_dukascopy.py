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

START_DATE = datetime(2025, 1, 1)
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
RECORD_FMT  = '>IIIIIf'


def download_bi5(url: str) -> bytes | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                return data if len(data) > 0 else None

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print(f"    HTTP {e.code} | تلاش {attempt}/{MAX_RETRIES}")

        except Exception as e:
            print(f"    خطا: {type(e).__name__} | تلاش {attempt}/{MAX_RETRIES}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    return None


def parse_bi5(raw_compressed: bytes, date: datetime, point: int) -> list:
    """
    فرمت bi5 دوکاسکوپی:
      bytes 0-3  : uint32 = میلی‌ثانیه از ابتدای روز
      bytes 4-7  : uint32 = open  × point
      bytes 8-11 : uint32 = high  × point  
      bytes 12-15: uint32 = low   × point
      bytes 16-19: uint32 = close × point
      bytes 20-23: float32 = volume

    نکته مهم: چون تایم‌فریم M1 است، t_ms باید مضرب 60000 باشد
    مثال: کندل ساعت 10:30 → t_ms = 37800000
    """
    try:
        raw = lzma.decompress(raw_compressed)
    except lzma.LZMAError:
        return []

    if len(raw) < RECORD_SIZE:
        return []

    total_records = len(raw) // RECORD_SIZE
    candles = []

    for i in range(total_records):
        offset = i * RECORD_SIZE
        chunk  = raw[offset: offset + RECORD_SIZE]

        try:
            t_ms, o, h, l, c, v = struct.unpack(RECORD_FMT, chunk)
        except struct.error:
            continue

        # ── فیلترهای اعتبارسنجی ──
        if o == 0 or c == 0:
            continue
        if h < l:
            continue
        if t_ms >= 86_400_000:
            continue

        # گرد کردن به دقیقه (M1 = هر 60 ثانیه یک کندل)
        # t_ms ممکنه مثلاً 3600123 باشه که باید 3600000 بشه
        t_ms_rounded = (t_ms // 60_000) * 60_000

        ts = date + timedelta(milliseconds=t_ms_rounded)

        # فیلتر کندل‌های تعطیل (volume=0 و OHLC یکسان)
        if v == 0.0 and o == h == l == c:
            continue

        candles.append([
            ts.strftime('%Y-%m-%d %H:%M:%S'),
            round(o / point, 6),
            round(h / point, 6),
            round(l / point, 6),
            round(c / point, 6),
            round(v, 2),
        ])

    # حذف duplicate های همان دقیقه (نگه داشتن آخری)
    seen = {}
    for row in candles:
        seen[row[0]] = row
    
    return list(seen.values())


def download_instrument(symbol: str, point: int, start: datetime, end: datetime):
    out_path = os.path.join(
        OUTPUT_DIR,
        f"{symbol}_M1_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        if size > 10 * 1024 * 1024:
            print(f"✅ فایل موجود: {out_path} ({size/1024/1024:.0f} MB)")
            return
        print(f"⚠️ فایل ناقص ({size/1024:.0f} KB)، دانلود مجدد...")
        os.remove(out_path)

    print(f"\n{'='*60}")
    print(f"▶ دانلود: {symbol} | {start.date()} → {end.date()}")
    total_days = (end - start).days
    print(f"  کل روز: {total_days} | تخمین: ~{total_days*0.4/60:.0f} دقیقه")
    print(f"{'='*60}")

    fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        current        = start
        downloaded     = 0
        skipped        = 0
        total_candles  = 0

        while current < end:
            url = BASE_URL.format(
                symbol=symbol,
                year=current.year,
                month=current.month - 1,  # دوکاسکوپی: ژانویه=00
                day=current.day
            )

            raw_data = download_bi5(url)

            if raw_data:
                candles = parse_bi5(raw_data, current, point)
                if candles:
                    writer.writerows(candles)
                    total_candles += len(candles)
                    downloaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1

            current += timedelta(days=1)
            time.sleep(0.4)

            done = downloaded + skipped
            if done % 60 == 0:
                pct = done / total_days * 100
                print(
                    f"  [{pct:5.1f}%] "
                    f"روز: {done}/{total_days} | "
                    f"دانلود: {downloaded} | "
                    f"خالی: {skipped} | "
                    f"کندل: {total_candles:,}"
                )

    # گزارش نهایی
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n✅ {symbol} کامل:")
    print(f"   حجم: {size_mb:.1f} MB | کندل: {total_candles:,}")
    print(f"   روز با داده: {downloaded} | خالی: {skipped}")

    # نمایش نمونه برای تایید
    df = pd.read_csv(out_path, nrows=5)
    print(f"\n   نمونه اول:\n{df.to_string()}")
    df_all = pd.read_csv(out_path)
    print(f"\n   نمونه آخر:\n{df_all.tail(3).to_string()}")

    # آمار timestamp
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
    daily = df_all.groupby(df_all['timestamp'].dt.date).size()
    print(f"\n   کندل/روز → میانگین: {daily.mean():.0f} | max: {daily.max()} | min: {daily.min()}")
    print(f"   ساعت‌های یکتا: {df_all['timestamp'].dt.hour.nunique()}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"دانلود دوکاسکوپی: {START_DATE.date()} → {END_DATE.date()}")
    print(f"جفت ارزها: {list(INSTRUMENTS.keys())}")
    print(f"{'='*60}\n")

    for symbol, config in INSTRUMENTS.items():
        download_instrument(
            symbol=symbol,
            point=config['point'],
            start=START_DATE,
            end=END_DATE,
        )
        if symbol != list(INSTRUMENTS.keys())[-1]:
            print(f"\n⏳ ۱۵ ثانیه تاخیر...")
            time.sleep(15)

    print(f"\n{'='*60}")
    print("✅ همه دانلودها کامل شد")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.csv'):
            path = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(path) / 1024 / 1024
            lines = sum(1 for _ in open(path)) - 1
            print(f"  {f} | {lines:,} کندل | {size:.0f} MB")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
