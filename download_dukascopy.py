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
RETRY_DELAY = 10  # ثانیه
# =================================================

BASE_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


def download_bi5(url: str, retries: int = MAX_RETRIES) -> bytes | None:
    """دانلود یک فایل bi5 با retry"""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) == 0:
                    return None  # روز تعطیل یا بدون داده
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # داده وجود ندارد
            print(f"  HTTP {e.code} - تلاش {attempt}/{retries}")
        except Exception as e:
            print(f"  خطا: {e} - تلاش {attempt}/{retries}")
        
        if attempt < retries:
            time.sleep(RETRY_DELAY * attempt)
    
    return None


def parse_bi5(data: bytes, date: datetime, point: int) -> list:
    """
    پارس فایل bi5 دوکاسکوپی
    فرمت هر رکورد: 5 عدد int32 = 20 بایت
    [time_ms, open, high, low, close, volume_float]
    اما فرمت واقعی: time(int), O(int), H(int), L(int), C(int), V(float) = 24 بایت
    """
    try:
        raw = lzma.decompress(data)
    except Exception as e:
        print(f"  خطا در decompress: {e}")
        return []
    
    candles = []
    record_size = 24  # 5 * int32 + 1 * float32
    
    for i in range(0, len(raw) - record_size + 1, record_size):
        chunk = raw[i:i + record_size]
        if len(chunk) < record_size:
            break
        
        try:
            t, o, h, l, c = struct.unpack('>iiiii', chunk[:20])
            v = struct.unpack('>f', chunk[20:24])[0]
        except struct.error:
            continue
        
        # زمان = میلی‌ثانیه از ابتدای روز
        ts = date + timedelta(milliseconds=t)
        
        open_p  = o / point
        high_p  = h / point
        low_p   = l / point
        close_p = c / point
        
        candles.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'open':  round(open_p, 6),
            'high':  round(high_p, 6),
            'low':   round(low_p, 6),
            'close': round(close_p, 6),
            'volume': round(v, 2),
        })
    
    return candles


def download_instrument(symbol: str, point: int, start: datetime, end: datetime):
    """دانلود کامل یک جفت ارز"""
    out_path = os.path.join(OUTPUT_DIR, f"{symbol}_M1_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")
    
    if os.path.exists(out_path):
        print(f"✅ فایل موجود است (از کش): {out_path}")
        return
    
    print(f"\n{'='*50}")
    print(f"شروع دانلود: {symbol} از {start.date()} تا {end.date()}")
    print(f"{'='*50}")
    
    all_candles = []
    current = start
    total_days = (end - start).days
    downloaded_days = 0
    skipped_days = 0
    
    while current < end:
        url = BASE_URL.format(
            symbol=symbol,
            year=current.year,
            month=current.month - 1,  # دوکاسکوپی ماه را از 0 شروع می‌کند
            day=current.day
        )
        
        data = download_bi5(url)
        
        if data:
            candles = parse_bi5(data, current, point)
            all_candles.extend(candles)
            downloaded_days += 1
            if downloaded_days % 30 == 0:
                pct = (downloaded_days + skipped_days) / total_days * 100
                print(f"  پیشرفت: {pct:.1f}% | روزهای دانلود شده: {downloaded_days} | بدون داده: {skipped_days}")
        else:
            skipped_days += 1
        
        current += timedelta(days=1)
        time.sleep(0.3)  # کمی تاخیر برای جلوگیری از ban
    
    # ذخیره CSV
    if all_candles:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            writer.writeheader()
            writer.writerows(all_candles)
        
        print(f"\n✅ ذخیره شد: {out_path}")
        print(f"   تعداد کندل: {len(all_candles):,}")
        print(f"   حجم فایل: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
    else:
        print(f"❌ هیچ داده‌ای دریافت نشد برای {symbol}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for symbol, config in INSTRUMENTS.items():
        download_instrument(
            symbol=symbol,
            point=config['point'],
            start=START_DATE,
            end=END_DATE
        )
        print(f"\n⏳ ۵ ثانیه تاخیر قبل از جفت ارز بعدی...")
        time.sleep(5)
    
    print("\n" + "="*50)
    print("✅ همه دانلودها کامل شد")
    print("="*50)
    import subprocess
    subprocess.run(['ls', '-lah', OUTPUT_DIR])


if __name__ == '__main__':
    main()
