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
                return data if len(data) > 0 else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print(f"  HTTP {e.code} | تلاش {attempt}/{retries}")
        except Exception as e:
            print(f"  خطا: {type(e).__name__}: {e} | تلاش {attempt}/{retries}")

        if attempt < retries:
            wait = RETRY_DELAY * attempt
            print(f"  صبر {wait} ثانیه...")
            time.sleep(wait)

    return None


def parse_bi5(data: bytes, date: datetime, point: int, debug: bool = False) -> list:
    """
    فرمت دوکاسکوپی bi5:
    هر رکورد 24 بایت:
      [0:4]   time_ms  : uint32 big-endian = میلی‌ثانیه از ابتدای روز (0 تا 86,399,999)
      [4:8]   open     : uint32 big-endian = قیمت × point  
      [8:12]  high     : uint32 big-endian
      [12:16] low      : uint32 big-endian
      [16:20] close    : uint32 big-endian
      [20:24] volume   : float32 big-endian
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

    if debug:
        # نمایش raw bytes اولین رکورد
        first_chunk = raw[:record_size]
        print(f"  RAW bytes[0:24]: {first_chunk.hex()}")
        
        # parse با unsigned
        t_u = struct.unpack('>I', first_chunk[0:4])[0]
        o_u = struct.unpack('>I', first_chunk[4:8])[0]
        h_u = struct.unpack('>I', first_chunk[8:12])[0]
        l_u = struct.unpack('>I', first_chunk[12:16])[0]
        c_u = struct.unpack('>I', first_chunk[16:20])[0]
        v_f = struct.unpack('>f', first_chunk[20:24])[0]
        
        ts_check = date + timedelta(milliseconds=t_u)
        print(f"  DEBUG unsigned | t={t_u}ms={t_u/1000:.1f}s | "
              f"time={ts_check.strftime('%H:%M:%S')} | "
              f"O={o_u/point:.5f} H={h_u/point:.5f} "
              f"L={l_u/point:.5f} C={c_u/point:.5f} V={v_f:.1f}")

    candles = []
    
    # ✅ فرمت درست: time و OHLC همه unsigned int32
    fmt = '>IIIIIf'  # uint32 × 5 + float32
    
    for i in range(total_records):
        offset = i * record_size
        chunk = raw[offset: offset + record_size]
        
        if len(chunk) < record_size:
            break

        try:
            t, o, h, l, c, v = struct.unpack(fmt, chunk)
        except struct.error:
            continue

        # فیلتر رکوردهای نامعتبر
        if o == 0 or h == 0 or l == 0 or c == 0:
            continue
        if h < l:
            continue
        # time باید در بازه یک روز باشد (0 تا 86,399,999 ms)
        if t >= 86_400_000:
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
        if size > 1024 * 500:  # بیشتر از 500KB
            print(f"✅ فایل موجود: {out_path} ({size/1024/1024:.1f} MB)")
            return
        else:
            print(f"⚠️ فایل ناقص ({size} bytes)، دانلود مجدد...")
            os.remove(out_path)

    print(f"\n{'='*60}")
    print(f"دانلود: {symbol} | {start.date()} → {end.date()}")
    print(f"{'='*60}")

    all_candles = []
    current = start
    total_days = (end - start).days
    downloaded = 0
    skipped = 0
    debug_done = False  # فقط برای اولین روز موفق debug کن

    while current < end:
        # ⚠️ دوکاسکوپی ماه را از 0 شروع می‌کند
        url = BASE_URL.format(
            symbol=symbol,
            year=current.year,
            month=current.month - 1,
            day=current.day
        )

        data = download_bi5(url)

        if data and len(data) > 0:
            # debug فقط برای اولین روز
            do_debug = not debug_done
            candles = parse_bi5(data, current, point, debug=do_debug)
            
            if candles:
                if do_debug:
                    print(f"  ✅ اولین کندل: {candles[0]}")
                    print(f"  ✅ آخرین کندل: {candles[-1]}")
                    print(f"  ✅ تعداد کندل این روز: {len(candles)}")
                    debug_done = True
                    
                all_candles.extend(candles)
                downloaded += 1
            else:
                skipped += 1
        else:
            skipped += 1

        current += timedelta(days=1)
        time.sleep(0.4)

        done = downloaded + skipped
        if done % 50 == 0:
            pct = done / total_days * 100
            print(f"  پیشرفت: {pct:.0f}% | دانلود: {downloaded} روز | "
                  f"کندل: {len(all_candles):,}")

    # ذخیره
    print(f"\n  ذخیره {len(all_candles):,} کندل...")

    if all_candles:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fieldnames = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_candles)

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"✅ {out_path}")
        print(f"   کندل: {len(all_candles):,} | حجم: {size_mb:.1f} MB")
        print(f"   روز دانلود: {downloaded} | بدون داده: {skipped}")
        
        # نمایش نمونه
        import pandas as pd
        df_check = pd.read_csv(out_path)
        print(f"\n   نمونه اول:")
        print(df_check.head(5).to_string())
        print(f"\n   نمونه آخر:")
        print(df_check.tail(3).to_string())
        
        # چک توزیع timestamp
        df_check['timestamp'] = pd.to_datetime(df_check['timestamp'])
        df_check['date'] = df_check['timestamp'].dt.date
        daily_counts = df_check.groupby('date').size()
        print(f"\n   میانگین کندل در روز: {daily_counts.mean():.0f}")
        print(f"   حداکثر کندل در روز: {daily_counts.max()}")
        print(f"   حداقل کندل در روز: {daily_counts.min()}")
        
    else:
        print(f"❌ هیچ داده‌ای برای {symbol} دریافت نشد!")


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

    print(f"\n{'='*60}")
    print("✅ دانلود کامل شد")
    print(f"{'='*60}")
    
    import subprocess
    subprocess.run(['ls', '-lah', OUTPUT_DIR])


if __name__ == '__main__':
    main()
