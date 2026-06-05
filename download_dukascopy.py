import os
import struct
import lzma
import time
import csv
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

# ==================== تنظیمات ====================
INSTRUMENTS = {
    'EURUSD': {'point': 100000},
    'GBPUSD': {'point': 100000},
}

# تست 1 ساله ایمن و پایدار
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 1, 1)

OUTPUT_DIR = './data'
# =================================================

BASE_URL = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"
RECORD_SIZE = 24
RECORD_FMT  = '>IIIIIf'

# ساخت سشن قدرتمند برای جلوگیری از مسدودی
session = requests.Session()
retry = Retry(connect=5, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

def download_bi5(url: str) -> bytes | None:
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None

def parse_bi5(raw_compressed: bytes, date: datetime, point: int) -> list:
    try:
        raw = lzma.decompress(raw_compressed)
    except Exception:
        return []

    if len(raw) < RECORD_SIZE:
        return []

    candles = []
    for i in range(len(raw) // RECORD_SIZE):
        chunk = raw[i * RECORD_SIZE : (i + 1) * RECORD_SIZE]
        try:
            t_ms, o, h, l, c, v = struct.unpack(RECORD_FMT, chunk)
        except struct.error:
            continue

        if o == 0 or c == 0 or h < l or t_ms >= 86_400_000:
            continue

        t_ms_rounded = (t_ms // 60_000) * 60_000
        ts = date + timedelta(milliseconds=t_ms_rounded)

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

    seen = {row[0]: row for row in candles}
    return list(seen.values())

def download_instrument(symbol: str, point: int, start: datetime, end: datetime):
    out_path = os.path.join(OUTPUT_DIR, f"{symbol}_M1_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv")
    
    print(f"\n▶ شروع دانلود: {symbol}")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        current = start
        total_candles = 0

        while current < end:
            url = BASE_URL.format(
                symbol=symbol,
                year=current.year,
                month=current.month - 1, # Dukascopy months are 0-indexed
                day=current.day
            )

            raw_data = download_bi5(url)
            if raw_data:
                candles = parse_bi5(raw_data, current, point)
                if candles:
                    writer.writerows(candles)
                    total_candles += len(candles)
            
            current += timedelta(days=1)
            time.sleep(0.1) # مکث کوتاه برای احترام به سرور

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"✅ {symbol} کامل شد: {total_candles:,} کندل | حجم: {size_mb:.1f} MB")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sym, config in INSTRUMENTS.items():
        download_instrument(sym, config['point'], START_DATE, END_DATE)
