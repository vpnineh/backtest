# debug_bi5.py
import urllib.request
import lzma
import struct

url = "https://datafeed.dukascopy.com/datafeed/EURUSD/2022/00/03/BID_candles_min_1.bi5"
headers = {'User-Agent': 'Mozilla/5.0', 'Accept-Encoding': 'identity'}

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, timeout=30) as r:
    raw_compressed = r.read()

print(f"حجم compressed: {len(raw_compressed)} bytes")

raw = lzma.decompress(raw_compressed)
print(f"حجم decompressed: {len(raw)} bytes")
print(f"تعداد رکورد: {len(raw) // 24}")

# نمایش raw hex اولین 3 رکورد
for i in range(3):
    chunk = raw[i*24:(i+1)*24]
    print(f"\nرکورد {i}: {chunk.hex()}")
    
    # تست فرمت‌های مختلف
    fmt1 = struct.unpack('>IIIIIf', chunk)
    fmt2 = struct.unpack('>iiiiii', chunk)  
    fmt3 = struct.unpack('>Iiiiif', chunk)
    
    print(f"  >IIIIIf : t={fmt1[0]}ms={fmt1[0]/1000:.1f}s  O={fmt1[1]/100000:.5f}  C={fmt1[4]/100000:.5f}  V={fmt1[5]:.2f}")
    print(f"  >iiiiii : t={fmt2[0]}ms  O={fmt2[1]/100000:.5f}  C={fmt2[4]/100000:.5f}")
    print(f"  >Iiiiif : t={fmt3[0]}ms={fmt3[0]/1000:.1f}s  O={fmt3[1]/100000:.5f}  C={fmt3[4]/100000:.5f}")
