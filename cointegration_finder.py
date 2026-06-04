import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import itertools
import logging
from typing import List, Tuple, Dict
import os

# تنظیمات لاگینگ برای سرور
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PairsCointegrationFinder:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()

    def fetch_data(self) -> None:
        """دانلود دیتای قیمت بسته شدن با استفاده از سشن مرورگر جعلی"""
        logging.info(f"شروع دانلود دیتا برای {len(self.tickers)} نماد...")
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            # ارسال سشن به همراه درخواست دانلود
            df = yf.download(self.tickers, start=self.start_date, end=self.end_date, session=session)['Close']
            
            self.data = df.dropna()
            logging.info(f"دیتا با موفقیت دریافت شد. ابعاد دیتا: {self.data.shape}")
        except Exception as e:
            logging.error(f"خطا در دریافت دیتا: {e}")
            raise

    def find_cointegrated_pairs(self, significance_level: float = 0.05) -> pd.DataFrame:
        """انجام تست انگل-گرنجر روی تمام ترکیبات ممکن از نمادها"""
        n = self.data.shape[1]
        keys = self.data.columns
        pairs_data = []

        logging.info("شروع محاسبه همگرایی (Cointegration)...")
        
        # ساخت تمام ترکیبات دوتایی ممکن از لیست نمادها
        all_combinations = list(itertools.combinations(keys, 2))
        
        for pair in all_combinations:
            sym_1, sym_2 = pair
            series_1 = self.data[sym_1].values
            series_2 = self.data[sym_2].values

            # انجام تست آماری
            try:
                score, pvalue, _ = ts.coint(series_1, series_2)
                
                # اگر p-value کمتر از سطح معنی‌داری باشد، همگرایی تایید می‌شود
                if pvalue < significance_level:
                    pairs_data.append({
                        'Asset_1': sym_1,
                        'Asset_2': sym_2,
                        'P_Value': round(pvalue, 5),
                        'T_Score': round(score, 3)
                    })
            except Exception as e:
                logging.warning(f"خطا در پردازش جفت {sym_1} و {sym_2}: {e}")
                continue

        # تبدیل نتایج به دیتافریم و مرتب‌سازی بر اساس بهترین P-Value
        results_df = pd.DataFrame(pairs_data)
        if not results_df.empty:
            results_df = results_df.sort_values(by='P_Value', ascending=True).reset_index(drop=True)
            
        logging.info(f"تعداد {len(results_df)} جفت همگرا پیدا شد.")
        return results_df

if __name__ == "__main__":
    # نمادهای جفت‌ارزها در یاهو فایننس با پسوند X= مشخص می‌شوند
    forex_tickers = [
        'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 
        'USDCAD=X', 'USDCHF=X', 'USDJPY=X', 'EURGBP=X', 
        'EURAUD=X', 'GBPAUD=X', 'AUDNZD=X', 'AUDCAD=X'
    ]
    
    # تنظیم بازه زمانی (۳ سال گذشته برای دیتای روزانه ایده‌آل است)
    START = "2021-01-01"
    END = "2024-01-01"

    finder = PairsCointegrationFinder(tickers=forex_tickers, start_date=START, end_date=END)
    
    # اجرای فرآیند
    finder.fetch_data()
    best_pairs = finder.find_cointegrated_pairs()
    
    # ذخیره نتایج در فایل CSV برای استخراج در گیت‌هاب
    output_file = "cointegrated_pairs_report.csv"
    best_pairs.to_csv(output_file, index=False)
    
    print("\n--- بهترین جفت‌ارزهای پیدا شده ---")
    print(best_pairs.head(10)) # نمایش ۱۰ جفت برتر در کنسول
    logging.info(f"گزارش نهایی در فایل {output_file} ذخیره شد.")
