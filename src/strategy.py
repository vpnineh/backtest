# src/strategy.py
import polars as pl
from loguru import logger
from src.config import StrategyParams

class TrendFollowingStrategy:
    def __init__(self, params: StrategyParams):
        self.params = params

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Calculating Trend Following indicators (EMA, Donchian, ATR)...")
        p = self.params

        # 1. Trend Filter (EMA) - بر اساس close همان کندل.
        #    این خودش مشکلی ندارد چون فقط برای ساخت raw_signal استفاده
        #    می‌شود و raw_signal در ادامه یک کندل به جلو shift می‌شود.
        df = df.with_columns(
            pl.col("close").ewm_mean(span=p.ema_trend_period, adjust=False).alias("ema_trend")
        )

        # 2. Donchian Channels (Breakout levels)
        # rolling_max/min شامل کندل جاری هم می‌شود، اما shift(1) این را
        # جبران می‌کند => donchian_upper[N] = max(high[N-donchian_period..N-1])
        df = df.with_columns([
            pl.col("high").rolling_max(window_size=p.donchian_period).shift(1).alias("donchian_upper"),
            pl.col("low").rolling_min(window_size=p.donchian_period).shift(1).alias("donchian_lower")
        ])

        # 3. ATR (Average True Range) خام - بر اساس high/low کندل جاری
        df = df.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr_hl"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr_hc"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr_lc")
        ]).with_columns([
            pl.max_horizontal(["tr_hl", "tr_hc", "tr_lc"]).alias("tr")
        ]).with_columns([
            pl.col("tr").rolling_mean(window_size=p.atr_period).alias("atr_raw")
        ])

        # 🔥 FIX (Look-ahead Bias #1 - ATR):
        # atr_raw در ردیف N از high/low خود ردیف N ساخته می‌شود که تا
        # close نهایی آن ردیف مشخص نیست. با shift(1)، مقدار "atr" در
        # ردیف N برابر می‌شود با آخرین ATR کاملاً شناخته‌شده در لحظه‌ی
        # OPEN ردیف N (یعنی محاسبه‌شده از داده‌های بسته‌شده تا ردیف N-1).
        # همین مقدار هم برای فاصله‌ی SL اولیه و هم برای trailing در طول
        # engine استفاده می‌شود.
        df = df.with_columns(
            pl.col("atr_raw").shift(1).alias("atr")
        )

        # 4. Time Filter
        # این mask فقط تعیین می‌کند "چه زمانی مجاز است سیگنال جدید صادر
        # شود"، دیگر برای فیلتر کردن کل دیتافریم استفاده نمی‌شود؛ چون
        # engine باید تمام کندل‌ها (از جمله خارج از ساعت لندن) را برای
        # مدیریت درست trailing stop معاملات باز ببیند.
        df = df.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        active_mask = (pl.col("hour") >= p.london_start_hour) & (pl.col("hour") <= p.london_end_hour)

        # 5. Raw entry conditions - بر اساس اطلاعاتی که فقط در لحظه‌ی
        #    CLOSE شدن کندل جاری قابل شناخت هستند.
        buy_cond = (pl.col("close") > pl.col("donchian_upper")) & (pl.col("close") > pl.col("ema_trend")) & active_mask
        sell_cond = (pl.col("close") < pl.col("donchian_lower")) & (pl.col("close") < pl.col("ema_trend")) & active_mask

        df = df.with_columns(
            pl.when(buy_cond).then(1).when(sell_cond).then(-1).otherwise(0).alias("raw_signal")
        )

        # 🔥 FIX (Look-ahead Bias #2 - این باگ اصلی و بحرانی بود):
        # raw_signal در ردیف N فقط بعد از بسته شدن کامل ردیف N قابل
        # مشاهده است. engine قبلاً معامله را با OPEN همان ردیفی باز
        # می‌کرد که raw_signal آن != 0 بود که معادل معامله کردن با قیمتی
        # است که زمانی قبل از تولید سیگنال رخ داده. با shift(1) سیگنال
        # یک کندل به جلو منتقل می‌شود تا ورود واقعی روی OPEN کندل N+1
        # انجام شود - اولین لحظه‌ای که این اطلاعات واقعاً قابل اجراست.
        df = df.with_columns(
            pl.col("raw_signal").shift(1).fill_null(0).alias("signal")
        )

        # Drop warmup nulls (شامل نال‌های ناشی از shift های جدید هم می‌شود)
        df = df.drop_nulls(subset=["ema_trend", "donchian_upper", "donchian_lower", "atr"])

        logger.success(
            f"Signal generation done. Total rows for engine (no time-based row filtering): {df.height}"
        )
        return df
