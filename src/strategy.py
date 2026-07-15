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
        
        # 1. Trend Filter (EMA)
        df = df.with_columns(
            pl.col("close").ewm_mean(span=p.ema_trend_period, adjust=False).alias("ema_trend")
        )
        
        # 2. Donchian Channels (Breakout levels)
        df = df.with_columns([
            pl.col("high").rolling_max(window_size=p.donchian_period).shift(1).alias("donchian_upper"),
            pl.col("low").rolling_min(window_size=p.donchian_period).shift(1).alias("donchian_lower")
        ])
        
        # 3. ATR (Average True Range) for dynamic stops
        df = df.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr_hl"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr_hc"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr_lc")
        ]).with_columns([
            pl.max_horizontal(["tr_hl", "tr_hc", "tr_lc"]).alias("tr")
        ]).with_columns([
            pl.col("tr").rolling_mean(window_size=p.atr_period).alias("atr")
        ])
        
        # 4. Time Filter
        df = df.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        active_mask = (pl.col("hour") >= p.london_start_hour) & (pl.col("hour") <= p.london_end_hour)
        
        # 5. Generate Entry Signals
        # Buy: Close breaks above Donchian Upper AND is above EMA trend
        buy_cond = (pl.col("close") > pl.col("donchian_upper")) & (pl.col("close") > pl.col("ema_trend")) & active_mask
        
        # Sell: Close breaks below Donchian Lower AND is below EMA trend
        sell_cond = (pl.col("close") < pl.col("donchian_lower")) & (pl.col("close") < pl.col("ema_trend")) & active_mask
        
        df = df.with_columns(
            pl.when(buy_cond).then(1).when(sell_cond).then(-1).otherwise(0).alias("signal")
        )
        
        # Drop warmup nulls
        df = df.drop_nulls(subset=["ema_trend", "donchian_upper", "atr"])
        
        # Filter to active hours to speed up engine
        df_filtered = df.filter(active_mask | (pl.col("signal") != 0))
        
        logger.success(f"Signal generation done. Active rows for engine: {df_filtered.height}")
        return df_filtered
