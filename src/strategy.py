# src/strategy.py
import polars as pl
from loguru import logger
from src.config import StrategyParams

class MeanReversionStrategy:
    def __init__(self, params: StrategyParams):
        self.params = params

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Calculating indicators...")
        p = self.params
        
        df = df.with_columns([
            pl.col("close").ewm_mean(span=p.ema_trend_period, adjust=False).alias("ema_trend"),
            pl.col("close").rolling_mean(window_size=p.bb_period).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=p.bb_period).alias("bb_std"),
            pl.col("close").diff().alias("delta"),
            pl.col("datetime").dt.hour().alias("hour")
        ])
        
        df = df.with_columns([
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(pl.col("delta").abs()).otherwise(0.0).alias("loss"),
        ]).with_columns([
            pl.col("gain").ewm_mean(span=p.rsi_period, adjust=False).alias("avg_gain"),
            pl.col("loss").ewm_mean(span=p.rsi_period, adjust=False).alias("avg_loss"),
        ]).with_columns([
            (100 - (100 / (1 + (pl.col("avg_gain") / pl.col("avg_loss"))))).alias("rsi"),
            (pl.col("bb_middle") + (p.bb_std_dev * pl.col("bb_std"))).alias("bb_upper"),
            (pl.col("bb_middle") - (p.bb_std_dev * pl.col("bb_std"))).alias("bb_lower"),
        ])
        
        active_mask = (pl.col("hour") >= p.london_start_hour) & (pl.col("hour") <= p.london_end_hour)
        
        buy_cond = (pl.col("close") > pl.col("ema_trend")) & (pl.col("close") <= pl.col("bb_lower")) & (pl.col("rsi") <= p.rsi_oversold) & active_mask
        sell_cond = (pl.col("close") < pl.col("ema_trend")) & (pl.col("close") >= pl.col("bb_upper")) & (pl.col("rsi") >= p.rsi_overbought) & active_mask
        
        df = df.with_columns(
            pl.when(buy_cond).then(1).when(sell_cond).then(-1).otherwise(0).alias("signal")
        )
        
        df = df.drop_nulls(subset=["ema_trend", "rsi"])
        df_filtered = df.filter(active_mask | (pl.col("signal") != 0))
        
        logger.success(f"Signal generation done. Active rows for engine: {df_filtered.height}")
        return df_filtered
