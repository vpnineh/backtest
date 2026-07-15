# src/strategy.py
import polars as pl
from loguru import logger
from src.config import StrategyParams

class MeanReversionStrategy:
    def __init__(self, params: StrategyParams):
        self.params = params

    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Calculating Mean Reversion indicators (Bollinger Bands, RSI)...")
        p = self.params

        # 1. Bollinger Bands
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=p.bb_period).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=p.bb_period).alias("bb_std"),
        ]).with_columns([
            (pl.col("bb_middle") + (p.bb_std_dev * pl.col("bb_std"))).alias("bb_upper"),
            (pl.col("bb_middle") - (p.bb_std_dev * pl.col("bb_std"))).alias("bb_lower"),
        ])

        # 2. RSI
        df = df.with_columns([
            pl.col("close").diff().alias("delta"),
        ]).with_columns([
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(pl.col("delta").abs()).otherwise(0.0).alias("loss"),
        ]).with_columns([
            pl.col("gain").ewm_mean(span=p.rsi_period, adjust=False).alias("avg_gain"),
            pl.col("loss").ewm_mean(span=p.rsi_period, adjust=False).alias("avg_loss"),
        ]).with_columns([
            (100 - (100 / (1 + (pl.col("avg_gain") / pl.col("avg_loss"))))).alias("rsi"),
        ])

        # 3. Time Filter
        df = df.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        active_mask = (pl.col("hour") >= p.london_start_hour) & (pl.col("hour") <= p.london_end_hour)

        # 4. Raw entry conditions
        # Buy: Price touches lower band AND RSI is oversold
        buy_cond = (pl.col("close") <= pl.col("bb_lower")) & (pl.col("rsi") <= p.rsi_oversold) & active_mask
        
        # Sell: Price touches upper band AND RSI is overbought
        sell_cond = (pl.col("close") >= pl.col("bb_upper")) & (pl.col("rsi") >= p.rsi_overbought) & active_mask

        df = df.with_columns(
            pl.when(buy_cond).then(1).when(sell_cond).then(-1).otherwise(0).alias("raw_signal")
        )

        # 🔥 CRITICAL: Shift signal by 1 candle to eliminate look-ahead bias
        # Signal generated at close of candle N will be executed at open of candle N+1
        df = df.with_columns(
            pl.col("raw_signal").shift(1).fill_null(0).alias("signal")
        )

        # Drop warmup nulls
        df = df.drop_nulls(subset=["bb_middle", "bb_upper", "bb_lower", "rsi"])

        logger.success(f"Signal generation done. Total rows for engine: {df.height}")
        return df
