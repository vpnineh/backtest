# src/etl.py
import zipfile
import re
import polars as pl
from pathlib import Path
from loguru import logger
from src.config import BacktestSettings

# Mapping timeframe config to Polars duration string
TF_MAP = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",
    "D1": "1d"
}

def read_m1_data(data_dir: Path, start_year: int, end_year: int) -> pl.DataFrame:
    """Reads and concatenates all M1 ZIP files within the year range."""
    schema = {
        "datetime": pl.Utf8, "open": pl.Float32, "high": pl.Float32, 
        "low": pl.Float32, "close": pl.Float32, "volume": pl.Int32
    }
    
    # We ALWAYS look for M1 files since that's our base data
    zip_files = sorted(data_dir.glob("*M1*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No M1 ZIP files found in {data_dir}. Base data is required.")
        
    chunks = []
    for zip_file in zip_files:
        match = re.search(r'(\d{4})\.zip$', zip_file.name)
        if not match: continue
        year = int(match.group(1))
        if not (start_year <= year <= end_year): continue
        
        logger.info(f"Reading base M1 data: {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as z:
            csv_name = z.namelist()[0] 
            with z.open(csv_name) as f:
                df_chunk = pl.read_csv(f, has_header=False, ignore_errors=True)
                if df_chunk.shape[1] == 5:
                    df_chunk.columns = ["datetime", "open", "high", "low", "close"]
                    df_chunk = df_chunk.with_columns(pl.lit(0).alias("volume").cast(pl.Int32))
                else:
                    df_chunk = df_chunk[:, :6]
                    df_chunk.columns = list(schema.keys())
                chunks.append(df_chunk.select(list(schema.keys())).cast(schema))

    logger.info("Concatenating M1 chunks and parsing datetimes...")
    df = pl.concat(chunks).unique(subset=["datetime"]).sort("datetime")
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, "%Y%m%d %H%M%S").alias("datetime")
    )
    return df

def extract_histdata_to_parquet(settings: BacktestSettings) -> Path:
    """
    Reads M1 data. If target timeframe is M1, saves directly.
    If target timeframe is higher (M5, H1, etc.), resamples M1 data on the fly.
    """
    data_dir = settings.data_dir
    output_file = data_dir / settings.parquet_filename
    
    if output_file.exists():
        logger.info(f"Processed data exists at {output_file}. Skipping ETL.")
        return output_file

    logger.info(f"Starting ETL for {settings.symbol} | Target Timeframe: {settings.timeframe} | Years: {settings.start_year}-{settings.end_year}")
    
    # 1. Always read M1 first (Base Data)
    df_m1 = read_m1_data(data_dir, settings.start_year, settings.end_year)
    
    # 2. If target timeframe is M1, just save and return
    if settings.timeframe == "M1":
        df_m1.write_parquet(output_file, compression="zstd")
        logger.success(f"ETL Complete! Saved M1 data to {output_file}")
        return output_file
        
    # 3. If target timeframe is higher (M5, H1, etc.), resample M1 data
    logger.info(f"Resampling M1 data to {settings.timeframe}...")
    tf_duration = TF_MAP.get(settings.timeframe)
    if not tf_duration:
        raise ValueError(f"Unsupported timeframe: {settings.timeframe}. Supported: {list(TF_MAP.keys())}")
        
    # Polars group_by_dynamic is blazing fast for OHLCV resampling
    df_resampled = df_m1.group_by_dynamic(
        index_column="datetime",
        every=tf_duration
    ).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume")
    ])
    
    # Drop rows where open is null (happens during weekends/holidays with no data)
    df_resampled = df_resampled.drop_nulls(subset=["open"])
    
    df_resampled.write_parquet(output_file, compression="zstd")
    logger.success(f"ETL Complete! Resampled and saved to {output_file}")
    return output_file
