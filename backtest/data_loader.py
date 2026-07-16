"""
data_loader.py
================
Loads raw HistData.com M1 ASCII files (either plain .csv or the
HISTDATA_COM_ASCII_*.zip archives) from the repo's `data/` folder and
resamples them to any target timeframe.

Supported file naming conventions actually present in the repo:

    DAT_ASCII_<SYMBOL>_M1_<YEAR>.csv                       (plain csv)
    HISTDATA_COM_ASCII_<SYMBOL>_M1<YEAR>.zip                (zip, contains a csv inside)

HistData ASCII M1 row format (no header):
    YYYYMMDD HHMMSS;OPEN;HIGH;LOW;CLOSE;VOLUME

All timestamps in HistData files are in EST (UTC-5, no DST). We keep
everything in that timezone-naive "broker time" -- consistent for the
whole backtest, which is all that matters for session filters etc.

IMPORTANT (anti-lookahead):
  Resampling here only ever aggregates *past* M1 bars into a coarser
  bar. The last, possibly-still-forming, bar of the requested range is
  dropped so that no partially-built candle is ever fed to the engine
  as if it were closed.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

_TF_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def _read_histdata_buffer(buf) -> pd.DataFrame:
    df = pd.read_csv(
        buf,
        sep=";",
        header=None,
        names=["dt", "open", "high", "low", "close", "volume"],
        dtype={"dt": str, "open": "float64", "high": "float64",
               "low": "float64", "close": "float64", "volume": "float64"},
        engine="c",
    )
    df["datetime"] = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def load_year(data_dir: Path, symbol: str, year: int) -> Optional[pd.DataFrame]:
    """Load a single year of M1 data for `symbol`. Returns None if missing."""
    csv_path = data_dir / f"DAT_ASCII_{symbol}_M1_{year}.csv"
    zip_path = data_dir / f"HISTDATA_COM_ASCII_{symbol}_M1{year}.zip"

    if csv_path.exists():
        return _read_histdata_buffer(csv_path)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            inner_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not inner_names:
                raise FileNotFoundError(f"No CSV found inside {zip_path.name}")
            with zf.open(inner_names[0]) as f:
                return _read_histdata_buffer(f)

    return None


def load_range(data_dir: Path, symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Load & concatenate M1 data for `symbol` across [start_year, end_year]."""
    frames = []
    missing_years = []
    for y in range(start_year, end_year + 1):
        df = load_year(data_dir, symbol, y)
        if df is None:
            missing_years.append(y)
            continue
        frames.append(df)

    if missing_years:
        print(f"[data_loader] WARNING - missing data files for {symbol}, years: {missing_years}")

    if not frames:
        raise FileNotFoundError(
            f"No M1 data found for symbol={symbol} in range {start_year}-{end_year} "
            f"under {data_dir}. Expected files like "
            f"DAT_ASCII_{symbol}_M1_{start_year}.csv or "
            f"HISTDATA_COM_ASCII_{symbol}_M1{start_year}.zip"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    return df


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample M1 OHLCV to the requested timeframe. Drops the final
    (possibly incomplete) bar so the caller never sees an unfinished candle."""
    timeframe = timeframe.upper()
    if timeframe == "M1":
        return df.reset_index(drop=True)

    rule = _TF_MAP.get(timeframe)
    if rule is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {list(_TF_MAP)}")

    d = df.set_index("datetime")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = d.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])

    if len(out) > 1:
        last_bar_start = out.index[-1]
        period_end = last_bar_start + pd.tseries.frequencies.to_offset(rule)
        last_tick = d.index.max()
        # if the data doesn't extend to (at least) the end of the last bucket,
        # that bucket is not a closed candle -> drop it.
        if last_tick < period_end - pd.Timedelta(minutes=1):
            out = out.iloc[:-1]

    return out.reset_index()


def load_and_resample(data_dir: Path, symbol: str, timeframe: str,
                       start_year: int, end_year: int) -> pd.DataFrame:
    raw = load_range(data_dir, symbol, start_year, end_year)
    return resample(raw, timeframe)
