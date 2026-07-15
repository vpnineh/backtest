"""
Data loader: reads M1 CSV files (both plain CSV and ZIP),
builds resampled OHLCV for M5, M15, H4.

FORMAT expected (HistData):
  20100103 170001,1.60743,1.60743,1.60743,1.60743,0
  or with semicolon separator in some files.

No look-ahead: resampling always uses closed bars.
"""

import os
import io
import zipfile
import logging
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTDATA_COLS = ["datetime", "open", "high", "low", "close", "volume"]

RESAMPLE_MAP = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}


# ---------------------------------------------------------------------------
# Low-level reader
# ---------------------------------------------------------------------------

def _read_single_csv(filepath: str) -> pd.DataFrame:
    """
    Read one HistData M1 CSV file (plain or inside ZIP).
    Returns DataFrame with DatetimeIndex and OHLCV columns.
    """
    content = _extract_content(filepath)
    if content is None:
        raise FileNotFoundError(f"Cannot read: {filepath}")

    # Detect separator
    first_line = content.split("\n")[0]
    sep = ";" if ";" in first_line else ","

    df = pd.read_csv(
        io.StringIO(content),
        sep=sep,
        header=None,
        names=HISTDATA_COLS,
        dtype={
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "volume": np.float64,
        },
    )

    # Parse datetime - handle both "20100103 170001" and "20100103 17:00:01"
    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str).str.strip(),
        format="%Y%m%d %H%M%S",
        errors="coerce",
    )

    # Fallback format
    mask = df["datetime"].isna()
    if mask.any():
        df.loc[mask, "datetime"] = pd.to_datetime(
            df.loc[mask, "datetime"].astype(str).str.strip(),
            format="%Y%m%d %H:%M:%S",
            errors="coerce",
        )

    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()

    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep="last")]

    return df[["open", "high", "low", "close", "volume"]]


def _extract_content(filepath: str) -> Optional[str]:
    """Extract text content from plain CSV or ZIP file."""
    path = Path(filepath)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(filepath, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                logger.error(f"No CSV inside ZIP: {filepath}")
                return None
            # Usually one CSV per ZIP
            with zf.open(csv_names[0]) as f:
                return f.read().decode("utf-8", errors="replace")
    else:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Multi-year loader
# ---------------------------------------------------------------------------

def load_pair_data(
    data_dir: str,
    pair: str,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Load all available years for a pair and concatenate.

    File naming conventions supported:
      DAT_ASCII_EURGBP_M1_2010.csv
      HISTDATA_COM_ASCII_EURGBP_M12010.zip

    Returns M1 DataFrame with DatetimeIndex (UTC-naive, HistData is UTC).
    """
    data_path = Path(data_dir)
    frames = []

    if years is None:
        years = list(range(2010, 2026))

    for year in years:
        filepath = _find_file(data_path, pair, year)
        if filepath is None:
            logger.debug(f"No data file for {pair} {year}, skipping.")
            continue

        try:
            df = _read_single_csv(str(filepath))
            if len(df) == 0:
                logger.warning(f"Empty file: {filepath}")
                continue
            frames.append(df)
            logger.info(f"Loaded {pair} {year}: {len(df):,} bars from {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            continue

    if not frames:
        raise ValueError(f"No data loaded for {pair}. Check data directory: {data_dir}")

    combined = pd.concat(frames, axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Sanity checks
    combined = combined[(combined["high"] >= combined["low"])]
    combined = combined[(combined["close"] > 0)]

    logger.info(
        f"{pair} total: {len(combined):,} M1 bars | "
        f"{combined.index[0]} → {combined.index[-1]}"
    )
    return combined


def _find_file(data_path: Path, pair: str, year: int) -> Optional[Path]:
    """Try all known naming conventions."""
    candidates = [
        data_path / f"DAT_ASCII_{pair}_M1_{year}.csv",
        data_path / f"HISTDATA_COM_ASCII_{pair}_M1{year}.zip",
        data_path / f"HISTDATA_COM_ASCII_{pair}_M12{str(year)[1:]}.zip",
    ]
    # Also try lowercase
    for c in list(candidates):
        candidates.append(Path(str(c).replace(pair, pair.lower())))

    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Resampler - NO LOOK-AHEAD
# ---------------------------------------------------------------------------

def resample_ohlcv(df_m1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample M1 data to higher timeframe.
    Uses 'left' label and 'left' closed so bar[t] contains
    data from t to t+tf-1. This is the correct non-lookahead approach.

    The resulting bar at time T represents the candle that CLOSED at T+period.
    When we iterate, we always look at the PREVIOUS completed bar.
    """
    rule = RESAMPLE_MAP.get(timeframe)
    if rule is None:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use: {list(RESAMPLE_MAP.keys())}")

    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }

    resampled = df_m1.resample(rule, label="left", closed="left").agg(agg)
    resampled = resampled.dropna(subset=["open", "close"])
    resampled = resampled[resampled["high"] >= resampled["low"]]

    return resampled


# ---------------------------------------------------------------------------
# EURGBP synthetic builder
# ---------------------------------------------------------------------------

def build_synthetic_pair(
    df_base1: pd.DataFrame,   # e.g., EURUSD M1
    df_base2: pd.DataFrame,   # e.g., GBPUSD M1
    pair_name: str = "EURGBP",
    operation: str = "divide",  # EURGBP = EURUSD / GBPUSD
) -> pd.DataFrame:
    """
    Build a synthetic cross pair from two USD pairs.
    EURGBP = EURUSD / GBPUSD
    AUDNZD = AUDUSD / NZDUSD

    Uses only overlapping timestamps to avoid NaN propagation.
    This is ONLY needed if direct pair data is not available.
    """
    # Align on common index
    df1 = df_base1[["open", "high", "low", "close"]].copy()
    df2 = df_base2[["open", "high", "low", "close"]].copy()

    combined = df1.join(df2, lsuffix="_1", rsuffix="_2", how="inner")
    combined = combined.dropna()

    result = pd.DataFrame(index=combined.index)

    if operation == "divide":
        result["open"]  = combined["open_1"]  / combined["open_2"]
        result["close"] = combined["close_1"] / combined["close_2"]
        # High/low of synthetic: conservative approximation
        result["high"]  = combined["high_1"]  / combined["low_2"]
        result["low"]   = combined["low_1"]   / combined["high_2"]
    else:
        raise ValueError(f"Unknown operation: {operation}")

    result["volume"] = (
        df1["volume"].reindex(combined.index).fillna(0) +
        df2["volume"].reindex(combined.index).fillna(0)
    ) / 2

    return result


# ---------------------------------------------------------------------------
# Master data preparation
# ---------------------------------------------------------------------------

class DataStore:
    """
    Holds all timeframe data for all pairs.
    Pre-computed once before backtest loop.
    """

    def __init__(self, data_dir: str, pairs: List[str], years: Optional[List[int]] = None):
        self.data_dir = data_dir
        self.pairs = pairs
        self.years = years
        self._m1: Dict[str, pd.DataFrame] = {}
        self._tf_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    def load(self):
        """Load all pairs."""
        for pair in self.pairs:
            logger.info(f"Loading {pair}...")
            self._m1[pair] = load_pair_data(self.data_dir, pair, self.years)
            self._tf_cache[pair] = {}

    def get_tf(self, pair: str, tf: str) -> pd.DataFrame:
        """Get cached resampled data."""
        if tf not in self._tf_cache[pair]:
            logger.info(f"Resampling {pair} to {tf}...")
            self._tf_cache[pair][tf] = resample_ohlcv(self._m1[pair], tf)
        return self._tf_cache[pair][tf]

    def get_m1(self, pair: str) -> pd.DataFrame:
        return self._m1[pair]
