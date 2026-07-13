"""
DataLoader
==========
Loads historical OHLC(V) data from a single CSV file or a folder of CSV
files (e.g. one file per year, which is how most GitHub Forex data dumps
for 2010-2025 are organized), normalizes column names, sorts by time and
optionally filters by date range.
"""

from __future__ import annotations
import glob
import os
import zipfile
from pathlib import Path

import pandas as pd

from .configuration import DataConfig

_CANDIDATE_COLUMNS = {
    "time": ["time", "datetime", "date", "timestamp", "date_time", "local time"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "price"],
    "volume": ["volume", "vol", "tick_volume", "tickvol"],
}

# HistData.com ASCII M1 files ship as: DateTime;Open;High;Low;Close;Volume
# no header row, ';' separated, DateTime formatted as "YYYYMMDD HHMMSS".
_HISTDATA_COLUMNS = ["time", "open", "high", "low", "close", "volume"]
_HISTDATA_DATETIME_FORMAT = "%Y%m%d %H%M%S"


class DataLoader:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def _resolve_files(self) -> list[str]:
        p = Path(self.cfg.path)
        if p.is_file():
            return [str(p)]
        if p.is_dir():
            files = sorted(glob.glob(str(p / self.cfg.file_pattern)))
            if not files:
                # fall back: any csv or zip in the folder
                files = sorted(glob.glob(str(p / "*.csv"))) + sorted(glob.glob(str(p / "*.zip")))
            if not files:
                raise FileNotFoundError(
                    f"No CSV/ZIP files found under '{p}' matching '{self.cfg.file_pattern}'."
                )
            return files
        raise FileNotFoundError(f"Data path '{p}' does not exist.")

    @staticmethod
    def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
        lower = {c: c.strip().lower() for c in df.columns}
        df = df.rename(columns=lower)
        rename_map = {}
        for target, candidates in _CANDIDATE_COLUMNS.items():
            for cand in candidates:
                if cand in df.columns:
                    rename_map[cand] = target
                    break
        df = df.rename(columns=rename_map)
        missing = [c for c in ["time", "open", "high", "low", "close"] if c not in df.columns]
        if missing:
            raise ValueError(
                f"Could not find required column(s) {missing} in data. "
                f"Available columns: {list(df.columns)}"
            )
        return df

    @staticmethod
    def _is_histdata_file(fp: str) -> bool:
        name = os.path.basename(fp).upper()
        return "HISTDATA" in name or fp.lower().endswith(".zip")

    @classmethod
    def _read_histdata(cls, fp: str) -> pd.DataFrame:
        """Reads a HistData.com ASCII M1 file, either a raw .csv or a .zip
        wrapping one .csv (the exact format HISTDATA_COM_ASCII_*.zip ships)."""
        if fp.lower().endswith(".zip"):
            with zipfile.ZipFile(fp) as zf:
                inner_names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
                if not inner_names:
                    raise ValueError(f"No CSV/TXT file found inside zip '{fp}'.")
                with zf.open(inner_names[0]) as inner:
                    df = pd.read_csv(inner, sep=";", header=None, names=_HISTDATA_COLUMNS)
        else:
            df = pd.read_csv(fp, sep=";", header=None, names=_HISTDATA_COLUMNS)

        df["time"] = pd.to_datetime(df["time"], format=_HISTDATA_DATETIME_FORMAT, errors="coerce")
        return df

    def load(self) -> pd.DataFrame:
        files = self._resolve_files()
        frames = []
        for fp in files:
            if self._is_histdata_file(fp):
                df = self._read_histdata(fp)
            else:
                df = pd.read_csv(fp)
                df = self._map_columns(df)
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)
        data["time"] = pd.to_datetime(data["time"], utc=False, errors="coerce")
        data = data.dropna(subset=["time", "open", "high", "low", "close"])
        data = data.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

        if self.cfg.start_date:
            data = data[data["time"] >= pd.to_datetime(self.cfg.start_date)]
        if self.cfg.end_date:
            data = data[data["time"] <= pd.to_datetime(self.cfg.end_date)]

        if data.empty:
            raise ValueError("Loaded dataset is empty after filtering - check date range / files.")

        for col in ["open", "high", "low", "close"]:
            data[col] = data[col].astype(float)
        if "volume" not in data.columns:
            data["volume"] = 0.0

        return data.reset_index(drop=True)
