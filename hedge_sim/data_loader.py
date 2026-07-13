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

        start_date, end_date = self.cfg.resolve_date_range()
        if start_date:
            data = data[data["time"] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data["time"] <= pd.to_datetime(end_date)]

        if data.empty:
            raise ValueError("Loaded dataset is empty after filtering - check date range / files.")

        for col in ["open", "high", "low", "close"]:
            data[col] = data[col].astype(float)
        if "volume" not in data.columns:
            data["volume"] = 0.0

        data = self._sanity_filter(data)

        return data.reset_index(drop=True)

    @staticmethod
    def _sanity_filter(data: pd.DataFrame, max_bar_move_pct: float = 0.20,
                        max_hl_range_pct: float = 0.10) -> pd.DataFrame:
        """
        Drops corrupted/outlier rows that are common in real downloaded FX
        history (a decimal shifted by a stray digit, a duplicated header
        line, a garbage tick). A single bad row like this can otherwise
        blow up margin/exposure/drawdown stats by orders of magnitude
        without ever showing up as an obvious crash. Heuristics:
          - OHLC values must be internally consistent (low <= open,close <= high, low <= high)
          - high/low range within a bar can't exceed `max_hl_range_pct` of price
          - close-to-close jump between consecutive bars can't exceed `max_bar_move_pct`
        Flagged rows are dropped and a summary is printed.
        """
        before = len(data)
        consistent = (data["low"] <= data["open"]) & (data["open"] <= data["high"]) & \
                     (data["low"] <= data["close"]) & (data["close"] <= data["high"]) & \
                     (data["low"] <= data["high"]) & (data["open"] > 0) & (data["close"] > 0)

        hl_range_pct = (data["high"] - data["low"]) / data["close"].replace(0, pd.NA)
        sane_range = hl_range_pct.fillna(0) <= max_hl_range_pct

        prev_close = data["close"].shift(1)
        jump_pct = (data["close"] - prev_close).abs() / prev_close.replace(0, pd.NA)
        sane_jump = jump_pct.fillna(0) <= max_bar_move_pct

        keep = consistent & sane_range & sane_jump
        dropped = before - int(keep.sum())
        if dropped > 0:
            print(f"      [data sanity check] dropped {dropped} corrupted/outlier row(s) out of {before} "
                  f"(inconsistent OHLC, absurd intrabar range, or an implausible tick-to-tick jump).")
        return data[keep]
