from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import BacktestConfig


def _detect_separator(first_line: str) -> str:
    if ";" in first_line:
        return ";"
    if "," in first_line:
        return ","
    return r"\s+"


def _parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_text = date_series.astype(str).str.strip()
    time_text = (
        time_series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )

    combined = date_text + " " + time_text

    parsed = pd.to_datetime(
        combined,
        format="%Y%m%d %H%M%S",
        errors="coerce",
    )

    if parsed.isna().mean() > 0.50:
        parsed = pd.to_datetime(combined, errors="coerce")

    return parsed


def _read_stream(stream) -> pd.DataFrame:
    if isinstance(stream, bytes):
        stream = io.BytesIO(stream)

    first_line = stream.readline()

    if isinstance(first_line, bytes):
        first_line_text = first_line.decode("utf-8", errors="ignore")
    else:
        first_line_text = first_line

    stream.seek(0)
    separator = _detect_separator(first_line_text)

    df = pd.read_csv(
        stream,
        sep=separator,
        header=None,
        engine="python",
        comment="#",
    )

    if df.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns, received {df.shape[1]}"
        )

    # HistData format:
    # Date, Time, Open, High, Low, Close, Volume
    columns = [
        "date",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df.iloc[:, : min(df.shape[1], 7)]
    df.columns = columns[: df.shape[1]]

    if "volume" not in df.columns:
        df["volume"] = 0

    df["datetime"] = _parse_datetime(df["date"], df["time"])

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df[
        ["datetime", "open", "high", "low", "close", "volume"]
    ].dropna(subset=["datetime", "open", "high", "low", "close"])


def _read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".csv", ".txt"))
            and not name.endswith("/")
        ]

        if not candidates:
            raise ValueError(f"No CSV/TXT file inside {path}")

        member = candidates[0]

        with archive.open(member) as stream:
            return _read_stream(stream)


def _extract_year(path: Path) -> int | None:
    match = re.search(r"(20\d{2})", path.name)
    return int(match.group(1)) if match else None


def find_symbol_files(
    data_dir: Path,
    symbol: str,
    start_year: int,
    end_year: int,
) -> list[Path]:
    symbol = symbol.upper()

    files = [
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and symbol in path.name.upper()
        and path.suffix.lower() in {".zip", ".csv", ".txt"}
    ]

    selected = []

    for path in files:
        year = _extract_year(path)

        if year is not None and start_year <= year <= end_year:
            selected.append(path)

    return sorted(selected, key=lambda item: (_extract_year(item) or 0, item.name))


def _localize_to_utc(
    df: pd.DataFrame,
    source_timezone: str,
    output_timezone: str,
) -> pd.DataFrame:
    timestamps = pd.DatetimeIndex(df["datetime"])

    if timestamps.tz is None:
        timestamps = timestamps.tz_localize(
            source_timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )

    timestamps = timestamps.tz_convert(output_timezone)

    df = df.copy()
    df["datetime"] = timestamps
    return df.dropna(subset=["datetime"])


def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    valid = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
        & (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
    )

    invalid_count = int((~valid).sum())

    if invalid_count:
        logger.warning(f"Dropping {invalid_count:,} invalid OHLC rows.")

    return df.loc[valid].copy()


def load_m1_data(
    symbol: str,
    config: BacktestConfig,
) -> pd.DataFrame:
    cache_path = (
        config.cache_dir
        / f"{symbol}_M1_{config.start_year}_{config.end_year}.parquet"
    )

    if cache_path.exists():
        logger.info(f"Loading cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.set_index("datetime").sort_index()

    files = find_symbol_files(
        config.data_dir,
        symbol,
        config.start_year,
        config.end_year,
    )

    if not files:
        raise FileNotFoundError(
            f"No files found for {symbol} in {config.data_dir}"
        )

    frames = []

    for path in files:
        logger.info(f"Reading {path.name}")

        if path.suffix.lower() == ".zip":
            frame = _read_zip(path)
        else:
            with path.open("rb") as stream:
                frame = _read_stream(stream)

        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    df = _localize_to_utc(
        df,
        config.source_timezone,
        config.output_timezone,
    )

    df = validate_ohlc(df)

    df = (
        df.drop_duplicates(subset="datetime", keep="last")
        .sort_values("datetime")
        .set_index("datetime")
    )

    if config.save_cache:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_parquet(cache_path, index=False)
        logger.info(f"Saved cache: {cache_path}")

    logger.info(
        f"{symbol}: {len(df):,} M1 rows, "
        f"{df.index.min()} -> {df.index.max()}"
    )

    return df
