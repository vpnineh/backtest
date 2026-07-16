"""
fx_convert.py
==============
Converts quote-currency P&L into account currency (assumed USD) using
the actual GBPUSD / NZDUSD history that ships in the same data/ folder,
instead of a fixed/guessed conversion rate. This matters a lot for
honesty of the results: EURGBP pip value is fixed in GBP but floats in
USD, and AUDNZD pip value is fixed in NZD but floats in USD.

If the conversion pair's data is not present, we fall back to a fixed
approximate rate and print a clear warning -- we never silently pretend
to have precision we don't.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import data_loader

CONVERSION_PAIR = {
    "EURGBP": "GBPUSD",   # quote currency of EURGBP is GBP
    "AUDNZD": "NZDUSD",   # quote currency of AUDNZD is NZD
}

FALLBACK_RATE = {
    "EURGBP": 1.25,   # approximate long-run GBPUSD, used only if data missing
    "AUDNZD": 0.60,   # approximate long-run NZDUSD, used only if data missing
}


class ConversionSeries:
    """Holds a timestamp-sorted conversion rate series and gives
    point-in-time (never-future) lookups via merge_asof(direction='backward')."""

    def __init__(self, symbol: str, times: pd.Series, rates: Optional[pd.Series]):
        self.symbol = symbol
        self._df = None
        if rates is not None and len(rates) > 0:
            self._df = pd.DataFrame({"datetime": times, "rate": rates}).sort_values("datetime")

    def rate_at(self, timestamps: pd.Series) -> pd.Series:
        if self._df is None:
            return pd.Series(FALLBACK_RATE_DEFAULT, index=timestamps.index)
        left = pd.DataFrame({"datetime": timestamps}).reset_index()
        merged = pd.merge_asof(
            left.sort_values("datetime"), self._df, on="datetime", direction="backward"
        )
        merged = merged.sort_values("index").set_index("index")
        return merged["rate"]


FALLBACK_RATE_DEFAULT = 1.0


def build_conversion_series(data_dir: Path, symbol: str, timeframe: str,
                             start_year: int, end_year: int) -> ConversionSeries:
    conv_symbol = CONVERSION_PAIR.get(symbol)
    if conv_symbol is None:
        # symbol already quoted in USD, or unknown -> no conversion needed
        return ConversionSeries(symbol, pd.Series(dtype="datetime64[ns]"), None)

    try:
        conv_df = data_loader.load_and_resample(data_dir, conv_symbol, timeframe, start_year, end_year)
        return ConversionSeries(conv_symbol, conv_df["datetime"], conv_df["close"])
    except FileNotFoundError as e:
        print(f"[fx_convert] WARNING - {e}. Falling back to fixed rate "
              f"{FALLBACK_RATE.get(symbol)} for {symbol} P&L conversion to USD. "
              f"Results involving absolute $ P&L should be treated as approximate.")
        global FALLBACK_RATE_DEFAULT
        FALLBACK_RATE_DEFAULT = FALLBACK_RATE.get(symbol, 1.0)
        return ConversionSeries(symbol, pd.Series(dtype="datetime64[ns]"), None)
