# engine/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import io
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        
    def load_symbol(self, symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Load M1 data for a symbol from multiple years"""
        all_data = []
        
        for year in range(start_year, end_year + 1):
            df = self._load_year(symbol, year)
            if df is not None:
                all_data.append(df)
                logger.info(f"Loaded {symbol} {year}: {len(df)} bars")
        
        if not all_data:
            raise ValueError(f"No data found for {symbol}")
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Total {symbol}: {len(combined)} M1 bars")
        return combined
    
    def _load_year(self, symbol: str, year: int) -> pd.DataFrame:
        """Load single year - handles both CSV and ZIP"""
        
        # Try CSV first (for EURUSD, GBPUSD)
        csv_pattern = f"DAT_ASCII_{symbol}_M1_{year}.csv"
        csv_file = self.data_dir / csv_pattern
        
        if csv_file.exists():
            return self._read_csv(csv_file)
        
        # Try ZIP (for other symbols)
        zip_pattern = f"HISTDATA_COM_ASCII_{symbol}_M1{year}.zip"
        zip_file = self.data_dir / zip_pattern
        
        if zip_file.exists():
            return self._read_zip(zip_file, symbol, year)
        
        logger.warning(f"No file found for {symbol} {year}")
        return None
    
    def _read_csv(self, filepath: Path) -> pd.DataFrame:
        """Read HistData CSV format"""
        df = pd.read_csv(
            filepath,
            sep=';',
            names=['time', 'open', 'high', 'low', 'close', 'volume'],
            parse_dates=['time'],
            date_format='%Y%m%d %H%M%S'
        )
        return df
    
    def _read_zip(self, filepath: Path, symbol: str, year: int) -> pd.DataFrame:
        """Read HistData ZIP format"""
        with zipfile.ZipFile(filepath, 'r') as z:
            # HistData ZIP usually contains CSV with similar name
            csv_name = f"DAT_ASCII_{symbol}_M1_{year}.csv"
            
            # Try to find the CSV inside
            names = z.namelist()
            csv_file = None
            
            for name in names:
                if name.endswith('.csv'):
                    csv_file = name
                    break
            
            if csv_file is None:
                logger.error(f"No CSV found in {filepath}")
                return None
            
            with z.open(csv_file) as f:
                df = pd.read_csv(
                    io.BytesIO(f.read()),
                    sep=';',
                    names=['time', 'open', 'high', 'low', 'close', 'volume'],
                    parse_dates=['time'],
                    date_format='%Y%m%d %H%M%S'
                )
                return df

class TimeFrameConverter:
    @staticmethod
    def resample_to_m15(df: pd.DataFrame) -> pd.DataFrame:
        """Convert M1 to M15"""
        df = df.set_index('time')
        
        resampled = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled.reset_index()
    
    @staticmethod
    def resample_to_h1(df: pd.DataFrame) -> pd.DataFrame:
        """Convert M1 to H1"""
        df = df.set_index('time')
        
        resampled = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled.reset_index()
