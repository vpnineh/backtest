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
        logger.info(f"DataLoader initialized with directory: {self.data_dir.absolute()}")
        
    def load_symbol(self, symbol: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Load M1 data for a symbol from multiple years"""
        all_data = []
        
        for year in range(start_year, end_year + 1):
            df = self._load_year(symbol, year)
            if df is not None and len(df) > 0:
                all_data.append(df)
                logger.info(f"✓ Loaded {symbol} {year}: {len(df):,} bars")
            else:
                logger.warning(f"✗ No data for {symbol} {year}")
        
        if not all_data:
            raise ValueError(f"No data found for {symbol}")
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('time').reset_index(drop=True)
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['time'], keep='first')
        
        logger.info(f"✓ Total {symbol}: {len(combined):,} M1 bars from {combined['time'].min()} to {combined['time'].max()}")
        return combined
    
    def _load_year(self, symbol: str, year: int) -> pd.DataFrame:
        """Load single year - handles both CSV and ZIP"""
        
        # Try CSV first (for EURUSD, GBPUSD)
        csv_pattern = f"DAT_ASCII_{symbol}_M1_{year}.csv"
        csv_file = self.data_dir / csv_pattern
        
        if csv_file.exists():
            logger.debug(f"Found CSV: {csv_file}")
            return self._read_csv(csv_file)
        
        # Try ZIP (for other symbols)
        zip_pattern = f"HISTDATA_COM_ASCII_{symbol}_M1{year}.zip"
        zip_file = self.data_dir / zip_pattern
        
        if zip_file.exists():
            logger.debug(f"Found ZIP: {zip_file}")
            return self._read_zip(zip_file, symbol, year)
        
        logger.warning(f"No file found for {symbol} {year} (tried {csv_pattern} and {zip_pattern})")
        return None
    
    def _read_csv(self, filepath: Path) -> pd.DataFrame:
        """Read HistData CSV format"""
        try:
            # Try different separators
            for sep in [';', ',']:
                try:
                    df = pd.read_csv(
                        filepath,
                        sep=sep,
                        names=['time', 'open', 'high', 'low', 'close', 'volume'],
                        header=None
                    )
                    
                    # Try to parse datetime
                    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H%M%S', errors='coerce')
                    
                    # If no errors, we found the right format
                    if df['time'].isna().sum() == 0:
                        logger.debug(f"Successfully parsed with separator '{sep}'")
                        return df[['time', 'open', 'high', 'low', 'close', 'volume']].dropna()
                except:
                    continue
            
            # If both failed, try reading first line to see format
            with open(filepath, 'r') as f:
                first_line = f.readline()
                logger.error(f"Could not parse CSV. First line: {first_line}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading CSV {filepath}: {e}")
            return None
    
    def _read_zip(self, filepath: Path, symbol: str, year: int) -> pd.DataFrame:
        """Read HistData ZIP format"""
        try:
            with zipfile.ZipFile(filepath, 'r') as z:
                names = z.namelist()
                logger.debug(f"ZIP contains: {names}")
                
                # Find CSV file
                csv_file = None
                for name in names:
                    if name.endswith('.csv'):
                        csv_file = name
                        break
                
                if csv_file is None:
                    logger.error(f"No CSV found in {filepath}")
                    return None
                
                logger.debug(f"Reading {csv_file} from ZIP")
                
                with z.open(csv_file) as f:
                    content = f.read()
                    
                    # Try different encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            df = pd.read_csv(
                                io.BytesIO(content),
                                sep=';',
                                names=['time', 'open', 'high', 'low', 'close', 'volume'],
                                header=None,
                                encoding=encoding
                            )
                            
                            df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H%M%S', errors='coerce')
                            
                            if df['time'].isna().sum() == 0:
                                logger.debug(f"Successfully parsed with encoding '{encoding}'")
                                return df[['time', 'open', 'high', 'low', 'close', 'volume']].dropna()
                        except:
                            continue
                    
                    logger.error(f"Could not parse ZIP content with any encoding")
                    return None
                    
        except Exception as e:
            logger.error(f"Error reading ZIP {filepath}: {e}")
            return None

class TimeFrameConverter:
    @staticmethod
    def resample_to_m15(df: pd.DataFrame) -> pd.DataFrame:
        """Convert M1 to M15"""
        if 'time' not in df.columns:
            raise ValueError("DataFrame must have 'time' column")
        
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
        if 'time' not in df.columns:
            raise ValueError("DataFrame must have 'time' column")
        
        df = df.set_index('time')
        
        resampled = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        result = resampled.reset_index()
        logger.debug(f"Resampled to H1: {len(result)} bars")
        return result
