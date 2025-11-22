from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
import yfinance as yf

DATA_RAW = Path("data/raw")

@dataclass
class FetchSpec:
    tickers: list[str]
    start: str
    end: str
    interval: str = "1d"
    
    
def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure tz-naive DatetimeIndex and drop freq to avoid cache/load diffs
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    df.index = df.index.tz_localize(None)
    df.index.freq = None
    return df


# Obtain data from yfinance and put into parquet and manifest
def fetch_prices(spec: FetchSpec, force: bool = False) -> pd.DataFrame:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    # Get parquet and manifest paths
    pq = _parquet_path(spec)
    mf = _manifest_path(spec)
    if pq.exists() and not force:
        return pd.read_parquet(pq)
    
    # Get data
    data = yf.download(
        tickers=spec.tickers,
        start=spec.start,
        end=spec.end,
        interval=spec.interval,
        auto_adjust=True,
        progress=False
    )['Close']
    
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Save data to parquet
    data = data.sort_index().astype('float64')
    data = _normalize_index(data)
    data.to_parquet(pq)
    
    # Create and save manifest
    manifest = {
        'tickers': spec.tickers,
        'start': spec.start,
        'end': spec.end,
        'interval': spec.interval,
        'rows': int(data.shape[0]),
        'cols': list(data.columns),
        'library_versions': {
            'pandas': pd.__version__,
            'yfinance': yf.__version__,
        },
    }
    mf.write_text(json.dumps(manifest, indent=2))
    
    return data


# Create Manifest PATH
def _manifest_path(spec: FetchSpec) -> Path:
    name = f"{'-'.join(spec.tickers)}_{spec.start}_{spec.end}_{spec.interval}"
    return DATA_RAW / f"{name}.manifest.json"


# Create Parquet PATH
def _parquet_path(spec: FetchSpec) -> Path:
    name = f"{'-'.join(spec.tickers)}_{spec.start}_{spec.end}_{spec.interval}"
    return DATA_RAW / f"{name}.parquet"


# Try to find data for a given set of tickers if we have it in data folder
def load_latest_for(tickers: Iterable[str]) -> pd.DataFrame | None:
    want = set(tickers)
    
    # Grab all parquets
    candidates = sorted(
        DATA_RAW.glob('*.parquet'), 
        key = lambda p: p.stat().st_mtime, 
        reverse=True
    )
    
    # Sort through parquets to find match and return df
    for p in candidates:
        parts = p.stem.split('_')[0].split('-')
        if set(parts) == want:
            df = pd.read_parquet(p)
            return _normalize_index(df)

    # If no match found return None
    return None
    