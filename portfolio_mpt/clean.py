from __future__ import annotations
import pandas as pd
from pathlib import Path
import json

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def clean_price_file(raw_path: str | Path) -> Path:
    """
    Clean a single parquet file and return the path to the cleaned file.
    """
    
    raw_path = Path(raw_path)
    df = pd.read_parquet(raw_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    df = df[df.index.notnull()]
    df = df[~df.index.duplicated(keep="first")]

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index).ffill().bfill().astype("float64")

    clean_path = CLEAN_DIR / raw_path.name
    df.to_parquet(clean_path)

    # Write manifest
    manifest = {
        "source_file": raw_path.name,
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
    }
    mf_path = clean_path.parent / (clean_path.stem + ".manifest.json")
    mf_path.write_text(json.dumps(manifest, indent=2))

    return clean_path


def clean_all_raw():
    """
    Clean all files in data/raw and save results to data/clean.
    """
    
    raw_files = list(RAW_DIR.glob("*.parquet"))
    
    cleaned = []
    
    for p in raw_files:
        out = clean_price_file(p)
        cleaned.append(out)
    
    return cleaned