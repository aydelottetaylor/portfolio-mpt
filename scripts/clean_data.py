from __future__ import annotations
import pandas as pd
from pathlib import Path
import json

RAW_DIR = Path('data/raw')
CLEAN_DIR = Path('data/clean')
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_file(path: Path) -> pd.DataFrame:
    """
    Load a raw parquet file and return a DataFrame
    """
    df = pd.read_parquet(path)
    
    if not isinstance(df.index, pd.DateTimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
        
    df = df.sort_index()
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply some standard cleaning ops to price data. 
    """
    
    # Drop rows where index can't be parsed
    df = df[df.index.notnull()]
    
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    # Ensure the index is daily frequency - reindex to full daily range
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_index)
    
    # Fill missing values
    df = df.ffill().bfill()
    
    # Ensure float type
    df = df.astype('float64')
    
    return df


def write_manifest(clean_path: Path, df: pd.DataFrame, source_file: str):
    manifest = {
        "source_file": source_file,
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "cleaning_steps": [
            "parse datetime index",
            "deduplicate index",
            "reindex to daily frequency",
            "forward/backward fill missing values",
            "enforce float64",
        ],
    }
    mf_path = clean_path.with_suffix(".manifest.json")
    mf_path.write_text(json.dumps(manifest, indent=2))


def main():
    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        print("No raw parquet files found in data/raw/")
        return

    print(f"Found {len(raw_files)} raw files. Cleaning…")

    for raw_path in raw_files:
        print(f" → Cleaning {raw_path.name}")

        df_raw = load_raw_file(raw_path)
        df_clean = clean_dataframe(df_raw)

        clean_path = CLEAN_DIR / raw_path.name
        df_clean.to_parquet(clean_path)
        write_manifest(clean_path, df_clean, source_file=raw_path.name)

        print(f"   Saved cleaned file: {clean_path}")

    print("\nCleaning complete! Cleaned files stored in data/clean/")


if __name__ == "__main__":
    main()
