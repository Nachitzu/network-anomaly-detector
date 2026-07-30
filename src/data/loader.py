"""Load and concatenate CICIDS2017 flow CSV files into a single DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names.

    CICIDS2017 CSVs are known to ship with inconsistent whitespace in column
    headers (e.g. " Flow Duration", "Label "). Normalizing here avoids
    duplicate/mismatched columns when concatenating multiple files.
    """
    return df.rename(columns=lambda col: col.strip())


def list_csv_files(data_dir: str | Path, pattern: str = "*.csv") -> list[Path]:
    """Return a sorted list of CSV file paths matching `pattern` in `data_dir`.

    Raises:
        FileNotFoundError: if `data_dir` does not exist or is not a directory.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")
    return sorted(directory.glob(pattern))


def load_flows(data_dir: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate all CICIDS2017 CSV files in `data_dir`.

    Column names are stripped of surrounding whitespace before concatenation
    so that files with slightly different header formatting still align into
    a single, consistent set of columns.

    Args:
        data_dir: Directory containing one or more CICIDS2017 CSV files.
        pattern: Glob pattern used to select CSV files within `data_dir`.

    Returns:
        A single DataFrame with all matching CSV files concatenated.

    Raises:
        FileNotFoundError: if `data_dir` does not exist, or no file in it
            matches `pattern`.
    """
    csv_files = list_csv_files(data_dir, pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' found in {data_dir}")

    frames = [_normalize_columns(pd.read_csv(csv_file)) for csv_file in csv_files]
    return pd.concat(frames, ignore_index=True)
