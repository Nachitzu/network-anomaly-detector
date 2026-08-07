"""Load and concatenate CICIDS2017 flow CSV files into a single DataFrame."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names.

    CICIDS2017 CSVs are known to ship with inconsistent whitespace in column
    headers (e.g. " Flow Duration", "Label "). Normalizing here avoids
    duplicate/mismatched columns when concatenating multiple files.
    """
    return df.rename(columns=lambda col: col.strip())


def _match_normalized(columns: Sequence[str]) -> Callable[[str], bool]:
    """Build the `usecols` predicate that selects `columns` from raw headers.

    `pd.read_csv` decides which columns to materialize BEFORE
    `_normalize_columns` runs, so the predicate receives the raw header exactly
    as it appears in the file -- " Flow Duration" with its leading space in
    CICIDS2017, while a sibling column like "Flow Bytes/s" has none. Matching
    on the stripped form is what lets callers pass the clean names from
    `config.yaml` and still hit both.
    """
    wanted = {col.strip() for col in columns}
    return lambda raw_name: raw_name.strip() in wanted


def _require_columns(df: pd.DataFrame, expected: Sequence[str], source: Path) -> None:
    """Fail loudly if a requested column was not present in `source`.

    `usecols` with a callable silently ignores names that never match. Left
    unchecked, a renamed header would surface much later as an all-NaN column
    whose rows `sanitize_flows` would quietly drop -- an empty training set
    with no explanation. Raising here names the file and the columns instead.

    Raises:
        ValueError: if any of `expected` is missing from `df`.
    """
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"{source.name} is missing expected columns: {missing}")


def list_csv_files(data_dir: str | Path, pattern: str = "*.csv") -> list[Path]:
    """Return a sorted list of CSV file paths matching `pattern` in `data_dir`.

    Raises:
        FileNotFoundError: if `data_dir` does not exist or is not a directory.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory not found: {directory}")
    return sorted(directory.glob(pattern))


def load_flows(
    data_dir: str | Path,
    pattern: str = "*.csv",
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate all CICIDS2017 CSV files in `data_dir`.

    Column names are stripped of surrounding whitespace before concatenation
    so that files with slightly different header formatting still align into
    a single, consistent set of columns.

    Args:
        data_dir: Directory containing one or more CICIDS2017 CSV files.
        pattern: Glob pattern used to select CSV files within `data_dir`.
        columns: Optional whitelist of already-normalized column names (i.e.
            the clean names from `config.yaml`) to read. When given, only
            those columns are materialized, which is what keeps the full
            dataset's memory footprint proportional to the features actually
            used rather than to all 79 CICIDS2017 columns. Keyword-only, and
            `None` by default: filtering is opt-in, so the general-purpose
            "load everything" behaviour is unchanged for other callers.

    Returns:
        A single DataFrame with all matching CSV files concatenated.

    Raises:
        FileNotFoundError: if `data_dir` does not exist, or no file in it
            matches `pattern`.
        ValueError: if `columns` is given and any of them is absent from one
            of the matched files.
    """
    csv_files = list_csv_files(data_dir, pattern)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' found in {data_dir}")

    usecols = _match_normalized(columns) if columns is not None else None
    frames = []
    for csv_file in csv_files:
        frame = _normalize_columns(pd.read_csv(csv_file, usecols=usecols))
        if columns is not None:
            _require_columns(frame, columns, csv_file)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
