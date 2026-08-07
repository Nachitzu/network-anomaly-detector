"""Unit tests for src.data.loader.

Uses the tiny synthetic CSV fixtures in tests/fixtures/ (never the full
CICIDS2017 dataset) so tests stay fast and dependency-free.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import list_csv_files, load_flows

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_list_csv_files_returns_sorted_matching_paths() -> None:
    files = list_csv_files(FIXTURES_DIR, pattern="flows_part*.csv")

    assert [f.name for f in files] == ["flows_part1.csv", "flows_part2.csv"]


def test_list_csv_files_raises_for_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        list_csv_files(missing_dir)


def test_load_flows_concatenates_multiple_csv_files() -> None:
    df = load_flows(FIXTURES_DIR, pattern="flows_part*.csv")

    # 2 rows from flows_part1.csv + 2 rows from flows_part2.csv
    assert len(df) == 4


def test_load_flows_strips_whitespace_from_column_names() -> None:
    # flows_part1.csv has headers with stray leading/trailing spaces
    # (e.g. " Destination Port", " Flow Duration"); flows_part2.csv does not.
    # Both must align after normalization so concatenation doesn't introduce
    # duplicate/NaN-only columns.
    df = load_flows(FIXTURES_DIR, pattern="flows_part*.csv")

    assert "Destination Port" in df.columns
    assert "Flow Duration" in df.columns
    assert not any(col != col.strip() for col in df.columns)
    # No stray duplicate columns caused by mismatched whitespace.
    assert list(df.columns).count("Flow Duration") == 1


def test_load_flows_returns_dataframe_with_label_column() -> None:
    df = load_flows(FIXTURES_DIR, pattern="flows_part*.csv")

    assert isinstance(df, pd.DataFrame)
    assert "Label" in df.columns
    assert set(df["Label"]) == {"BENIGN", "DoS Hulk"}


def test_load_flows_raises_when_no_csv_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_flows(tmp_path, pattern="*.csv")


def test_load_flows_raises_for_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        load_flows(missing_dir)


# --- columns=: opt-in narrowing of what gets read --------------------------


def test_load_flows_without_columns_keeps_every_column() -> None:
    """Pins the default: filtering is opt-in, nothing is dropped implicitly."""
    df = load_flows(FIXTURES_DIR, pattern="flows_part*.csv")

    assert "Destination Port" in df.columns
    assert len(df.columns) > 2


def test_load_flows_columns_reads_only_the_requested_columns() -> None:
    df = load_flows(FIXTURES_DIR, pattern="flows_part*.csv", columns=["Flow Duration", "Label"])

    assert list(df.columns) == ["Flow Duration", "Label"]
    assert "Destination Port" not in df.columns


def test_load_flows_columns_matches_headers_with_stray_whitespace() -> None:
    """The whole point of the feature: `pd.read_csv` picks columns BEFORE
    names are normalized, and flows_part1.csv spells it " Flow Duration"
    while flows_part2.csv spells it "Flow Duration". Asking for the clean
    name must select it from both, with no NaN-only rows from a near miss.
    """
    df = load_flows(FIXTURES_DIR, pattern="flows_part*.csv", columns=["Flow Duration", "Label"])

    assert len(df) == 4
    assert not df.isna().any().any()


def test_load_flows_columns_matches_unfiltered_load() -> None:
    """Narrowing must not alter a single value -- only what is materialized."""
    requested = ["Flow Duration", "Label"]

    narrow = load_flows(FIXTURES_DIR, pattern="flows_part*.csv", columns=requested)
    wide = load_flows(FIXTURES_DIR, pattern="flows_part*.csv")[requested]

    pd.testing.assert_frame_equal(narrow, wide)


def test_load_flows_raises_when_a_file_lacks_a_requested_column(tmp_path: Path) -> None:
    """`usecols` with a callable ignores misses silently; we must not."""
    (tmp_path / "partial.csv").write_text("Flow Duration,Label\n5,BENIGN\n", encoding="utf-8")

    with pytest.raises(ValueError, match="partial.csv is missing expected columns"):
        load_flows(tmp_path, pattern="*.csv", columns=["Flow Duration", "Total Fwd Packets"])


def test_load_flows_columns_is_keyword_only() -> None:
    """Guards the positional call site in `run_comparison.load_prepared_dataset`."""
    with pytest.raises(TypeError):
        load_flows(FIXTURES_DIR, "flows_part*.csv", ["Flow Duration"])  # type: ignore[call-arg]
