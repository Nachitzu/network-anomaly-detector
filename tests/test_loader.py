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
