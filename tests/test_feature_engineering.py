"""Unit tests for src.data.feature_engineering.

Uses the tiny synthetic fixture at tests/fixtures/sample_flows.csv (24 rows:
14 BENIGN, 4 DoS Hulk, 3 PortScan, 3 DDoS) — never the full CICIDS2017
dataset — so tests stay fast and require no external data.

`test_prepare_dataset_has_no_label_leakage` is the project's non-negotiable
check (see README section 7): the training feature matrix must never
contain, or be derived from, the ground-truth label column, and the
benign-only training set must contain only BENIGN rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.feature_engineering import (
    Config,
    PreparedData,
    fit_scaler,
    load_config,
    prepare_dataset,
    required_columns,
    sanitize_flows,
    select_feature_matrix,
    split_benign_and_mixed,
)
from src.data.loader import load_flows

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent

FEATURE_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
]
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return load_flows(FIXTURES_DIR, pattern="sample_flows.csv")


@pytest.fixture()
def config() -> dict[str, Any]:
    return {
        "features": {
            "numeric_columns": FEATURE_COLUMNS,
            "label_column": LABEL_COLUMN,
            "benign_label": BENIGN_LABEL,
        },
        "split": {"test_size": 0.3, "random_state": 42},
        "scaler": {"with_mean": True, "with_std": True},
    }


# --- load_config -------------------------------------------------------


def test_load_config_reads_real_project_config_yaml() -> None:
    config_path = REPO_ROOT / "config.yaml"

    loaded = load_config(config_path)

    assert loaded["features"]["label_column"] == "Label"
    assert loaded["features"]["benign_label"] == "BENIGN"
    assert isinstance(loaded["features"]["numeric_columns"], list)
    assert len(loaded["features"]["numeric_columns"]) > 0


def test_config_yaml_feature_list_excludes_label_column() -> None:
    """The feature list in config.yaml must never include the label column."""
    loaded = load_config(REPO_ROOT / "config.yaml")

    assert loaded["features"]["label_column"] not in loaded["features"]["numeric_columns"]


# --- split_benign_and_mixed ---------------------------------------------


def test_split_train_set_contains_only_benign_rows(sample_df: pd.DataFrame) -> None:
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL, test_size=0.3
    )

    assert len(split.train_df) > 0
    assert (split.train_df[LABEL_COLUMN] == BENIGN_LABEL).all()


def test_split_test_set_is_mixed_benign_and_attack(sample_df: pd.DataFrame) -> None:
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL, test_size=0.3
    )

    test_labels = set(split.test_df[LABEL_COLUMN])
    assert BENIGN_LABEL in test_labels
    assert test_labels - {BENIGN_LABEL}  # at least one attack category present


def test_split_all_attack_rows_go_to_test_set(sample_df: pd.DataFrame) -> None:
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL, test_size=0.3
    )

    n_attack_in_df = (sample_df[LABEL_COLUMN] != BENIGN_LABEL).sum()
    n_attack_in_test = (split.test_df[LABEL_COLUMN] != BENIGN_LABEL).sum()
    assert n_attack_in_test == n_attack_in_df
    assert not (split.train_df[LABEL_COLUMN] != BENIGN_LABEL).any()


def test_split_row_counts_preserved(sample_df: pd.DataFrame) -> None:
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL, test_size=0.3
    )

    assert len(split.train_df) + len(split.test_df) == len(sample_df)


def test_split_raises_for_missing_label_column(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        split_benign_and_mixed(
            sample_df, label_column="NotAColumn", benign_label=BENIGN_LABEL
        )


def test_split_with_fewer_than_two_benign_rows_uses_all_for_training(
    sample_df: pd.DataFrame,
) -> None:
    """Boundary case: too few BENIGN rows to hold any out for testing.

    All available BENIGN rows go to training instead of raising or crashing
    inside train_test_split (which requires at least 2 samples per split).
    """
    tiny_df = pd.concat(
        [
            sample_df[sample_df[LABEL_COLUMN] == BENIGN_LABEL].iloc[:1],
            sample_df[sample_df[LABEL_COLUMN] != BENIGN_LABEL],
        ],
        ignore_index=True,
    )

    split = split_benign_and_mixed(
        tiny_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL, test_size=0.3
    )

    assert len(split.train_df) == 1
    assert (split.train_df[LABEL_COLUMN] == BENIGN_LABEL).all()
    # No benign rows held out for test since there weren't enough to split.
    assert (split.test_df[LABEL_COLUMN] != BENIGN_LABEL).all()


def test_split_raises_clear_error_when_no_benign_rows(sample_df: pd.DataFrame) -> None:
    """All-attack input: a benign-only training set is impossible, so the
    pipeline must fail with a purpose-built message, not a downstream crash.
    """
    attack_only = sample_df[sample_df[LABEL_COLUMN] != BENIGN_LABEL]

    with pytest.raises(ValueError, match="No BENIGN rows"):
        split_benign_and_mixed(
            attack_only, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL
        )


# --- sanitize_flows -------------------------------------------------------


def test_sanitize_flows_drops_rows_with_non_finite_feature_values(
    sample_df: pd.DataFrame,
) -> None:
    """CICIDS2017 zero-duration flows carry Infinity/NaN in rate columns;
    such rows must be dropped so StandardScaler never sees non-finite input.
    """
    dirty = sample_df.copy()
    dirty.loc[dirty.index[0], "Flow Bytes/s"] = np.inf
    dirty.loc[dirty.index[1], "Flow Packets/s"] = -np.inf
    dirty.loc[dirty.index[2], "Flow IAT Mean"] = np.nan

    cleaned = sanitize_flows(dirty, FEATURE_COLUMNS)

    assert len(cleaned) == len(sample_df) - 3
    assert np.isfinite(cleaned[FEATURE_COLUMNS].to_numpy()).all()


def test_sanitize_flows_keeps_clean_data_unchanged(sample_df: pd.DataFrame) -> None:
    cleaned = sanitize_flows(sample_df, FEATURE_COLUMNS)

    assert len(cleaned) == len(sample_df)


def test_sanitize_flows_raises_for_unknown_strategy(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown cleaning strategy"):
        sanitize_flows(sample_df, FEATURE_COLUMNS, strategy="impute")


def test_sanitize_flows_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    """Guards the caller's frame: dropping rows must never touch the original.

    Pinned explicitly because the implementation trades a whole-frame `.copy()`
    for a boolean mask -- cheap on 79 columns x 2.8M rows, but only safe if
    non-mutation is enforced rather than assumed.
    """
    dirty = sample_df.copy()
    dirty.loc[0, "Flow Bytes/s"] = np.inf
    before = dirty.copy()

    sanitize_flows(dirty, FEATURE_COLUMNS)

    pd.testing.assert_frame_equal(dirty, before)


# --- required_columns ------------------------------------------------------


def test_required_columns_is_numeric_columns_plus_label(config: dict[str, Any]) -> None:
    cfg = Config.model_validate(config)

    assert required_columns(cfg) == [*FEATURE_COLUMNS, LABEL_COLUMN]


def test_required_columns_deduplicates_preserving_order() -> None:
    """A config that repeats the label among the features still yields a clean
    whitelist; reporting that mistake stays `select_feature_matrix`'s job.
    """
    cfg = Config.model_validate(
        {
            "features": {
                "numeric_columns": ["Flow Duration", "Label", "Flow Duration"],
                "label_column": "Label",
                "benign_label": "BENIGN",
            }
        }
    )

    assert required_columns(cfg) == ["Flow Duration", "Label"]


def test_prepare_dataset_drops_non_finite_rows_before_scaling(
    sample_df: pd.DataFrame, config: dict[str, Any]
) -> None:
    """Regression: without sanitization, StandardScaler raises ValueError on a
    non-finite value. prepare_dataset must drop the offending row and emit
    fully finite train/test matrices.
    """
    dirty = sample_df.copy()
    dirty.loc[dirty.index[0], "Flow Bytes/s"] = np.inf

    prepared = prepare_dataset(dirty, config)

    assert np.isfinite(prepared.x_train).all()
    assert np.isfinite(prepared.x_test).all()


# --- select_feature_matrix -----------------------------------------------


def test_select_feature_matrix_excludes_label_column(sample_df: pd.DataFrame) -> None:
    features = select_feature_matrix(sample_df, FEATURE_COLUMNS, LABEL_COLUMN)

    assert LABEL_COLUMN not in features.columns
    assert list(features.columns) == FEATURE_COLUMNS


def test_select_feature_matrix_raises_if_label_in_feature_list(
    sample_df: pd.DataFrame,
) -> None:
    tainted_feature_list = [*FEATURE_COLUMNS, LABEL_COLUMN]

    with pytest.raises(ValueError, match="must not be part of the feature list"):
        select_feature_matrix(sample_df, tainted_feature_list, LABEL_COLUMN)


def test_select_feature_matrix_raises_for_missing_columns(
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Missing expected feature columns"):
        select_feature_matrix(sample_df, ["Nonexistent Column"], LABEL_COLUMN)


# --- fit_scaler -----------------------------------------------------------


def test_fit_scaler_matches_manual_mean_and_std_of_training_data(
    sample_df: pd.DataFrame,
) -> None:
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL, test_size=0.3
    )
    train_features = select_feature_matrix(split.train_df, FEATURE_COLUMNS, LABEL_COLUMN)

    scaler = fit_scaler(train_features)

    expected_mean = train_features.to_numpy().mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-8)


def test_fit_scaler_is_unaffected_by_attack_outliers_in_full_dataset(
    sample_df: pd.DataFrame,
) -> None:
    """Even though the full dataset contains extreme DDoS/DoS Hulk outliers,
    a scaler fit only on the benign training split must not reflect them.
    """
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL, test_size=0.3
    )
    train_features = select_feature_matrix(split.train_df, FEATURE_COLUMNS, LABEL_COLUMN)
    full_features = select_feature_matrix(sample_df, FEATURE_COLUMNS, LABEL_COLUMN)

    scaler = fit_scaler(train_features)

    # "Flow Packets/s" has extreme DDoS outliers (tens of thousands) that would
    # massively inflate mean/std if the scaler had seen them.
    idx = FEATURE_COLUMNS.index("Flow Packets/s")
    full_mean = full_features.to_numpy()[:, idx].mean()
    assert scaler.mean_[idx] < full_mean


# --- prepare_dataset / no label leakage (non-negotiable) ------------------


def test_prepare_dataset_returns_expected_shapes(
    sample_df: pd.DataFrame, config: dict[str, Any]
) -> None:
    prepared = prepare_dataset(sample_df, config)

    assert isinstance(prepared, PreparedData)
    assert isinstance(prepared.x_train, np.ndarray)
    assert isinstance(prepared.x_test, np.ndarray)
    assert prepared.x_train.shape[1] == len(FEATURE_COLUMNS)
    assert prepared.x_test.shape[1] == len(FEATURE_COLUMNS)
    assert prepared.x_train.shape[0] > 0
    assert len(prepared.y_test) == prepared.x_test.shape[0]


def test_prepare_dataset_has_no_label_leakage(
    sample_df: pd.DataFrame, config: dict[str, Any]
) -> None:
    """The project's non-negotiable check: the training feature matrix must
    never contain, or be derived from, the ground-truth label column, and
    the benign-only training set must contain only BENIGN rows.
    """
    prepared = prepare_dataset(sample_df, config)

    # 1. The label column must never appear among the selected feature names.
    assert LABEL_COLUMN not in prepared.feature_names

    # 2. x_train must have exactly len(feature_names) columns -- no hidden
    #    extra column smuggling in label information.
    assert prepared.x_train.shape[1] == len(prepared.feature_names)

    # 3. There must be no y_train / training-labels artifact at all: training
    #    is unsupervised by construction, not just "labels present but unused".
    assert not hasattr(prepared, "y_train")

    # 4. Cross-check against the underlying split: the exact number of rows
    #    used for training can never exceed the number of BENIGN rows in the
    #    raw input, proving training data was benign-only at the source.
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL,
        test_size=config["split"]["test_size"], random_state=config["split"]["random_state"],
    )
    n_benign_in_source = (sample_df[LABEL_COLUMN] == BENIGN_LABEL).sum()
    assert prepared.x_train.shape[0] == len(split.train_df)
    assert prepared.x_train.shape[0] <= n_benign_in_source
    assert (split.train_df[LABEL_COLUMN] == BENIGN_LABEL).all()

    # 5. Ground-truth labels ARE present, but only attached to the test set.
    assert set(prepared.y_test) - {BENIGN_LABEL}  # at least one attack label
    assert BENIGN_LABEL in set(prepared.y_test)


def test_prepare_dataset_raises_if_config_feature_list_includes_label(
    sample_df: pd.DataFrame, config: dict[str, Any]
) -> None:
    tainted_config = {
        "features": {
            "numeric_columns": [*FEATURE_COLUMNS, LABEL_COLUMN],
            "label_column": LABEL_COLUMN,
            "benign_label": BENIGN_LABEL,
        },
        "split": config["split"],
        "scaler": config["scaler"],
    }

    with pytest.raises(ValueError, match="must not be part of the feature list"):
        prepare_dataset(sample_df, tainted_config)


def test_prepare_dataset_scaler_is_fit_on_train_only(
    sample_df: pd.DataFrame, config: dict[str, Any]
) -> None:
    prepared = prepare_dataset(sample_df, config)

    assert isinstance(prepared.scaler, StandardScaler)
    # Reconstruct the train feature matrix independently and confirm the
    # returned scaler's statistics match it exactly (not the full/mixed data).
    split = split_benign_and_mixed(
        sample_df, label_column=LABEL_COLUMN, benign_label=BENIGN_LABEL,
        test_size=config["split"]["test_size"], random_state=config["split"]["random_state"],
    )
    train_features = select_feature_matrix(split.train_df, FEATURE_COLUMNS, LABEL_COLUMN)
    expected_mean = train_features.to_numpy().mean(axis=0)

    np.testing.assert_allclose(prepared.scaler.mean_, expected_mean, rtol=1e-8)


def test_prepare_dataset_raises_clear_error_when_no_benign_rows(
    config: dict[str, Any],
) -> None:
    """End-to-end: all-attack input surfaces the clear 'No BENIGN rows' error
    rather than an opaque StandardScaler failure on an empty training matrix.
    """
    sample_df = load_flows(FIXTURES_DIR, pattern="sample_flows.csv")
    attack_only = sample_df[sample_df[LABEL_COLUMN] != BENIGN_LABEL]

    with pytest.raises(ValueError, match="No BENIGN rows"):
        prepare_dataset(attack_only, config)


def test_prepare_dataset_handles_all_benign_input(config: dict[str, Any]) -> None:
    """All-benign input must not crash; y_test simply has no positive class."""
    sample_df = load_flows(FIXTURES_DIR, pattern="sample_flows.csv")
    benign_only = sample_df[sample_df[LABEL_COLUMN] == BENIGN_LABEL]

    prepared = prepare_dataset(benign_only, config)

    assert prepared.x_train.shape[0] > 0
    assert np.isfinite(prepared.x_test).all()
    assert set(prepared.y_test) == {BENIGN_LABEL}


# --- Config (typed config.yaml view) --------------------------------------


def test_config_applies_defaults_when_optional_sections_omitted() -> None:
    """A minimal config (features only) fills split/scaler/cleaning from the
    single-source defaults, so callers never re-declare fallback literals.
    """
    cfg = Config.model_validate(
        {
            "features": {
                "numeric_columns": FEATURE_COLUMNS,
                "label_column": LABEL_COLUMN,
                "benign_label": BENIGN_LABEL,
            }
        }
    )

    assert cfg.split.test_size == 0.3
    assert cfg.split.random_state == 42
    assert cfg.scaler.with_mean is True
    assert cfg.scaler.with_std is True
    assert cfg.cleaning.invalid_values == "drop"


def test_config_ignores_unknown_top_level_keys() -> None:
    """Unknown sections (e.g. `data`, future model hyperparameters) are ignored
    so config.yaml can carry keys this model does not yet describe.
    """
    cfg = Config.model_validate(
        {
            "data": {"raw_dir": "data/raw"},
            "features": {
                "numeric_columns": FEATURE_COLUMNS,
                "label_column": LABEL_COLUMN,
                "benign_label": BENIGN_LABEL,
            },
        }
    )

    assert cfg.features.label_column == LABEL_COLUMN


def test_prepare_dataset_accepts_validated_config_instance(
    sample_df: pd.DataFrame,
) -> None:
    """prepare_dataset works with an already-validated Config, not just a dict."""
    cfg = Config.model_validate(load_config(REPO_ROOT / "config.yaml"))

    prepared = prepare_dataset(sample_df, cfg)

    assert prepared.x_train.shape[1] == len(cfg.features.numeric_columns)
