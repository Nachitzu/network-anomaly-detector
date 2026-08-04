"""Feature selection, scaling, and benign/attack train-test split for CICIDS2017 flows.

This is an unsupervised anomaly detection setup: models train ONLY on
BENIGN-labeled traffic. Ground-truth labels are used exclusively to build the
mixed evaluation test set and are never present in, nor derivable from, the
training feature matrix. `select_feature_matrix` and `prepare_dataset` enforce
this constraint programmatically; `tests/test_feature_engineering.py` verifies
it explicitly (the project's non-negotiable "no label leakage" check).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Single source of truth for the reproducible-split defaults, shared by the
# typed `SplitConfig` model and the `split_benign_and_mixed` signature so the
# two can never silently drift apart.
DEFAULT_TEST_SIZE = 0.3
DEFAULT_RANDOM_STATE = 42


class FeaturesConfig(BaseModel):
    """`features` section of config.yaml: which columns feed the models."""

    numeric_columns: list[str]
    label_column: str
    benign_label: str


class SplitConfig(BaseModel):
    """`split` section: benign train / mixed test split parameters."""

    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE


class ScalerConfig(BaseModel):
    """`scaler` section: StandardScaler parameters."""

    with_mean: bool = True
    with_std: bool = True


class CleaningConfig(BaseModel):
    """`cleaning` section: how non-finite feature values are handled."""

    invalid_values: str = "drop"


class IsolationForestConfig(BaseModel):
    """`models.isolation_forest` section: `sklearn.ensemble.IsolationForest`
    hyperparameters, sourced from config.yaml rather than hardcoded (README
    section 5.2).
    """

    n_estimators: int = Field(default=100, gt=0)
    # sklearn requires 0 < contamination <= 0.5; fail fast on a bad config value
    # rather than deep inside IsolationForest.fit().
    contamination: float = Field(default=0.1, gt=0, le=0.5)
    max_samples: int | float | str = "auto"
    random_state: int = DEFAULT_RANDOM_STATE


class AutoencoderConfig(BaseModel):
    """`models.autoencoder` section: the PyTorch feedforward encoder-decoder's
    hyperparameters, sourced from config.yaml rather than hardcoded (README
    section 5.3).
    """

    # Encoder hidden layer widths, e.g. [32, 16, 8]; the decoder mirrors these
    # in reverse. The input/output layer width is `input_dim`, inferred at fit
    # time from the training data rather than duplicated here.
    hidden_dims: list[int] = Field(default_factory=lambda: [32, 16, 8])
    epochs: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=8, gt=0)
    # Percentile of held-out BENIGN validation reconstruction error used as the
    # anomaly threshold -- NEVER computed from attack data (README section 5.3).
    threshold_percentile: float = Field(default=95.0, gt=0, lt=100)
    # Fraction of the benign training set held out to fit the threshold.
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = DEFAULT_RANDOM_STATE

    @field_validator("hidden_dims")
    @classmethod
    def _hidden_dims_positive_and_non_empty(cls, v: list[int]) -> list[int]:
        """Reject empty layers or non-positive widths, which would otherwise
        produce a cryptic optimizer error or a silently degenerate model."""
        if not v or any(dim <= 0 for dim in v):
            raise ValueError("hidden_dims must be a non-empty list of positive ints")
        return v


class ModelsConfig(BaseModel):
    """`models` section: hyperparameters for each `BaseDetector` implementation."""

    isolation_forest: IsolationForestConfig = Field(default_factory=IsolationForestConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)


class Config(BaseModel):
    """Typed, validated view of config.yaml.

    Centralizes every default in one place (fail-fast on malformed input) so
    downstream functions never re-declare `.get()` fallbacks. Unknown top-level
    keys (e.g. `data`) are ignored here.
    """

    features: FeaturesConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    scaler: ScalerConfig = Field(default_factory=ScalerConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the project's YAML configuration file (dataset paths, feature
    list, split ratios, and scaler parameters).
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def sanitize_flows(
    df: pd.DataFrame,
    feature_columns: list[str],
    strategy: str = "drop",
) -> pd.DataFrame:
    """Remove non-finite values from the selected feature columns.

    CICIDS2017 encodes zero-duration flows with ``Infinity``/``NaN`` in rate
    columns such as ``"Flow Bytes/s"``/``"Flow Packets/s"``. ``StandardScaler``
    cannot fit or transform non-finite input, so such rows must be handled
    before scaling. Only values in `feature_columns` are considered; non-finite
    values in unused columns are ignored so they never trigger a spurious drop.

    Args:
        df: Full flow DataFrame (label and extra columns allowed).
        feature_columns: Feature columns whose finiteness matters, from config.
        strategy: How to handle offending rows. Only ``"drop"`` is supported in
            v1 (discard rows with a non-finite feature value).

    Returns:
        A new DataFrame with a fresh 0..N index and no non-finite feature value.

    Raises:
        ValueError: if `strategy` is not a supported value.
    """
    if strategy != "drop":
        raise ValueError(f"Unknown cleaning strategy '{strategy}'; expected 'drop'")

    present = [col for col in feature_columns if col in df.columns]
    cleaned = df.copy()
    cleaned[present] = cleaned[present].replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=present)
    return cleaned.reset_index(drop=True)


@dataclass(frozen=True)
class BenignMixedSplit:
    """Result of splitting flows into a benign-only train set and a mixed test set."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame


def split_benign_and_mixed(
    df: pd.DataFrame,
    label_column: str,
    benign_label: str,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> BenignMixedSplit:
    """Split flows into a BENIGN-only training set and a mixed labeled test set.

    - `train_df`: exclusively BENIGN rows (a `(1 - test_size)` fraction of all
      BENIGN rows). This is the only set unsupervised models may train on.
    - `test_df`: the held-out BENIGN fraction PLUS every attack row, so
      evaluation always sees a realistic mix of normal and malicious traffic.

    Args:
        df: Full flow DataFrame, including the label column.
        label_column: Name of the ground-truth label column.
        benign_label: Exact label value identifying normal traffic.
        test_size: Fraction of BENIGN rows held out for the test set.
        random_state: Seed for the reproducible split.

    Returns:
        A `BenignMixedSplit` with `train_df` (benign-only) and `test_df` (mixed).

    Raises:
        ValueError: if `label_column` is not present in `df`, or if `df`
            contains no BENIGN rows (an unsupervised benign-only training set
            cannot be built without them).
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")

    benign_mask = df[label_column] == benign_label
    benign_df = df.loc[benign_mask]
    attack_df = df.loc[~benign_mask]

    if benign_df.empty:
        raise ValueError(
            f"No BENIGN rows (label '{benign_label}') found; cannot build a "
            f"benign-only training set"
        )

    if len(benign_df) < 2:
        # Too few benign rows to hold any out; use all of them for training.
        benign_train_df = benign_df
        benign_test_df = benign_df.iloc[0:0]
    else:
        benign_train_df, benign_test_df = train_test_split(
            benign_df,
            test_size=test_size,
            random_state=random_state,
        )

    train_df = benign_train_df.reset_index(drop=True)
    test_df = pd.concat([benign_test_df, attack_df], ignore_index=True)

    return BenignMixedSplit(train_df=train_df, test_df=test_df)


def select_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    """Select exactly the configured numeric feature columns from `df`.

    Acts as the enforcement point for the "no label leakage" constraint: it
    refuses to build a feature matrix that includes the label column, and
    never includes any column beyond the ones explicitly listed.

    Args:
        df: Source DataFrame (may include the label column and others).
        feature_columns: Exact list of numeric feature column names to keep,
            sourced from `config.yaml` (never hardcoded).
        label_column: Name of the ground-truth label column, checked for
            accidental inclusion in `feature_columns`.

    Returns:
        A DataFrame containing only `feature_columns`, in that order.

    Raises:
        ValueError: if `label_column` is present in `feature_columns`, or if
            any `feature_columns` entry is missing from `df`.
    """
    if label_column in feature_columns:
        raise ValueError(
            f"Label column '{label_column}' must not be part of the feature list"
        )
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    return df.loc[:, feature_columns].copy()


def fit_scaler(
    train_features: pd.DataFrame,
    with_mean: bool = True,
    with_std: bool = True,
) -> StandardScaler:
    """Fit a StandardScaler using ONLY the training feature matrix.

    Callers must never pass test/attack rows here — the scaler's mean/scale
    must reflect benign training traffic exclusively.
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    scaler.fit(train_features.to_numpy())
    return scaler


@dataclass(frozen=True)
class PreparedData:
    """Fully prepared, model-ready train/test arrays.

    `x_train` is intentionally the only training artifact: it is a plain
    `np.ndarray` with no label column and no accompanying `y_train`, because
    unsupervised training never uses ground-truth labels. `y_test` is the
    only place labels appear, reserved for evaluation.
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_test: pd.Series
    feature_names: list[str]
    scaler: StandardScaler


def prepare_dataset(df: pd.DataFrame, config: dict[str, Any] | Config) -> PreparedData:
    """Run the full feature engineering pipeline: raw flows -> model-ready arrays.

    Steps:
        1. Drop rows with non-finite values in any selected feature column.
        2. Split flows into a BENIGN-only train set and a mixed labeled test set.
        3. Select the configured numeric feature columns (label excluded).
        4. Fit a `StandardScaler` on the training features only.
        5. Transform both train and test feature matrices with that scaler.

    Note:
        `y_test` holds the raw string labels (e.g. "BENIGN", "DoS Hulk"); the
        evaluation stage must binarize it against `benign_label` before
        computing precision/recall/ROC-AUC.

    Args:
        df: Full flow DataFrame (e.g. from `src.data.loader.load_flows`).
        config: Parsed `config.yaml` contents (see `load_config`) as a dict, or
            an already-validated `Config` instance.

    Returns:
        A `PreparedData` instance ready to feed into `BaseDetector.fit`/`score`.
    """
    cfg = config if isinstance(config, Config) else Config.model_validate(config)

    feature_columns = cfg.features.numeric_columns
    label_column = cfg.features.label_column
    benign_label = cfg.features.benign_label

    df = sanitize_flows(df, feature_columns, strategy=cfg.cleaning.invalid_values)

    split = split_benign_and_mixed(
        df,
        label_column=label_column,
        benign_label=benign_label,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
    )

    train_features = select_feature_matrix(split.train_df, feature_columns, label_column)
    test_features = select_feature_matrix(split.test_df, feature_columns, label_column)

    scaler = fit_scaler(
        train_features,
        with_mean=cfg.scaler.with_mean,
        with_std=cfg.scaler.with_std,
    )

    x_train = scaler.transform(train_features.to_numpy())
    x_test = scaler.transform(test_features.to_numpy())
    y_test = split.test_df[label_column].reset_index(drop=True)

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=list(feature_columns),
        scaler=scaler,
    )
