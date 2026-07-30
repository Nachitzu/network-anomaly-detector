"""Unit tests for src.models.autoencoder_model.

Uses the tiny synthetic fixture at tests/fixtures/sample_flows.csv (24 rows:
14 BENIGN, 4 DoS Hulk, 3 PortScan, 3 DDoS) via the existing
`src.data.feature_engineering.prepare_dataset` pipeline -- never the full
CICIDS2017 dataset -- so this suite stays fast, CPU-only, and requires no
external data or GPU.

Non-negotiable checks exercised here (README sections 2 and 5.3):
- `AutoencoderDetector` implements `BaseDetector` exactly (no labels in
  `fit`; it trains only on `X_benign`).
- The anomaly threshold is a percentile of reconstruction error measured on
  a held-out BENIGN validation split carved out of `X_benign` inside `fit`,
  and is never affected by attack data seen later via `score`.
- Known attack rows in the fixture receive HIGHER anomaly (reconstruction
  error) scores, on average, than benign rows.
- Training is deterministic on CPU given a fixed `random_state`.
- `top_contributing_features` ranks by per-feature squared reconstruction
  error and guards against being called before `fit`, non-positive `k`, and
  non-finite input.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.feature_engineering import (
    AutoencoderConfig,
    Config,
    PreparedData,
    load_config,
    prepare_dataset,
)
from src.data.loader import load_flows
from src.models.autoencoder_model import AutoencoderDetector
from src.models.base_detector import BaseDetector

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

# Small, fast, but numerous-enough epochs for the training loop to converge
# reliably to a low-error reconstruction of the (tiny) benign fixture, so the
# attack-vs-benign score comparison isn't flaky run to run.
_AUTOENCODER_HYPERPARAMS: dict[str, Any] = {
    "hidden_dims": [8, 4],
    "epochs": 200,
    "learning_rate": 0.01,
    "batch_size": 4,
    "threshold_percentile": 90.0,
    "validation_fraction": 0.2,
    "random_state": 42,
}


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return load_flows(FIXTURES_DIR, pattern="sample_flows.csv")


@pytest.fixture()
def config_dict() -> dict[str, Any]:
    return {
        "features": {
            "numeric_columns": FEATURE_COLUMNS,
            "label_column": LABEL_COLUMN,
            "benign_label": BENIGN_LABEL,
        },
        "split": {"test_size": 0.3, "random_state": 42},
        "scaler": {"with_mean": True, "with_std": True},
        "models": {"autoencoder": _AUTOENCODER_HYPERPARAMS},
    }


@pytest.fixture()
def prepared(sample_df: pd.DataFrame, config_dict: dict[str, Any]) -> PreparedData:
    return prepare_dataset(sample_df, config_dict)


@pytest.fixture()
def autoencoder_config() -> AutoencoderConfig:
    return AutoencoderConfig(**_AUTOENCODER_HYPERPARAMS)


# --- BaseDetector interface conformance ------------------------------------


def test_is_a_base_detector_subclass() -> None:
    assert issubclass(AutoencoderDetector, BaseDetector)


def test_instantiation_succeeds_implementing_every_abstract_method() -> None:
    """If any abstract method were missing, instantiation itself would raise
    TypeError -- this is the cheapest possible "implements the interface"
    check.
    """
    detector = AutoencoderDetector()

    assert isinstance(detector, BaseDetector)
    assert detector.name == "autoencoder"


def test_fit_signature_takes_only_the_feature_matrix_no_labels() -> None:
    sig = inspect.signature(AutoencoderDetector.fit)
    assert list(sig.parameters) == ["self", "X_benign"]


# --- fit() trains without labels, on CPU, deterministically -----------------


def test_fit_accepts_only_x_and_trains_without_error(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)

    detector.fit(prepared.x_train)  # must not raise; no y argument exists to pass

    assert detector._model is not None
    assert detector._threshold is not None


def test_training_is_deterministic_given_seed(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    """CPU training with a fixed random_state must be reproducible: two
    independently fitted detectors on the same benign data yield the same
    threshold and the same scores on the same test batch.
    """
    detector_a = AutoencoderDetector(autoencoder_config)
    detector_b = AutoencoderDetector(autoencoder_config)

    detector_a.fit(prepared.x_train)
    detector_b.fit(prepared.x_train)

    assert detector_a.threshold == pytest.approx(detector_b.threshold)
    np.testing.assert_allclose(
        detector_a.score(prepared.x_test), detector_b.score(prepared.x_test)
    )


# --- score() ------------------------------------------------------------


def test_score_returns_one_value_per_row(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    scores = detector.score(prepared.x_test)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (prepared.x_test.shape[0],)


def test_score_raises_before_fit() -> None:
    detector = AutoencoderDetector()

    with pytest.raises(RuntimeError):
        detector.score(np.zeros((1, len(FEATURE_COLUMNS))))


def test_attack_rows_score_higher_than_benign_rows_on_average(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    """The project's core sanity check for the Autoencoder (README section
    5.3): fit on benign-only x_train, score the mixed x_test, and confirm
    known attack rows (DoS Hulk / PortScan / DDoS in the fixture) are flagged
    as more anomalous, on average, than held-out benign rows.

    The 24-row fixture is tiny, so this compares group means (not a strict
    per-row guarantee) with a fixed seed and enough epochs for the network to
    reliably converge on the benign reconstruction -- this keeps the
    assertion stable rather than flaky.
    """
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    scores = detector.score(prepared.x_test)
    is_benign = (prepared.y_test == BENIGN_LABEL).to_numpy()

    benign_scores = scores[is_benign]
    attack_scores = scores[~is_benign]

    assert len(benign_scores) > 0
    assert len(attack_scores) > 0
    assert attack_scores.mean() > benign_scores.mean()


def test_is_anomaly_matches_score_above_threshold(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    scores = detector.score(prepared.x_test)
    expected = scores > detector.threshold

    np.testing.assert_array_equal(detector.is_anomaly(prepared.x_test), expected)


# --- threshold: benign-validation-only, never attack data -------------------


def test_threshold_is_percentile_of_benign_validation_errors_only(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    assert detector._validation_errors is not None
    expected_threshold = np.percentile(
        detector._validation_errors, autoencoder_config.threshold_percentile
    )
    assert detector.threshold == pytest.approx(expected_threshold)


def test_threshold_unaffected_by_scoring_attack_data(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    """The threshold is fixed at fit() time from the benign validation split
    carved out of X_benign. Later calling score() on a mixed batch containing
    attack rows (prepared.x_test) must never mutate it -- proving no attack
    data can leak into the threshold after the fact.
    """
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)
    threshold_before = detector.threshold

    is_benign = (prepared.y_test == BENIGN_LABEL).to_numpy()
    assert (~is_benign).any()  # sanity: x_test really does contain attack rows
    detector.score(prepared.x_test)  # mixed benign + attack; must not mutate state

    assert detector.threshold == threshold_before


def test_threshold_raises_before_fit() -> None:
    detector = AutoencoderDetector()

    with pytest.raises(RuntimeError):
        _ = detector.threshold


# --- hyperparameters sourced from config.yaml -------------------------------


def test_hyperparameters_are_read_from_config_not_hardcoded(
    config_dict: dict[str, Any],
) -> None:
    cfg = Config.model_validate(config_dict)
    detector = AutoencoderDetector(cfg.models.autoencoder)

    assert detector._config.hidden_dims == [8, 4]
    assert detector._config.epochs == 200
    assert detector._config.threshold_percentile == 90.0


def test_default_config_is_used_when_none_provided() -> None:
    detector = AutoencoderDetector()
    defaults = AutoencoderConfig()

    assert detector._config.hidden_dims == defaults.hidden_dims
    assert detector._config.epochs == defaults.epochs
    assert detector._config.threshold_percentile == defaults.threshold_percentile


def test_real_config_yaml_declares_autoencoder_hyperparameters() -> None:
    loaded = load_config(REPO_ROOT / "config.yaml")

    assert "models" in loaded
    section = loaded["models"]["autoencoder"]
    assert "hidden_dims" in section
    assert "epochs" in section
    assert "threshold_percentile" in section
    assert "validation_fraction" in section


def test_real_config_yaml_validates_into_typed_config() -> None:
    cfg = Config.model_validate(load_config(REPO_ROOT / "config.yaml"))

    assert cfg.models.autoencoder.epochs > 0
    detector = AutoencoderDetector(cfg.models.autoencoder)
    assert detector._config.hidden_dims == cfg.models.autoencoder.hidden_dims


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"threshold_percentile": 0},
        {"threshold_percentile": 100},
        {"validation_fraction": 0},
        {"validation_fraction": 1},
        {"batch_size": 0},
        {"learning_rate": 0},
    ],
)
def test_autoencoder_config_rejects_out_of_range_values(kwargs: dict[str, Any]) -> None:
    """Bad hyperparameters must fail fast at config validation, not deep
    inside the training loop.
    """
    with pytest.raises(ValidationError):
        AutoencoderConfig(**kwargs)


@pytest.mark.parametrize("bad_dims", [[], [0, 4], [16, -8]])
def test_autoencoder_config_rejects_invalid_hidden_dims(bad_dims: list[int]) -> None:
    """Empty or non-positive layer widths would otherwise yield a cryptic
    optimizer error (empty list) or a silently degenerate model (zero width).
    """
    with pytest.raises(ValidationError):
        AutoencoderConfig(hidden_dims=bad_dims)


# --- top_contributing_features() (reconstruction-error explainability) -----


def test_top_contributing_features_returns_exactly_k_known_feature_names(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    top = detector.top_contributing_features(prepared.x_test[0], prepared.feature_names, k=5)

    assert len(top) == 5
    assert len(set(top)) == 5  # no duplicates
    assert all(name in prepared.feature_names for name in top)


def test_top_contributing_features_ranks_the_most_extreme_feature_first(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    feature_names = prepared.feature_names
    extreme_idx = feature_names.index("Flow Packets/s")
    # A typical (near-zero, scaled) benign row, with one feature pushed to an
    # extreme the network never saw during training: its squared
    # reconstruction error dwarfs every other (well-reconstructed) feature.
    x = np.zeros(len(feature_names))
    x[extreme_idx] = 1000.0

    top = detector.top_contributing_features(x, feature_names, k=3)

    assert top[0] == "Flow Packets/s"
    assert len(top) == 3


def test_top_contributing_features_raises_before_fit() -> None:
    detector = AutoencoderDetector()

    with pytest.raises(RuntimeError):
        detector.top_contributing_features(np.zeros(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, k=3)


@pytest.mark.parametrize("bad_k", [0, -1])
def test_top_contributing_features_rejects_non_positive_k(
    bad_k: int, prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    with pytest.raises(ValueError, match="k must be positive"):
        detector.top_contributing_features(prepared.x_test[0], prepared.feature_names, k=bad_k)


def test_top_contributing_features_rejects_non_finite_x(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    x = prepared.x_test[0].copy()
    x[0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        detector.top_contributing_features(x, prepared.feature_names, k=2)


def test_top_contributing_features_caps_k_at_feature_count(
    prepared: PreparedData, autoencoder_config: AutoencoderConfig
) -> None:
    detector = AutoencoderDetector(autoencoder_config)
    detector.fit(prepared.x_train)

    top = detector.top_contributing_features(
        prepared.x_test[0], prepared.feature_names, k=1000
    )

    assert len(top) == len(prepared.feature_names)
