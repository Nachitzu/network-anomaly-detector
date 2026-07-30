"""Unit tests for src.models.isolation_forest_model.

Uses the tiny synthetic fixture at tests/fixtures/sample_flows.csv (24 rows:
14 BENIGN, 4 DoS Hulk, 3 PortScan, 3 DDoS) via the existing
`src.data.feature_engineering.prepare_dataset` pipeline — never the full
CICIDS2017 dataset — so this suite stays fast and requires no external data.

Non-negotiable checks exercised here (README sections 2 and 5.2):
- `IsolationForestDetector` implements `BaseDetector` exactly (no labels in
  `fit`).
- Known attack rows in the fixture receive HIGHER anomaly scores, on
  average, than benign rows.
- Hyperparameters (`n_estimators`, `contamination`, `max_samples`) come from
  `config.yaml` / the typed `Config` model, never hardcoded.
- `top_contributing_features` ranks by |z-score| against the benign training
  distribution and guards against zero-variance features.
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
    Config,
    IsolationForestConfig,
    PreparedData,
    load_config,
    prepare_dataset,
)
from src.data.loader import load_flows
from src.models.base_detector import BaseDetector
from src.models.isolation_forest_model import IsolationForestDetector

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
def config_dict() -> dict[str, Any]:
    return {
        "features": {
            "numeric_columns": FEATURE_COLUMNS,
            "label_column": LABEL_COLUMN,
            "benign_label": BENIGN_LABEL,
        },
        "split": {"test_size": 0.3, "random_state": 42},
        "scaler": {"with_mean": True, "with_std": True},
        "models": {
            "isolation_forest": {
                "n_estimators": 37,
                "contamination": 0.2,
                "max_samples": "auto",
                "random_state": 7,
            }
        },
    }


@pytest.fixture()
def prepared(sample_df: pd.DataFrame, config_dict: dict[str, Any]) -> PreparedData:
    return prepare_dataset(sample_df, config_dict)


# --- BaseDetector interface conformance ------------------------------------


def test_is_a_base_detector_subclass() -> None:
    assert issubclass(IsolationForestDetector, BaseDetector)


def test_instantiation_succeeds_implementing_every_abstract_method() -> None:
    """If any abstract method were missing, instantiation itself would raise
    TypeError -- this is the cheapest possible "implements the interface"
    check.
    """
    detector = IsolationForestDetector()

    assert isinstance(detector, BaseDetector)
    assert detector.name == "isolation_forest"


def test_fit_signature_takes_only_the_feature_matrix_no_labels() -> None:
    sig = inspect.signature(IsolationForestDetector.fit)
    assert list(sig.parameters) == ["self", "X_benign"]


# --- fit() trains without labels --------------------------------------------


def test_fit_accepts_only_x_and_trains_without_error(prepared: PreparedData) -> None:
    detector = IsolationForestDetector()

    detector.fit(prepared.x_train)  # must not raise; no y argument exists to pass

    assert detector._train_mean is not None
    assert detector._train_std is not None
    assert detector._train_mean.shape == (len(prepared.feature_names),)


# --- score() ----------------------------------------------------------------


def test_score_returns_one_value_per_row(prepared: PreparedData) -> None:
    detector = IsolationForestDetector()
    detector.fit(prepared.x_train)

    scores = detector.score(prepared.x_test)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (prepared.x_test.shape[0],)


def test_attack_rows_score_higher_than_benign_rows_on_average(
    prepared: PreparedData,
) -> None:
    """The project's core sanity check for Isolation Forest (README section
    5.2): fit on benign-only x_train, score the mixed x_test, and confirm
    known attack rows (DoS Hulk / PortScan / DDoS in the fixture) are flagged
    as more anomalous, on average, than held-out benign rows.
    """
    detector = IsolationForestDetector()
    detector.fit(prepared.x_train)

    scores = detector.score(prepared.x_test)
    is_benign = (prepared.y_test == BENIGN_LABEL).to_numpy()

    benign_scores = scores[is_benign]
    attack_scores = scores[~is_benign]

    assert len(benign_scores) > 0
    assert len(attack_scores) > 0
    assert attack_scores.mean() > benign_scores.mean()


# --- hyperparameters sourced from config.yaml -------------------------------


def test_hyperparameters_are_read_from_config_not_hardcoded(
    config_dict: dict[str, Any],
) -> None:
    cfg = Config.model_validate(config_dict)
    detector = IsolationForestDetector(cfg.models.isolation_forest)

    assert detector._estimator.n_estimators == 37
    assert detector._estimator.contamination == 0.2
    assert detector._estimator.max_samples == "auto"
    assert detector._estimator.random_state == 7


def test_default_config_is_used_when_none_provided() -> None:
    detector = IsolationForestDetector()
    defaults = IsolationForestConfig()

    assert detector._estimator.n_estimators == defaults.n_estimators
    assert detector._estimator.contamination == defaults.contamination


def test_real_config_yaml_declares_isolation_forest_hyperparameters() -> None:
    loaded = load_config(REPO_ROOT / "config.yaml")

    assert "models" in loaded
    section = loaded["models"]["isolation_forest"]
    assert "n_estimators" in section
    assert "contamination" in section
    assert "max_samples" in section


def test_real_config_yaml_validates_into_typed_config() -> None:
    cfg = Config.model_validate(load_config(REPO_ROOT / "config.yaml"))

    assert cfg.models.isolation_forest.n_estimators > 0
    detector = IsolationForestDetector(cfg.models.isolation_forest)
    assert detector._estimator.n_estimators == cfg.models.isolation_forest.n_estimators


# --- top_contributing_features() (z-score explainability) ------------------


def test_top_contributing_features_returns_exactly_k_known_feature_names(
    prepared: PreparedData,
) -> None:
    detector = IsolationForestDetector()
    detector.fit(prepared.x_train)

    top = detector.top_contributing_features(prepared.x_test[0], prepared.feature_names, k=5)

    assert len(top) == 5
    assert len(set(top)) == 5  # no duplicates
    assert all(name in prepared.feature_names for name in top)


def test_top_contributing_features_ranks_the_most_extreme_feature_first(
    prepared: PreparedData,
) -> None:
    detector = IsolationForestDetector()
    detector.fit(prepared.x_train)

    feature_names = prepared.feature_names
    extreme_idx = feature_names.index("Flow Packets/s")
    # Start exactly at the training mean (z-score 0 everywhere), then make
    # one feature obviously extreme.
    x = detector._train_mean.copy()
    x[extreme_idx] += 1000.0

    top = detector.top_contributing_features(x, feature_names, k=3)

    assert top[0] == "Flow Packets/s"
    assert len(top) == 3


def test_top_contributing_features_raises_before_fit() -> None:
    detector = IsolationForestDetector()

    with pytest.raises(RuntimeError):
        detector.top_contributing_features(np.zeros(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, k=3)


def test_top_contributing_features_caps_k_at_feature_count(prepared: PreparedData) -> None:
    detector = IsolationForestDetector()
    detector.fit(prepared.x_train)

    top = detector.top_contributing_features(
        prepared.x_test[0], prepared.feature_names, k=1000
    )

    assert len(top) == len(prepared.feature_names)


# --- review hardening: input guards & config validation -------------------


def _fitted_detector() -> tuple[IsolationForestDetector, list[str]]:
    """A detector fitted on a tiny arbitrary benign array (guard tests only
    need a fitted state, not the full CICIDS2017 pipeline)."""
    rng = np.random.default_rng(0)
    detector = IsolationForestDetector()
    detector.fit(rng.normal(size=(20, 4)))
    return detector, ["a", "b", "c", "d"]


@pytest.mark.parametrize("bad_k", [0, -1])
def test_top_contributing_features_rejects_non_positive_k(bad_k: int) -> None:
    """Negative k previously slid through slice semantics and returned a
    scrambled subset instead of failing; it must raise now.
    """
    detector, names = _fitted_detector()

    with pytest.raises(ValueError, match="k must be positive"):
        detector.top_contributing_features(np.zeros(4), names, k=bad_k)


def test_top_contributing_features_rejects_non_finite_x() -> None:
    """Explainability runs on live, unsanitized input at inference time, so a
    non-finite feature must fail loudly rather than silently rank last.
    """
    detector, names = _fitted_detector()
    x = np.array([1.0, np.inf, 2.0, 3.0])

    with pytest.raises(ValueError, match="finite"):
        detector.top_contributing_features(x, names, k=2)


@pytest.mark.parametrize(
    "kwargs",
    [{"contamination": 0.0}, {"contamination": 0.9}, {"n_estimators": 0}],
)
def test_isolation_forest_config_rejects_out_of_range_values(
    kwargs: dict[str, Any],
) -> None:
    """Bad hyperparameters must fail fast at config validation, not deep inside
    IsolationForest.fit().
    """
    with pytest.raises(ValidationError):
        IsolationForestConfig(**kwargs)


def test_top_contributing_features_guards_against_zero_variance_feature() -> None:
    """A feature that is constant across all benign training rows has zero
    std; z-score computation must not raise or return NaN/Inf.
    """
    detector = IsolationForestDetector()
    X_benign = np.zeros((10, 3))
    X_benign[:, 0] = 5.0  # constant -> zero variance
    X_benign[:, 1] = np.linspace(0, 1, 10)
    X_benign[:, 2] = np.linspace(0, 100, 10)
    detector.fit(X_benign)

    # Feature 0 exactly matches its (constant) training value.
    x = np.array([5.0, 0.9, 99.0])

    top = detector.top_contributing_features(x, ["const", "b", "c"], k=3)

    assert set(top) == {"const", "b", "c"}
    assert top[-1] == "const"  # zero deviation on the zero-variance feature ranks last
