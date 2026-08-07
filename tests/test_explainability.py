"""Unit tests for src.evaluation.explainability.

Uses the tiny synthetic fixture at tests/fixtures/sample_flows.csv via the
existing `src.data.feature_engineering.prepare_dataset` pipeline -- never the
full CICIDS2017 dataset -- so this suite stays fast and CPU-only.

Non-negotiable checks exercised here (README sections 5.1, 5.3, 5.5):
- `explain_flagged_anomalies` maps ONLY flagged (`is_anomaly == True`) rows to
  their top-k contributing features, for BOTH detectors, through the shared
  `BaseDetector` interface only.
- Each flagged row gets exactly `min(k, n_features)` feature names, drawn from
  `feature_names`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import Config, PreparedData, prepare_dataset
from src.data.loader import load_flows
from src.evaluation.explainability import explain_flagged_anomalies
from src.models.autoencoder_model import AutoencoderDetector
from src.models.base_detector import BaseDetector
from src.models.isolation_forest_model import IsolationForestDetector

FIXTURES_DIR = Path(__file__).parent / "fixtures"

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

# Aggressive-but-fast hyperparameters: a low threshold_percentile guarantees
# at least some test rows are flagged, so this suite doesn't depend on
# borderline model behavior to exercise the "flagged" code path.
_AUTOENCODER_HYPERPARAMS: dict[str, Any] = {
    "hidden_dims": [8, 4],
    "epochs": 20,
    "learning_rate": 0.01,
    "batch_size": 4,
    "threshold_percentile": 50.0,
    "validation_fraction": 0.2,
    "random_state": 42,
}
_ISOLATION_FOREST_HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 37,
    "contamination": 0.2,
    "max_samples": "auto",
    "threshold_percentile": 50.0,
    "random_state": 7,
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
        "models": {
            "isolation_forest": _ISOLATION_FOREST_HYPERPARAMS,
            "autoencoder": _AUTOENCODER_HYPERPARAMS,
        },
    }


@pytest.fixture()
def prepared(sample_df: pd.DataFrame, config_dict: dict[str, Any]) -> PreparedData:
    return prepare_dataset(sample_df, config_dict)


@pytest.fixture()
def fitted_isolation_forest(prepared: PreparedData, config_dict: dict[str, Any]) -> BaseDetector:
    cfg = Config.model_validate(config_dict)
    detector: BaseDetector = IsolationForestDetector(cfg.models.isolation_forest)
    detector.fit(prepared.x_train)
    return detector


@pytest.fixture()
def fitted_autoencoder(prepared: PreparedData, config_dict: dict[str, Any]) -> BaseDetector:
    cfg = Config.model_validate(config_dict)
    detector: BaseDetector = AutoencoderDetector(cfg.models.autoencoder)
    detector.fit(prepared.x_train)
    return detector


@pytest.fixture(params=["isolation_forest", "autoencoder"])
def fitted_detector(
    request: pytest.FixtureRequest,
    fitted_isolation_forest: BaseDetector,
    fitted_autoencoder: BaseDetector,
) -> BaseDetector:
    return fitted_isolation_forest if request.param == "isolation_forest" else fitted_autoencoder


# --- explain_flagged_anomalies(): only flagged rows, exactly min(k, n) names -


def test_explain_flagged_anomalies_only_includes_flagged_rows(
    fitted_detector: BaseDetector, prepared: PreparedData
) -> None:
    explanations = explain_flagged_anomalies(
        fitted_detector, prepared.x_test, prepared.feature_names, k=3
    )

    flags = fitted_detector.is_anomaly(prepared.x_test)
    expected_indices = set(np.flatnonzero(flags).tolist())

    assert set(explanations.keys()) == expected_indices
    assert len(expected_indices) > 0  # sanity: the low threshold flags >= 1 row


def test_explain_flagged_anomalies_returns_exactly_min_k_features(
    fitted_detector: BaseDetector, prepared: PreparedData
) -> None:
    k = 3
    explanations = explain_flagged_anomalies(
        fitted_detector, prepared.x_test, prepared.feature_names, k=k
    )

    expected_count = min(k, len(prepared.feature_names))
    for feature_list in explanations.values():
        assert len(feature_list) == expected_count
        assert len(set(feature_list)) == expected_count  # no duplicates
        assert all(name in prepared.feature_names for name in feature_list)


def test_explain_flagged_anomalies_caps_k_at_feature_count(
    fitted_detector: BaseDetector, prepared: PreparedData
) -> None:
    explanations = explain_flagged_anomalies(
        fitted_detector, prepared.x_test, prepared.feature_names, k=1000
    )

    assert len(explanations) > 0
    for feature_list in explanations.values():
        assert len(feature_list) == len(prepared.feature_names)


def test_explain_flagged_anomalies_empty_when_nothing_flagged(
    fitted_detector: BaseDetector, prepared: PreparedData
) -> None:
    """An all-benign-looking batch (the benign training data itself, already
    well-reconstructed / inlying) should flag few or no rows; whatever rows
    ARE flagged must still satisfy the same top-k contract, and an empty
    result must be a valid, non-erroring dict.
    """
    explanations = explain_flagged_anomalies(
        fitted_detector, prepared.x_train, prepared.feature_names, k=3
    )

    assert isinstance(explanations, dict)
    for row_index, feature_list in explanations.items():
        assert isinstance(row_index, int)
        assert len(feature_list) == min(3, len(prepared.feature_names))


def test_explain_flagged_anomalies_raises_before_fit() -> None:
    detector = IsolationForestDetector()

    with pytest.raises(RuntimeError):
        explain_flagged_anomalies(
            detector, np.zeros((2, len(FEATURE_COLUMNS))), FEATURE_COLUMNS, k=3
        )


class _SpyDetector(BaseDetector):
    """Records which explainability entry point gets used."""

    name = "spy"

    def __init__(self) -> None:
        self.batch_calls = 0
        self.per_row_calls = 0

    def fit(self, X_benign: np.ndarray) -> None:
        return None

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.ones(X.shape[0])

    @property
    def threshold(self) -> float:
        return 0.5

    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:
        self.per_row_calls += 1
        return feature_names[:k]

    def top_contributing_features_batch(
        self, X: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[list[str]]:
        self.batch_calls += 1
        return [feature_names[:k] for _ in range(X.shape[0])]


def test_explain_flagged_anomalies_asks_for_every_row_at_once() -> None:
    """Regression guard: explanations used to be a Python loop over flagged
    rows, one call each -- unusable on the ~250k a full run flags.
    """
    detector = _SpyDetector()
    X = np.zeros((25, len(FEATURE_COLUMNS)))

    explanations = explain_flagged_anomalies(detector, X, FEATURE_COLUMNS, k=3)

    assert len(explanations) == 25  # every row scores above the threshold
    assert detector.batch_calls == 1
    assert detector.per_row_calls == 0
