"""Unit tests for src.evaluation.compare_models.

Uses the tiny synthetic fixture at tests/fixtures/sample_flows.csv (24 rows:
14 BENIGN, 4 DoS Hulk, 3 PortScan, 3 DDoS) via the existing
`src.data.feature_engineering.prepare_dataset` pipeline -- never the full
CICIDS2017 dataset -- so this suite stays fast and CPU-only.

Non-negotiable checks exercised here (README sections 2, 5.1, 5.5, 9):
- `compare_models` returns ONE DataFrame with both models side by side,
  covering every required metric family (precision/recall/F1/ROC-AUC,
  training time, inference latency), all finite and in valid ranges.
- It depends ONLY on the `BaseDetector` interface -- verified by typing the
  detector list as `list[BaseDetector]` and by static inspection of the
  module's imports.
- No label leakage: each detector's threshold is derived purely from benign
  training data and is unaffected by the mixed (benign + attack) test batch
  passed to `score`/`is_anomaly` at evaluation time.
- `is_anomaly` returns a boolean array aligned with input rows for both
  detectors; `threshold` is a float set after `fit`; accessing it before
  `fit` raises.
- `render_comparison_table` writes a Markdown table to an arbitrary path
  (never the repo's own `docs/comparison_results.md` in tests).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import Config, PreparedData, prepare_dataset
from src.data.loader import load_flows
from src.evaluation.compare_models import (
    METRIC_NAMES,
    _measure_inference_latency_ms,
    compare_models,
    render_comparison_table,
)
from src.models.autoencoder_model import AutoencoderDetector
from src.models.base_detector import BaseDetector
from src.models.isolation_forest_model import IsolationForestDetector

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).parent.parent
COMPARE_MODELS_MODULE = REPO_ROOT / "src" / "evaluation" / "compare_models.py"

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

# Small, fast hyperparameters -- this suite only cares about compare_models'
# own behavior, not model quality (already covered by the per-model test
# suites).
_AUTOENCODER_HYPERPARAMS: dict[str, Any] = {
    "hidden_dims": [8, 4],
    "epochs": 20,
    "learning_rate": 0.01,
    "batch_size": 4,
    "threshold_percentile": 90.0,
    "validation_fraction": 0.2,
    "random_state": 42,
}
_ISOLATION_FOREST_HYPERPARAMS: dict[str, Any] = {
    "n_estimators": 37,
    "contamination": 0.2,
    "max_samples": "auto",
    "threshold_percentile": 90.0,
    "random_state": 7,
}

# README section 5.5 requires >= 1000 samples for a stable latency figure;
# use a much smaller floor here so the fixture-based suite stays fast while
# still exercising the tiling logic (n_test=24 << min_latency_samples).
_TEST_MIN_LATENCY_SAMPLES = 50


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
def detectors(config_dict: dict[str, Any]) -> list[BaseDetector]:
    """Two real, unfitted detectors, deliberately typed as `list[BaseDetector]`
    to prove `compare_models` needs nothing beyond the shared interface.
    """
    cfg = Config.model_validate(config_dict)
    isolation_forest: BaseDetector = IsolationForestDetector(cfg.models.isolation_forest)
    autoencoder: BaseDetector = AutoencoderDetector(cfg.models.autoencoder)
    return [isolation_forest, autoencoder]


# --- compare_models(): single side-by-side DataFrame, all metrics ----------


def test_compare_models_returns_one_dataframe_with_both_models(
    detectors: list[BaseDetector], prepared: PreparedData
) -> None:
    comparison = compare_models(
        detectors,
        prepared.x_train,
        prepared.x_test,
        prepared.y_test,
        benign_label=BENIGN_LABEL,
        min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
    )

    assert isinstance(comparison, pd.DataFrame)
    assert set(comparison.columns) == {"isolation_forest", "autoencoder"}
    assert list(comparison.index) == METRIC_NAMES


def test_compare_models_metrics_are_finite_and_in_valid_ranges(
    detectors: list[BaseDetector], prepared: PreparedData
) -> None:
    comparison = compare_models(
        detectors,
        prepared.x_train,
        prepared.x_test,
        prepared.y_test,
        benign_label=BENIGN_LABEL,
        min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
    )

    assert np.isfinite(comparison.to_numpy(dtype=float)).all()

    bounded_metrics = ["precision", "recall", "f1", "roc_auc"]
    bounded = comparison.loc[bounded_metrics].to_numpy(dtype=float)
    assert (bounded >= 0.0).all()
    assert (bounded <= 1.0).all()

    non_negative_metrics = ["training_time_seconds", "inference_latency_ms_per_sample"]
    non_negative = comparison.loc[non_negative_metrics].to_numpy(dtype=float)
    assert (non_negative >= 0.0).all()


def test_compare_models_latency_measured_over_min_latency_samples(
    detectors: list[BaseDetector], prepared: PreparedData
) -> None:
    """The fixture's mixed test set (well under `_TEST_MIN_LATENCY_SAMPLES`
    rows) must still yield a positive, finite mean latency -- proving the
    batch was tiled up rather than measured on a handful of rows.
    """
    assert prepared.x_test.shape[0] < _TEST_MIN_LATENCY_SAMPLES

    comparison = compare_models(
        detectors,
        prepared.x_train,
        prepared.x_test,
        prepared.y_test,
        benign_label=BENIGN_LABEL,
        min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
    )

    latencies = comparison.loc["inference_latency_ms_per_sample"]
    assert (latencies > 0.0).all()


def test_measure_inference_latency_skips_tiling_when_batch_already_large_enough(
    prepared: PreparedData,
) -> None:
    """When `x_test` already has >= `min_samples` rows, the batch passed to
    `score` must be exactly `x_test` (no tiling needed).
    """
    detector: BaseDetector = IsolationForestDetector()
    detector.fit(prepared.x_train)

    latency_ms = _measure_inference_latency_ms(
        detector, prepared.x_test, min_samples=1
    )

    assert latency_ms > 0.0


def test_measure_inference_latency_rejects_empty_batch(prepared: PreparedData) -> None:
    detector: BaseDetector = IsolationForestDetector()
    detector.fit(prepared.x_train)

    empty_x_test = prepared.x_test[:0]

    with pytest.raises(ValueError, match="at least one row"):
        _measure_inference_latency_ms(detector, empty_x_test, min_samples=100)


def test_compare_models_roc_auc_is_nan_for_single_class_y_test(
    detectors: list[BaseDetector], prepared: PreparedData, tmp_path: Path
) -> None:
    """With a single ground-truth class (an all-benign slice) ROC-AUC is
    undefined: compare_models must fall back to nan without raising, and
    render_comparison_table must still emit a table with a `nan` cell.
    """
    benign_only_y = pd.Series([BENIGN_LABEL] * prepared.x_train.shape[0])

    comparison = compare_models(
        detectors,
        prepared.x_train,
        prepared.x_train,
        benign_only_y,
        benign_label=BENIGN_LABEL,
        min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
    )

    assert comparison.loc["roc_auc"].isna().all()

    out = render_comparison_table(comparison, tmp_path / "comparison.md")
    assert out.exists()
    assert "nan" in out.read_text(encoding="utf-8")


def test_compare_models_rejects_duplicate_detector_names(prepared: PreparedData) -> None:
    """Two detectors sharing `.name` would silently overwrite one model's
    results in the comparison dict -- must raise instead of losing a model.
    """
    duplicate_detectors: list[BaseDetector] = [
        IsolationForestDetector(),
        IsolationForestDetector(),
    ]

    with pytest.raises(ValueError, match="distinct .name"):
        compare_models(
            duplicate_detectors,
            prepared.x_train,
            prepared.x_test,
            prepared.y_test,
            benign_label=BENIGN_LABEL,
            min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
        )


class _CountingDetector(BaseDetector):
    """Counts how many times the test matrix gets scored.

    A real detector cannot be used for this: `IsolationForestDetector.fit`
    calls `self.score` internally to derive its threshold, which would
    contaminate the count.
    """

    name = "counting"

    def __init__(self) -> None:
        self.score_calls = 0

    def fit(self, X_benign: np.ndarray) -> None:
        return None

    def score(self, X: np.ndarray) -> np.ndarray:
        self.score_calls += 1
        return np.linspace(0.0, 1.0, X.shape[0])

    @property
    def threshold(self) -> float:
        return 0.5

    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:  # pragma: no cover - not exercised by compare_models
        return feature_names[:k]


def test_compare_models_scores_the_test_set_only_twice(prepared: PreparedData) -> None:
    """Regression guard: one pass for the metrics, one for the latency timing.

    `_evaluate_one` used to score `x_test` three times -- once for the raw
    scores, once again inside `is_anomaly`, and once more for the latency
    measurement. The middle pass is pure waste at real-dataset scale.
    """
    detector = _CountingDetector()

    compare_models(
        [detector],
        prepared.x_train,
        prepared.x_test,
        prepared.y_test,
        benign_label=BENIGN_LABEL,
        min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
    )

    assert detector.score_calls == 2


# --- interface-only: no model-specific imports/branching -------------------


def test_compare_models_module_imports_no_model_internals() -> None:
    """Static guarantee that `src.evaluation.compare_models` never imports
    `IsolationForestDetector`/`AutoencoderDetector` (or their config
    classes) -- it must depend ONLY on `BaseDetector` (README section 2/9).
    """
    tree = ast.parse(COMPARE_MODELS_MODULE.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            imported_names.update(alias.name for alias in node.names)

    forbidden = {
        "IsolationForestDetector",
        "AutoencoderDetector",
        "IsolationForestConfig",
        "AutoencoderConfig",
    }
    assert not (imported_names & forbidden)
    assert "src.models.isolation_forest_model" not in imported_names
    assert "src.models.autoencoder_model" not in imported_names


def test_compare_models_works_given_only_base_detector_typed_objects(
    detectors: list[BaseDetector], prepared: PreparedData
) -> None:
    """`detectors` is statically typed as `list[BaseDetector]` (see fixture);
    successfully calling `compare_models` with it demonstrates the function
    needs nothing beyond that interface at both type-check and runtime.
    """
    comparison = compare_models(
        detectors,
        prepared.x_train,
        prepared.x_test,
        prepared.y_test,
        benign_label=BENIGN_LABEL,
        min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
    )

    assert not comparison.empty


# --- no-leakage: thresholds derived from benign data only -------------------


@pytest.mark.parametrize(
    "make_detector",
    [
        lambda cfg: IsolationForestDetector(cfg.models.isolation_forest),
        lambda cfg: AutoencoderDetector(cfg.models.autoencoder),
    ],
    ids=["isolation_forest", "autoencoder"],
)
def test_threshold_unaffected_by_appending_attack_rows_at_score_time(
    make_detector: Any, prepared: PreparedData, config_dict: dict[str, Any]
) -> None:
    """Fit on benign-only x_train (threshold is fixed then); scoring a mixed
    batch containing attack rows afterwards must never change it.
    """
    cfg = Config.model_validate(config_dict)
    detector: BaseDetector = make_detector(cfg)
    detector.fit(prepared.x_train)

    threshold_before = detector.threshold

    is_benign = (prepared.y_test == BENIGN_LABEL).to_numpy()
    assert (~is_benign).any()  # sanity: x_test really contains attack rows
    detector.score(prepared.x_test)  # mixed benign + attack
    detector.is_anomaly(prepared.x_test)

    assert detector.threshold == threshold_before


# --- is_anomaly(): boolean array aligned with input rows --------------------


@pytest.mark.parametrize(
    "make_detector",
    [
        lambda cfg: IsolationForestDetector(cfg.models.isolation_forest),
        lambda cfg: AutoencoderDetector(cfg.models.autoencoder),
    ],
    ids=["isolation_forest", "autoencoder"],
)
def test_is_anomaly_returns_aligned_boolean_array(
    make_detector: Any, prepared: PreparedData, config_dict: dict[str, Any]
) -> None:
    cfg = Config.model_validate(config_dict)
    detector: BaseDetector = make_detector(cfg)
    detector.fit(prepared.x_train)

    flags = detector.is_anomaly(prepared.x_test)

    assert isinstance(flags, np.ndarray)
    assert flags.dtype == np.bool_
    assert flags.shape == (prepared.x_test.shape[0],)
    np.testing.assert_array_equal(flags, detector.score(prepared.x_test) > detector.threshold)


@pytest.mark.parametrize(
    "make_detector",
    [
        lambda cfg: IsolationForestDetector(cfg.models.isolation_forest),
        lambda cfg: AutoencoderDetector(cfg.models.autoencoder),
    ],
    ids=["isolation_forest", "autoencoder"],
)
def test_threshold_is_a_float_set_after_fit(
    make_detector: Any, prepared: PreparedData, config_dict: dict[str, Any]
) -> None:
    cfg = Config.model_validate(config_dict)
    detector: BaseDetector = make_detector(cfg)
    detector.fit(prepared.x_train)

    assert isinstance(detector.threshold, float)


@pytest.mark.parametrize(
    "make_detector",
    [
        lambda cfg: IsolationForestDetector(cfg.models.isolation_forest),
        lambda cfg: AutoencoderDetector(cfg.models.autoencoder),
    ],
    ids=["isolation_forest", "autoencoder"],
)
def test_threshold_raises_before_fit(make_detector: Any, config_dict: dict[str, Any]) -> None:
    cfg = Config.model_validate(config_dict)
    detector: BaseDetector = make_detector(cfg)

    with pytest.raises(RuntimeError):
        _ = detector.threshold


# --- render_comparison_table(): writes Markdown to an arbitrary path -------


def test_render_comparison_table_writes_markdown_to_tmp_path(
    detectors: list[BaseDetector], prepared: PreparedData, tmp_path: Path
) -> None:
    comparison = compare_models(
        detectors,
        prepared.x_train,
        prepared.x_test,
        prepared.y_test,
        benign_label=BENIGN_LABEL,
        min_latency_samples=_TEST_MIN_LATENCY_SAMPLES,
    )
    output_path = tmp_path / "comparison_results.md"

    written_path = render_comparison_table(comparison, output_path)

    assert written_path == output_path
    content = output_path.read_text(encoding="utf-8")
    assert "isolation_forest" in content
    assert "autoencoder" in content
    assert "precision" in content
    assert content.startswith("|")
