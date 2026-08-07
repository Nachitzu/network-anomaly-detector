"""Unit tests for src.evaluation.run_comparison.

The runner is exercised end to end against the 24-row synthetic fixture with
fixture-scale hyperparameters -- never `data/raw/` and never the repo's own
`docs/comparison_results.md`, so the suite stays fast, CPU-only, and cannot
overwrite a real run's artifact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.feature_engineering import Config
from src.evaluation.run_comparison import (
    build_detectors,
    main,
    recall_by_attack_type,
    run,
)
from src.models.base_detector import BaseDetector

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
BENIGN_LABEL = "BENIGN"


class _StubDetector(BaseDetector):
    """A detector whose decisions are fixed up front.

    `recall_by_attack_type` is pure aggregation logic, so it is tested against
    a deterministic stub rather than a real model: that keeps these assertions
    about the grouping itself instead of some real detector's separability on
    a handful of synthetic rows.
    """

    name = "stub"

    def __init__(self, flags: np.ndarray) -> None:
        self._flags = flags

    def fit(self, X_benign: np.ndarray) -> None:  # pragma: no cover - not used here
        return None

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._flags.astype(np.float64)

    @property
    def threshold(self) -> float:
        return 0.5

    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:  # pragma: no cover - not used here
        return feature_names[:k]


@pytest.fixture()
def config_dict() -> dict[str, Any]:
    """A fixture-scale config: same shape as config.yaml, tiny hyperparameters."""
    return {
        "data": {"raw_dir": str(FIXTURES_DIR), "csv_glob": "sample_flows.csv"},
        "features": {
            "numeric_columns": FEATURE_COLUMNS,
            "label_column": "Label",
            "benign_label": BENIGN_LABEL,
        },
        "split": {"test_size": 0.3, "random_state": 42},
        "cleaning": {"invalid_values": "drop"},
        "scaler": {"with_mean": True, "with_std": True},
        "models": {
            "isolation_forest": {
                "n_estimators": 20,
                "contamination": 0.2,
                "max_samples": "auto",
                "threshold_percentile": 90.0,
                "random_state": 42,
            },
            "autoencoder": {
                "hidden_dims": [8, 4],
                "epochs": 20,
                "learning_rate": 0.01,
                "batch_size": 4,
                "threshold_percentile": 90.0,
                "validation_fraction": 0.2,
                "random_state": 42,
            },
        },
    }


@pytest.fixture()
def config_path(config_dict: dict[str, Any], tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    return path


# --- build_detectors -------------------------------------------------------


def test_build_detectors_returns_two_distinctly_named_base_detectors(
    config_dict: dict[str, Any],
) -> None:
    detectors = build_detectors(Config.model_validate(config_dict))

    assert len(detectors) == 2
    assert all(isinstance(detector, BaseDetector) for detector in detectors)
    assert len({detector.name for detector in detectors}) == 2


# --- recall_by_attack_type -------------------------------------------------


def test_recall_by_attack_type_excludes_benign_and_scores_each_label() -> None:
    """Benign rows are excluded; each attack label gets its own detection rate.

    Flags below: the benign row is flagged (a false positive, which must not
    affect recall), one of two `DoS Hulk` rows is caught, and `PortScan` is
    missed entirely.
    """
    x_test = np.zeros((4, 1))
    y_test = pd.Series([BENIGN_LABEL, "DoS Hulk", "DoS Hulk", "PortScan"])
    detector = _StubDetector(np.array([1, 1, 0, 0]))

    per_type = recall_by_attack_type(detector, x_test, y_test, BENIGN_LABEL)

    assert set(per_type.index) == {"DoS Hulk", "PortScan"}
    assert per_type["DoS Hulk"] == pytest.approx(0.5)
    assert per_type["PortScan"] == pytest.approx(0.0)


def test_recall_by_attack_type_is_one_when_every_attack_row_is_flagged() -> None:
    x_test = np.zeros((2, 1))
    y_test = pd.Series(["DoS Hulk", "DoS Hulk"])
    detector = _StubDetector(np.array([1, 1]))

    per_type = recall_by_attack_type(detector, x_test, y_test, BENIGN_LABEL)

    assert per_type["DoS Hulk"] == pytest.approx(1.0)


def test_recall_by_attack_type_sorts_best_detected_attack_first() -> None:
    x_test = np.zeros((4, 1))
    y_test = pd.Series(["PortScan", "PortScan", "DoS Hulk", "DoS Hulk"])
    detector = _StubDetector(np.array([0, 0, 1, 1]))

    per_type = recall_by_attack_type(detector, x_test, y_test, BENIGN_LABEL)

    assert list(per_type.index) == ["DoS Hulk", "PortScan"]


# --- run(): full end-to-end report -----------------------------------------


def test_run_writes_report_with_all_sections(config_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "comparison_results.md"

    comparison = run(config_path, raw_dir=FIXTURES_DIR, output_path=output)

    assert output.exists()
    report = output.read_text(encoding="utf-8")
    for heading in (
        "# Model comparison results",
        "## Run context",
        "## Side-by-side metrics",
        "## Detection rate by attack type",
        "## Example explanations",
        "## Reading these numbers",
    ):
        assert heading in report

    # Both models must appear as columns of the single side-by-side table.
    for name in comparison.columns:
        assert str(name) in report


def test_run_returns_comparison_with_both_models_and_all_metrics(
    config_path: Path, tmp_path: Path
) -> None:
    comparison = run(config_path, raw_dir=FIXTURES_DIR, output_path=tmp_path / "out.md")

    assert list(comparison.columns) == ["isolation_forest", "autoencoder"]
    assert "f1" in comparison.index
    assert "roc_auc" in comparison.index


def test_run_defaults_raw_dir_to_config_data_section(config_path: Path, tmp_path: Path) -> None:
    """With `raw_dir=None` the runner must fall back to config's `data.raw_dir`."""
    output = tmp_path / "out.md"

    comparison = run(config_path, raw_dir=None, output_path=output)

    assert output.exists()
    assert not comparison.empty


def test_run_writes_only_to_the_requested_path(config_path: Path, tmp_path: Path) -> None:
    """A test run must write only where it is told, never to the repo's docs/."""
    output = tmp_path / "nested" / "report.md"

    run(config_path, raw_dir=FIXTURES_DIR, output_path=output)

    assert output.exists()
    assert output.parent.name == "nested"


# --- main(): command-line entry point --------------------------------------


def test_main_honors_command_line_arguments(
    config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--config`, `--raw-dir` and `--output` must all reach `run`."""
    output = tmp_path / "from_cli.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_comparison",
            "--config",
            str(config_path),
            "--raw-dir",
            str(FIXTURES_DIR),
            "--output",
            str(output),
        ],
    )

    main()

    assert output.exists()
    assert "# Model comparison results" in output.read_text(encoding="utf-8")
