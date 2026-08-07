"""Side-by-side evaluation of two (or more) `BaseDetector` implementations.

This module depends ONLY on the `BaseDetector` interface (`fit`, `score`,
`is_anomaly`, `.name`) -- it never imports or special-cases
`IsolationForestDetector`/`AutoencoderDetector` internals (README section 2,
"One shared interface", and section 9's "identical `BaseDetector` interface"
hard constraint).

Ground-truth labels (`y_test`, raw strings such as `"BENIGN"`/`"DoS Hulk"`)
are used EXCLUSIVELY here, at evaluation time, to binarize against
`benign_label` and compute precision/recall/F1/ROC-AUC -- never during
`fit` (README section 2, "Unsupervised by construction").
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.base_detector import BaseDetector

# README section 5.5: inference latency must be measured over a batch of at
# least this many samples for a stable ms/sample figure, even when the
# labeled test set itself (e.g. a small fixture) is much smaller.
DEFAULT_MIN_LATENCY_SAMPLES = 1000

# Row order of the returned comparison DataFrame (one row per metric family,
# one column per model), matching README section 5.5's required metrics.
METRIC_NAMES = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "training_time_seconds",
    "inference_latency_ms_per_sample",
]


def _binarize_labels(y_test: pd.Series, benign_label: str) -> np.ndarray:
    """Binarize raw string labels into `0` (benign) / `1` (anomaly).

    Labels are used ONLY here, at evaluation time -- never inside any
    detector's `fit`.
    """
    return np.asarray(np.asarray(y_test) != benign_label, dtype=np.int64)


def _measure_inference_latency_ms(
    detector: BaseDetector,
    x_test: np.ndarray,
    min_samples: int,
) -> float:
    """Mean inference latency in ms/sample, over a batch of >= `min_samples` rows.

    `x_test` is tiled (repeated) up to `min_samples` rows when it is smaller,
    so a tiny labeled test set (e.g. the 24-row test fixture) still yields a
    stable per-sample latency figure (README section 5.5).

    Raises:
        ValueError: if `x_test` has zero rows.
    """
    n_test = x_test.shape[0]
    if n_test == 0:
        raise ValueError("x_test must contain at least one row")

    if n_test >= min_samples:
        batch = x_test
    else:
        repeats = int(np.ceil(min_samples / n_test))
        batch = np.tile(x_test, (repeats, 1))[:min_samples]

    start = time.perf_counter()
    detector.score(batch)
    elapsed_seconds = time.perf_counter() - start

    return (elapsed_seconds / float(batch.shape[0])) * 1000.0


def _evaluate_one(
    detector: BaseDetector,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_true_binary: np.ndarray,
    min_latency_samples: int,
) -> dict[str, float]:
    """Fit, time, and score a single detector; compute its full metric set."""
    start = time.perf_counter()
    detector.fit(x_train)
    training_time_seconds = time.perf_counter() - start

    scores = detector.score(x_test)
    predictions = detector.is_anomaly(x_test).astype(int)

    precision = precision_score(y_true_binary, predictions, zero_division=0)
    recall = recall_score(y_true_binary, predictions, zero_division=0)
    f1 = f1_score(y_true_binary, predictions, zero_division=0)
    # ROC-AUC is undefined with a single ground-truth class; guard rather
    # than let sklearn raise, since a labeled test set is expected to (and,
    # per the split strategy, always does) contain both benign and attack
    # rows in real usage.
    roc_auc = (
        roc_auc_score(y_true_binary, scores)
        if len(np.unique(y_true_binary)) > 1
        else float("nan")
    )

    latency_ms = _measure_inference_latency_ms(detector, x_test, min_latency_samples)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "training_time_seconds": training_time_seconds,
        "inference_latency_ms_per_sample": latency_ms,
    }


def compare_models(
    detectors: Sequence[BaseDetector],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: pd.Series,
    benign_label: str,
    min_latency_samples: int = DEFAULT_MIN_LATENCY_SAMPLES,
) -> pd.DataFrame:
    """Fit, evaluate, and directly compare each detector on the same data.

    For each detector (any `BaseDetector`, used purely through that
    interface): fits it on the benign-only `x_train` (timing the wall clock),
    then computes precision/recall/F1 (via `is_anomaly`, i.e. that model's own
    threshold), ROC-AUC (threshold-independent, from raw `score`), and mean
    inference latency (ms/sample, over a batch of at least
    `min_latency_samples` rows).

    Args:
        detectors: Two (or more) fitted-or-unfitted `BaseDetector` instances,
            each with a distinct `.name`. `fit` is (re-)called on each here so
            training time can be measured consistently for all detectors.
        x_train: Benign-only training feature matrix, already scaled upstream
            by the shared feature-engineering pipeline (same features/scaling
            used to build every detector in `detectors`).
        x_test: Mixed (benign + attack) test feature matrix.
        y_test: Raw ground-truth string labels aligned with `x_test`'s rows
            (e.g. `"BENIGN"`, `"DoS Hulk"`). Used ONLY here, for evaluation.
        benign_label: The exact label value identifying normal traffic.
        min_latency_samples: Minimum batch size for the latency measurement.

    Returns:
        A single `pd.DataFrame` with one row per metric (`METRIC_NAMES`) and
        one column per detector, named after `detector.name` -- a direct,
        side-by-side comparison table, not separate per-model output.

    Raises:
        ValueError: if two or more `detectors` share the same `.name` --
            silently overwriting one model's results in the comparison table
            would otherwise go unnoticed.
    """
    names = [detector.name for detector in detectors]
    if len(set(names)) != len(names):
        raise ValueError(f"detectors must have distinct .name values, got: {names}")

    y_true_binary = _binarize_labels(y_test, benign_label)

    results = {
        detector.name: _evaluate_one(detector, x_train, x_test, y_true_binary, min_latency_samples)
        for detector in detectors
    }

    return pd.DataFrame(results).reindex(METRIC_NAMES)


def _dataframe_to_markdown(comparison: pd.DataFrame) -> str:
    """Render `comparison` as a GitHub-flavored Markdown table (no extra deps)."""
    header = "| metric | " + " | ".join(str(col) for col in comparison.columns) + " |"
    separator = "| " + " | ".join(["---"] * (len(comparison.columns) + 1)) + " |"
    rows = [
        "| "
        + " | ".join([str(metric), *(f"{value:.6g}" for value in row)])
        + " |"
        for metric, row in zip(comparison.index, comparison.to_numpy(), strict=True)
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def render_comparison_table(comparison: pd.DataFrame, output_path: str | Path) -> Path:
    """Write `comparison` to `output_path` as a Markdown table.

    Args:
        comparison: The DataFrame returned by `compare_models`.
        output_path: Destination file (parent directories are created if
            needed). Production usage points this at
            `docs/comparison_results.md`; tests must use a `tmp_path`.

    Returns:
        The `Path` that was written, for convenience.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dataframe_to_markdown(comparison), encoding="utf-8")
    return path
