"""Per-anomaly explainability: top-N contributing features for flagged rows.

Depends ONLY on the `BaseDetector` interface (`is_anomaly`,
`top_contributing_features`) -- works identically for `IsolationForestDetector`
(z-score deviation) and `AutoencoderDetector` (per-feature reconstruction
error) without ever branching on which model produced the detector (README
section 5.1/5.5).
"""
from __future__ import annotations

import numpy as np

from src.models.base_detector import BaseDetector


def explain_flagged_anomalies(
    detector: BaseDetector,
    X: np.ndarray,
    feature_names: list[str],
    k: int = 5,
) -> dict[int, list[str]]:
    """Map each row flagged as anomalous to its top-`k` contributing features.

    Args:
        detector: Any fitted `BaseDetector`.
        X: Feature matrix to explain (e.g. the labeled test set).
        feature_names: Column names aligned with `X`'s columns.
        k: Number of top feature names requested per row (capped at
            `len(feature_names)` by `top_contributing_features`).

    Returns:
        A dict mapping each row's integer position in `X` (only rows where
        `detector.is_anomaly(X)` is `True`) to its
        `min(k, len(feature_names))` top contributing feature names. Rows not
        flagged as anomalous are omitted entirely.

    Raises:
        RuntimeError: if `detector` has not been fitted yet (propagated from
            `is_anomaly`/`top_contributing_features`).
    """
    flagged = detector.is_anomaly(X)
    return {
        int(row_index): detector.top_contributing_features(X[row_index], feature_names, k=k)
        for row_index in np.flatnonzero(flagged)
    }
