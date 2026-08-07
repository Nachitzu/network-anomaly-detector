"""Per-anomaly explainability: top-N contributing features for flagged rows.

Depends ONLY on the `BaseDetector` interface (`is_anomaly`,
`top_contributing_features_batch`) -- works identically for
`IsolationForestDetector` (z-score deviation) and `AutoencoderDetector`
(per-feature reconstruction error) without ever branching on which model
produced the detector (README section 5.1/5.5).

Explanations are requested for every flagged row in one call, so the cost is
one vectorized operation rather than one per anomaly. That matters: a full
CICIDS2017 run flags hundreds of thousands of flows.
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
    flagged_indices = np.flatnonzero(detector.is_anomaly(X))
    explanations = detector.top_contributing_features_batch(X[flagged_indices], feature_names, k=k)
    return {
        int(row_index): names
        for row_index, names in zip(flagged_indices, explanations, strict=True)
    }
