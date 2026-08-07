"""Shared ranking step behind every detector's explainability output.

`IsolationForestDetector` and `AutoencoderDetector` disagree on *what* counts
as a feature's contribution -- a z-score against the benign training
distribution versus a per-feature reconstruction error -- but they agree
completely on what to do with those numbers: take the `k` largest per row,
descending, capped at the number of feature names. That shared tail lives here
so it exists exactly once, and so the single-row and whole-matrix code paths
cannot drift apart.

Deliberately free of any `BaseDetector` import, so models can depend on it
without a cycle.
"""

from __future__ import annotations

import numpy as np


def as_explanation_matrix(x: np.ndarray, k: int) -> np.ndarray:
    """Validate an explainability request and normalize it to a `(n, d)` matrix.

    Accepts either a single flow's feature vector (shape `(d,)`) or a batch of
    them (shape `(n, d)`), always returning the 2-D form as float64 so callers
    have one shape to reason about.

    Args:
        x: One feature vector or a matrix of them.
        k: Number of top features the caller intends to request.

    Returns:
        `x` as a float64 array of shape `(n, d)`.

    Raises:
        ValueError: if `k` is not positive, or `x` holds non-finite values.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    matrix = np.asarray(x, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("x must contain only finite values")

    return matrix


def rank_top_features(
    contributions: np.ndarray, feature_names: list[str], k: int
) -> list[list[str]]:
    """Name the `k` highest-contributing features of each row, descending.

    Args:
        contributions: Per-feature contribution scores, shape `(n, d)`. Higher
            means "explains this row's anomaly score more".
        feature_names: Names aligned with the columns of `contributions`.
        k: How many names to return per row, capped at `len(feature_names)`.

    Returns:
        One list of feature names per row of `contributions`.

    Note:
        Ties are broken by ascending feature index, via a stable sort. That
        makes the ranking independent of how many rows are ranked at once and
        of where a row sits in the batch -- without it, the unstable default
        could name different features for the same flow depending on the size
        of the batch it happened to arrive in. Exact ties are real here: a
        feature with zero variance in training scores exactly 0 for every flow.
    """
    top_k = min(k, len(feature_names))
    ranked_indices = np.argsort(-contributions, axis=-1, kind="stable")[:, :top_k]
    return [[feature_names[index] for index in row] for row in ranked_indices]
