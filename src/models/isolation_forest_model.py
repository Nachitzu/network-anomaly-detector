"""Isolation Forest anomaly detector implementing `BaseDetector`.

Wraps `sklearn.ensemble.IsolationForest`. Hyperparameters (`n_estimators`,
`contamination`, `max_samples`) are sourced from `config.yaml`'s
`models.isolation_forest` section (validated via `IsolationForestConfig` in
`src.data.feature_engineering`) — never hardcoded (README section 5.2).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from src.data.feature_engineering import IsolationForestConfig
from src.models.base_detector import BaseDetector

# Guards per-feature z-score division against zero-variance training features
# (e.g. a flag column that is constant across all benign training rows).
_ZERO_VARIANCE_EPSILON = 1e-8


class IsolationForestDetector(BaseDetector):
    """`sklearn.ensemble.IsolationForest`-backed unsupervised anomaly detector."""

    name = "isolation_forest"

    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        """Build the detector from validated hyperparameters.

        Args:
            config: `models.isolation_forest` section of `config.yaml`,
                already validated into an `IsolationForestConfig`. Defaults
                to `IsolationForestConfig()`'s own defaults if omitted.
        """
        self._config = config or IsolationForestConfig()
        self._estimator = IsolationForest(
            n_estimators=self._config.n_estimators,
            contamination=self._config.contamination,
            max_samples=self._config.max_samples,
            random_state=self._config.random_state,
        )
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None

    def fit(self, X_benign: np.ndarray) -> None:
        """Train the forest using ONLY benign traffic. No labels involved.

        Also stores the per-feature training mean/std, needed later by
        `top_contributing_features` for z-score-based explainability.
        """
        self._estimator.fit(X_benign)
        self._train_mean = X_benign.mean(axis=0)
        self._train_std = X_benign.std(axis=0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score per row. Higher = more anomalous.

        `IsolationForest.decision_function` returns HIGHER values for normal
        (inlier) points and LOWER (more negative) values for anomalies, so the
        sign is flipped here to match the project-wide convention shared with
        the Autoencoder (README section 5.2).
        """
        return np.asarray(-self._estimator.decision_function(X), dtype=np.float64)

    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:
        """Rank features by |z-score| against the benign training distribution.

        Args:
            x: A single flow's feature vector (shape `(n_features,)`).
            feature_names: Names aligned with `x`'s columns.
            k: Number of top feature names to return.

        Returns:
            The `k` feature names with the largest `|(x - train_mean) / train_std|`,
            descending. Zero-variance training features are guarded against
            division by zero.

        Raises:
            RuntimeError: if called before `fit`.
            ValueError: if `k` is not positive or `x` contains non-finite values.
        """
        if self._train_mean is None or self._train_std is None:
            raise RuntimeError("IsolationForestDetector.fit() must be called before "
                                "top_contributing_features()")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        x_flat = np.asarray(x, dtype=float).reshape(-1)
        if not np.all(np.isfinite(x_flat)):
            raise ValueError("x must contain only finite values")
        safe_std = np.where(self._train_std == 0, _ZERO_VARIANCE_EPSILON, self._train_std)
        z_scores = np.abs((x_flat - self._train_mean) / safe_std)

        top_k = min(k, len(feature_names))
        ranked_indices = np.argsort(-z_scores)[:top_k]
        return [feature_names[i] for i in ranked_indices]
