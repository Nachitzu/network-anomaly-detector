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
from src.models.feature_ranking import as_explanation_matrix, rank_top_features

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
        self._threshold: float | None = None

    def fit(self, X_benign: np.ndarray) -> None:
        """Train the forest using ONLY benign traffic. No labels involved.

        Also stores the per-feature training mean/std, needed later by
        `top_contributing_features` for z-score-based explainability, and
        derives `self._threshold` as the configured percentile of this same
        benign training data's own scores -- mirroring the Autoencoder's
        benign-percentile threshold (README section 5.3) while staying
        unsupervised: the threshold is NEVER computed from attack data.
        """
        self._estimator.fit(X_benign)
        self._train_mean = X_benign.mean(axis=0)
        self._train_std = X_benign.std(axis=0)

        benign_scores = self.score(X_benign)
        self._threshold = float(
            np.percentile(benign_scores, self._config.threshold_percentile)
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score per row. Higher = more anomalous.

        `IsolationForest.decision_function` returns HIGHER values for normal
        (inlier) points and LOWER (more negative) values for anomalies, so the
        sign is flipped here to match the project-wide convention shared with
        the Autoencoder (README section 5.2).

        Raises:
            RuntimeError: if called before `fit` (checked explicitly here,
                rather than relying on sklearn's own `NotFittedError`, so the
                `BaseDetector` contract raises the same exception type across
                every detector).
        """
        if self._train_mean is None:
            raise RuntimeError("IsolationForestDetector.fit() must be called before score()")
        return np.asarray(-self._estimator.decision_function(X), dtype=np.float64)

    @property
    def threshold(self) -> float:
        """The fitted anomaly threshold (percentile of benign training scores).

        Raises:
            RuntimeError: if accessed before `fit`.
        """
        if self._threshold is None:
            raise RuntimeError("IsolationForestDetector.fit() must be called before threshold")
        return self._threshold

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
        return self.top_contributing_features_batch(
            np.asarray(x, dtype=float).reshape(1, -1), feature_names, k=k
        )[0]

    def top_contributing_features_batch(
        self, X: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[list[str]]:
        """Vectorized `top_contributing_features` over a whole matrix.

        The z-score is a pure element-wise expression against two per-feature
        vectors, so it broadcasts over `(n, n_features)` unchanged -- one
        expression for any number of rows, instead of one Python call each.
        `top_contributing_features` routes through here with `n = 1`, so there
        is a single implementation and the two can never disagree.

        Args:
            X: One flow per row (shape `(n, n_features)`).
            feature_names: Names aligned with `X`'s columns.
            k: Number of top feature names per row.

        Returns:
            One list of `min(k, len(feature_names))` names per row of `X`.

        Raises:
            RuntimeError: if called before `fit`.
            ValueError: if `k` is not positive or `X` contains non-finite values.
        """
        if self._train_mean is None or self._train_std is None:
            raise RuntimeError("IsolationForestDetector.fit() must be called before "
                                "top_contributing_features()")

        matrix = as_explanation_matrix(X, k)
        safe_std = np.where(self._train_std == 0, _ZERO_VARIANCE_EPSILON, self._train_std)
        z_scores = np.abs((matrix - self._train_mean) / safe_std)

        return rank_top_features(z_scores, feature_names, k)
