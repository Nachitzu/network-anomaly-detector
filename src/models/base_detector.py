"""Shared abstract interface for unsupervised anomaly detectors.

Both `IsolationForestDetector` and the future `AutoencoderDetector` implement
this exact interface (see README section 5.1), so evaluation and comparison
code (`src/evaluation/compare_models.py`) can depend only on `BaseDetector`
and never special-case a model's internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Common contract for unsupervised network anomaly detectors.

    Models train ONLY on benign traffic: `fit` never receives, and must never
    require, ground-truth labels. This is a hard constraint of the project
    (see README section 2, "Unsupervised by construction").
    """

    name: str

    @abstractmethod
    def fit(self, X_benign: np.ndarray) -> None:
        """Train using ONLY benign traffic. No labels involved."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score per row. Higher = more anomalous.

        Comparable across models after normalization in evaluation.
        """

    @property
    @abstractmethod
    def threshold(self) -> float:
        """The fitted anomaly threshold, above which `score(X)` counts as anomalous.

        Every subclass derives its threshold EXCLUSIVELY from benign data at
        `fit` time (e.g. a percentile of benign scores/reconstruction error) --
        never from attack data or ground-truth labels (README section 2).

        Raises:
            RuntimeError: if accessed before `fit`.
        """

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        """Binary anomaly decision per row: `score(X) > threshold`.

        Concrete on `BaseDetector` so both detectors share the exact same
        decision rule -- subclasses only need to provide `score` and
        `threshold`, never override this method (README section 5.1/5.5:
        `compare_models` gets a binary decision from both models via this one
        polymorphic call).

        Raises:
            RuntimeError: if accessed before `fit` (propagated from `score`
                and/or `threshold`).
        """
        return np.asarray(self.score(X) > self.threshold, dtype=bool)

    @abstractmethod
    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:
        """Explainability hook: which features drove this specific anomaly score."""
