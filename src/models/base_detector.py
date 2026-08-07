"""Shared abstract interface for unsupervised anomaly detectors.

Both `IsolationForestDetector` and the future `AutoencoderDetector` implement
this exact interface (see README section 5.1), so evaluation and comparison
code (`src/evaluation/compare_models.py`) can depend only on `BaseDetector`
and never special-case a model's internals.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models.feature_ranking import as_explanation_matrix


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

    def is_anomaly_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply the decision rule to scores that were ALREADY computed.

        This is the single place in the project where the decision operator
        lives; `is_anomaly` delegates here. It exists so callers that already
        hold `score(X)` -- notably `compare_models._evaluate_one`, which needs
        both the raw scores (for ROC-AUC) and the binary decision (for
        precision/recall/F1) -- get that decision without scoring the same
        matrix twice, and without re-spelling `> threshold` at the call site.

        Raises:
            RuntimeError: if called before `fit` (propagated from `threshold`).
        """
        return np.asarray(scores > self.threshold, dtype=bool)

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
        return self.is_anomaly_from_scores(self.score(X))

    @abstractmethod
    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:
        """Explainability hook: which features drove this specific anomaly score."""

    def top_contributing_features_batch(
        self, X: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[list[str]]:
        """`top_contributing_features` for a whole matrix, one list per row.

        Concrete, with a row-by-row default, so implementing the single-row
        hook is still all a subclass owes this interface. Subclasses MAY
        override it with a vectorized equivalent -- and both shipped detectors
        do, because at real-dataset scale the per-row loop is the difference
        between explaining a sample and explaining every flagged flow. An
        override must produce exactly what this default produces.

        Raises:
            RuntimeError: if called before `fit`.
            ValueError: if `k` is not positive, or `X` holds non-finite values.
        """
        matrix = as_explanation_matrix(X, k)
        return [self.top_contributing_features(row, feature_names, k=k) for row in matrix]
