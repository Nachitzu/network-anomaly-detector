"""Unit tests for the shared `BaseDetector` contract itself.

Two things are pinned here, both of which matter more now that `BaseDetector`
carries concrete methods and not just abstract ones:

- A *minimal* subclass -- one that implements only the four abstract members --
  gets working `is_anomaly` and `is_anomaly_from_scores` for free, without
  overriding anything.
- The concrete detectors do NOT override the shared decision rule, so there is
  exactly one definition of "anomalous" in the project.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.feature_engineering import AutoencoderConfig, IsolationForestConfig
from src.models.autoencoder_model import AutoencoderDetector
from src.models.base_detector import BaseDetector
from src.models.isolation_forest_model import IsolationForestDetector

FEATURE_NAMES = ["a", "b", "c", "d"]


class MinimalDetector(BaseDetector):
    """The smallest possible `BaseDetector`: only the four abstract members.

    Deliberately overrides nothing else, so any test passing here proves the
    concrete helpers on the ABC work for an arbitrary future subclass.
    """

    name = "minimal"

    def __init__(self, scores: np.ndarray, threshold: float = 0.5) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)
        self._threshold = threshold

    def fit(self, X_benign: np.ndarray) -> None:
        return None

    def score(self, X: np.ndarray) -> np.ndarray:
        return self._scores

    @property
    def threshold(self) -> float:
        return self._threshold

    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:
        # Rank by raw feature value, descending -- enough to tell rows apart.
        order = np.argsort(-np.asarray(x, dtype=float), kind="stable")
        return [feature_names[i] for i in order[: min(k, len(feature_names))]]


@pytest.fixture()
def benign_train() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, size=(60, len(FEATURE_NAMES)))


@pytest.fixture(params=["isolation_forest", "autoencoder"])
def fitted_detector(request: pytest.FixtureRequest, benign_train: np.ndarray) -> BaseDetector:
    detector: BaseDetector
    if request.param == "isolation_forest":
        detector = IsolationForestDetector(IsolationForestConfig(n_estimators=20, random_state=42))
    else:
        detector = AutoencoderDetector(
            AutoencoderConfig(hidden_dims=[8, 4], epochs=15, batch_size=8, random_state=42)
        )
    detector.fit(benign_train)
    return detector


# --- a minimal subclass gets the shared behaviour for free -----------------


def test_minimal_subclass_is_instantiable_without_overriding_helpers() -> None:
    detector = MinimalDetector(np.array([0.1, 0.9]))

    assert isinstance(detector, BaseDetector)
    assert type(detector).is_anomaly is BaseDetector.is_anomaly


def test_minimal_subclass_is_anomaly_applies_the_inherited_rule() -> None:
    detector = MinimalDetector(np.array([0.1, 0.9]), threshold=0.5)

    flags = detector.is_anomaly(np.zeros((2, len(FEATURE_NAMES))))

    np.testing.assert_array_equal(flags, np.array([False, True]))


# --- is_anomaly_from_scores ------------------------------------------------


def test_is_anomaly_from_scores_matches_is_anomaly(
    fitted_detector: BaseDetector, benign_train: np.ndarray
) -> None:
    """The reused-scores path must agree exactly with the score-again path."""
    scores = fitted_detector.score(benign_train)

    np.testing.assert_array_equal(
        fitted_detector.is_anomaly_from_scores(scores),
        fitted_detector.is_anomaly(benign_train),
    )


def test_is_anomaly_from_scores_returns_bool_array_of_matching_length(
    fitted_detector: BaseDetector, benign_train: np.ndarray
) -> None:
    scores = fitted_detector.score(benign_train)

    flags = fitted_detector.is_anomaly_from_scores(scores)

    assert flags.dtype == np.bool_
    assert flags.shape == (benign_train.shape[0],)


def test_is_anomaly_from_scores_raises_before_fit() -> None:
    detector = IsolationForestDetector()

    with pytest.raises(RuntimeError):
        detector.is_anomaly_from_scores(np.array([0.1, 0.2]))


# --- one, and only one, definition of "anomalous" --------------------------


@pytest.mark.parametrize("detector_cls", [IsolationForestDetector, AutoencoderDetector])
def test_concrete_detectors_do_not_override_the_decision_rule(
    detector_cls: type[BaseDetector],
) -> None:
    """Pins the "never override this method" rule stated in `BaseDetector`."""
    assert detector_cls.is_anomaly is BaseDetector.is_anomaly
    assert detector_cls.is_anomaly_from_scores is BaseDetector.is_anomaly_from_scores


# --- top_contributing_features_batch ---------------------------------------


def test_minimal_subclass_gets_the_default_row_by_row_batch() -> None:
    """A subclass that implements only the single-row hook still gets a batch."""
    detector = MinimalDetector(np.array([0.0, 0.0]))
    X = np.array([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])

    assert type(detector).top_contributing_features_batch is (
        BaseDetector.top_contributing_features_batch
    )
    assert detector.top_contributing_features_batch(X, FEATURE_NAMES, k=2) == [
        ["a", "b"],
        ["d", "c"],
    ]


def test_batch_matches_the_per_row_loop_it_replaces(
    fitted_detector: BaseDetector, benign_train: np.ndarray
) -> None:
    """The vectorized overrides must agree with the row-by-row default exactly.

    This is the contract every override owes `BaseDetector`; for the shipped
    detectors it holds structurally, since their single-row method simply calls
    the batch one with `n = 1`.
    """
    batched = fitted_detector.top_contributing_features_batch(benign_train, FEATURE_NAMES, k=3)
    looped = [
        fitted_detector.top_contributing_features(row, FEATURE_NAMES, k=3) for row in benign_train
    ]

    assert batched == looped


def test_batch_returns_empty_list_for_an_empty_matrix(fitted_detector: BaseDetector) -> None:
    """No flagged rows must not need a special case at the call site."""
    empty = np.zeros((0, len(FEATURE_NAMES)))

    assert fitted_detector.top_contributing_features_batch(empty, FEATURE_NAMES, k=3) == []


@pytest.mark.parametrize("bad_k", [0, -1])
def test_batch_rejects_non_positive_k(fitted_detector: BaseDetector, bad_k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        fitted_detector.top_contributing_features_batch(
            np.zeros((2, len(FEATURE_NAMES))), FEATURE_NAMES, k=bad_k
        )


def test_batch_rejects_non_finite_values(fitted_detector: BaseDetector) -> None:
    X = np.zeros((2, len(FEATURE_NAMES)))
    X[1, 0] = np.inf

    with pytest.raises(ValueError, match="finite"):
        fitted_detector.top_contributing_features_batch(X, FEATURE_NAMES, k=2)


@pytest.mark.parametrize("detector_cls", [IsolationForestDetector, AutoencoderDetector])
def test_batch_raises_before_fit(detector_cls: type[BaseDetector]) -> None:
    detector = detector_cls()

    with pytest.raises(RuntimeError):
        detector.top_contributing_features_batch(
            np.zeros((2, len(FEATURE_NAMES))), FEATURE_NAMES, k=2
        )
