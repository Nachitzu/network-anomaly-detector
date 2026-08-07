"""Unit tests for src.models.feature_ranking.

The ranking tail is shared by both detectors, so it is tested directly on
plain arrays rather than only through a fitted model.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.feature_ranking import as_explanation_matrix, rank_top_features

FEATURE_NAMES = ["a", "b", "c", "d"]


# --- as_explanation_matrix -------------------------------------------------


def test_as_explanation_matrix_promotes_a_single_row_to_two_dimensions() -> None:
    matrix = as_explanation_matrix(np.array([1.0, 2.0, 3.0, 4.0]), k=2)

    assert matrix.shape == (1, 4)
    assert matrix.dtype == np.float64


def test_as_explanation_matrix_leaves_a_matrix_shaped_input_alone() -> None:
    matrix = as_explanation_matrix(np.zeros((3, 4)), k=2)

    assert matrix.shape == (3, 4)


def test_as_explanation_matrix_accepts_an_empty_batch() -> None:
    """No flagged rows is a normal outcome, not an error."""
    matrix = as_explanation_matrix(np.zeros((0, 4)), k=2)

    assert matrix.shape == (0, 4)


@pytest.mark.parametrize("bad_k", [0, -1])
def test_as_explanation_matrix_rejects_non_positive_k(bad_k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        as_explanation_matrix(np.zeros((2, 4)), k=bad_k)


def test_as_explanation_matrix_rejects_non_finite_values() -> None:
    x = np.array([[1.0, np.inf, 3.0, 4.0]])

    with pytest.raises(ValueError, match="finite"):
        as_explanation_matrix(x, k=2)


# --- rank_top_features -----------------------------------------------------


def test_rank_top_features_orders_each_row_descending() -> None:
    contributions = np.array([[0.1, 0.4, 0.2, 0.3], [9.0, 1.0, 5.0, 0.0]])

    ranked = rank_top_features(contributions, FEATURE_NAMES, k=2)

    assert ranked == [["b", "d"], ["a", "c"]]


def test_rank_top_features_caps_k_at_the_number_of_feature_names() -> None:
    ranked = rank_top_features(np.array([[0.1, 0.4, 0.2, 0.3]]), FEATURE_NAMES, k=1000)

    assert len(ranked[0]) == len(FEATURE_NAMES)
    assert set(ranked[0]) == set(FEATURE_NAMES)


def test_rank_top_features_returns_one_list_per_row() -> None:
    ranked = rank_top_features(np.zeros((3, 4)), FEATURE_NAMES, k=2)

    assert len(ranked) == 3
    assert all(len(row) == 2 for row in ranked)


def test_rank_top_features_returns_empty_for_an_empty_batch() -> None:
    assert rank_top_features(np.zeros((0, 4)), FEATURE_NAMES, k=2) == []


def test_rank_top_features_breaks_ties_by_ascending_feature_index() -> None:
    """Exact ties are real -- a zero-variance feature scores 0 for every flow.

    A stable sort makes the winner deterministic and, crucially, the same
    regardless of how many rows are ranked in one call.
    """
    contributions = np.array([[0.5, 0.5, 0.5, 0.5]])

    ranked = rank_top_features(contributions, FEATURE_NAMES, k=4)

    assert ranked[0] == FEATURE_NAMES


def test_rank_top_features_is_independent_of_batch_size() -> None:
    """The same row must rank identically alone and inside a larger batch."""
    row = np.array([0.5, 0.5, 0.2, 0.5])
    batch = np.vstack([np.array([9.0, 0.0, 0.0, 0.0]), row, np.array([0.0, 0.0, 9.0, 0.0])])

    alone = rank_top_features(row.reshape(1, -1), FEATURE_NAMES, k=3)[0]
    within_batch = rank_top_features(batch, FEATURE_NAMES, k=3)[1]

    assert alone == within_batch
