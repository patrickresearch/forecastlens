import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidDecisionConfigError
from forecastlens.decision.cost_matrix import CostMatrix


def test_from_matrix_basic():
    cm = CostMatrix.from_matrix([[0, 1], [2, 0]], bucket_labels=["low", "high"])
    assert cm.n_buckets == 2
    assert cm.cost(0, 1) == 1.0
    assert cm.cost(1, 0) == 2.0
    assert cm.cost(0, 0) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(InvalidDecisionConfigError):
        CostMatrix.from_matrix([[0, 1]], bucket_labels=["low", "high"])


def test_nan_matrix_raises():
    with pytest.raises(InvalidDecisionConfigError):
        CostMatrix.from_matrix([[0, np.nan], [1, 0]], bucket_labels=["low", "high"])


def test_linear_distance_hand_computed():
    # 4 buckets, under_cost=2 (predicted too low), over_cost=5 (predicted too high)
    cm = CostMatrix.linear_distance(["a", "b", "c", "d"], under_cost=2.0, over_cost=5.0)
    # predicted bucket 0, actual bucket 3 -> under by 3 steps -> 2*3=6
    assert cm.cost(0, 3) == 6.0
    # predicted bucket 3, actual bucket 0 -> over by 3 steps -> 5*3=15
    assert cm.cost(3, 0) == 15.0
    # predicted bucket 1, actual bucket 1 -> correct -> 0
    assert cm.cost(1, 1) == 0.0
    # predicted bucket 2, actual bucket 1 -> over by 1 -> 5
    assert cm.cost(2, 1) == 5.0


def test_linear_distance_negative_costs_raise():
    with pytest.raises(InvalidDecisionConfigError):
        CostMatrix.linear_distance(["a", "b"], under_cost=-1.0, over_cost=1.0)


def test_matrix_is_read_only():
    cm = CostMatrix.from_matrix([[0, 1], [1, 0]], bucket_labels=["a", "b"])
    with pytest.raises(ValueError):
        cm.matrix[0, 0] = 99.0
