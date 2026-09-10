import pytest

from forecastlens.core.exceptions import ZeroNaiveCostError
from forecastlens.decision.value_score import relative_value_score


def test_model_beats_naive():
    # cost_naive=100, cost_model=60 -> (100-60)/100 = 0.4
    assert relative_value_score(cost_naive=100.0, cost_model=60.0) == pytest.approx(0.4)


def test_model_free_gives_score_one():
    assert relative_value_score(cost_naive=50.0, cost_model=0.0) == pytest.approx(1.0)


def test_model_worse_than_naive_is_negative():
    # cost_naive=50, cost_model=75 -> (50-75)/50 = -0.5
    assert relative_value_score(cost_naive=50.0, cost_model=75.0) == pytest.approx(-0.5)


def test_score_can_exceed_one_when_naive_cost_near_zero():
    # cost_naive=1, cost_model=-4 (a "benefit" cost matrix entry) -> (1 - -4)/1 = 5
    assert relative_value_score(cost_naive=1.0, cost_model=-4.0) == pytest.approx(5.0)


def test_zero_naive_cost_raises():
    with pytest.raises(ZeroNaiveCostError):
        relative_value_score(cost_naive=0.0, cost_model=10.0)
