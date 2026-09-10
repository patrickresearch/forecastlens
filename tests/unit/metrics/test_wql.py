import pytest

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.wql import mean_weighted_quantile_loss, weighted_quantile_loss


def test_single_quantile_hand_computed():
    # y_true=[10, 20], q=0.5 preds=[8, 22]
    # pinball: point1 y>=f: 0.5*2=1; point2 y<f: 0.5*2=1 -> sum=2
    # wQL = 2 * 2 / sum(|y_true|) = 4 / 30
    result = weighted_quantile_loss([10.0, 20.0], {0.5: [8.0, 22.0]})
    assert result == pytest.approx({0.5: 4.0 / 30.0})


def test_multiple_quantiles_and_mean():
    y_true = [10.0, 20.0]
    quantiles = {0.5: [8.0, 22.0], 0.1: [8.0, 22.0]}
    per_q = weighted_quantile_loss(y_true, quantiles)
    assert set(per_q) == {0.5, 0.1}
    mean_wql = mean_weighted_quantile_loss(y_true, quantiles)
    assert mean_wql == pytest.approx(sum(per_q.values()) / 2)


def test_zero_sum_y_true_raises():
    with pytest.raises(InvalidForecastResultError):
        weighted_quantile_loss([0.0, 0.0], {0.5: [1.0, -1.0]})


def test_empty_quantiles_raises():
    with pytest.raises(InvalidForecastResultError):
        weighted_quantile_loss([1.0], {})
