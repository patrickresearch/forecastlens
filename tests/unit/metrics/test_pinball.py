import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.pinball import pinball_loss


def test_over_prediction_hand_computed():
    # y=10, f=12 (over-predicted, y < f) -> (1-q)*(f-y) = 0.1 * 2 = 0.2
    assert pinball_loss([10.0], [12.0], quantile=0.9) == pytest.approx(0.2)


def test_under_prediction_hand_computed():
    # y=10, f=8 (under-predicted, y >= f) -> q*(y-f) = 0.9 * 2 = 1.8
    assert pinball_loss([10.0], [8.0], quantile=0.9) == pytest.approx(1.8)


def test_exact_prediction_is_zero():
    assert pinball_loss([5.0], [5.0], quantile=0.3) == pytest.approx(0.0)


def test_median_quantile_is_half_absolute_error():
    # At q=0.5, pinball loss reduces to 0.5 * |y - f| regardless of sign.
    y = np.array([1.0, 10.0])
    f = np.array([4.0, 2.0])
    expected = 0.5 * np.abs(y - f)
    result = pinball_loss(y, f, quantile=0.5, reduction="none")
    np.testing.assert_allclose(result, expected)


def test_reduction_modes_consistent():
    y = [1.0, 2.0, 3.0]
    f = [1.5, 1.5, 1.5]
    per_point = pinball_loss(y, f, 0.7, reduction="none")
    assert pinball_loss(y, f, 0.7, reduction="mean") == pytest.approx(np.mean(per_point))
    assert pinball_loss(y, f, 0.7, reduction="sum") == pytest.approx(np.sum(per_point))


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.1, 1.2])
def test_quantile_out_of_range_raises(quantile):
    with pytest.raises(InvalidForecastResultError):
        pinball_loss([1.0], [2.0], quantile=quantile)


def test_shape_mismatch_raises():
    with pytest.raises(InvalidForecastResultError):
        pinball_loss([1.0, 2.0], [1.0], quantile=0.5)


def test_unknown_reduction_raises():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], quantile=0.5, reduction="bogus")
