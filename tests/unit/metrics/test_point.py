import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.point import forecast_accuracy, mae, rmse, wape


def test_mae_hand_computed():
    # |1-2|+|2-2|+|3-2|+|4-2| = 1+0+1+2 = 4 -> mean = 1.0
    assert mae([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]) == pytest.approx(1.0)


def test_rmse_hand_computed():
    # squared errors: 1, 0, 1, 4 -> mean=1.5 -> sqrt=1.224744871...
    assert rmse([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]) == pytest.approx(np.sqrt(1.5))


def test_wape_hand_computed():
    # sum(|diff|)=4, sum(|y_true|)=1+2+3+4=10 -> 0.4
    assert wape([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]) == pytest.approx(0.4)


def test_forecast_accuracy_is_one_minus_wape():
    y_true, y_pred = [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]
    assert forecast_accuracy(y_true, y_pred) == pytest.approx(1.0 - wape(y_true, y_pred))


def test_perfect_forecast():
    y = [1.0, 2.0, 3.0]
    assert mae(y, y) == pytest.approx(0.0)
    assert rmse(y, y) == pytest.approx(0.0)
    assert wape(y, y) == pytest.approx(0.0)
    assert forecast_accuracy(y, y) == pytest.approx(1.0)


def test_rmse_penalizes_large_errors_more_than_mae():
    # One large outlier: RMSE should exceed MAE (Jensen's inequality on |x| vs x^2).
    y_true = [0.0, 0.0, 0.0, 0.0]
    y_pred = [1.0, 1.0, 1.0, 10.0]
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_accuracy_negative_when_forecast_worse_than_predicting_zero():
    # WAPE > 1 -> accuracy < 0
    assert forecast_accuracy([1.0, 1.0], [10.0, 10.0]) < 0


@pytest.mark.parametrize("fn", [mae, rmse, wape])
def test_shape_mismatch_raises(fn):
    with pytest.raises(InvalidForecastResultError):
        fn([1.0, 2.0], [1.0])


def test_wape_zero_denominator_raises():
    with pytest.raises(InvalidForecastResultError):
        wape([0.0, 0.0], [1.0, 1.0])
