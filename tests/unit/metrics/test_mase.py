import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.mase import mase


def test_hand_computed_non_seasonal():
    # y_train = [1,2,3,4,5] -> naive lag-1 errors all 1 -> scale = 1
    # y_true=[6,7], y_pred=[6.5,7.5] -> mae = 0.5 -> mase = 0.5
    result = mase(y_true=[6.0, 7.0], y_pred=[6.5, 7.5], y_train=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(0.5)


def test_perfect_forecast_is_zero():
    result = mase(y_true=[6.0, 7.0], y_pred=[6.0, 7.0], y_train=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(0.0)


def test_seasonal_naive_scale():
    # seasonality=2, y_train = [1, 10, 2, 11, 3] -> lag-2 errors: |2-1|,|11-10|,|3-2| = [1,1,1]
    result = mase(
        y_true=[4.0], y_pred=[6.0], y_train=[1.0, 10.0, 2.0, 11.0, 3.0], seasonality=2
    )
    assert result == pytest.approx(2.0)  # mae=2, scale=1


def test_constant_y_train_raises_zero_division():
    with pytest.raises(InvalidForecastResultError):
        mase(y_true=[1.0], y_pred=[2.0], y_train=[5.0, 5.0, 5.0])


def test_insufficient_y_train_raises():
    with pytest.raises(InvalidForecastResultError):
        mase(y_true=[1.0], y_pred=[2.0], y_train=[5.0], seasonality=1)


def test_invalid_seasonality_raises():
    with pytest.raises(InvalidForecastResultError):
        mase(y_true=[1.0], y_pred=[2.0], y_train=[1.0, 2.0, 3.0], seasonality=0)


def test_shape_mismatch_raises():
    with pytest.raises(InvalidForecastResultError):
        mase(y_true=[1.0, 2.0], y_pred=[1.0], y_train=[1.0, 2.0, 3.0])


@pytest.mark.parametrize("seasonality", [1, 12])
def test_reference_sktime_mean_absolute_scaled_error(seasonality):
    sktime_metrics = pytest.importorskip("sktime.performance_metrics.forecasting")
    rng = np.random.default_rng(0)
    y_train = rng.normal(size=50).cumsum() + 100
    y_true = rng.normal(size=10).cumsum() + y_train[-1]
    y_pred = y_true + rng.normal(scale=0.5, size=10)

    ours = mase(y_true, y_pred, y_train, seasonality=seasonality)

    reference = sktime_metrics.MeanAbsoluteScaledError(sp=seasonality)
    theirs = reference(
        y_true=np.asarray(y_true), y_pred=np.asarray(y_pred), y_train=np.asarray(y_train)
    )
    assert ours == pytest.approx(theirs, rel=1e-8)
