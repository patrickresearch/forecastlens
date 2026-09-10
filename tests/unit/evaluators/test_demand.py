import numpy as np
import pandas as pd
import pytest

from forecastlens.core.exceptions import DegenerateSeriesError, MismatchedLengthError
from forecastlens.core.result import ForecastResult
from forecastlens.evaluators.demand import DemandForecastEvaluator


def _forecast(point=None, quantiles=None, n=4):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    return ForecastResult(timestamps=timestamps, freq="D", point=point, quantiles=quantiles)


def test_wmape_bias_intermittency_hand_computed():
    y_true = [10.0, 0.0, 20.0, 0.0]
    forecast = _forecast(point=[12.0, 2.0, 18.0, 1.0])
    report = DemandForecastEvaluator().evaluate(forecast, y_true)

    # denom = 30; wmape = (2+2+2+1)/30; bias = (2+2-2+1)/30
    assert report.wmape == pytest.approx(7.0 / 30.0)
    assert report.bias == pytest.approx(3.0 / 30.0)
    assert report.intermittency_rate == pytest.approx(0.5)
    assert report.n_periods == 4
    assert report.service_level is None


def test_service_level_hand_computed():
    y_true = [10.0, 0.0, 20.0, 0.0]
    # 0.5-quantile just needs to be <= the 0.9-quantile pointwise (ForecastResult's
    # only cross-quantile constraint); it need not equal y_true.
    forecast = _forecast(
        quantiles={0.5: [5.0, 2.0, 15.0, 2.0], 0.9: [8.0, 5.0, 25.0, 5.0]}
    )
    report = DemandForecastEvaluator(service_level_quantile=0.9).evaluate(forecast, y_true)
    # 10<=8 False, 0<=5 True, 20<=25 True, 0<=5 True -> 3/4
    assert report.service_level == pytest.approx(0.75)
    assert report.service_level_quantile == 0.9


def test_all_zero_y_true_raises():
    forecast = _forecast(point=[1.0, 1.0, 1.0, 1.0])
    with pytest.raises(DegenerateSeriesError):
        DemandForecastEvaluator().evaluate(forecast, [0.0, 0.0, 0.0, 0.0])


def test_length_mismatch_raises():
    forecast = _forecast(point=[1.0, 1.0, 1.0, 1.0])
    with pytest.raises(MismatchedLengthError):
        DemandForecastEvaluator().evaluate(forecast, [1.0, 2.0])


@pytest.mark.parametrize("bad_level", [0.0, 1.0, -0.1, 1.5])
def test_invalid_service_level_quantile_raises(bad_level):
    with pytest.raises(DegenerateSeriesError):
        DemandForecastEvaluator(service_level_quantile=bad_level)


def test_median_used_when_quantiles_present():
    forecast = _forecast(quantiles={0.5: [10.0, 10.0, 10.0, 10.0]})
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    report = DemandForecastEvaluator().evaluate(forecast, y_true)
    assert report.wmape == pytest.approx(0.0)
    assert report.bias == pytest.approx(0.0)
