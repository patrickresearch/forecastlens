import numpy as np
import pandas as pd
import pytest

darts = pytest.importorskip("darts")

from darts import TimeSeries  # noqa: E402
from darts.models import ExponentialSmoothing, NaiveDrift  # noqa: E402

from forecastlens.adapters.darts import DartsAdapter  # noqa: E402
from forecastlens.core.exceptions import UnrecognizedAdapterFormatError  # noqa: E402


def _univariate_series(n=50):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    values = np.sin(np.arange(n) / 5.0) * 10 + 100
    return TimeSeries.from_times_and_values(dates, values)


def test_deterministic_prediction_becomes_point_forecast():
    series = _univariate_series()
    model = NaiveDrift()
    model.fit(series)
    prediction = model.predict(5)

    result = DartsAdapter.from_timeseries(prediction, model_name=type(model).__name__)

    assert not result.has_quantiles
    assert result.point is not None
    assert result.horizon == 5
    assert result.freq == "D"
    np.testing.assert_array_equal(result.timestamps, prediction.time_index)
    np.testing.assert_allclose(result.point, prediction.values(copy=False)[:, 0])
    assert result.metadata["source"] == "darts"
    assert result.metadata["model"] == "NaiveDrift"


def test_probabilistic_prediction_becomes_quantiles():
    series = _univariate_series()
    model = ExponentialSmoothing()
    model.fit(series)
    prediction = model.predict(5, num_samples=200)

    result = DartsAdapter.from_timeseries(prediction, quantile_levels=[0.1, 0.5, 0.9])

    assert result.has_quantiles
    assert result.point is None
    assert result.quantile_levels() == [0.1, 0.5, 0.9]
    # Empirical quantiles from 200 samples should bracket the mean reasonably.
    all_values = prediction.all_values(copy=False)[:, 0, :]
    mean = all_values.mean(axis=1)
    assert np.all(result.quantile(0.1) <= mean + 1e-6)
    assert np.all(result.quantile(0.9) >= mean - 1e-6)


def test_multivariate_series_raises():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    mv_series = TimeSeries.from_times_and_values(dates, rng.random((30, 2)))
    model = NaiveDrift()
    model.fit(mv_series)
    prediction = model.predict(5)

    with pytest.raises(UnrecognizedAdapterFormatError):
        DartsAdapter.from_timeseries(prediction)


def test_range_indexed_series_raises():
    values = np.arange(30, dtype=float)
    series = TimeSeries.from_values(values)
    model = NaiveDrift()
    model.fit(series)
    prediction = model.predict(5)

    with pytest.raises(UnrecognizedAdapterFormatError):
        DartsAdapter.from_timeseries(prediction)


def test_non_daily_frequency_round_trips():
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    values = np.arange(24, dtype=float) + 100
    series = TimeSeries.from_times_and_values(dates, values)
    model = NaiveDrift()
    model.fit(series)
    prediction = model.predict(3)

    result = DartsAdapter.from_timeseries(prediction)
    assert result.freq == "MS"
