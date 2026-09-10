import dataclasses

import numpy as np
import pandas as pd
import pytest

from forecastlens.core import ForecastResult, InvalidForecastResultError, MissingQuantilesError


def _timestamps(n: int = 3) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_point_only_construction():
    result = ForecastResult(timestamps=_timestamps(), freq="D", point=[1.0, 2.0, 3.0])
    assert result.horizon == 3
    assert not result.has_quantiles
    np.testing.assert_array_equal(result.median(), [1.0, 2.0, 3.0])


def test_quantiles_only_construction():
    result = ForecastResult(
        timestamps=_timestamps(),
        freq="D",
        quantiles={0.1: [0.5, 1.5, 2.5], 0.5: [1.0, 2.0, 3.0], 0.9: [1.5, 2.5, 3.5]},
    )
    assert result.has_quantiles
    assert result.quantile_levels() == [0.1, 0.5, 0.9]
    np.testing.assert_array_equal(result.median(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.quantile(0.1), [0.5, 1.5, 2.5])


def test_missing_both_point_and_quantiles_raises():
    with pytest.raises(InvalidForecastResultError):
        ForecastResult(timestamps=_timestamps(), freq="D")


def test_point_shape_mismatch_raises():
    with pytest.raises(InvalidForecastResultError):
        ForecastResult(timestamps=_timestamps(3), freq="D", point=[1.0, 2.0])


def test_quantile_shape_mismatch_raises():
    with pytest.raises(InvalidForecastResultError):
        ForecastResult(timestamps=_timestamps(3), freq="D", quantiles={0.5: [1.0, 2.0]})


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_quantile_level_out_of_range_raises(level):
    with pytest.raises(InvalidForecastResultError):
        ForecastResult(timestamps=_timestamps(), freq="D", quantiles={level: [1.0, 2.0, 3.0]})


def test_quantile_crossing_raises():
    # 0.9-quantile below 0.1-quantile at the second timestamp: physically inconsistent.
    with pytest.raises(InvalidForecastResultError):
        ForecastResult(
            timestamps=_timestamps(),
            freq="D",
            quantiles={0.1: [0.5, 5.0, 2.5], 0.9: [1.5, 1.0, 3.5]},
        )


def test_median_falls_back_to_point_when_no_0_5_quantile():
    result = ForecastResult(
        timestamps=_timestamps(),
        freq="D",
        point=[1.0, 2.0, 3.0],
        quantiles={0.1: [0.5, 1.5, 2.5], 0.9: [1.5, 2.5, 3.5]},
    )
    np.testing.assert_array_equal(result.median(), [1.0, 2.0, 3.0])


def test_median_raises_without_point_or_0_5_quantile():
    result = ForecastResult(
        timestamps=_timestamps(),
        freq="D",
        quantiles={0.1: [0.5, 1.5, 2.5], 0.9: [1.5, 2.5, 3.5]},
    )
    with pytest.raises(MissingQuantilesError):
        result.median()


def test_quantile_raises_for_missing_level():
    result = ForecastResult(timestamps=_timestamps(), freq="D", point=[1.0, 2.0, 3.0])
    with pytest.raises(MissingQuantilesError):
        result.quantile(0.5)


def test_is_frozen():
    result = ForecastResult(timestamps=_timestamps(), freq="D", point=[1.0, 2.0, 3.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.freq = "W"


def test_point_array_is_read_only():
    result = ForecastResult(timestamps=_timestamps(), freq="D", point=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        result.point[0] = 99.0


def test_timestamps_accepts_list_of_dates():
    result = ForecastResult(
        timestamps=["2024-01-01", "2024-01-02"], freq="D", point=[1.0, 2.0]
    )
    assert isinstance(result.timestamps, pd.DatetimeIndex)


def test_invalid_timestamps_raise():
    with pytest.raises(InvalidForecastResultError):
        ForecastResult(timestamps=["not-a-date", "also-not"], freq="D", point=[1.0, 2.0])
