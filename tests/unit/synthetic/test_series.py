import pandas as pd
import pytest

from forecastlens.core.exceptions import InvalidGeneratorConfigError
from forecastlens.synthetic.series import GeneratedSeries


def _timestamps(n=3):
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_valid_construction():
    series = GeneratedSeries(
        timestamps=_timestamps(), values=[1.0, 2.0, 3.0], regime_labels=[0, 0, 1], changepoints=(2,)
    )
    assert series.changepoints == (2,)


def test_values_length_mismatch_raises():
    with pytest.raises(InvalidGeneratorConfigError):
        GeneratedSeries(
            timestamps=_timestamps(), values=[1.0, 2.0], regime_labels=[0, 0, 0], changepoints=()
        )


def test_regime_labels_length_mismatch_raises():
    with pytest.raises(InvalidGeneratorConfigError):
        GeneratedSeries(
            timestamps=_timestamps(),
            values=[1.0, 2.0, 3.0],
            regime_labels=[0, 0],
            changepoints=(),
        )


def test_out_of_range_changepoint_raises():
    with pytest.raises(InvalidGeneratorConfigError):
        GeneratedSeries(
            timestamps=_timestamps(),
            values=[1.0, 2.0, 3.0],
            regime_labels=[0, 0, 0],
            changepoints=(5,),
        )
