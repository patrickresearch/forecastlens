import pandas as pd
import pytest

from forecastlens.adapters.raw import RawAdapter
from forecastlens.core.exceptions import UnrecognizedAdapterFormatError


def _df():
    return pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=3, freq="D"),
            "point": [1.0, 2.0, 3.0],
            "q10": [0.5, 1.5, 2.5],
            "q90": [1.5, 2.5, 3.5],
        }
    )


def test_point_only():
    result = RawAdapter.from_dataframe(_df(), timestamp_col="ds", freq="D", point_col="point")
    assert not result.has_quantiles
    assert result.metadata["source"] == "raw"


def test_quantiles_only():
    result = RawAdapter.from_dataframe(
        _df(), timestamp_col="ds", freq="D", quantile_cols={0.1: "q10", 0.9: "q90"}
    )
    assert result.quantile_levels() == [0.1, 0.9]


def test_point_and_quantiles():
    result = RawAdapter.from_dataframe(
        _df(),
        timestamp_col="ds",
        freq="D",
        point_col="point",
        quantile_cols={0.1: "q10", 0.9: "q90"},
    )
    assert result.point is not None
    assert result.has_quantiles


def test_missing_timestamp_col_raises():
    with pytest.raises(UnrecognizedAdapterFormatError):
        RawAdapter.from_dataframe(_df(), timestamp_col="nope", freq="D", point_col="point")


def test_missing_point_col_raises():
    with pytest.raises(UnrecognizedAdapterFormatError):
        RawAdapter.from_dataframe(_df(), timestamp_col="ds", freq="D", point_col="nope")


def test_missing_quantile_col_raises():
    with pytest.raises(UnrecognizedAdapterFormatError):
        RawAdapter.from_dataframe(
            _df(), timestamp_col="ds", freq="D", quantile_cols={0.1: "nope"}
        )


def test_neither_point_nor_quantiles_raises():
    with pytest.raises(UnrecognizedAdapterFormatError):
        RawAdapter.from_dataframe(_df(), timestamp_col="ds", freq="D")
