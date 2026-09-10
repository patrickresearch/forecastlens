"""Adapter for plain pandas DataFrames -- no forecasting framework required.

Covers any model whose output can be put in a table: one timestamp column,
plus either a point-forecast column or one column per quantile level.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from forecastlens.core.exceptions import UnrecognizedAdapterFormatError
from forecastlens.core.result import ForecastResult


class RawAdapter:
    """Normalizes a tabular forecast output into a `ForecastResult`."""

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        timestamp_col: str,
        freq: str,
        point_col: str | None = None,
        quantile_cols: Mapping[float, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ForecastResult:
        """Build a `ForecastResult` from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        timestamp_col : str
            Column holding forecast horizon timestamps.
        freq : str
            Pandas frequency alias the timestamps are sampled at.
        point_col : str, optional
            Column holding the point forecast.
        quantile_cols : Mapping[float, str], optional
            Quantile level -> column name.
        metadata : Mapping[str, Any], optional
        """
        if point_col is None and not quantile_cols:
            raise UnrecognizedAdapterFormatError(
                "RawAdapter.from_dataframe needs at least one of `point_col` or `quantile_cols`."
            )
        if timestamp_col not in df.columns:
            raise UnrecognizedAdapterFormatError(
                f"timestamp_col {timestamp_col!r} not found in DataFrame "
                f"columns {list(df.columns)}."
            )

        timestamps = pd.DatetimeIndex(df[timestamp_col])

        point = None
        if point_col is not None:
            if point_col not in df.columns:
                raise UnrecognizedAdapterFormatError(
                    f"point_col {point_col!r} not found in DataFrame columns {list(df.columns)}."
                )
            point = df[point_col].to_numpy(dtype=float)

        quantiles = None
        if quantile_cols:
            missing = [col for col in quantile_cols.values() if col not in df.columns]
            if missing:
                raise UnrecognizedAdapterFormatError(
                    f"quantile_cols not found in DataFrame columns: {missing}."
                )
            quantiles = {
                level: df[col].to_numpy(dtype=float) for level, col in quantile_cols.items()
            }

        combined_metadata = {"source": "raw", **dict(metadata or {})}
        return ForecastResult(
            timestamps=timestamps,
            freq=freq,
            point=point,
            quantiles=quantiles,
            metadata=combined_metadata,
        )
