"""Adapter normalizing a `neuralforecast` prediction DataFrame into a `ForecastResult`.

Requires the `neuralforecast` extra (`pip install forecastlens[neuralforecast]`).

`NeuralForecast.predict()` returns a DataFrame with one column per model
output; for probabilistic losses (e.g. `MQLoss`), those columns follow an
internal naming convention (`f"{model_name}{output_name}"`, e.g.
`"NHITS-lo-80.0"`) that maps to quantile levels only via the loss object's
own `quantiles`/`output_names` attributes -- parsing the column name string
itself would be brittle across neuralforecast versions, so this adapter
introspects the trained model's loss instead of guessing from strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from forecastlens.core.exceptions import UnrecognizedAdapterFormatError
from forecastlens.core.result import ForecastResult

if TYPE_CHECKING:
    from neuralforecast.common._base_model import BaseModel


class NeuralForecastAdapter:
    """Normalizes the output of `NeuralForecast.predict()`."""

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        model: BaseModel,
        model_name: str | None = None,
        unique_id: str | None = None,
        timestamp_col: str = "ds",
        freq: str = "D",
        metadata: dict[str, Any] | None = None,
    ) -> ForecastResult:
        """Convert one series' rows of a `NeuralForecast.predict()` DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Output of `nf.predict()` (or `nf.predict(df=...)`).
        model : neuralforecast model instance
            The trained model that produced `df`'s forecast column(s) (e.g.
            `nf.models[0]`) -- used to read the quantile-level -> column-name
            mapping from `model.loss`, not to make new predictions.
        model_name : str, optional
            Column-name prefix `neuralforecast` used; defaults to
            `type(model).__name__`, matching the default `predict()` behavior.
        unique_id : str, optional
            Which series to extract if `df` contains multiple. Required
            whenever `df` has more than one distinct `unique_id`.
        timestamp_col : str
        freq : str
            Pandas frequency alias for the resulting `ForecastResult`.
        metadata : dict, optional
        """
        if "unique_id" not in df.columns:
            raise UnrecognizedAdapterFormatError(
                "Expected a neuralforecast prediction DataFrame with a 'unique_id' column."
            )
        if timestamp_col not in df.columns:
            raise UnrecognizedAdapterFormatError(
                f"timestamp_col {timestamp_col!r} not found in DataFrame "
                f"columns {list(df.columns)}."
            )

        ids = df["unique_id"].unique()
        if unique_id is None:
            if len(ids) != 1:
                raise UnrecognizedAdapterFormatError(
                    f"df contains {len(ids)} distinct unique_id values {list(ids)}; "
                    "pass `unique_id` to select one series."
                )
            unique_id = ids[0]
        elif unique_id not in ids:
            raise UnrecognizedAdapterFormatError(f"unique_id {unique_id!r} not found in df.")

        series_df = df[df["unique_id"] == unique_id].sort_values(timestamp_col)
        name = model_name or type(model).__name__
        loss = model.loss

        combined_metadata = {
            "source": "neuralforecast",
            "model": name,
            "unique_id": unique_id,
            **(metadata or {}),
        }
        timestamps = pd.DatetimeIndex(series_df[timestamp_col])

        if not hasattr(loss, "quantiles"):
            point_col = name
            if point_col not in series_df.columns:
                raise UnrecognizedAdapterFormatError(
                    f"Expected point-forecast column {point_col!r}, "
                    f"got columns {list(series_df.columns)}."
                )
            point = series_df[point_col].to_numpy(dtype=float)
            return ForecastResult(
                timestamps=timestamps, freq=freq, point=point, metadata=combined_metadata
            )

        # loss.quantiles is a float32 torch tensor -- e.g. 0.1 round-trips as
        # 0.10000000149011612. Round back to a sane precision so levels compare
        # cleanly against the values users actually specified.
        levels = [round(float(q), 6) for q in loss.quantiles]
        quantiles: dict[float, Any] = {}
        for level, suffix in zip(levels, loss.output_names, strict=True):
            col = f"{name}{suffix}"
            if col not in series_df.columns:
                raise UnrecognizedAdapterFormatError(
                    f"Expected quantile column {col!r} for level {level}, "
                    f"got columns {list(series_df.columns)}."
                )
            quantiles[level] = series_df[col].to_numpy(dtype=float)

        return ForecastResult(
            timestamps=timestamps, freq=freq, quantiles=quantiles, metadata=combined_metadata
        )
