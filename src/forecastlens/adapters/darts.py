"""Adapter normalizing a `darts.TimeSeries` forecast into a `ForecastResult`.

Requires the `darts` extra (`pip install forecastlens[darts]`) -- imported
lazily inside `DartsAdapter` methods so the core package stays dependency-light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forecastlens.adapters.base import DEFAULT_QUANTILE_LEVELS, quantiles_from_samples
from forecastlens.core.exceptions import UnrecognizedAdapterFormatError
from forecastlens.core.result import ForecastResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from darts import TimeSeries


class DartsAdapter:
    """Normalizes the output of `darts` forecasting models."""

    @staticmethod
    def from_timeseries(
        prediction: TimeSeries,
        model_name: str | None = None,
        quantile_levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
        metadata: dict[str, Any] | None = None,
    ) -> ForecastResult:
        """Convert a `darts.TimeSeries` prediction (from `model.predict(...)`).

        A deterministic prediction (`num_samples=1`, the darts default) becomes
        a point forecast. A probabilistic prediction (`num_samples > 1`)
        becomes empirical quantiles at `quantile_levels`, computed from the
        underlying sample paths.

        Parameters
        ----------
        prediction : darts.TimeSeries
        model_name : str, optional
            Name of the model that produced `prediction`, for metadata only --
            a `TimeSeries` carries no reference back to the model that
            produced it, so this must be supplied by the caller (e.g.
            `type(model).__name__`) rather than guessed.
        quantile_levels : sequence of float in (0, 1)
            Only used for probabilistic (sampled) predictions.
        metadata : dict, optional

        Raises
        ------
        UnrecognizedAdapterFormatError
            If `prediction` is multivariate or not datetime-indexed --
            neither is supported by `ForecastResult` in this version.
        """
        if not prediction.has_datetime_index:
            raise UnrecognizedAdapterFormatError(
                "DartsAdapter requires a datetime-indexed TimeSeries; got a "
                "range/integer-indexed series instead."
            )
        if prediction.n_components != 1:
            raise UnrecognizedAdapterFormatError(
                f"DartsAdapter supports univariate series only, got "
                f"{prediction.n_components} components. Adapt each component separately."
            )

        timestamps = prediction.time_index
        freq = prediction.freq_str
        combined_metadata = {"source": "darts", "model": model_name, **(metadata or {})}

        if prediction.is_deterministic:
            point = prediction.values(copy=False)[:, 0]
            return ForecastResult(
                timestamps=timestamps, freq=freq, point=point, metadata=combined_metadata
            )

        # all_values(): shape (horizon, n_components, n_samples) -> (n_samples, horizon)
        samples = prediction.all_values(copy=False)[:, 0, :].T
        quantiles = quantiles_from_samples(samples, levels=quantile_levels)
        return ForecastResult(
            timestamps=timestamps, freq=freq, quantiles=quantiles, metadata=combined_metadata
        )
