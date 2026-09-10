"""Framework-agnostic forecast output shared by every adapter and evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecastlens.core.exceptions import InvalidForecastResultError, MissingQuantilesError

# Quantiles are allowed to be non-monotonic only within floating-point noise;
# anything beyond this is a genuine crossing and points at a broken forecast.
_QUANTILE_CROSSING_TOLERANCE = 1e-8


@dataclass(frozen=True)
class ForecastResult:
    """Normalized forecast output produced by adapters, consumed by evaluators.

    Immutable by construction so an evaluator can never mutate an adapter's
    output out from under another consumer.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Forecast horizon timestamps, one entry per forecasted period.
    freq : str
        Pandas frequency alias (e.g. "D", "MS") the timestamps are sampled at.
    point : np.ndarray, optional
        Point forecast, one value per timestamp. At least one of `point` or
        `quantiles` is required.
    quantiles : Mapping[float, np.ndarray], optional
        Quantile level (in (0, 1)) -> forecast array, one value per timestamp.
    metadata : Mapping[str, Any]
        Free-form adapter/model metadata (e.g. model name, source framework).
    """

    timestamps: pd.DatetimeIndex
    freq: str
    point: np.ndarray | None = None
    quantiles: Mapping[float, np.ndarray] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            timestamps = pd.DatetimeIndex(self.timestamps)
        except (TypeError, ValueError) as exc:
            raise InvalidForecastResultError(f"Could not parse `timestamps`: {exc}") from exc
        object.__setattr__(self, "timestamps", timestamps)
        n = len(timestamps)

        if self.point is None and not self.quantiles:
            raise InvalidForecastResultError(
                "ForecastResult needs at least one of `point` or `quantiles`."
            )

        if self.point is not None:
            point = np.asarray(self.point, dtype=float)
            if point.shape != (n,):
                raise InvalidForecastResultError(
                    f"`point` has shape {point.shape}, expected ({n},) to match `timestamps`."
                )
            if not np.all(np.isfinite(point)):
                raise InvalidForecastResultError(
                    "`point` contains NaN/inf -- likely a gap or divergence in the "
                    "upstream model output, not a valid forecast."
                )
            point.setflags(write=False)
            object.__setattr__(self, "point", point)

        if self.quantiles:
            normalized: dict[float, np.ndarray] = {}
            for level, values in self.quantiles.items():
                level = float(level)
                if not 0.0 < level < 1.0:
                    raise InvalidForecastResultError(f"Quantile level {level} outside (0, 1).")
                arr = np.asarray(values, dtype=float)
                if arr.shape != (n,):
                    raise InvalidForecastResultError(
                        f"Quantile {level} has shape {arr.shape}, expected ({n},)."
                    )
                if not np.all(np.isfinite(arr)):
                    raise InvalidForecastResultError(
                        f"Quantile {level} contains NaN/inf -- likely a gap or divergence "
                        "in the upstream model output, not a valid forecast."
                    )
                arr.setflags(write=False)
                normalized[level] = arr
            object.__setattr__(self, "quantiles", dict(sorted(normalized.items())))
            self._check_no_quantile_crossing()

    def _check_no_quantile_crossing(self) -> None:
        assert self.quantiles is not None
        levels = list(self.quantiles.keys())
        if len(levels) < 2:
            return
        stacked = np.stack([self.quantiles[lvl] for lvl in levels], axis=0)
        diffs = np.diff(stacked, axis=0)
        if np.any(diffs < -_QUANTILE_CROSSING_TOLERANCE):
            raise InvalidForecastResultError(
                "Quantile crossing detected: a higher quantile level yields a lower "
                "forecast value than a lower quantile level at the same timestamp."
            )

    @property
    def horizon(self) -> int:
        return len(self.timestamps)

    @property
    def has_quantiles(self) -> bool:
        return bool(self.quantiles)

    def quantile_levels(self) -> list[float]:
        return list(self.quantiles.keys()) if self.quantiles else []

    def quantile(self, level: float) -> np.ndarray:
        if not self.quantiles or level not in self.quantiles:
            raise MissingQuantilesError(
                f"Quantile level {level} not available. Present levels: {self.quantile_levels()}."
            )
        return self.quantiles[level]

    def median(self) -> np.ndarray:
        """Best available central forecast: quantile 0.5 if present, else `point`."""
        if self.quantiles and 0.5 in self.quantiles:
            return self.quantiles[0.5]
        if self.point is not None:
            return self.point
        raise MissingQuantilesError("Neither a 0.5 quantile nor a point forecast is available.")
