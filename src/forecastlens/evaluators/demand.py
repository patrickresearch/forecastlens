"""Evaluator for material/OEM demand forecasts: WMAPE, bias, service level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecastlens.core.exceptions import DegenerateSeriesError, MismatchedLengthError
from forecastlens.core.result import ForecastResult


@dataclass(frozen=True)
class DemandEvaluationReport:
    wmape: float
    bias: float
    intermittency_rate: float
    n_periods: int
    service_level: float | None
    service_level_quantile: float | None


class DemandForecastEvaluator:
    """WMAPE/bias/service-level evaluation, robust to intermittent (many-zero) demand.

    Uses `forecast.median()` (the 0.5 quantile if present, else the point
    forecast) as the central estimate for WMAPE/bias. WMAPE -- unlike plain
    MAPE -- divides by the sum of actuals rather than each actual
    individually, so it stays well-defined when some periods have zero
    demand; only a fully-zero evaluation window is undefined (fail-loud
    rather than a silent NaN/inf).

    Parameters
    ----------
    service_level_quantile : float, optional
        If given, `evaluate()` also reports the empirical service level --
        the fraction of periods where actual demand did not exceed the
        forecast's quantile at this level (e.g. 0.9 for a P90 safety-stock
        target). Requires the forecast to carry that quantile.
    """

    def __init__(self, service_level_quantile: float | None = None) -> None:
        if service_level_quantile is not None and not 0.0 < service_level_quantile < 1.0:
            raise DegenerateSeriesError(
                f"service_level_quantile must be in (0, 1), got {service_level_quantile}."
            )
        self.service_level_quantile = service_level_quantile

    def evaluate(self, forecast: ForecastResult, y_true: np.ndarray) -> DemandEvaluationReport:
        y_true_arr = np.asarray(y_true, dtype=float)
        if forecast.horizon != len(y_true_arr):
            raise MismatchedLengthError(
                f"forecast horizon ({forecast.horizon}) != len(y_true) ({len(y_true_arr)})."
            )

        central = forecast.median()
        denom = float(np.sum(np.abs(y_true_arr)))
        if denom == 0.0:
            raise DegenerateSeriesError(
                "Cannot compute WMAPE/bias: y_true is all zero over the entire "
                "evaluation window."
            )

        wmape = float(np.sum(np.abs(y_true_arr - central)) / denom)
        bias = float(np.sum(central - y_true_arr) / denom)
        intermittency_rate = float(np.mean(y_true_arr == 0.0))

        service_level = None
        if self.service_level_quantile is not None:
            target = forecast.quantile(self.service_level_quantile)
            service_level = float(np.mean(y_true_arr <= target))

        return DemandEvaluationReport(
            wmape=wmape,
            bias=bias,
            intermittency_rate=intermittency_rate,
            n_periods=len(y_true_arr),
            service_level=service_level,
            service_level_quantile=self.service_level_quantile,
        )
