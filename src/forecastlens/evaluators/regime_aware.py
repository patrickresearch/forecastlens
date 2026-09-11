"""Regime-conditional evaluator for energy/commodity-style forecasts.

Splits CRPS/WQL/calibration/MAE/RMSE/WAPE by the regime detected in the
*realized* series -- the point isn't how well the forecast predicts
regimes, it's whether accuracy holds up once you stop averaging away the
very regime shifts that make the forecasting problem hard in the first
place (roadmap.md's core motivation). The `overall_*` fields are exactly
what you'd get applying each metric the ordinary way, over the whole
period with no segmentation -- comparing them against `per_regime` is the
point of this evaluator, not an afterthought.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from forecastlens.core.exceptions import MismatchedLengthError, MissingQuantilesError
from forecastlens.metrics.crps import crps_from_quantiles
from forecastlens.metrics.point import forecast_accuracy, mae, rmse, wape
from forecastlens.metrics.wql import mean_weighted_quantile_loss

if TYPE_CHECKING:
    from forecastlens.core.result import ForecastResult
    from forecastlens.regime.base import RegimeDetector


def _calibration(y_true: np.ndarray, quantiles: Mapping[float, np.ndarray]) -> dict[float, float]:
    """Empirical coverage per quantile level: mean(y_true <= quantile forecast).

    A well-calibrated level-q quantile forecast should show coverage close to q.
    """
    return {level: float(np.mean(y_true <= arr)) for level, arr in quantiles.items()}


@dataclass(frozen=True)
class RegimeMetrics:
    """CRPS/WQL/calibration/MAE/RMSE/WAPE/accuracy restricted to one detected regime's periods."""

    regime_label: int
    n_periods: int
    crps: float
    wql: float
    calibration: dict[float, float]
    mae: float
    rmse: float
    wape: float
    accuracy: float


@dataclass(frozen=True)
class RegimeAwareEvaluationReport:
    overall_crps: float
    overall_wql: float
    overall_calibration: dict[float, float]
    overall_mae: float
    overall_rmse: float
    overall_wape: float
    overall_accuracy: float
    per_regime: tuple[RegimeMetrics, ...]
    regime_labels: np.ndarray
    changepoints: tuple[int, ...]


class RegimeAwareEvaluator:
    """Conditions CRPS/WQL/calibration on regimes detected in `y_true`.

    Parameters
    ----------
    regime_detector : RegimeDetector
        Either `VolatilityRegimeDetector` or `CUSUMDetector` (or any other
        implementation of the shared protocol) -- swappable without
        changing how this evaluator works.
    """

    def __init__(self, regime_detector: RegimeDetector) -> None:
        self.regime_detector = regime_detector

    def evaluate(self, forecast: ForecastResult, y_true: np.ndarray) -> RegimeAwareEvaluationReport:
        y_true_arr = np.asarray(y_true, dtype=float)
        if forecast.horizon != len(y_true_arr):
            raise MismatchedLengthError(
                f"forecast horizon ({forecast.horizon}) != len(y_true) ({len(y_true_arr)})."
            )
        if not forecast.has_quantiles:
            raise MissingQuantilesError(
                "RegimeAwareEvaluator needs a quantile forecast to compute CRPS/WQL/calibration."
            )
        quantiles = forecast.quantiles
        assert quantiles is not None

        detection = self.regime_detector.detect(y_true_arr)
        regimes = detection.regime_labels
        median = forecast.median()

        overall_crps = float(np.mean(crps_from_quantiles(y_true_arr, quantiles)))
        overall_wql = mean_weighted_quantile_loss(y_true_arr, quantiles)
        overall_calibration = _calibration(y_true_arr, quantiles)
        overall_wape = wape(y_true_arr, median)

        per_regime = []
        for label in sorted(set(regimes.tolist())):
            mask = regimes == label
            y_true_sub = y_true_arr[mask]
            median_sub = median[mask]
            quantiles_sub = {level: arr[mask] for level, arr in quantiles.items()}
            per_regime.append(
                RegimeMetrics(
                    regime_label=int(label),
                    n_periods=int(mask.sum()),
                    crps=float(np.mean(crps_from_quantiles(y_true_sub, quantiles_sub))),
                    wql=mean_weighted_quantile_loss(y_true_sub, quantiles_sub),
                    calibration=_calibration(y_true_sub, quantiles_sub),
                    mae=mae(y_true_sub, median_sub),
                    rmse=rmse(y_true_sub, median_sub),
                    wape=wape(y_true_sub, median_sub),
                    accuracy=forecast_accuracy(y_true_sub, median_sub),
                )
            )

        return RegimeAwareEvaluationReport(
            overall_crps=overall_crps,
            overall_wql=overall_wql,
            overall_calibration=overall_calibration,
            overall_mae=mae(y_true_arr, median),
            overall_rmse=rmse(y_true_arr, median),
            overall_wape=overall_wape,
            overall_accuracy=1.0 - overall_wape,
            per_regime=tuple(per_regime),
            regime_labels=regimes,
            changepoints=detection.changepoints,
        )
