"""Point-forecast error metrics: MAE, RMSE, WAPE, and a WAPE-derived accuracy figure."""

from __future__ import annotations

import numpy as np

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.pinball import ArrayLike


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise InvalidForecastResultError(
            f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}."
        )


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    _validate_shapes(y_true_arr, y_pred_arr)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    _validate_shapes(y_true_arr, y_pred_arr)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted Absolute Percentage Error: sum(|y_true - y_pred|) / sum(|y_true|).

    Unlike plain MAPE, WAPE divides by the *sum* of actuals rather than each
    actual individually, so a handful of near-zero periods can't blow up the
    score -- the same reasoning `DemandForecastEvaluator`'s WMAPE relies on
    (WAPE and WMAPE are the same formula under different domain names).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    _validate_shapes(y_true_arr, y_pred_arr)
    denom = float(np.sum(np.abs(y_true_arr)))
    if denom == 0.0:
        raise InvalidForecastResultError("Cannot compute WAPE: sum(|y_true|) is zero.")
    return float(np.sum(np.abs(y_true_arr - y_pred_arr))) / denom


def forecast_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """1 - WAPE, the common FP&A/demand-planning "forecast accuracy %" convention.

    Not related to classification accuracy. Can go negative when WAPE > 1
    (a forecast worse than predicting zero everywhere).
    """
    return 1.0 - wape(y_true, y_pred)
