"""Mean Absolute Scaled Error (Hyndman & Koehler, 2006)."""

from __future__ import annotations

import numpy as np

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.pinball import ArrayLike


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 1,
) -> float:
    """MASE = MAE(forecast) / MAE(in-sample naive seasonal forecast).

    The scale comes from `y_train` (the history available before the
    forecast period), never from `y_true`/`y_pred` — scaling by the test
    period would leak information about the very errors being measured.

    Parameters
    ----------
    y_true, y_pred : array-like, same shape
    y_train : array-like
        In-sample history used only to compute the naive-forecast scale.
    seasonality : int
        Lag of the naive forecast (1 for non-seasonal data, e.g. 12 for
        monthly data with yearly seasonality).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise InvalidForecastResultError(
            f"y_true shape {y_true_arr.shape} != y_pred shape {y_pred_arr.shape}."
        )
    if seasonality < 1:
        raise InvalidForecastResultError(f"seasonality must be >= 1, got {seasonality}.")
    if y_train_arr.ndim != 1 or len(y_train_arr) <= seasonality:
        raise InvalidForecastResultError(
            f"y_train needs more than `seasonality` ({seasonality}) observations to "
            f"compute the naive-forecast scale, got {len(y_train_arr)}."
        )

    naive_errors = np.abs(y_train_arr[seasonality:] - y_train_arr[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale == 0.0:
        raise InvalidForecastResultError(
            "In-sample naive forecast has zero error on y_train — MASE scale "
            "would divide by zero (e.g. a perfectly constant/seasonal series)."
        )

    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    return mae / scale
