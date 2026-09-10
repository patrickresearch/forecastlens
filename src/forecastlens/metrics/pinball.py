"""Pinball (quantile) loss — the building block for WQL and quantile-based CRPS."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from forecastlens.core.exceptions import InvalidForecastResultError

ArrayLike = Sequence[float] | np.ndarray


def pinball_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    quantile: float,
    *,
    reduction: str = "mean",
) -> float | np.ndarray:
    """Pinball loss at a single quantile level.

    L_q(y, f) = q * (y - f)        if y >= f
              = (1 - q) * (f - y)  otherwise

    Parameters
    ----------
    y_true, y_pred : array-like, same shape
    quantile : float in (0, 1)
    reduction : {"mean", "sum", "none"}

    Returns
    -------
    float for "mean"/"sum", per-element `np.ndarray` for "none".
    """
    if not 0.0 < quantile < 1.0:
        raise InvalidForecastResultError(f"quantile must be in (0, 1), got {quantile}.")

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise InvalidForecastResultError(
            f"y_true shape {y_true_arr.shape} != y_pred shape {y_pred_arr.shape}."
        )

    diff = y_true_arr - y_pred_arr
    loss: np.ndarray = np.maximum(quantile * diff, (quantile - 1.0) * diff)

    if reduction == "mean":
        return float(np.mean(loss))
    if reduction == "sum":
        return float(np.sum(loss))
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction {reduction!r}, expected 'mean', 'sum', or 'none'.")
