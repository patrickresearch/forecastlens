"""Weighted Quantile Loss (WQL) — normalized pinball loss, as used e.g. in GluonTS."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.pinball import ArrayLike, pinball_loss


def weighted_quantile_loss(
    y_true: ArrayLike, quantiles: Mapping[float, ArrayLike]
) -> dict[float, float]:
    """Per-quantile weighted quantile loss.

    wQL_q = 2 * sum_t pinball_q(y_t, f_t) / sum_t |y_t|

    Normalizing by sum(|y_true|) makes the score comparable across series
    of different scale.
    """
    if not quantiles:
        raise InvalidForecastResultError(
            "weighted_quantile_loss needs at least one quantile level."
        )

    y_true_arr = np.asarray(y_true, dtype=float)
    denom = float(np.sum(np.abs(y_true_arr)))
    if denom == 0.0:
        raise InvalidForecastResultError(
            "Cannot compute weighted quantile loss: sum(|y_true|) is zero."
        )

    return {
        float(level): 2.0 * float(pinball_loss(y_true_arr, preds, level, reduction="sum")) / denom
        for level, preds in quantiles.items()
    }


def mean_weighted_quantile_loss(y_true: ArrayLike, quantiles: Mapping[float, ArrayLike]) -> float:
    """Average of `weighted_quantile_loss` across all quantile levels."""
    per_quantile = weighted_quantile_loss(y_true, quantiles)
    return float(np.mean(list(per_quantile.values())))
