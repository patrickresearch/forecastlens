"""Continuous Ranked Probability Score, from a quantile grid or from ensemble samples."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from forecastlens.core.exceptions import InvalidForecastResultError
from forecastlens.metrics.pinball import ArrayLike, pinball_loss

_MIN_QUANTILE_LEVELS_FOR_CRPS = 2


def crps_from_quantiles(y_true: ArrayLike, quantiles: Mapping[float, ArrayLike]) -> np.ndarray:
    """CRPS approximated from a finite quantile grid.

    CRPS(F, y) = 2 * integral_0^1 pinball_tau(F^-1(tau), y) dtau, approximated
    as the mean of `2 * pinball_tau` over the given quantile levels (a
    Riemann-sum estimate of the integral). Accuracy improves with a denser
    grid; a 2-level grid is a coarse lower bound, not a precise score.
    """
    if len(quantiles) < _MIN_QUANTILE_LEVELS_FOR_CRPS:
        raise InvalidForecastResultError(
            f"crps_from_quantiles needs at least {_MIN_QUANTILE_LEVELS_FOR_CRPS} quantile "
            f"levels to approximate the integral, got {len(quantiles)}."
        )

    levels = sorted(quantiles)
    per_level_loss = np.stack(
        [pinball_loss(y_true, quantiles[level], level, reduction="none") for level in levels],
        axis=0,
    )
    result: np.ndarray = 2.0 * np.mean(per_level_loss, axis=0)
    return result


def crps_from_samples(y_true: ArrayLike, samples: ArrayLike) -> np.ndarray:
    """Exact empirical CRPS from an ensemble of forecast samples.

    CRPS = mean_i |x_i - y| - mean_{i,j} |x_i - x_j| / 2

    `samples` has shape (n_samples, n_timestamps); `y_true` has shape
    (n_timestamps,). Matches `properscoring.crps_ensemble`'s default
    (non-fair) estimator.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.ndim != 2 or samples_arr.shape[1] != y_true_arr.shape[0]:
        raise InvalidForecastResultError(
            f"samples must have shape (n_samples, {y_true_arr.shape[0]}), "
            f"got {samples_arr.shape}."
        )

    n_samples = samples_arr.shape[0]
    term_data = np.mean(np.abs(samples_arr - y_true_arr[None, :]), axis=0)
    pairwise = np.abs(samples_arr[:, None, :] - samples_arr[None, :, :])
    term_spread = np.sum(pairwise, axis=(0, 1)) / (2.0 * n_samples * n_samples)
    result: np.ndarray = term_data - term_spread
    return result
