"""Shared helpers for adapters normalizing framework output to `ForecastResult`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from forecastlens.core.exceptions import UnrecognizedAdapterFormatError

DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def quantiles_from_samples(
    samples: np.ndarray, levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS
) -> dict[float, np.ndarray]:
    """Empirical quantiles from a probabilistic forecast's sample paths.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, horizon)
    levels : sequence of float in (0, 1)

    Returns
    -------
    dict mapping quantile level -> array of shape (horizon,).
    """
    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.ndim != 2:
        raise UnrecognizedAdapterFormatError(
            f"Expected samples with shape (n_samples, horizon), got shape {samples_arr.shape}."
        )
    return {float(level): np.quantile(samples_arr, level, axis=0) for level in levels}
