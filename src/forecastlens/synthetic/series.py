"""Shared output container for synthetic generators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forecastlens.core.exceptions import InvalidGeneratorConfigError


@dataclass(frozen=True)
class GeneratedSeries:
    """Output of a synthetic generator, with known ground truth attached.

    For tests and demos only -- never a substitute for real data in
    production forecasting.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
    values : np.ndarray
    regime_labels : np.ndarray
        Integer regime/phase id per timestamp (all zero if the generator
        has no regime switching).
    changepoints : tuple[int, ...]
        Positional indices where a regime switch or structural/behavioral
        change occurs -- ground truth for regime-detector validation.
    metadata : Mapping[str, Any]
        Generator name, seed, and any preset-specific ground truth (e.g.
        promo/jump indices).
    """

    timestamps: pd.DatetimeIndex
    values: np.ndarray
    regime_labels: np.ndarray
    changepoints: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if len(self.values) != n:
            raise InvalidGeneratorConfigError(
                f"values length ({len(self.values)}) != timestamps length ({n})."
            )
        if len(self.regime_labels) != n:
            raise InvalidGeneratorConfigError(
                f"regime_labels length ({len(self.regime_labels)}) != timestamps length ({n})."
            )
        for cp in self.changepoints:
            if not 0 <= cp < n:
                raise InvalidGeneratorConfigError(
                    f"changepoint {cp} out of range for series of length {n}."
                )
