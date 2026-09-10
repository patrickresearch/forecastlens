"""Rolling-volatility-ratio regime detector -- the simplest, most transparent baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastlens.core.exceptions import InsufficientDataError, InvalidDetectorConfigError
from forecastlens.regime.base import RegimeDetectionResult


@dataclass(frozen=True)
class VolatilityRegimeDetector:
    """Flags a period as high-volatility when short-window vol exceeds long-window vol.

    regime(t) = 1 (high-vol) if std(diff)_short(t) / std(diff)_long(t) > threshold_multiplier
                else 0

    Both rolling windows are trailing (use only values up to and including t),
    so this is safe to run inside a walk-forward loop -- no lookahead.

    Parameters
    ----------
    short_window : int
        Window (in periods) for the fast volatility estimate.
    long_window : int
        Window for the slow baseline volatility estimate; must exceed `short_window`.
    threshold_multiplier : float
        Ratio above which a period is classified high-volatility.
    """

    short_window: int = 10
    long_window: int = 60
    threshold_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if self.short_window < 2:
            raise InvalidDetectorConfigError(f"short_window must be >= 2, got {self.short_window}.")
        if self.long_window <= self.short_window:
            raise InvalidDetectorConfigError(
                f"long_window ({self.long_window}) must be > short_window ({self.short_window})."
            )
        if self.threshold_multiplier <= 0:
            raise InvalidDetectorConfigError("threshold_multiplier must be > 0.")

    def detect(self, values: np.ndarray) -> RegimeDetectionResult:
        values_arr = np.asarray(values, dtype=float)
        n = len(values_arr)
        if n < self.long_window + 2:
            raise InsufficientDataError(
                f"Need at least long_window + 2 ({self.long_window + 2}) values, got {n}."
            )

        diffs = pd.Series(np.diff(values_arr))
        short_vol = diffs.rolling(self.short_window, min_periods=self.short_window).std()
        long_vol = diffs.rolling(self.long_window, min_periods=self.long_window).std()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = short_vol / long_vol
        # NaN (undefined ratio, e.g. 0/0 before both windows fill or during a flat
        # stretch) compares False, so those periods default to the low-vol regime.
        is_high = (ratio > self.threshold_multiplier).to_numpy(dtype=bool)

        regime_labels = np.zeros(n, dtype=int)
        regime_labels[1:] = is_high.astype(int)

        changepoints = tuple(
            int(t) for t in range(1, n) if regime_labels[t] != regime_labels[t - 1]
        )
        return RegimeDetectionResult(
            changepoints=changepoints,
            regime_labels=regime_labels,
            metadata={"detector": "VolatilityRegimeDetector"},
        )
