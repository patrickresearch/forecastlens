"""Two-sided CUSUM (Page's test) change-point detector for mean-level shifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecastlens.core.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidDetectorConfigError,
)
from forecastlens.regime.base import RegimeDetectionResult


@dataclass(frozen=True)
class CUSUMDetector:
    """Detects mean-level shifts via a two-sided cumulative sum (Page's test).

    Baseline mean/std are estimated causally from the first `warmup` periods.
    After each detected changepoint, the reference mean re-anchors to the
    triggering value and the cumulative sums reset, so later shifts remain
    detectable -- a formal alternative to `VolatilityRegimeDetector`'s
    threshold heuristic, targeting level rather than volatility changes.

    Parameters
    ----------
    warmup : int
        Number of initial periods used to estimate the baseline mean/std.
    threshold_std : float
        Detection threshold `h`, in units of the warm-up standard deviation.
    drift_std : float
        Slack/reference value `k` (in units of warm-up std) subtracted from
        each deviation before accumulating -- suppresses drift from noise.
    """

    warmup: int = 30
    threshold_std: float = 5.0
    drift_std: float = 0.5

    def __post_init__(self) -> None:
        if self.warmup < 2:
            raise InvalidDetectorConfigError(f"warmup must be >= 2, got {self.warmup}.")
        if self.threshold_std <= 0:
            raise InvalidDetectorConfigError("threshold_std must be > 0.")
        if self.drift_std < 0:
            raise InvalidDetectorConfigError("drift_std must be >= 0.")

    def detect(self, values: np.ndarray) -> RegimeDetectionResult:
        values_arr = np.asarray(values, dtype=float)
        n = len(values_arr)
        if n <= self.warmup + 1:
            raise InsufficientDataError(
                f"Need more than warmup + 1 ({self.warmup + 1}) values, got {n}."
            )

        mu_warmup = float(np.mean(values_arr[: self.warmup]))
        sigma = float(np.std(values_arr[: self.warmup], ddof=1))
        if sigma == 0.0:
            raise DegenerateSeriesError(
                "Warm-up window has zero variance -- CUSUM threshold/drift are undefined."
            )

        h = self.threshold_std * sigma
        k = self.drift_std * sigma

        regime_labels = np.zeros(n, dtype=int)
        changepoints: list[int] = []
        current_regime = 0
        mu_ref = mu_warmup
        g_pos = 0.0
        g_neg = 0.0

        for t in range(self.warmup, n):
            g_pos = max(0.0, g_pos + (values_arr[t] - mu_ref) - k)
            g_neg = min(0.0, g_neg + (values_arr[t] - mu_ref) + k)
            if g_pos > h or g_neg < -h:
                changepoints.append(t)
                current_regime += 1
                g_pos = 0.0
                g_neg = 0.0
                mu_ref = values_arr[t]
            regime_labels[t] = current_regime

        return RegimeDetectionResult(
            changepoints=tuple(changepoints),
            regime_labels=regime_labels,
            metadata={"detector": "CUSUMDetector", "mu_warmup": mu_warmup, "sigma_warmup": sigma},
        )
