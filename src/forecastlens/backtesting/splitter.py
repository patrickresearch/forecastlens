"""Leak-safe walk-forward splitter for time-series backtesting."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from dataclasses import dataclass

import numpy as np

from forecastlens.core.exceptions import ForecastLensError, InsufficientDataError, LeakageError


@dataclass(frozen=True)
class Split:
    """One walk-forward fold: positional indices into the original series."""

    train_idx: np.ndarray
    test_idx: np.ndarray


class LeakSafeWalkForwardSplitter:
    """Walk-forward CV splitter that guarantees max(train_idx) < min(test_idx).

    Operates purely on position, not calendar time: irregular frequencies or
    gaps in the input timestamps need no special handling, as long as the
    timestamps are sorted ascending.

    Parameters
    ----------
    horizon : int
        Number of periods in each test fold.
    min_train_size : int
        Minimum number of periods in the first training fold.
    step : int
        Number of periods to advance the test window between folds.
    max_train_size : int, optional
        If given, use a rolling window of this size instead of an expanding
        window that keeps all history from the start of the series.
    gap : int
        Number of periods excluded between the end of training and the
        start of the test fold (e.g. to model a reporting lag).
    """

    def __init__(
        self,
        horizon: int,
        min_train_size: int,
        step: int = 1,
        max_train_size: int | None = None,
        gap: int = 0,
    ) -> None:
        if horizon < 1:
            raise ForecastLensError(f"horizon must be >= 1, got {horizon}.")
        if min_train_size < 1:
            raise ForecastLensError(f"min_train_size must be >= 1, got {min_train_size}.")
        if step < 1:
            raise ForecastLensError(f"step must be >= 1, got {step}.")
        if gap < 0:
            raise ForecastLensError(f"gap must be >= 0, got {gap}.")
        if max_train_size is not None and max_train_size < 1:
            raise ForecastLensError(f"max_train_size must be >= 1, got {max_train_size}.")

        self.horizon = horizon
        self.min_train_size = min_train_size
        self.step = step
        self.max_train_size = max_train_size
        self.gap = gap

    def split(self, timestamps: Sized) -> Iterator[Split]:
        """Yield `Split`s of positional indices into `timestamps`, oldest first.

        Raises `InsufficientDataError` if not a single valid split fits.
        """
        n = len(timestamps)
        if (
            hasattr(timestamps, "is_monotonic_increasing")
            and not timestamps.is_monotonic_increasing
        ):
            raise ForecastLensError(
                "timestamps must be sorted ascending for walk-forward splitting."
            )

        produced_any = False
        test_start = self.min_train_size + self.gap
        while test_start + self.horizon <= n:
            train_end = test_start - self.gap
            train_start = (
                0 if self.max_train_size is None else max(0, train_end - self.max_train_size)
            )
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_start + self.horizon)

            if train_idx.size == 0:
                raise ForecastLensError(
                    "Computed an empty training window -- min_train_size/gap/max_train_size "
                    "combination leaves no training data."
                )
            if train_idx.max() >= test_idx.min():
                # Should be unreachable for gap >= 0; guards against a future
                # refactor of the index arithmetic above silently reintroducing a leak.
                raise LeakageError(
                    f"Leak detected: train_idx max ({train_idx.max()}) >= "
                    f"test_idx min ({test_idx.min()})."
                )

            produced_any = True
            yield Split(train_idx=train_idx, test_idx=test_idx)
            test_start += self.step

        if not produced_any:
            required = self.min_train_size + self.gap + self.horizon
            raise InsufficientDataError(
                f"n={n} samples cannot produce a single split with min_train_size="
                f"{self.min_train_size}, gap={self.gap}, horizon={self.horizon} "
                f"(need at least {required})."
            )

    def n_splits(self, n_samples: int) -> int:
        """Number of splits `split()` yields for a series of length `n_samples`."""
        first_test_start = self.min_train_size + self.gap
        last_valid_test_start = n_samples - self.horizon
        if last_valid_test_start < first_test_start:
            return 0
        return (last_valid_test_start - first_test_start) // self.step + 1
