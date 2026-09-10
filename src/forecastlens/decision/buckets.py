"""Rolling-quantile decision buckets, computed causally (no lookahead)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastlens.core.exceptions import InvalidDecisionConfigError


@dataclass(frozen=True)
class BucketDefinition:
    """Splits a series into `n_buckets` regions using rolling, causal quantile edges.

    The edge values at time `t` are computed from history strictly before
    `t` (a length-`window` window, shifted by one period), so they carry
    exactly the same leak-safety guarantee as `LeakSafeWalkForwardSplitter`:
    changing data at or after `t` can never change the edges at `t`. The
    first `min_periods` observations of any series have no valid edges yet
    (too little history) and are reported as skipped by callers, not
    silently assigned a bucket.

    Parameters
    ----------
    n_buckets : int
        Number of buckets (>= 2).
    window : int
        Rolling window size, in periods, used to compute quantile edges.
    min_periods : int
        Minimum history required before the first valid edge; must be <= window.
    labels : tuple[str, ...], optional
        Custom bucket labels; defaults to "bucket_0", "bucket_1", ...
    """

    n_buckets: int
    window: int
    min_periods: int
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_buckets < 2:
            raise InvalidDecisionConfigError(f"n_buckets must be >= 2, got {self.n_buckets}.")
        if self.min_periods < 2:
            raise InvalidDecisionConfigError(f"min_periods must be >= 2, got {self.min_periods}.")
        if self.window < self.min_periods:
            raise InvalidDecisionConfigError(
                f"window ({self.window}) must be >= min_periods ({self.min_periods})."
            )
        if self.labels is not None and len(self.labels) != self.n_buckets:
            raise InvalidDecisionConfigError(
                f"labels has {len(self.labels)} entries, expected {self.n_buckets}."
            )

    def bucket_labels(self) -> tuple[str, ...]:
        return self.labels or tuple(f"bucket_{i}" for i in range(self.n_buckets))

    def compute_edges(self, history: Sequence[float] | np.ndarray) -> np.ndarray:
        """Causal rolling quantile edges, one row per period of `history`.

        Returns an array of shape (len(history), n_buckets - 1); rows before
        `min_periods` observations are available are all-NaN.
        """
        history_arr = np.asarray(history, dtype=float)
        levels = np.linspace(0.0, 1.0, self.n_buckets + 1)[1:-1]
        # shift(1): the edge used to classify period t must only see periods < t.
        shifted = pd.Series(history_arr).shift(1)
        rolling = shifted.rolling(self.window, min_periods=self.min_periods)
        edges = np.stack([rolling.quantile(level).to_numpy() for level in levels], axis=1)
        # Guard against near-tie numerical noise producing a non-monotonic row.
        return np.sort(edges, axis=1)

    def assign(self, values: Sequence[float] | np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Bucket index per value, using the matching row of `edges`.

        Returns -1 where the value is NaN or the edge row is not yet valid
        (insufficient history) -- callers must treat -1 as "undefined", not
        as a real bucket.
        """
        values_arr = np.asarray(values, dtype=float)
        n = len(values_arr)
        if edges.shape[0] != n:
            raise InvalidDecisionConfigError(
                f"edges has {edges.shape[0]} rows, expected {n} to match `values`."
            )
        bucket_idx = np.full(n, -1, dtype=int)
        for t in range(n):
            if np.isnan(values_arr[t]) or np.any(np.isnan(edges[t])):
                continue
            bucket_idx[t] = int(np.searchsorted(edges[t], values_arr[t], side="right"))
        return bucket_idx
