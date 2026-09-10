"""Cost matrix over decision buckets: cost(predicted_bucket, actual_bucket)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from forecastlens.core.exceptions import InvalidDecisionConfigError


@dataclass(frozen=True)
class CostMatrix:
    """Cost incurred when the model predicts one bucket but another occurs.

    `matrix[i, j]` is the cost of predicting bucket `i` when the actual
    bucket was `j`. Diagonal entries (correct predictions) are typically
    zero but this is not enforced -- some workflows cost even a correct
    "high" call (e.g. hedging fees), which is a legitimate use case.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_buckets, n_buckets)
    bucket_labels : tuple[str, ...]
        Human-readable label per bucket index, same order as `matrix`.
    """

    matrix: np.ndarray
    bucket_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        n = len(self.bucket_labels)
        if matrix.shape != (n, n):
            raise InvalidDecisionConfigError(
                f"matrix has shape {matrix.shape}, expected ({n}, {n}) to match {n} bucket_labels."
            )
        if not np.all(np.isfinite(matrix)):
            raise InvalidDecisionConfigError("matrix must not contain NaN/inf.")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bucket_labels", tuple(self.bucket_labels))

    @property
    def n_buckets(self) -> int:
        return len(self.bucket_labels)

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, bucket_labels: Sequence[str]) -> CostMatrix:
        """Explicit cost matrix, e.g. for asymmetric or hand-tuned business costs."""
        return cls(matrix=np.asarray(matrix, dtype=float), bucket_labels=tuple(bucket_labels))

    @classmethod
    def linear_distance(
        cls, bucket_labels: Sequence[str], under_cost: float, over_cost: float
    ) -> CostMatrix:
        """Cost grows linearly with bucket distance, at different rates per direction.

        cost(i, j) = under_cost * (j - i)  if predicted too low  (j > i)
                   = over_cost  * (i - j)  if predicted too high (i > j)
                   = 0                     if correct

        Convenience for the common case where under- and over-estimation
        carry different per-bucket-step penalties (e.g. running out of
        stock vs. holding excess inventory).
        """
        if under_cost < 0 or over_cost < 0:
            raise InvalidDecisionConfigError("under_cost/over_cost must be >= 0.")
        n = len(bucket_labels)
        matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if j > i:
                    matrix[i, j] = under_cost * (j - i)
                elif i > j:
                    matrix[i, j] = over_cost * (i - j)
        return cls(matrix=matrix, bucket_labels=tuple(bucket_labels))

    def cost(self, predicted_idx: int, actual_idx: int) -> float:
        return float(self.matrix[predicted_idx, actual_idx])
