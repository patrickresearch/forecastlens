"""Decision-relevant bucket accuracy: a cost-weighted confusion matrix over buckets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from forecastlens.core.exceptions import InsufficientDataError, InvalidDecisionConfigError
from forecastlens.decision.buckets import BucketDefinition
from forecastlens.decision.cost_matrix import CostMatrix
from forecastlens.decision.value_score import relative_value_score


@dataclass(frozen=True)
class BucketAccuracyReport:
    """Result of `DecisionRelevantBucketAccuracy.evaluate`.

    `total_cost`, `cost_naive`, and `value_score` are `None` whenever no
    `CostMatrix` was supplied -- callers get confusion-matrix/hit-rate only,
    never an implicit "all errors cost the same" assumption (ROADMAP.md 1.5).
    """

    confusion_matrix: np.ndarray
    bucket_labels: tuple[str, ...]
    hit_rate: float
    n_evaluated: int
    n_skipped_warmup: int
    total_cost: float | None
    cost_naive: float | None
    value_score: float | None


class DecisionRelevantBucketAccuracy:
    """Classifies forecast and realized values into shared, causally-defined buckets.

    Bucket edges come from the realized series' own rolling history (the
    real-world quantity whose regions matter for the decision), so both the
    forecast and the realized value at time `t` are judged against the same
    yardstick. The naive baseline for `value_score` is a one-step
    persistence forecast (`realized[t-1]`), matching MASE's convention.

    Parameters
    ----------
    bucket_definition : BucketDefinition
    cost_matrix : CostMatrix, optional
        Without one, `evaluate()` returns only the confusion matrix and hit rate.
    """

    def __init__(
        self, bucket_definition: BucketDefinition, cost_matrix: CostMatrix | None = None
    ) -> None:
        if cost_matrix is not None and cost_matrix.n_buckets != bucket_definition.n_buckets:
            raise InvalidDecisionConfigError(
                f"cost_matrix has {cost_matrix.n_buckets} buckets, "
                f"bucket_definition has {bucket_definition.n_buckets}."
            )
        self.bucket_definition = bucket_definition
        self.cost_matrix = cost_matrix

    def evaluate(
        self,
        forecast_values: Sequence[float] | np.ndarray,
        realized_values: Sequence[float] | np.ndarray,
    ) -> BucketAccuracyReport:
        forecast_arr = np.asarray(forecast_values, dtype=float)
        realized_arr = np.asarray(realized_values, dtype=float)
        if forecast_arr.shape != realized_arr.shape:
            raise InvalidDecisionConfigError(
                f"forecast_values shape {forecast_arr.shape} != "
                f"realized_values shape {realized_arr.shape}."
            )

        edges = self.bucket_definition.compute_edges(realized_arr)
        predicted_idx = self.bucket_definition.assign(forecast_arr, edges)
        actual_idx = self.bucket_definition.assign(realized_arr, edges)

        valid = (predicted_idx >= 0) & (actual_idx >= 0)
        n_evaluated = int(np.sum(valid))
        n_skipped = int(len(valid) - n_evaluated)
        if n_evaluated == 0:
            raise InsufficientDataError(
                "No period has enough history for a valid bucket assignment -- "
                "increase the series length or lower BucketDefinition.min_periods."
            )

        n_buckets = self.bucket_definition.n_buckets
        confusion = np.zeros((n_buckets, n_buckets), dtype=int)
        for p, a in zip(predicted_idx[valid], actual_idx[valid], strict=True):
            confusion[p, a] += 1
        hit_rate = float(np.trace(confusion) / n_evaluated)

        total_cost = cost_naive = value_score = None
        if self.cost_matrix is not None:
            naive_forecast = np.full(len(realized_arr), np.nan)
            naive_forecast[1:] = realized_arr[:-1]
            naive_idx = self.bucket_definition.assign(naive_forecast, edges)

            # Compare model vs. naive over the periods valid for *both*, so
            # value_score reflects a fair, apples-to-apples cost difference.
            comparable = valid & (naive_idx >= 0)
            if not np.any(comparable):
                raise InsufficientDataError(
                    "No period has a valid naive-baseline comparison "
                    "(needs both a bucket assignment and a prior-period value)."
                )
            total_cost = float(
                sum(
                    self.cost_matrix.cost(p, a)
                    for p, a in zip(predicted_idx[comparable], actual_idx[comparable], strict=True)
                )
            )
            cost_naive = float(
                sum(
                    self.cost_matrix.cost(p, a)
                    for p, a in zip(naive_idx[comparable], actual_idx[comparable], strict=True)
                )
            )
            value_score = relative_value_score(cost_naive, total_cost)

        return BucketAccuracyReport(
            confusion_matrix=confusion,
            bucket_labels=self.bucket_definition.bucket_labels(),
            hit_rate=hit_rate,
            n_evaluated=n_evaluated,
            n_skipped_warmup=n_skipped,
            total_cost=total_cost,
            cost_naive=cost_naive,
            value_score=value_score,
        )
