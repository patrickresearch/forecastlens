import numpy as np
import pytest

from forecastlens.core.exceptions import InsufficientDataError, InvalidDecisionConfigError
from forecastlens.decision.bucket_accuracy import DecisionRelevantBucketAccuracy
from forecastlens.decision.buckets import BucketDefinition
from forecastlens.decision.cost_matrix import CostMatrix


def _setup():
    # Same alternating-10/20 series as test_buckets.py's hand-computed edges:
    # edges are NaN for t<4, then a stable median of 15.0 for t>=4.
    realized = np.array([10, 20, 10, 20, 10, 20, 10, 20, 10, 20], dtype=float)
    # Forecast deliberately mismatches at t=4,7,8,9 and matches at t=5,6.
    forecast = np.array([0, 0, 0, 0, 20, 20, 10, 10, 20, 10], dtype=float)
    bucket_def = BucketDefinition(n_buckets=2, window=4, min_periods=4)
    return forecast, realized, bucket_def


def test_hand_computed_confusion_matrix_and_hit_rate():
    forecast, realized, bucket_def = _setup()
    model = DecisionRelevantBucketAccuracy(bucket_def)
    report = model.evaluate(forecast, realized)

    assert report.n_evaluated == 6
    assert report.n_skipped_warmup == 4
    np.testing.assert_array_equal(report.confusion_matrix, [[1, 2], [2, 1]])
    assert report.hit_rate == pytest.approx(2 / 6)
    assert report.total_cost is None
    assert report.cost_naive is None
    assert report.value_score is None


def test_hand_computed_cost_and_value_score():
    forecast, realized, bucket_def = _setup()
    cost_matrix = CostMatrix.from_matrix([[0, 1], [1, 0]], bucket_labels=["low", "high"])
    model = DecisionRelevantBucketAccuracy(bucket_def, cost_matrix=cost_matrix)
    report = model.evaluate(forecast, realized)

    assert report.total_cost == pytest.approx(4.0)
    assert report.cost_naive == pytest.approx(6.0)
    assert report.value_score == pytest.approx((6.0 - 4.0) / 6.0)


def test_shape_mismatch_raises():
    _, realized, bucket_def = _setup()
    model = DecisionRelevantBucketAccuracy(bucket_def)
    with pytest.raises(InvalidDecisionConfigError):
        model.evaluate(np.zeros(3), realized)


def test_cost_matrix_bucket_count_mismatch_raises():
    bucket_def = BucketDefinition(n_buckets=3, window=4, min_periods=4)
    cost_matrix = CostMatrix.from_matrix([[0, 1], [1, 0]], bucket_labels=["low", "high"])
    with pytest.raises(InvalidDecisionConfigError):
        DecisionRelevantBucketAccuracy(bucket_def, cost_matrix=cost_matrix)


def test_insufficient_data_raises():
    bucket_def = BucketDefinition(n_buckets=2, window=50, min_periods=50)
    model = DecisionRelevantBucketAccuracy(bucket_def)
    with pytest.raises(InsufficientDataError):
        model.evaluate(np.arange(10, dtype=float), np.arange(10, dtype=float))
