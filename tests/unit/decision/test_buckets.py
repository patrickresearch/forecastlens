import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidDecisionConfigError
from forecastlens.decision.buckets import BucketDefinition


def test_hand_computed_edges_and_assignment():
    # history alternates 10,20,...; shifted-by-1 rolling median (window=4) needs
    # 4 periods of history before it's defined, then stabilizes at 15 forever
    # since every 4-window contains two 10s and two 20s.
    history = np.array([10, 20, 10, 20, 10, 20, 10, 20, 10, 20], dtype=float)
    bucket_def = BucketDefinition(n_buckets=2, window=4, min_periods=4)

    edges = bucket_def.compute_edges(history)
    assert edges.shape == (10, 1)
    assert np.all(np.isnan(edges[:4]))
    np.testing.assert_allclose(edges[4:, 0], 15.0)

    bucket_idx = bucket_def.assign(history, edges)
    np.testing.assert_array_equal(bucket_idx, [-1, -1, -1, -1, 0, 1, 0, 1, 0, 1])


def test_bucket_labels_default_and_custom():
    default = BucketDefinition(n_buckets=3, window=5, min_periods=5)
    assert default.bucket_labels() == ("bucket_0", "bucket_1", "bucket_2")

    custom = BucketDefinition(n_buckets=3, window=5, min_periods=5, labels=("low", "mid", "high"))
    assert custom.bucket_labels() == ("low", "mid", "high")


def test_nan_value_is_undefined_bucket():
    history = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 7.0])
    bucket_def = BucketDefinition(n_buckets=2, window=3, min_periods=3)
    edges = bucket_def.compute_edges(history)
    bucket_idx = bucket_def.assign(history, edges)
    assert bucket_idx[5] == -1


def test_edges_are_prefix_invariant_leak_safety():
    rng = np.random.default_rng(0)
    full_history = rng.normal(size=40).cumsum() + 100
    bucket_def = BucketDefinition(n_buckets=4, window=10, min_periods=10)

    edges_full = bucket_def.compute_edges(full_history)
    edges_prefix = bucket_def.compute_edges(full_history[:20])

    np.testing.assert_allclose(edges_full[:20], edges_prefix, equal_nan=True)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"n_buckets": 1, "window": 5, "min_periods": 5},
        {"n_buckets": 3, "window": 5, "min_periods": 1},
        {"n_buckets": 3, "window": 5, "min_periods": 10},
        {"n_buckets": 3, "window": 5, "min_periods": 5, "labels": ("a", "b")},
    ],
)
def test_invalid_construction_raises(bad_kwargs):
    with pytest.raises(InvalidDecisionConfigError):
        BucketDefinition(**bad_kwargs)


def test_assign_length_mismatch_raises():
    bucket_def = BucketDefinition(n_buckets=2, window=3, min_periods=3)
    edges = bucket_def.compute_edges(np.arange(10, dtype=float))
    with pytest.raises(InvalidDecisionConfigError):
        bucket_def.assign(np.arange(5, dtype=float), edges)
