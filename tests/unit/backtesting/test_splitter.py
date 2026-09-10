import numpy as np
import pandas as pd
import pytest

from forecastlens.backtesting.splitter import LeakSafeWalkForwardSplitter
from forecastlens.core.exceptions import ForecastLensError, InsufficientDataError


def test_expanding_window_hand_computed():
    splitter = LeakSafeWalkForwardSplitter(horizon=2, min_train_size=4, step=1)
    splits = list(splitter.split(range(10)))

    assert len(splits) == 5
    expected_test_starts = [4, 5, 6, 7, 8]
    for split, test_start in zip(splits, expected_test_starts, strict=True):
        np.testing.assert_array_equal(split.train_idx, np.arange(0, test_start))
        np.testing.assert_array_equal(split.test_idx, np.arange(test_start, test_start + 2))


def test_rolling_window_hand_computed():
    splitter = LeakSafeWalkForwardSplitter(horizon=2, min_train_size=3, max_train_size=3, step=1)
    splits = list(splitter.split(range(10)))

    assert len(splits) == 6
    for split in splits:
        assert len(split.train_idx) == 3
    np.testing.assert_array_equal(splits[0].train_idx, [0, 1, 2])
    np.testing.assert_array_equal(splits[1].train_idx, [1, 2, 3])
    np.testing.assert_array_equal(splits[-1].train_idx, [5, 6, 7])


def test_gap_excludes_indices_between_train_and_test():
    splitter = LeakSafeWalkForwardSplitter(horizon=2, min_train_size=4, gap=2, step=1)
    first = next(iter(splitter.split(range(12))))

    np.testing.assert_array_equal(first.train_idx, [0, 1, 2, 3])
    np.testing.assert_array_equal(first.test_idx, [6, 7])
    # indices 4, 5 are the gap: neither train nor test.
    assert 4 not in first.train_idx and 4 not in first.test_idx
    assert 5 not in first.train_idx and 5 not in first.test_idx


def test_step_greater_than_one():
    splitter = LeakSafeWalkForwardSplitter(horizon=2, min_train_size=4, step=3)
    splits = list(splitter.split(range(14)))
    test_starts = [split.test_idx[0] for split in splits]
    assert test_starts == [4, 7, 10]


@pytest.mark.parametrize(
    ("horizon", "min_train_size", "step", "max_train_size", "gap", "n"),
    [
        (1, 1, 1, None, 0, 20),
        (3, 5, 2, None, 0, 30),
        (2, 4, 1, 4, 1, 25),
        (5, 10, 5, 8, 3, 100),
    ],
)
def test_no_leak_invariant_holds_across_configs(
    horizon, min_train_size, step, max_train_size, gap, n
):
    splitter = LeakSafeWalkForwardSplitter(
        horizon=horizon,
        min_train_size=min_train_size,
        step=step,
        max_train_size=max_train_size,
        gap=gap,
    )
    splits = list(splitter.split(range(n)))
    assert len(splits) > 0
    for split in splits:
        assert split.train_idx.max() < split.test_idx.min()


def test_non_monotonic_timestamps_raise():
    timestamps = pd.DatetimeIndex(
        ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]
    )
    splitter = LeakSafeWalkForwardSplitter(horizon=1, min_train_size=2)
    with pytest.raises(ForecastLensError):
        list(splitter.split(timestamps))


def test_insufficient_data_raises():
    splitter = LeakSafeWalkForwardSplitter(horizon=5, min_train_size=100)
    with pytest.raises(InsufficientDataError):
        list(splitter.split(range(10)))


@pytest.mark.parametrize(
    ("horizon", "min_train_size", "step", "n"),
    [(1, 1, 1, 20), (3, 5, 2, 30), (2, 4, 3, 17)],
)
def test_n_splits_matches_materialized_count(horizon, min_train_size, step, n):
    splitter = LeakSafeWalkForwardSplitter(
        horizon=horizon, min_train_size=min_train_size, step=step
    )
    assert splitter.n_splits(n) == len(list(splitter.split(range(n))))


def test_n_splits_zero_when_insufficient_data():
    splitter = LeakSafeWalkForwardSplitter(horizon=5, min_train_size=100)
    assert splitter.n_splits(10) == 0


def test_earlier_folds_unaffected_by_appending_future_data():
    """Leak-safety regression guard: extending the series with more recent
    data must not change any previously computed fold's train/test indices."""
    splitter = LeakSafeWalkForwardSplitter(horizon=2, min_train_size=4, step=1)

    splits_short = list(splitter.split(range(10)))
    splits_long = list(splitter.split(range(20)))

    for short, long in zip(splits_short, splits_long[: len(splits_short)], strict=True):
        np.testing.assert_array_equal(short.train_idx, long.train_idx)
        np.testing.assert_array_equal(short.test_idx, long.test_idx)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"horizon": 0, "min_train_size": 1},
        {"horizon": 1, "min_train_size": 0},
        {"horizon": 1, "min_train_size": 1, "step": 0},
        {"horizon": 1, "min_train_size": 1, "gap": -1},
        {"horizon": 1, "min_train_size": 1, "max_train_size": 0},
    ],
)
def test_invalid_construction_params_raise(bad_kwargs):
    with pytest.raises(ForecastLensError):
        LeakSafeWalkForwardSplitter(**bad_kwargs)
