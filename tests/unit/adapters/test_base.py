import numpy as np
import pytest

from forecastlens.adapters.base import quantiles_from_samples
from forecastlens.core.exceptions import UnrecognizedAdapterFormatError


def test_quantiles_from_samples_hand_computed():
    # samples shape (n_samples=5, horizon=2); column 0 is [1,2,3,4,5], column 1 is [10,20,30,40,50].
    samples = np.array(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]
    )
    result = quantiles_from_samples(samples, levels=[0.5])
    # median of [1,2,3,4,5] is 3; median of [10,20,30,40,50] is 30.
    np.testing.assert_allclose(result[0.5], [3.0, 30.0])


def test_quantiles_from_samples_multiple_levels():
    samples = np.tile(np.arange(1, 101, dtype=float).reshape(-1, 1), (1, 1))
    result = quantiles_from_samples(samples, levels=[0.1, 0.5, 0.9])
    assert set(result) == {0.1, 0.5, 0.9}
    assert result[0.1][0] < result[0.5][0] < result[0.9][0]


def test_wrong_ndim_raises():
    with pytest.raises(UnrecognizedAdapterFormatError):
        quantiles_from_samples(np.array([1.0, 2.0, 3.0]))
