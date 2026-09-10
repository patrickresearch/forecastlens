import numpy as np
import pytest

from forecastlens.core.exceptions import InsufficientDataError, InvalidDetectorConfigError
from forecastlens.regime.volatility import VolatilityRegimeDetector


def test_hand_computed_step_change_in_volatility():
    # diffs = [0]*8 + [10] + [0]*5 -> values step from 100 (9x) to 110 (6x).
    # short_window=2, long_window=5, threshold=1.5:
    #   short_vol spikes to 10/sqrt(2)=7.071 at the two windows straddling the jump
    #   (diffs-index 8, 9); long_vol there is std({0,0,0,0,10}, ddof=1) = sqrt(20)=4.472.
    #   ratio=1.581 > 1.5 at diffs-index 8 and 9 only -> values-index 9 and 10 flagged high.
    values = np.array([100.0] * 9 + [110.0] * 6)
    detector = VolatilityRegimeDetector(short_window=2, long_window=5, threshold_multiplier=1.5)
    result = detector.detect(values)

    expected_labels = np.zeros(15, dtype=int)
    expected_labels[9] = 1
    expected_labels[10] = 1
    np.testing.assert_array_equal(result.regime_labels, expected_labels)
    assert result.changepoints == (9, 11)


def test_constant_series_has_no_changepoints():
    values = np.full(50, 42.0)
    result = VolatilityRegimeDetector(short_window=3, long_window=10).detect(values)
    assert result.changepoints == ()
    assert np.all(result.regime_labels == 0)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"short_window": 1},
        {"short_window": 10, "long_window": 10},
        {"short_window": 10, "long_window": 5},
        {"threshold_multiplier": 0.0},
        {"threshold_multiplier": -1.0},
    ],
)
def test_invalid_construction_raises(bad_kwargs):
    with pytest.raises(InvalidDetectorConfigError):
        VolatilityRegimeDetector(**bad_kwargs)


def test_insufficient_data_raises():
    detector = VolatilityRegimeDetector(short_window=5, long_window=20)
    with pytest.raises(InsufficientDataError):
        detector.detect(np.arange(10, dtype=float))
