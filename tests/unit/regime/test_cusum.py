import numpy as np
import pytest

from forecastlens.core.exceptions import (
    DegenerateSeriesError,
    InsufficientDataError,
    InvalidDetectorConfigError,
)
from forecastlens.regime.cusum import CUSUMDetector


def test_hand_computed_single_shift():
    # warmup=4, values[:4]=[0,2,0,2] -> mu=1, sample std=sqrt(4/3)=1.1547.
    # threshold_std=3 -> h=3.4641; drift_std=0.5 -> k=0.57735.
    # t=4: x=1, g_pos=max(0, 0-0.57735)=0; g_neg=min(0, 0+0.57735)=0 -> no trigger.
    # t=5: x=1, same -> no trigger.
    # t=6: x=10, g_pos=max(0, (10-1)-0.57735)=8.42265 > h -> changepoint, reset, mu_ref=10.
    # t=7: x=10, g_pos=max(0, 0-0.57735)=0; g_neg=0 -> no trigger.
    values = np.array([0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 10.0, 10.0])
    detector = CUSUMDetector(warmup=4, threshold_std=3.0, drift_std=0.5)
    result = detector.detect(values)

    assert result.changepoints == (6,)
    np.testing.assert_array_equal(result.regime_labels, [0, 0, 0, 0, 0, 0, 1, 1])
    assert result.metadata["mu_warmup"] == pytest.approx(1.0)
    assert result.metadata["sigma_warmup"] == pytest.approx(np.sqrt(4 / 3))


def test_no_shift_no_changepoints():
    # Pure noise, no real shift. A larger warmup gives a stable sigma estimate;
    # CUSUM's false-alarm rate is a function of run length (ARL), not zero by
    # construction, so this checks one fixed seed stays quiet rather than
    # proving it always will.
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.0, scale=1.0, size=200)
    detector = CUSUMDetector(warmup=50, threshold_std=6.0, drift_std=0.5)
    result = detector.detect(values)
    assert result.changepoints == ()


def test_zero_variance_warmup_raises():
    values = np.concatenate([np.zeros(10), np.array([100.0] * 10)])
    detector = CUSUMDetector(warmup=10)
    with pytest.raises(DegenerateSeriesError):
        detector.detect(values)


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"warmup": 1},
        {"threshold_std": 0.0},
        {"threshold_std": -1.0},
        {"drift_std": -0.1},
    ],
)
def test_invalid_construction_raises(bad_kwargs):
    with pytest.raises(InvalidDetectorConfigError):
        CUSUMDetector(**bad_kwargs)


def test_insufficient_data_raises():
    detector = CUSUMDetector(warmup=30)
    with pytest.raises(InsufficientDataError):
        detector.detect(np.arange(20, dtype=float))
