import numpy as np
import pytest

from forecastlens.core.exceptions import InvalidDetectorConfigError
from forecastlens.regime.base import RegimeDetectionResult
from forecastlens.regime.clustering import ClusteredRegimeDetector


class _FakeDetector:
    """Stub returning fixed, sequentially-increasing raw segment labels --
    mimicking CUSUMDetector's behavior of never reusing a label."""

    def __init__(self, regime_labels):
        self._labels = np.asarray(regime_labels)

    def detect(self, values):
        changepoints = tuple(
            int(t) for t in range(1, len(self._labels)) if self._labels[t] != self._labels[t - 1]
        )
        return RegimeDetectionResult(
            changepoints=changepoints, regime_labels=self._labels, metadata={"detector": "fake"}
        )


def test_merges_recurring_levels_hand_computed():
    # 4 raw segments of 5 points each, means exactly 10, 50, 10, 50.
    # overall mean=30, sample std = sqrt(8000/19) ~= 20.52.
    # Segments sharing a mean have distance 0; the two groups are 40 apart --
    # any threshold between 0 and 40 (here 0.5*std ~= 10.26) cleanly separates
    # them into exactly 2 clusters regardless of linkage method specifics.
    values = np.array([10.0] * 5 + [50.0] * 5 + [10.0] * 5 + [50.0] * 5)
    raw_labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    detector = ClusteredRegimeDetector(_FakeDetector(raw_labels), distance_threshold=0.5)

    result = detector.detect(values)

    expected_labels = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)
    np.testing.assert_array_equal(result.regime_labels, expected_labels)
    assert result.changepoints == (5, 10, 15)
    assert result.metadata["n_raw_segments"] == 4
    assert result.metadata["n_clusters"] == 2
    assert result.metadata["wrapped_by"] == "ClusteredRegimeDetector"
    assert result.metadata["detector"] == "fake"  # base metadata preserved


def test_single_raw_segment_is_unchanged():
    values = np.arange(10, dtype=float)
    raw_labels = np.zeros(10, dtype=int)
    detector = ClusteredRegimeDetector(_FakeDetector(raw_labels))

    result = detector.detect(values)

    np.testing.assert_array_equal(result.regime_labels, raw_labels)
    assert result.changepoints == ()
    assert result.metadata["n_clusters"] == 1


def test_constant_series_merges_to_one_cluster():
    # overall_std == 0 -> degenerate case must not divide by zero.
    values = np.full(10, 5.0)
    raw_labels = np.array([0] * 3 + [1] * 3 + [2] * 4)
    detector = ClusteredRegimeDetector(_FakeDetector(raw_labels))

    result = detector.detect(values)

    assert result.metadata["n_clusters"] == 1
    np.testing.assert_array_equal(result.regime_labels, np.zeros(10, dtype=int))


def test_large_threshold_merges_everything():
    values = np.array([10.0] * 5 + [50.0] * 5 + [10.0] * 5 + [50.0] * 5)
    raw_labels = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
    # threshold far larger than the 40-unit gap between the two levels.
    detector = ClusteredRegimeDetector(_FakeDetector(raw_labels), distance_threshold=100.0)

    result = detector.detect(values)

    assert result.metadata["n_clusters"] == 1
    assert result.changepoints == ()


def test_invalid_distance_threshold_raises():
    with pytest.raises(InvalidDetectorConfigError):
        ClusteredRegimeDetector(_FakeDetector(np.zeros(5, dtype=int)), distance_threshold=0.0)
