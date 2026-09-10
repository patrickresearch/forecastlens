import numpy as np
import pandas as pd
import pytest

from forecastlens.core.exceptions import MismatchedLengthError, MissingQuantilesError
from forecastlens.core.result import ForecastResult
from forecastlens.evaluators.regime_aware import RegimeAwareEvaluator
from forecastlens.regime.base import RegimeDetectionResult


class _FakeDetector:
    """Stub RegimeDetector returning fixed labels, to isolate the evaluator's
    own aggregation logic from any particular detector's behavior."""

    def __init__(self, regime_labels, changepoints=()):
        self._labels = np.asarray(regime_labels)
        self._changepoints = changepoints

    def detect(self, values):
        return RegimeDetectionResult(
            changepoints=self._changepoints, regime_labels=self._labels, metadata={}
        )


def _forecast(quantiles, n=4):
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    return ForecastResult(timestamps=timestamps, freq="D", quantiles=quantiles)


def test_hand_computed_overall_and_per_regime_metrics():
    # y_true = [10,10,20,20]; quantiles 0.25/0.75 sit symmetrically +-2 around
    # y_true at every point, so pinball loss is 0.5 everywhere at both levels:
    #   crps = 2*mean([0.5,0.5]) = 1.0 for every point (overall and per regime)
    #   wql_level = 2*sum(pinball)/sum(|y_true|) -- see inline comments below
    #   calibration: 0.25-quantile never covers y_true (0.0), 0.75-quantile always does (1.0)
    y_true = np.array([10.0, 10.0, 20.0, 20.0])
    forecast = _forecast({0.25: [8.0, 8.0, 18.0, 18.0], 0.75: [12.0, 12.0, 22.0, 22.0]})
    detector = _FakeDetector(regime_labels=[0, 0, 1, 1], changepoints=(2,))

    report = RegimeAwareEvaluator(detector).evaluate(forecast, y_true)

    assert report.overall_crps == pytest.approx(1.0)
    assert report.overall_wql == pytest.approx(4.0 / 60.0)  # sum(pinball)=2.0/level, denom=60
    assert report.overall_calibration == pytest.approx({0.25: 0.0, 0.75: 1.0})
    assert report.changepoints == (2,)
    np.testing.assert_array_equal(report.regime_labels, [0, 0, 1, 1])

    assert len(report.per_regime) == 2
    regime0, regime1 = report.per_regime
    assert regime0.regime_label == 0
    assert regime0.n_periods == 2
    assert regime0.crps == pytest.approx(1.0)
    assert regime0.wql == pytest.approx(2.0 / 20.0)  # sum(pinball)=1.0/level, denom=20
    assert regime0.calibration == pytest.approx({0.25: 0.0, 0.75: 1.0})

    assert regime1.regime_label == 1
    assert regime1.n_periods == 2
    assert regime1.crps == pytest.approx(1.0)
    assert regime1.wql == pytest.approx(2.0 / 40.0)  # denom=40 for y_true=[20,20]
    assert regime1.calibration == pytest.approx({0.25: 0.0, 0.75: 1.0})


def test_requires_quantile_forecast():
    timestamps = pd.date_range("2024-01-01", periods=4, freq="D")
    forecast = ForecastResult(timestamps=timestamps, freq="D", point=[1.0, 2.0, 3.0, 4.0])
    detector = _FakeDetector(regime_labels=[0, 0, 0, 0])
    with pytest.raises(MissingQuantilesError):
        RegimeAwareEvaluator(detector).evaluate(forecast, [1.0, 2.0, 3.0, 4.0])


def test_length_mismatch_raises():
    forecast = _forecast({0.5: [1.0, 2.0, 3.0, 4.0]})
    detector = _FakeDetector(regime_labels=[0, 0, 0, 0])
    with pytest.raises(MismatchedLengthError):
        RegimeAwareEvaluator(detector).evaluate(forecast, [1.0, 2.0])
