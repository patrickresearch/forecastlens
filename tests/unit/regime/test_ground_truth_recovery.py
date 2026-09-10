"""Validate both detectors against synthetic presets' known ground-truth changepoints.

Per ROADMAP: does each detector recover the known regime changes within a
tolerance? This is a recall check on a heuristic baseline, not a precision
guarantee -- neither detector is expected to be noise-free, and CUSUM in
particular reacts to genuine sustained drift (see the smooth-trend test),
not only to discrete regime changes.
"""

from __future__ import annotations

from forecastlens.regime.cusum import CUSUMDetector
from forecastlens.regime.volatility import VolatilityRegimeDetector
from forecastlens.synthetic.price import (
    gradual_demand_trend,
    regime_switch_markov,
    seasonal_volatility_cluster,
    structural_break,
    supply_shock_spike,
)


def _recall_within_tolerance(true_changepoints, detected_changepoints, tolerance):
    if not true_changepoints:
        return None
    hits = sum(
        any(abs(d - t) <= tolerance for d in detected_changepoints) for t in true_changepoints
    )
    return hits / len(true_changepoints)


def test_volatility_detector_finds_supply_shock_spike():
    series = supply_shock_spike()
    detector = VolatilityRegimeDetector(short_window=10, long_window=60, threshold_multiplier=2.0)
    result = detector.detect(series.values)
    recall = _recall_within_tolerance(series.changepoints, result.changepoints, tolerance=5)
    assert recall is not None and recall > 0.0


def test_volatility_detector_finds_seasonal_volatility_cluster():
    series = seasonal_volatility_cluster()
    detector = VolatilityRegimeDetector(short_window=10, long_window=60, threshold_multiplier=2.0)
    result = detector.detect(series.values)
    recall = _recall_within_tolerance(series.changepoints, result.changepoints, tolerance=5)
    assert recall is not None and recall > 0.0


def test_volatility_detector_no_false_positives_on_smooth_trend():
    series = gradual_demand_trend()
    assert series.changepoints == ()  # sanity: this preset is a negative control
    detector = VolatilityRegimeDetector(short_window=10, long_window=60, threshold_multiplier=2.0)
    result = detector.detect(series.values)
    assert result.changepoints == ()


def test_cusum_detector_finds_structural_break():
    series = structural_break()
    detector = CUSUMDetector(warmup=80, threshold_std=10.0, drift_std=1.5)
    result = detector.detect(series.values)
    recall = _recall_within_tolerance(series.changepoints, result.changepoints, tolerance=10)
    assert recall == 1.0


def test_cusum_detector_finds_most_regime_switch_markov_changepoints():
    series = regime_switch_markov()
    detector = CUSUMDetector(warmup=60, threshold_std=8.0, drift_std=1.0)
    result = detector.detect(series.values)
    recall = _recall_within_tolerance(series.changepoints, result.changepoints, tolerance=10)
    assert recall is not None and recall >= 0.5
