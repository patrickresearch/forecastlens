# Regime-aware evaluation

## The problem this solves

A single averaged error metric over a whole backtest period hides that the period contained different market regimes — a calm stretch and a volatile one, say. A model that's excellent in calm periods and mediocre in volatile ones can post the same average CRPS as a model that's mediocre everywhere. Averaging is exactly what erases the difference between those two models, and the difference is usually the one that matters for a real decision.

`RegimeAwareEvaluator` doesn't fix a forecast's accuracy — it makes the regime-dependent structure of that accuracy visible instead of averaging it away.

## Two swappable detectors, one interface

Both detectors implement `regime.base.RegimeDetector` (`detect(values) -> RegimeDetectionResult`), so `RegimeAwareEvaluator` doesn't care which one it holds.

**`VolatilityRegimeDetector`** — the simplest, most transparent baseline. Classifies each period high/low-volatility via a short-vs-long rolling volatility ratio (`std(diff)_short / std(diff)_long > threshold_multiplier`). Both windows are trailing, so it's safe inside a walk-forward loop. It reacts to changes in *volatility*, not level — a pure level shift with unchanged volatility can go undetected (this is expected, not a bug: see `tests/unit/regime/test_ground_truth_recovery.py::test_volatility_detector_no_false_positives_on_smooth_trend`, where it correctly stays quiet on a smooth trend with no volatility change).

**`CUSUMDetector`** — a two-sided cumulative sum (Page's test) for mean-level shifts. Baseline mean/std come from a causal warm-up window; after each detected changepoint the reference re-anchors to the triggering value so later shifts stay detectable. It reacts to *level* changes and, because it's a mean-shift detector, will also react to genuine sustained drift (a real trend), not only to discrete regime changes — that's a property of what CUSUM measures, not a defect.

Because they react to different things, running both on the same series and comparing — as the [regime-aware quickstart notebook](https://github.com/forecastlens/forecastlens/blob/main/examples/02_regime_aware_energy_quickstart.ipynb) does — can surface a materially different regime breakdown from each. **The regime-aware evaluation is only as informative as the regime definition feeding it.** Neither detector is an oracle; both are heuristics validated against synthetic ground truth (`tests/unit/regime/test_ground_truth_recovery.py`) by recall within a tolerance window, not by zero false positives.

## What the report contains

`RegimeAwareEvaluator.evaluate()` requires a quantile forecast (CRPS/WQL need a distribution, not a point) and returns:

- `overall_crps`, `overall_wql` — computed over the whole series.
- `per_regime`: one `RegimeMetrics` per detected regime label, each with its own `crps`, `wql`, and `n_periods`.
- `calibration`: for each quantile level `q`, the empirical coverage `mean(y_true <= quantile_forecast)`. A well-calibrated level-`q` quantile should show coverage close to `q`; a level-0.9 quantile with 0.6 coverage means the forecast is systematically under-covering that regime.
- `changepoints`: the detector's own changepoint indices, straight from `RegimeDetectionResult`.

Regime detection always runs on the **realized** series, not the forecast — the question this evaluator answers is "how did the forecast perform during each real-world regime," which requires the regime labels to come from what actually happened, not from what was predicted.
