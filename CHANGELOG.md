# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Unreleased

Initial pre-release. API may still change without notice.

### Added

- Project scaffolding: `src/` layout, `pyproject.toml` with optional extras (`darts`, `neuralforecast`, `viz`, `docs`), pre-commit config, CI (lint, unit, integration, docs-deploy).
- `core.ForecastResult`: immutable, framework-agnostic forecast representation with point/quantile validation, quantile-crossing detection, and NaN/inf rejection.
- `metrics`: CRPS, MASE, WQL, and pinball loss, each validated against a reference implementation (`properscoring`, `sktime`, or an analytic closed form) in addition to hand-computed unit tests.
- `backtesting.LeakSafeWalkForwardSplitter`: causal walk-forward CV with expanding/rolling windows, gap support, and a hard leak assertion.
- `synthetic`: `PriceRegimeGenerator` and `DemandGenerator` engines with 10 presets (5 each) exposing known ground-truth regime changes, for validating `regime/` and `decision/` without real data.
- `regime`: `VolatilityRegimeDetector` and `CUSUMDetector`, both implementing a shared, swappable interface, validated against the synthetic presets' ground truth.
- `adapters`: `DartsAdapter`, `NeuralForecastAdapter`, and a framework-free `RawAdapter`, all normalizing into `ForecastResult`; adapter integration tests train real (tiny) models rather than mocking.
- `decision`: `DecisionRelevantBucketAccuracy` (cost-weighted confusion matrix over causally-defined buckets) and `ProcurementTimingModel` (buy-now-vs-wait simulation with a euro-per-unit result), plus the shared `relative_value_score` — explicitly a simplified MVP metric, not the academic Murphy value score.
- `evaluators`: `DemandForecastEvaluator` (WMAPE, bias, intermittency, service level) and `RegimeAwareEvaluator` (regime-conditional CRPS/WQL/calibration).
- Documentation: README quickstarts, `mkdocs-material` API reference and concept guides, two executed end-to-end tutorial notebooks (`examples/`).
