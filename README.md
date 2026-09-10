# ForecastLens

[![CI](https://github.com/forecastlens/forecastlens/actions/workflows/ci.yml/badge.svg)](https://github.com/forecastlens/forecastlens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/forecastlens.svg)](https://pypi.org/project/forecastlens/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A diagnostic layer on top of your existing time-series forecasting stack (`darts`, `neuralforecast`, or anything else you can put in a DataFrame). ForecastLens doesn't train models — it tells you things your forecasting framework doesn't:

- **Regime-aware evaluation.** A single averaged CRPS/MAE across a whole backtest period hides that it was earned across several different market regimes. ForecastLens breaks accuracy down *by regime*, not just by time.
- **Leak-safe backtesting.** A causal, walk-forward splitter and rolling decision buckets that are provably unable to see future data — with tests that assert it.
- **An economic decision layer.** Forecasts translated into concrete decisions ("buy now vs. wait") and scored in currency per unit, not just abstract error metrics.

**Status:** pre-alpha (`0.1.0`), API not yet stable.

## Installation

```bash
pip install forecastlens                  # core only (numpy, pandas, scipy)
pip install forecastlens[darts]           # + darts adapter
pip install forecastlens[neuralforecast]  # + neuralforecast adapter
pip install forecastlens[viz]             # + plotting
pip install forecastlens[all]             # everything
```

## Quickstart: economic value of waiting on a forecast

No API keys, no real data needed — `forecastlens.synthetic` ships regime-switching presets with known ground truth for exactly this kind of demo.

```python
import numpy as np
from forecastlens.core import ForecastResult
from forecastlens.decision import ProcurementTimingModel, ThresholdTimingRule
from forecastlens.synthetic import load_preset

series = load_preset("supply_shock_spike")
start, horizon = 30, 30  # the procurement window opens right at the price spike's peak
baseline_forecast = series.values[:20].mean()  # our model expects reversion to the pre-shock level
forecast = ForecastResult(
    timestamps=series.timestamps[start : start + horizon],
    freq="D",
    point=np.full(horizon, baseline_forecast),
)
realized_prices = series.values[start : start + horizon]

model = ProcurementTimingModel(
    decision_rule=ThresholdTimingRule(wait_if_forecast_below_current_by=0.03),
    deadline_horizon=horizon,
    units=1000,
)
report = model.evaluate(forecast, realized_prices)
print(f"Savings per unit: {report.savings_per_unit:.2f}")   # 21.94
print(f"Total savings:    {report.total_savings:.2f}")      # 21940.29
print(f"Bought on day:    {report.buy_day}")                # 11, not forced by the deadline
```

The model waited 11 days for the price to revert toward its forecast baseline instead of buying at the spike — a result expressed directly in currency, no statistics vocabulary required.

## Quickstart: regime-conditional accuracy

```python
import numpy as np
from forecastlens.core import ForecastResult
from forecastlens.evaluators import RegimeAwareEvaluator
from forecastlens.regime import CUSUMDetector
from forecastlens.synthetic import load_preset

series = load_preset("structural_break")
y_true = series.values

# A toy quantile forecast: one-step persistence with a fixed +-5 band.
median = np.roll(y_true, 1)
median[0] = y_true[0]
forecast = ForecastResult(
    timestamps=series.timestamps,
    freq="D",
    quantiles={0.1: median - 5.0, 0.5: median, 0.9: median + 5.0},
)

detector = CUSUMDetector(warmup=80, threshold_std=10.0, drift_std=1.5)
report = RegimeAwareEvaluator(detector).evaluate(forecast, y_true)

print(f"Overall CRPS: {report.overall_crps:.3f}")
for regime in report.per_regime:
    print(f"Regime {regime.regime_label}: n={regime.n_periods}, CRPS={regime.crps:.3f}")
```

The overall CRPS averages away exactly the thing a regime shift analysis needs: some regimes here score close to 0.94, others above 1.4 — a >50% spread invisible in the single averaged number.

## Using darts or neuralforecast output directly

```python
from forecastlens.adapters import DartsAdapter, NeuralForecastAdapter

# darts: model.predict(...) -> TimeSeries
result = DartsAdapter.from_timeseries(prediction, model_name="ExponentialSmoothing")

# neuralforecast: nf.predict() -> DataFrame
result = NeuralForecastAdapter.from_dataframe(predictions_df, model=nf.models[0])
```

Both normalize into the same `ForecastResult` that every metric, evaluator, and decision model in this package consumes — deterministic predictions become a point forecast, probabilistic ones become quantiles.

## What's in the box

| Module | Purpose |
|---|---|
| `core` | `ForecastResult` — the shared, immutable forecast representation |
| `adapters` | Normalize `darts`, `neuralforecast`, or any tabular output into `ForecastResult` |
| `metrics` | CRPS, MASE, WQL, pinball loss — each validated against a reference implementation |
| `backtesting` | `LeakSafeWalkForwardSplitter` — causal, provably leak-free walk-forward CV |
| `regime` | `VolatilityRegimeDetector`, `CUSUMDetector` — swappable regime/changepoint detection |
| `decision` | `DecisionRelevantBucketAccuracy`, `ProcurementTimingModel` — forecasts priced as decisions |
| `evaluators` | `DemandForecastEvaluator`, `RegimeAwareEvaluator` — use-case-specific reports |
| `synthetic` | 10 regime-switching price/demand presets with known ground truth, zero setup |

See `examples/` for full end-to-end tutorials, one per use case (demand planning, energy/commodity regime shifts).

## A note on `relative_value_score`

`decision.relative_value_score` is a **deliberately simplified MVP metric** — `(cost_naive - cost_model) / cost_naive` — not the academic Murphy value score (which normalizes against a perfect-foresight forecast on a 0-1 scale). Ours can go negative or exceed 1. See the docstring in [`decision/value_score.py`](src/forecastlens/decision/value_score.py) for the full rationale.

## License

Apache 2.0 — see [LICENSE](LICENSE).
