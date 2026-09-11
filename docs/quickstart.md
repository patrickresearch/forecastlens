# Quickstart

Both examples below use `forecastlens.synthetic` — zero setup, no API keys, and every preset ships with known ground truth (regime changes, promo dates, disruption windows) so you can validate against a controlled series before pointing ForecastLens at real data. Full runnable notebooks live in [`examples/`](https://github.com/patrickresearch/forecastlens/tree/main/examples).

## Economic value of waiting on a forecast

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
print(report.savings_per_unit)   # 21.94
print(report.total_savings)      # 21940.29
print(report.buy_day)            # 11, not forced by the deadline
```

The model waits for the price to revert toward its forecast baseline instead of buying at the spike, and reports the result directly in currency — see [Economic value layer](concepts/economic-value.md) for the design rationale (leak-safety, why a single static forecast is enough, the deadline mechanic).

## Regime-conditional accuracy

```python
import numpy as np
from forecastlens.core import ForecastResult
from forecastlens.evaluators import RegimeAwareEvaluator
from forecastlens.regime import CUSUMDetector
from forecastlens.synthetic import load_preset

series = load_preset("structural_break")
y_true = series.values

median = np.roll(y_true, 1)
median[0] = y_true[0]
forecast = ForecastResult(
    timestamps=series.timestamps,
    freq="D",
    quantiles={0.1: median - 5.0, 0.5: median, 0.9: median + 5.0},
)

detector = CUSUMDetector(warmup=80, threshold_std=10.0, drift_std=1.5)
report = RegimeAwareEvaluator(detector).evaluate(forecast, y_true)

print(report.overall_crps)
for regime in report.per_regime:
    print(regime.regime_label, regime.n_periods, regime.crps)
```

See [Regime-aware evaluation](concepts/regime-aware.md) for how the two built-in detectors differ and what the per-regime calibration numbers mean.

## Using darts or neuralforecast directly

```python
from forecastlens.adapters import DartsAdapter, NeuralForecastAdapter

result = DartsAdapter.from_timeseries(prediction, model_name="ExponentialSmoothing")
result = NeuralForecastAdapter.from_dataframe(predictions_df, model=nf.models[0])
```

Both normalize into the same `ForecastResult` every metric, evaluator, and decision model in this package consumes.
