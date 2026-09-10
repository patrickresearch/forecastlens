# ForecastLens

A diagnostic layer on top of your existing time-series forecasting stack (`darts`, `neuralforecast`, or anything you can put in a DataFrame). ForecastLens doesn't train models — it tells you things your forecasting framework doesn't:

- **Regime-aware evaluation.** A single averaged CRPS/MAE across a backtest period hides that it was earned across several different market regimes. ForecastLens breaks accuracy down *by regime*, not just by time — see [Regime-aware evaluation](concepts/regime-aware.md).
- **Leak-safe backtesting.** A causal, walk-forward splitter and rolling decision buckets provably unable to see future data, with tests that assert it.
- **An economic decision layer.** Forecasts translated into concrete decisions ("buy now vs. wait") and scored in currency per unit, not just abstract error metrics — see [Economic value layer](concepts/economic-value.md).

**Status:** pre-alpha (`0.1.0`), API not yet stable.

## Installation

```bash
pip install forecastlens                  # core only (numpy, pandas, scipy)
pip install forecastlens[darts]           # + darts adapter
pip install forecastlens[neuralforecast]  # + neuralforecast adapter
pip install forecastlens[all]             # everything
```

Start with the [Quickstart](quickstart.md), or jump straight to the [API Reference](api/core.md).
