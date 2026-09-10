# ForecastLens

Diagnostic layer for time-series forecasting frameworks (`darts`, `neuralforecast`): regime-aware calibration, leak-safe backtesting, and an economic decision-value layer that translates forecasts into concrete decisions (e.g. "buy now vs. trust the forecast") measured in currency per unit, instead of abstract error metrics alone.

**Status:** pre-alpha, API not yet stable. Full documentation, quickstart, and tutorials land in a later milestone (see `CHANGELOG.md`).

## Installation

```bash
pip install forecastlens              # core only (numpy, pandas, scipy)
pip install forecastlens[darts]       # + darts adapter
pip install forecastlens[neuralforecast]  # + neuralforecast adapter
pip install forecastlens[all]         # everything
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
