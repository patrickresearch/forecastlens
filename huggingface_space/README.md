---
title: ForecastLens Demo
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.26.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# ForecastLens Demo

Interactive demo of two ForecastLens diagnostics, running entirely on `forecastlens.synthetic` presets (zero setup, known ground truth):

- **Regime-Aware Evaluation** — compares `VolatilityRegimeDetector` and `CUSUMDetector` against a price series' true regime changes, and shows per-regime CRPS/WQL.
- **Procurement Timing** — simulates `ProcurementTimingModel`'s buy-now-vs-wait decision and reports the result in currency per unit.

This Space uses ForecastLens to **diagnose** forecasts, not to train new forecasting models. See the [main repository](https://github.com/forecastlens/forecastlens) for the package itself.
