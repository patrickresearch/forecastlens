# ForecastLens Web Demo

Interactive demo of two ForecastLens diagnostics, running entirely on `forecastlens.synthetic` presets (zero setup, known ground truth):

- **Regime-Aware Evaluation** — compares `VolatilityRegimeDetector` and `CUSUMDetector` against a price series' true regime changes, and shows per-regime CRPS/WQL.
- **Procurement Timing** — simulates `ProcurementTimingModel`'s buy-now-vs-wait decision and reports the result in currency per unit.

This demo uses ForecastLens to **diagnose** forecasts, not to train new forecasting models. See the [main repository](https://github.com/forecastlens/forecastlens) for the package itself.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Opens on `http://localhost:7860`.

## Deploy to Render (free tier)

1. Push this repo to GitHub (already done).
2. On [render.com](https://render.com), "New +" → "Web Service" → connect this repo.
3. Render picks up [`render.yaml`](../render.yaml) automatically (build command, start command, Python version already configured).
4. Deploy. Render's free tier spins the service down after 15 minutes of inactivity — the first request after that takes 30-60s to wake it back up, which is expected, not a bug.

`requirements.txt` installs `forecastlens` straight from this GitHub repo since it isn't on PyPI yet — swap that line for `forecastlens` once the package is released.
