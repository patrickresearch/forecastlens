"""ForecastLens web demo.

Uses ForecastLens to *diagnose* forecasts against synthetic data with known
ground truth -- it does not train new forecasting models. Kept entirely
outside src/ and out of the PyPI package: a separate deploy target with its
own requirements.txt, deployable to Render (see render.yaml) or run locally.
"""

from __future__ import annotations

import os

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from forecastlens.core import ForecastResult
from forecastlens.core.exceptions import ForecastLensError
from forecastlens.decision import ProcurementTimingModel, ThresholdTimingRule
from forecastlens.evaluators import RegimeAwareEvaluator
from forecastlens.regime import ClusteredRegimeDetector, CUSUMDetector, VolatilityRegimeDetector
from forecastlens.synthetic import PRICE_PRESETS, load_preset

_KPI_HEADERS = ["segment", "n_periods", "MAE", "RMSE", "WAPE", "accuracy", "CRPS", "WQL"]


def _rolling_normal_forecast(
    values: np.ndarray, window: int = 20, levels: tuple[float, ...] = (0.1, 0.5, 0.9)
):
    """Causal rolling-mean/std Gaussian quantile forecast -- a transparent baseline, not a model.

    Self-adapting to whichever preset's own scale and local volatility is fed
    in, so the same function produces a plausible-looking forecast line for
    every preset without per-preset tuning.
    """
    shifted = pd.Series(values).shift(1)
    roll_mean = shifted.rolling(window, min_periods=window).mean()
    roll_std = shifted.rolling(window, min_periods=window).std()
    valid = (roll_mean.notna() & roll_std.notna() & (roll_std > 0)).to_numpy()
    idx = np.flatnonzero(valid)
    mu, sigma = roll_mean.to_numpy()[idx], roll_std.to_numpy()[idx]
    quantiles = {level: stats.norm.ppf(level, loc=mu, scale=sigma) for level in levels}
    return idx, quantiles


def _toggle_detector_sliders(detector_name: str):
    is_vol = detector_name == "VolatilityRegimeDetector"
    return gr.update(visible=is_vol), gr.update(visible=not is_vol)


def run_regime_evaluation(
    preset_name: str,
    detector_name: str,
    vol_threshold: float,
    cusum_threshold: float,
    merge_sensitivity: float,
):
    series = load_preset(preset_name)
    idx, quantiles = _rolling_normal_forecast(series.values)
    forecast_timestamps = series.timestamps[idx]
    forecast = ForecastResult(timestamps=forecast_timestamps, freq="D", quantiles=quantiles)
    y_true = series.values[idx]
    n = len(y_true)

    if detector_name == "VolatilityRegimeDetector":
        base_detector = VolatilityRegimeDetector(
            short_window=10, long_window=60, threshold_multiplier=vol_threshold
        )
    else:
        # Scale warmup to the series length: a fixed absolute warmup (e.g. 60)
        # eats too much of a short preset's history to ever detect anything.
        warmup = max(20, n // 8)
        base_detector = CUSUMDetector(warmup=warmup, threshold_std=cusum_threshold, drift_std=0.5)

    # A detector like CUSUM assigns a new label to every changepoint, even
    # when a level recurs -- cluster those raw segments by mean level so the
    # KPI table shows a handful of genuinely distinct regimes, not 20+ rows.
    detector = ClusteredRegimeDetector(base_detector, distance_threshold=merge_sensitivity)

    try:
        detection = detector.detect(y_true)
        report = RegimeAwareEvaluator(detector).evaluate(forecast, y_true)
    except ForecastLensError as exc:
        empty_fig, _ = plt.subplots(figsize=(10, 4.5))
        return empty_fig, f"Could not evaluate with these settings: {exc}", []
    n_raw_segments = detection.metadata["n_raw_segments"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(series.timestamps, series.values, color="black", linewidth=1, label="actual price")
    ax.plot(
        forecast_timestamps,
        quantiles[0.5],
        color="tab:orange",
        linewidth=1.5,
        label="forecast (median)",
    )
    ax.fill_between(
        forecast_timestamps,
        quantiles[0.1],
        quantiles[0.9],
        color="tab:orange",
        alpha=0.2,
        label="10-90% forecast band",
    )
    for cp in series.changepoints:
        ax.axvline(series.timestamps[cp], color="tab:blue", alpha=0.35, linestyle="--")
    for cp in report.changepoints:
        ax.axvline(forecast_timestamps[cp], color="tab:red", alpha=0.7, linestyle=":")
    ax.set_title(f"{preset_name}: true changepoints (blue) vs. merged detected regimes (red)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    rows = [
        [
            "Whole period (no segmentation)",
            n,
            round(report.overall_mae, 3),
            round(report.overall_rmse, 3),
            round(report.overall_wape, 3),
            round(report.overall_accuracy, 3),
            round(report.overall_crps, 3),
            round(report.overall_wql, 4),
        ]
    ]
    for r in report.per_regime:
        rows.append(
            [
                f"Regime {r.regime_label}",
                r.n_periods,
                round(r.mae, 3),
                round(r.rmse, 3),
                round(r.wape, 3),
                round(r.accuracy, 3),
                round(r.crps, 3),
                round(r.wql, 4),
            ]
        )

    mae_values = [r.mae for r in report.per_regime]
    summary = (
        f"Detector found {n_raw_segments} raw segments, merged into "
        f"{len(report.per_regime)} statistically distinct regimes "
        f"(ground truth: {len(series.changepoints)} true changes).\n"
        f"MAE by regime ranges {min(mae_values):.3f} - {max(mae_values):.3f}, vs. "
        f"{report.overall_mae:.3f} for the whole period lumped together -- averaging "
        f"can hide real accuracy differences between regimes."
    )
    return fig, summary, rows


def run_procurement_timing(
    preset_name: str, start: int, horizon: int, wait_threshold: float, deadline: int, units: float
):
    series = load_preset(preset_name)
    horizon = int(horizon)
    start = int(min(start, len(series.values) - horizon - 1))
    baseline_forecast = float(series.values[max(0, start - 20) : start].mean())
    forecast = ForecastResult(
        timestamps=series.timestamps[start : start + horizon],
        freq="D",
        point=np.full(horizon, baseline_forecast),
    )
    realized = series.values[start : start + horizon]

    model = ProcurementTimingModel(
        decision_rule=ThresholdTimingRule(wait_if_forecast_below_current_by=wait_threshold),
        deadline_horizon=int(min(deadline, horizon)),
        units=units,
    )
    report = model.evaluate(forecast, realized)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(realized)), realized, marker="o", label="realized price")
    ax.axhline(baseline_forecast, color="gray", linestyle="--", label="forecast baseline")
    ax.axvline(report.buy_day, color="tab:green", label=f"bought on day {report.buy_day}")
    ax.set_xlabel("day")
    ax.legend()
    fig.tight_layout()

    summary = (
        f"Buy now:            {report.cost_buy_now_per_unit:.2f} / unit\n"
        f"Forecast-informed:  {report.cost_forecast_informed_per_unit:.2f} / unit\n"
        f"Savings per unit:   {report.savings_per_unit:.2f}\n"
        f"Total savings:      {report.total_savings:.2f}\n"
        f"Bought on day:      {report.buy_day} "
        f"(forced by deadline: {report.forced_by_deadline})"
    )
    return fig, summary


_GLOSSARY_MD = """
### Forecast quality (point metrics)

- **MAE** (Mean Absolute Error) -- average `|actual - forecast|`, in the
  series' own units. Easy to explain, but treats a 2-unit error the same
  whether the series is at 10 or 10,000.
- **RMSE** (Root Mean Squared Error) -- like MAE, but squares errors before
  averaging, so a few large misses dominate the score more than many small
  ones.
- **WAPE** (Weighted Absolute Percentage Error) --
  `sum(|actual - forecast|) / sum(|actual|)`. Unlike plain MAPE, dividing by
  the *sum* rather than each point individually keeps it well-defined even
  when some actuals are zero.
- **Accuracy** -- `1 - WAPE`, the common demand-planning "forecast accuracy
  %" convention. *Not* the same "accuracy" as in classification; can go
  negative for a forecast worse than predicting zero everywhere.

### Forecast quality (probabilistic / quantile metrics)

- **CRPS** (Continuous Ranked Probability Score) -- generalizes MAE to a
  full predicted distribution, not just a point. Lower is better; rewards
  both an accurate center *and* well-sized uncertainty.
- **WQL** (Weighted Quantile Loss) -- averages pinball loss across quantile
  levels, normalized like WAPE so it's comparable across series of
  different scale.
- **Calibration** -- for a level-`q` quantile forecast, the fraction of
  actuals that fell at or below it. Should be close to `q` for a
  well-calibrated forecast; far off means the forecast is systematically
  over- or under-confident.

### Regime detection

- **VolatilityRegimeDetector** -- flags a period as high-volatility when a
  short-window rolling volatility clearly exceeds a long-window one. Reacts
  to *volatility* changes, not level shifts.
- **CUSUMDetector** -- a cumulative-sum test for *mean-level* shifts; also
  reacts to sustained real trends, not only discrete jumps. It also assigns
  a *new* label to every changepoint, even if the series returns to a level
  it already visited -- a long series can end up with 20+ raw segments.
- **Regime merge sensitivity** -- `ClusteredRegimeDetector` groups a
  detector's raw segments by mean level (agglomerative clustering, no need
  to pick a target number of clusters up front), so recurring levels
  collapse back into one regime. Higher sensitivity merges more
  aggressively; the summary reports both the raw segment count and the
  merged regime count. Neither base detector is an oracle -- see the "true
  (blue) vs. merged detected (red)" changepoints on the plot for how it
  actually did here.

### Why "whole period" vs. "per regime"

The table's first row applies every metric the ordinary way: one number
over the entire series, no segmentation -- exactly what most dashboards
report. The rows below condition the same metrics on the regime detected
at each period. The gap between them is the whole point of this project:
an averaged error can look fine while hiding a regime where the forecast
was actually much worse (or much better).

### Procurement Timing tab

- **`savings_per_unit`** -- price paid buying immediately on day 0 minus
  the price paid under the wait-then-buy strategy. Positive means waiting
  paid off.
- **Deadline** -- the strategy is forced to buy by this day regardless of
  the forecast, so it can never simulate "waiting forever."
- **`relative_value_score`** (used elsewhere in the package, not shown
  here) is a deliberately simplified MVP metric, *not* the academic Murphy
  value score -- see
  [`decision/value_score.py`](https://github.com/patrickresearch/forecastlens/blob/main/src/forecastlens/decision/value_score.py).
"""

with gr.Blocks(title="ForecastLens Demo") as demo:
    gr.Markdown(
        "# ForecastLens Demo\n"
        "Diagnoses forecasts against synthetic data with known ground truth -- "
        "does not train new forecasting models. "
        "[GitHub](https://github.com/patrickresearch/forecastlens)"
    )

    with gr.Tab("Regime-Aware Evaluation"):
        with gr.Row():
            preset = gr.Dropdown(
                sorted(PRICE_PRESETS), value="regime_switch_markov", label="Price preset"
            )
            detector = gr.Dropdown(
                ["VolatilityRegimeDetector", "CUSUMDetector"],
                value="CUSUMDetector",
                label="Detector",
            )
        with gr.Row():
            vol_threshold = gr.Slider(
                1.05,
                3.0,
                value=1.3,
                step=0.05,
                label="Volatility threshold_multiplier",
                visible=False,
            )
            cusum_threshold = gr.Slider(
                2.0, 12.0, value=5.0, step=0.5, label="CUSUM threshold_std", visible=True
            )
        detector.change(_toggle_detector_sliders, [detector], [vol_threshold, cusum_threshold])
        merge_sensitivity = gr.Slider(
            0.1,
            1.2,
            value=0.5,
            step=0.05,
            label="Regime merge sensitivity (higher = merges more raw segments)",
        )

        run_btn = gr.Button("Evaluate")
        plot_out = gr.Plot()
        summary_out = gr.Textbox(label="Summary", lines=3)
        table_out = gr.Dataframe(headers=_KPI_HEADERS, label="KPIs: whole period vs. per regime")
        run_btn.click(
            run_regime_evaluation,
            [preset, detector, vol_threshold, cusum_threshold, merge_sensitivity],
            [plot_out, summary_out, table_out],
        )

    with gr.Tab("Procurement Timing"):
        with gr.Row():
            preset2 = gr.Dropdown(
                sorted(PRICE_PRESETS), value="supply_shock_spike", label="Price preset"
            )
            start = gr.Slider(0, 200, value=30, step=1, label="Window start day")
            horizon = gr.Slider(5, 60, value=30, step=1, label="Horizon (days)")
        with gr.Row():
            wait_threshold = gr.Slider(0.0, 0.2, value=0.03, label="Wait threshold (fraction)")
            deadline = gr.Slider(5, 60, value=30, step=1, label="Deadline (days)")
            units = gr.Number(value=1000, label="Units")
        run_btn2 = gr.Button("Simulate")
        plot_out2 = gr.Plot()
        summary_out2 = gr.Textbox(label="Report", lines=5)
        run_btn2.click(
            run_procurement_timing,
            [preset2, start, horizon, wait_threshold, deadline, units],
            [plot_out2, summary_out2],
        )

    with gr.Tab("Glossary"):
        gr.Markdown(_GLOSSARY_MD)

if __name__ == "__main__":
    # Render (and most PaaS hosts) assign the port via $PORT and expect the
    # process to bind 0.0.0.0; falls back to Gradio's local default otherwise.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
