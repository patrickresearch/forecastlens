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
from forecastlens.decision import ProcurementTimingModel, ThresholdTimingRule
from forecastlens.evaluators import RegimeAwareEvaluator
from forecastlens.regime import CUSUMDetector, VolatilityRegimeDetector
from forecastlens.synthetic import PRICE_PRESETS, load_preset


def _rolling_normal_forecast(
    values: np.ndarray, window: int = 20, levels: tuple[float, ...] = (0.1, 0.5, 0.9)
):
    """Causal rolling-mean/std Gaussian quantile forecast -- a transparent baseline, not a model."""
    shifted = pd.Series(values).shift(1)
    roll_mean = shifted.rolling(window, min_periods=window).mean()
    roll_std = shifted.rolling(window, min_periods=window).std()
    valid = (roll_mean.notna() & roll_std.notna() & (roll_std > 0)).to_numpy()
    idx = np.flatnonzero(valid)
    mu, sigma = roll_mean.to_numpy()[idx], roll_std.to_numpy()[idx]
    quantiles = {level: stats.norm.ppf(level, loc=mu, scale=sigma) for level in levels}
    return idx, quantiles


def run_regime_evaluation(preset_name: str, detector_name: str, threshold: float):
    series = load_preset(preset_name)
    idx, quantiles = _rolling_normal_forecast(series.values)
    forecast = ForecastResult(timestamps=series.timestamps[idx], freq="D", quantiles=quantiles)
    y_true = series.values[idx]

    if detector_name == "VolatilityRegimeDetector":
        detector = VolatilityRegimeDetector(
            short_window=10, long_window=60, threshold_multiplier=threshold
        )
    else:
        detector = CUSUMDetector(warmup=60, threshold_std=threshold, drift_std=1.0)

    report = RegimeAwareEvaluator(detector).evaluate(forecast, y_true)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.timestamps, series.values, color="black", linewidth=1, label="price")
    for cp in series.changepoints:
        ax.axvline(series.timestamps[cp], color="tab:blue", alpha=0.3, linestyle="--")
    for cp in report.changepoints:
        ax.axvline(series.timestamps[idx[cp]], color="tab:red", alpha=0.6, linestyle=":")
    ax.set_title(f"{preset_name}: true changepoints (blue) vs. detected (red)")
    fig.tight_layout()

    summary = (
        f"Overall CRPS: {report.overall_crps:.3f}\n"
        f"Overall WQL: {report.overall_wql:.4f}\n"
        f"Detected changepoints: {len(report.changepoints)}"
    )
    rows = [
        [r.regime_label, r.n_periods, round(r.crps, 3), round(r.wql, 4)] for r in report.per_regime
    ]
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


with gr.Blocks(title="ForecastLens Demo") as demo:
    gr.Markdown(
        "# ForecastLens Demo\n"
        "Diagnoses forecasts against synthetic data with known ground truth -- "
        "does not train new forecasting models. "
        "[GitHub](https://github.com/forecastlens/forecastlens)"
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
            threshold = gr.Slider(1.0, 15.0, value=8.0, label="Detector threshold")
        run_btn = gr.Button("Evaluate")
        plot_out = gr.Plot()
        summary_out = gr.Textbox(label="Summary", lines=3)
        table_out = gr.Dataframe(
            headers=["regime", "n_periods", "crps", "wql"], label="Per-regime metrics"
        )
        run_btn.click(
            run_regime_evaluation, [preset, detector, threshold], [plot_out, summary_out, table_out]
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

if __name__ == "__main__":
    # Render (and most PaaS hosts) assign the port via $PORT and expect the
    # process to bind 0.0.0.0; falls back to Gradio's local default otherwise.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
