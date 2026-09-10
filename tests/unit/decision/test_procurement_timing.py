import numpy as np
import pandas as pd
import pytest

from forecastlens.core.exceptions import InvalidDecisionConfigError
from forecastlens.core.result import ForecastResult
from forecastlens.decision.procurement_timing import ProcurementTimingModel, ThresholdTimingRule


def _forecast(median_values):
    timestamps = pd.date_range("2024-01-01", periods=len(median_values), freq="D")
    return ForecastResult(
        timestamps=timestamps, freq="D", point=np.asarray(median_values, dtype=float)
    )


def test_deadline_forces_purchase_when_rule_always_says_wait():
    # Forecast keeps expecting a lower price further out than today's actual
    # price warrants waiting for, every single day -- so only the deadline forces the buy.
    forecast = _forecast([100.0, 95.0, 90.0, 92.0, 88.0])
    realized_prices = [105.0, 102.0, 98.0, 101.0, 97.0]
    model = ProcurementTimingModel(
        decision_rule=ThresholdTimingRule(wait_if_forecast_below_current_by=0.05),
        deadline_horizon=5,
        units=10.0,
    )
    report = model.evaluate(forecast, realized_prices)

    assert report.buy_day == 4
    assert report.forced_by_deadline is True
    assert report.waited_periods == 4
    assert report.cost_buy_now_per_unit == pytest.approx(105.0)
    assert report.cost_forecast_informed_per_unit == pytest.approx(97.0)
    assert report.savings_per_unit == pytest.approx(8.0)
    assert report.total_savings == pytest.approx(80.0)


def test_rule_triggers_early_buy_when_todays_price_already_beats_forecast():
    forecast = _forecast([100.0, 95.0, 90.0, 92.0, 88.0])
    # Day 1's actual price (80) already undercuts everything the forecast still expects.
    realized_prices = [105.0, 80.0, 98.0, 101.0, 97.0]
    model = ProcurementTimingModel(
        decision_rule=ThresholdTimingRule(wait_if_forecast_below_current_by=0.05),
        deadline_horizon=5,
        units=1.0,
    )
    report = model.evaluate(forecast, realized_prices)

    assert report.buy_day == 1
    assert report.forced_by_deadline is False
    assert report.cost_buy_now_per_unit == pytest.approx(105.0)
    assert report.cost_forecast_informed_per_unit == pytest.approx(80.0)
    assert report.savings_per_unit == pytest.approx(25.0)


def test_strategy_can_lose_to_buying_immediately():
    # Prices only rise -- waiting was the wrong call, and the model must report that honestly.
    forecast = _forecast([100.0, 95.0, 90.0])
    realized_prices = [100.0, 110.0, 120.0]
    model = ProcurementTimingModel(
        decision_rule=ThresholdTimingRule(wait_if_forecast_below_current_by=0.05),
        deadline_horizon=3,
        units=1.0,
    )
    report = model.evaluate(forecast, realized_prices)
    assert report.savings_per_unit < 0


def test_decision_unaffected_by_realized_prices_after_buy_day_leak_safety():
    forecast = _forecast([100.0, 95.0, 90.0, 92.0, 88.0])
    realized_prices = [105.0, 80.0, 98.0, 101.0, 97.0]
    model = ProcurementTimingModel(
        decision_rule=ThresholdTimingRule(wait_if_forecast_below_current_by=0.05),
        deadline_horizon=5,
        units=1.0,
    )
    baseline = model.evaluate(forecast, realized_prices)

    mutated_prices = [105.0, 80.0, 5.0, 999.0, -50.0]  # days after buy_day=1, wildly different
    mutated = model.evaluate(forecast, mutated_prices)

    assert mutated.buy_day == baseline.buy_day
    assert mutated.cost_forecast_informed_per_unit == baseline.cost_forecast_informed_per_unit
    assert mutated.savings_per_unit == baseline.savings_per_unit


@pytest.mark.parametrize(
    "bad_kwargs",
    [{"wait_if_forecast_below_current_by": 1.0}, {"wait_if_forecast_below_current_by": -0.1}],
)
def test_invalid_threshold_rule_raises(bad_kwargs):
    with pytest.raises(InvalidDecisionConfigError):
        ThresholdTimingRule(**bad_kwargs)


def test_invalid_deadline_horizon_raises():
    with pytest.raises(InvalidDecisionConfigError):
        ProcurementTimingModel(
            decision_rule=ThresholdTimingRule(0.05), deadline_horizon=0, units=1.0
        )


def test_invalid_units_raises():
    with pytest.raises(InvalidDecisionConfigError):
        ProcurementTimingModel(
            decision_rule=ThresholdTimingRule(0.05), deadline_horizon=3, units=0.0
        )


def test_deadline_exceeding_forecast_horizon_raises():
    forecast = _forecast([100.0, 95.0])
    model = ProcurementTimingModel(
        decision_rule=ThresholdTimingRule(0.05), deadline_horizon=5, units=1.0
    )
    with pytest.raises(InvalidDecisionConfigError):
        model.evaluate(forecast, [100.0, 95.0])


def test_deadline_exceeding_realized_prices_length_raises():
    forecast = _forecast([100.0, 95.0, 90.0, 92.0, 88.0])
    model = ProcurementTimingModel(
        decision_rule=ThresholdTimingRule(0.05), deadline_horizon=5, units=1.0
    )
    with pytest.raises(InvalidDecisionConfigError):
        model.evaluate(forecast, [100.0, 95.0])
