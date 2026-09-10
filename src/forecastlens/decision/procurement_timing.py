"""ProcurementTimingModel: "buy now" vs. forecast-informed waiting.

The strongest business story of the decision models -- a euro-per-unit
savings figure needing no statistical vocabulary to explain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from forecastlens.core.exceptions import InvalidDecisionConfigError

if TYPE_CHECKING:
    from forecastlens.core.result import ForecastResult


class TimingRule(Protocol):
    def should_wait(self, current_price: float, remaining_forecast: np.ndarray) -> bool: ...


@dataclass(frozen=True)
class ThresholdTimingRule:
    """Wait if the forecast still expects a meaningfully cheaper price ahead.

    should_wait = True whenever the lowest still-forecast price in the
    remaining window undercuts today's actual price by more than
    `wait_if_forecast_below_current_by` (a fraction, e.g. 0.03 = 3%).

    This only ever looks at `current_price` (today's own, now-known price)
    and the forecast that was already computed before the procurement
    window opened -- never at any later day's realized price, which is
    exactly the leak-safety guarantee `ProcurementTimingModel` relies on.
    """

    wait_if_forecast_below_current_by: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.wait_if_forecast_below_current_by < 1.0:
            raise InvalidDecisionConfigError(
                "wait_if_forecast_below_current_by must be in [0, 1), got "
                f"{self.wait_if_forecast_below_current_by}."
            )

    def should_wait(self, current_price: float, remaining_forecast: np.ndarray) -> bool:
        if remaining_forecast.size == 0:
            return False
        threshold = current_price * (1.0 - self.wait_if_forecast_below_current_by)
        return bool(np.min(remaining_forecast) < threshold)


@dataclass(frozen=True)
class ProcurementTimingReport:
    cost_buy_now_per_unit: float
    cost_forecast_informed_per_unit: float
    savings_per_unit: float
    total_savings: float
    buy_day: int
    waited_periods: int
    forced_by_deadline: bool


class ProcurementTimingModel:
    """Simulates buying immediately vs. waiting for a forecast-suggested better price.

    Walks forward one day at a time over `[0, deadline_horizon)`. On each
    day `t`, `current_price = realized_prices[t]` (today's own, now-known
    price) is compared by `decision_rule` against the *original* forecast's
    remaining horizon (`forecast.median()[t:]`) -- never against a later
    day's realized price. If the rule says wait, move to day `t + 1`; if it
    says buy, or the deadline is reached, purchase at `current_price`.

    Parameters
    ----------
    decision_rule : TimingRule
    deadline_horizon : int
        Periods after which a purchase is forced regardless of the rule --
        without this, the simulated strategy could "wait forever," which no
        real procurement process would allow.
    units : float
        Quantity purchased, for scaling `total_savings`.
    """

    def __init__(self, decision_rule: TimingRule, deadline_horizon: int, units: float) -> None:
        if deadline_horizon < 1:
            raise InvalidDecisionConfigError(
                f"deadline_horizon must be >= 1, got {deadline_horizon}."
            )
        if units <= 0:
            raise InvalidDecisionConfigError(f"units must be > 0, got {units}.")
        self.decision_rule = decision_rule
        self.deadline_horizon = deadline_horizon
        self.units = units

    def evaluate(
        self, forecast: ForecastResult, realized_prices: Sequence[float] | np.ndarray
    ) -> ProcurementTimingReport:
        realized_arr = np.asarray(realized_prices, dtype=float)
        forecast_median = forecast.median()

        if self.deadline_horizon > forecast_median.shape[0]:
            raise InvalidDecisionConfigError(
                f"deadline_horizon ({self.deadline_horizon}) exceeds the forecast "
                f"horizon ({forecast_median.shape[0]})."
            )
        if self.deadline_horizon > realized_arr.shape[0]:
            raise InvalidDecisionConfigError(
                f"deadline_horizon ({self.deadline_horizon}) exceeds the length of "
                f"realized_prices ({realized_arr.shape[0]})."
            )

        buy_day = self.deadline_horizon - 1
        forced_by_deadline = True
        for t in range(self.deadline_horizon):
            current_price = float(realized_arr[t])
            if t == self.deadline_horizon - 1:
                break  # deadline: forced purchase today regardless of the rule.
            remaining_forecast = forecast_median[t:]
            if not self.decision_rule.should_wait(current_price, remaining_forecast):
                buy_day = t
                forced_by_deadline = False
                break

        cost_buy_now = float(realized_arr[0])
        cost_forecast_informed = float(realized_arr[buy_day])
        savings_per_unit = cost_buy_now - cost_forecast_informed

        return ProcurementTimingReport(
            cost_buy_now_per_unit=cost_buy_now,
            cost_forecast_informed_per_unit=cost_forecast_informed,
            savings_per_unit=savings_per_unit,
            total_savings=savings_per_unit * self.units,
            buy_day=buy_day,
            waited_periods=buy_day,
            forced_by_deadline=forced_by_deadline,
        )
