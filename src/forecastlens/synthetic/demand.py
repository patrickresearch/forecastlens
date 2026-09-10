"""Synthetic demand series generator, plus 5 presets.

For tests and demos only: gives `regime/` and `decision/` a controlled
series with known ground-truth events (promos, disruptions, lifecycle
phases) to validate against, never a substitute for real demand data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastlens.core.exceptions import InvalidGeneratorConfigError
from forecastlens.synthetic.series import GeneratedSeries

_VALID_LIFECYCLES = ("none", "ramp_up", "ramp_down", "s_curve")

# Regime labels used in `regime_labels` / metadata.
_REGIME_NORMAL = 0
_REGIME_PROMO = 1
_REGIME_HOARDING = 2
_REGIME_DESTOCK = 3


@dataclass(frozen=True)
class DemandGenerator:
    """Baseline demand with seasonality, intermittency, promos, lifecycle, and disruption.

    demand_t = baseline * lifecycle_factor(t) * seasonal_factor(t) * event_factor(t) + noise_t,
    zeroed out with probability `intermittency_probability` and clipped at 0.

    Parameters
    ----------
    n_periods : int
    freq, start : pandas date_range arguments.
    baseline : float
    seasonality_amplitude, seasonality_period : float, int
        Multiplicative sinusoidal seasonality: `1 + amplitude * sin(2*pi*t/period)`.
    intermittency_probability : float in [0, 1]
        Per-period probability that demand is exactly zero (spare-parts-style).
    promo_probability, promo_size_multiplier : float
        Per-period probability of a one-off promo, and its demand multiplier.
    lifecycle : {"none", "ramp_up", "ramp_down", "s_curve"}
        "s_curve" is a rising logistic (around `lifecycle_midpoint`) times a
        falling logistic (around `lifecycle_decline_midpoint`), giving the
        classic introduction-growth-maturity-decline bump.
    lifecycle_midpoint, lifecycle_decline_midpoint : int, optional
    lifecycle_steepness : float
    disruption_at : int, optional
        Period index of a supply disruption: a hoarding spike for
        `hoarding_window` periods, followed by a destocking trough for
        `destock_window` periods, then a return to baseline.
    hoarding_window, hoarding_multiplier, destock_window, destock_multiplier : int, float
    noise_std : float
    seed : int, optional
    """

    n_periods: int
    freq: str = "D"
    start: str = "2020-01-01"
    baseline: float = 20.0
    seasonality_amplitude: float = 0.0
    seasonality_period: int = 365
    intermittency_probability: float = 0.0
    promo_probability: float = 0.0
    promo_size_multiplier: float = 1.0
    lifecycle: str = "none"
    lifecycle_midpoint: int | None = None
    lifecycle_decline_midpoint: int | None = None
    lifecycle_steepness: float = 0.1
    disruption_at: int | None = None
    hoarding_window: int = 0
    hoarding_multiplier: float = 1.0
    destock_window: int = 0
    destock_multiplier: float = 1.0
    noise_std: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_periods < 2:
            raise InvalidGeneratorConfigError(f"n_periods must be >= 2, got {self.n_periods}.")
        if self.baseline < 0:
            raise InvalidGeneratorConfigError("baseline must be >= 0.")
        if not 0.0 <= self.intermittency_probability <= 1.0:
            raise InvalidGeneratorConfigError("intermittency_probability must be in [0, 1].")
        if not 0.0 <= self.promo_probability <= 1.0:
            raise InvalidGeneratorConfigError("promo_probability must be in [0, 1].")
        if self.lifecycle not in _VALID_LIFECYCLES:
            raise InvalidGeneratorConfigError(
                f"lifecycle must be one of {_VALID_LIFECYCLES}, got {self.lifecycle!r}."
            )
        if self.lifecycle != "none" and self.lifecycle_midpoint is None:
            raise InvalidGeneratorConfigError(
                f"lifecycle={self.lifecycle!r} needs lifecycle_midpoint."
            )
        if self.lifecycle == "s_curve" and self.lifecycle_decline_midpoint is None:
            raise InvalidGeneratorConfigError(
                "lifecycle='s_curve' needs lifecycle_decline_midpoint."
            )
        if self.disruption_at is not None and not 0 <= self.disruption_at < self.n_periods:
            raise InvalidGeneratorConfigError(
                f"disruption_at must be in [0, {self.n_periods}), got {self.disruption_at}."
            )
        if self.hoarding_window < 0 or self.destock_window < 0:
            raise InvalidGeneratorConfigError("hoarding_window/destock_window must be >= 0.")
        if self.noise_std < 0:
            raise InvalidGeneratorConfigError("noise_std must be >= 0.")

    def _lifecycle_factor(self, t: int) -> float:
        if self.lifecycle == "none":
            return 1.0
        # __post_init__ guarantees the midpoint(s) are set whenever lifecycle != "none".
        assert self.lifecycle_midpoint is not None
        k = self.lifecycle_steepness
        if self.lifecycle == "ramp_up":
            return float(1.0 / (1.0 + np.exp(-k * (t - self.lifecycle_midpoint))))
        if self.lifecycle == "ramp_down":
            return float(1.0 - 1.0 / (1.0 + np.exp(-k * (t - self.lifecycle_midpoint))))
        # s_curve: rising logistic * falling logistic -> introduction/growth/decline bump.
        assert self.lifecycle_decline_midpoint is not None
        rise = 1.0 / (1.0 + np.exp(-k * (t - self.lifecycle_midpoint)))
        fall = 1.0 / (1.0 + np.exp(k * (t - self.lifecycle_decline_midpoint)))
        return float(rise * fall)

    def generate(self) -> GeneratedSeries:
        rng = np.random.default_rng(self.seed)
        n = self.n_periods
        timestamps = pd.date_range(self.start, periods=n, freq=self.freq)

        regimes = np.zeros(n, dtype=int)
        values = np.empty(n, dtype=float)
        promo_indices: list[int] = []

        for t in range(n):
            factor = self._lifecycle_factor(t)
            factor *= 1.0 + self.seasonality_amplitude * np.sin(
                2 * np.pi * t / self.seasonality_period
            )

            if self.promo_probability > 0 and rng.random() < self.promo_probability:
                factor *= self.promo_size_multiplier
                regimes[t] = _REGIME_PROMO
                promo_indices.append(t)

            if self.disruption_at is not None and t >= self.disruption_at:
                offset = t - self.disruption_at
                if offset < self.hoarding_window:
                    factor *= self.hoarding_multiplier
                    regimes[t] = _REGIME_HOARDING
                elif offset < self.hoarding_window + self.destock_window:
                    factor *= self.destock_multiplier
                    regimes[t] = _REGIME_DESTOCK

            demand = self.baseline * factor
            if self.noise_std > 0:
                demand += rng.normal(0.0, self.noise_std)
            if self.intermittency_probability > 0 and rng.random() < self.intermittency_probability:
                demand = 0.0
            values[t] = max(0.0, demand)

        changepoints: set[int] = set()
        if self.lifecycle in ("ramp_up", "ramp_down"):
            assert self.lifecycle_midpoint is not None
            changepoints.add(self.lifecycle_midpoint)
        elif self.lifecycle == "s_curve":
            assert self.lifecycle_midpoint is not None
            assert self.lifecycle_decline_midpoint is not None
            changepoints.add(self.lifecycle_midpoint)
            changepoints.add(self.lifecycle_decline_midpoint)
        if self.disruption_at is not None:
            changepoints.add(int(self.disruption_at))
            if self.hoarding_window > 0:
                changepoints.add(int(self.disruption_at + self.hoarding_window))
            if self.destock_window > 0:
                changepoints.add(
                    int(self.disruption_at + self.hoarding_window + self.destock_window)
                )
        changepoints = {c for c in changepoints if 0 <= c < n}

        metadata = {
            "generator": "DemandGenerator",
            "seed": self.seed,
            "promo_indices": tuple(promo_indices),
        }
        return GeneratedSeries(
            timestamps=timestamps,
            values=values,
            regime_labels=regimes,
            changepoints=tuple(sorted(changepoints)),
            metadata=metadata,
        )


def intermittent_spare_parts(n_periods: int = 365, seed: int | None = 10) -> GeneratedSeries:
    """Low-volume, mostly-zero demand typical of spare parts / slow movers."""
    return DemandGenerator(
        n_periods=n_periods,
        baseline=3.0,
        intermittency_probability=0.7,
        seasonality_amplitude=0.1,
        noise_std=0.5,
        seed=seed,
    ).generate()


def seasonal_consumer_good(n_periods: int = 730, seed: int | None = 11) -> GeneratedSeries:
    """Smooth, strongly seasonal demand -- a negative control for regime detectors."""
    return DemandGenerator(
        n_periods=n_periods,
        baseline=100.0,
        seasonality_amplitude=0.5,
        seasonality_period=365,
        noise_std=5.0,
        seed=seed,
    ).generate()


def promo_spike(n_periods: int = 365, seed: int | None = 12) -> GeneratedSeries:
    """Baseline demand with occasional sharp promo-driven spikes."""
    return DemandGenerator(
        n_periods=n_periods,
        baseline=50.0,
        promo_probability=0.03,
        promo_size_multiplier=4.0,
        noise_std=3.0,
        seed=seed,
    ).generate()


def product_lifecycle_ramp(n_periods: int = 500, seed: int | None = 13) -> GeneratedSeries:
    """Introduction-growth-maturity-decline bump typical of a product lifecycle."""
    return DemandGenerator(
        n_periods=n_periods,
        baseline=200.0,
        lifecycle="s_curve",
        lifecycle_midpoint=100,
        lifecycle_decline_midpoint=400,
        lifecycle_steepness=0.05,
        noise_std=5.0,
        seed=seed,
    ).generate()


def supply_disruption_hoarding(n_periods: int = 300, seed: int | None = 14) -> GeneratedSeries:
    """Disruption-driven hoarding spike followed by a destocking trough (bullwhip effect)."""
    return DemandGenerator(
        n_periods=n_periods,
        baseline=80.0,
        disruption_at=150,
        hoarding_window=15,
        hoarding_multiplier=3.0,
        destock_window=30,
        destock_multiplier=0.2,
        noise_std=4.0,
        seed=seed,
    ).generate()


DEMAND_PRESETS: dict[str, Callable[[], GeneratedSeries]] = {
    "intermittent_spare_parts": intermittent_spare_parts,
    "seasonal_consumer_good": seasonal_consumer_good,
    "promo_spike": promo_spike,
    "product_lifecycle_ramp": product_lifecycle_ramp,
    "supply_disruption_hoarding": supply_disruption_hoarding,
}
