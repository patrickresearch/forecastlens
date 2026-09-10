"""Regime-switching synthetic price series generator, plus 5 presets.

For tests and demos only: gives `regime/` and `decision/` a controlled
series with known ground-truth regime changes to validate against, never
a substitute for real market data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastlens.core.exceptions import InvalidGeneratorConfigError
from forecastlens.synthetic.series import GeneratedSeries


@dataclass(frozen=True)
class PriceRegimeGenerator:
    """Mean-reverting price process with optional Markov regime switching.

    x_t = x_{t-1} + speed * (target_t - x_{t-1}) + shock_t + jump_t

    `target_t` combines the active regime's level, a sinusoidal seasonal
    component, a linear trend, and a one-off structural break. `shock_t` is
    Gaussian noise whose variance can cluster via a simple ARCH(1)-style
    update; `jump_t` is a rare, larger Gaussian shock.

    Parameters
    ----------
    n_periods : int
    freq, start : pandas date_range arguments.
    base_level : float
        Long-run level target before regime/seasonal/trend adjustments.
    mean_reversion_speed : float
        Fraction of the gap to `target_t` closed each period (0, 1].
    volatility : float
        Base per-period noise standard deviation.
    seasonality_amplitude, seasonality_period : float, int
        Additive sinusoidal component on the target level.
    trend_slope : float
        Linear drift added to the target level per period.
    jump_probability, jump_size_std : float
        Per-period probability of an extra Gaussian jump, and its std dev.
    vol_cluster_persistence : float in [0, 1)
        ARCH(1)-style weight on the previous period's squared residual when
        computing the current period's variance; 0 disables clustering.
    regime_transition_matrix : (n_regimes, n_regimes) array, optional
        Row-stochastic Markov transition matrix. None means a single regime.
    regime_level_multipliers, regime_vol_multipliers : sequence of float, optional
        Per-regime multiplier on `base_level` / `volatility`.
    structural_break_at : int, optional
        Period index from which `structural_break_shift` is permanently
        added to the target level.
    structural_break_shift : float
    seed : int, optional
    """

    n_periods: int
    freq: str = "D"
    start: str = "2020-01-01"
    base_level: float = 100.0
    mean_reversion_speed: float = 0.1
    volatility: float = 1.0
    seasonality_amplitude: float = 0.0
    seasonality_period: int = 365
    trend_slope: float = 0.0
    jump_probability: float = 0.0
    jump_size_std: float = 0.0
    vol_cluster_persistence: float = 0.0
    regime_transition_matrix: np.ndarray | None = None
    regime_level_multipliers: Sequence[float] | None = None
    regime_vol_multipliers: Sequence[float] | None = None
    structural_break_at: int | None = None
    structural_break_shift: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.n_periods < 2:
            raise InvalidGeneratorConfigError(f"n_periods must be >= 2, got {self.n_periods}.")
        if not 0.0 < self.mean_reversion_speed <= 1.0:
            raise InvalidGeneratorConfigError(
                f"mean_reversion_speed must be in (0, 1], got {self.mean_reversion_speed}."
            )
        if self.volatility < 0:
            raise InvalidGeneratorConfigError("volatility must be >= 0.")
        if not 0.0 <= self.jump_probability <= 1.0:
            raise InvalidGeneratorConfigError("jump_probability must be in [0, 1].")
        if not 0.0 <= self.vol_cluster_persistence < 1.0:
            raise InvalidGeneratorConfigError("vol_cluster_persistence must be in [0, 1).")

        if self.regime_transition_matrix is not None:
            matrix = np.asarray(self.regime_transition_matrix, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise InvalidGeneratorConfigError("regime_transition_matrix must be square.")
            if not np.allclose(matrix.sum(axis=1), 1.0):
                raise InvalidGeneratorConfigError("regime_transition_matrix rows must sum to 1.")
            n_regimes = matrix.shape[0]
            if (
                self.regime_level_multipliers is not None
                and len(self.regime_level_multipliers) != n_regimes
            ):
                raise InvalidGeneratorConfigError(
                    "regime_level_multipliers length must match regime_transition_matrix size."
                )
            if (
                self.regime_vol_multipliers is not None
                and len(self.regime_vol_multipliers) != n_regimes
            ):
                raise InvalidGeneratorConfigError(
                    "regime_vol_multipliers length must match regime_transition_matrix size."
                )

        if (
            self.structural_break_at is not None
            and not 0 < self.structural_break_at < self.n_periods
        ):
            raise InvalidGeneratorConfigError(
                f"structural_break_at must be in (0, {self.n_periods}), "
                f"got {self.structural_break_at}."
            )

    def generate(self) -> GeneratedSeries:
        rng = np.random.default_rng(self.seed)
        n = self.n_periods
        timestamps = pd.date_range(self.start, periods=n, freq=self.freq)

        if self.regime_transition_matrix is not None:
            transition = np.asarray(self.regime_transition_matrix, dtype=float)
            n_regimes = transition.shape[0]
            level_mult = np.asarray(self.regime_level_multipliers or [1.0] * n_regimes, dtype=float)
            vol_mult = np.asarray(self.regime_vol_multipliers or [1.0] * n_regimes, dtype=float)
            regimes = np.zeros(n, dtype=int)
            for t in range(1, n):
                regimes[t] = rng.choice(n_regimes, p=transition[regimes[t - 1]])
        else:
            level_mult = np.array([1.0])
            vol_mult = np.array([1.0])
            regimes = np.zeros(n, dtype=int)

        values = np.empty(n, dtype=float)
        values[0] = self.base_level * level_mult[regimes[0]]
        jump_indices: list[int] = []

        for t in range(1, n):
            r = regimes[t]
            target = self.base_level * level_mult[r]
            target += self.seasonality_amplitude * np.sin(2 * np.pi * t / self.seasonality_period)
            target += self.trend_slope * t
            if self.structural_break_at is not None and t >= self.structural_break_at:
                target += self.structural_break_shift

            base_vol = self.volatility * vol_mult[r]
            if self.vol_cluster_persistence > 0 and base_vol > 0:
                prev_target = self.base_level * level_mult[regimes[t - 1]]
                resid_prev = values[t - 1] - prev_target
                running_var = (
                    1 - self.vol_cluster_persistence
                ) * base_vol**2 + self.vol_cluster_persistence * resid_prev**2
                vol_t = np.sqrt(running_var)
            else:
                vol_t = base_vol

            shock = rng.normal(0.0, vol_t) if vol_t > 0 else 0.0
            jump = 0.0
            if self.jump_probability > 0 and rng.random() < self.jump_probability:
                jump = rng.normal(0.0, self.jump_size_std)
                jump_indices.append(t)

            values[t] = (
                values[t - 1] + self.mean_reversion_speed * (target - values[t - 1]) + shock + jump
            )

        changepoints = {int(t) for t in range(1, n) if regimes[t] != regimes[t - 1]}
        if self.structural_break_at is not None:
            changepoints.add(int(self.structural_break_at))

        metadata = {
            "generator": "PriceRegimeGenerator",
            "seed": self.seed,
            "jump_indices": tuple(jump_indices),
            "n_regimes": len(level_mult),
        }
        return GeneratedSeries(
            timestamps=timestamps,
            values=values,
            regime_labels=regimes,
            changepoints=tuple(sorted(changepoints)),
            metadata=metadata,
        )


def supply_shock_spike(n_periods: int = 250, seed: int | None = 0) -> GeneratedSeries:
    """Brief, sharp price spike (e.g. a sudden supply disruption) that mean-reverts quickly."""
    transition = np.array([[0.98, 0.02], [0.6, 0.4]])
    return PriceRegimeGenerator(
        n_periods=n_periods,
        base_level=80.0,
        mean_reversion_speed=0.15,
        volatility=1.0,
        regime_transition_matrix=transition,
        regime_level_multipliers=[1.0, 1.6],
        regime_vol_multipliers=[1.0, 3.0],
        seed=seed,
    ).generate()


def seasonal_volatility_cluster(n_periods: int = 500, seed: int | None = 1) -> GeneratedSeries:
    """Seasonal price with sticky excursions into a high-volatility regime."""
    transition = np.array([[0.97, 0.03], [0.1, 0.9]])
    return PriceRegimeGenerator(
        n_periods=n_periods,
        base_level=100.0,
        mean_reversion_speed=0.05,
        volatility=1.0,
        seasonality_amplitude=8.0,
        seasonality_period=90,
        regime_transition_matrix=transition,
        regime_level_multipliers=[1.0, 1.0],
        regime_vol_multipliers=[1.0, 4.0],
        seed=seed,
    ).generate()


def gradual_demand_trend(n_periods: int = 400, seed: int | None = 2) -> GeneratedSeries:
    """Smooth linear drift, no discrete regimes -- a negative control for regime detectors."""
    return PriceRegimeGenerator(
        n_periods=n_periods,
        base_level=50.0,
        mean_reversion_speed=0.2,
        volatility=0.5,
        trend_slope=0.05,
        seed=seed,
    ).generate()


def regime_switch_markov(n_periods: int = 600, seed: int | None = 3) -> GeneratedSeries:
    """Three-regime Markov-switching price process (low / mid / high level and volatility)."""
    transition = np.array(
        [
            [0.95, 0.04, 0.01],
            [0.05, 0.90, 0.05],
            [0.02, 0.08, 0.90],
        ]
    )
    return PriceRegimeGenerator(
        n_periods=n_periods,
        base_level=100.0,
        mean_reversion_speed=0.08,
        volatility=1.5,
        regime_transition_matrix=transition,
        regime_level_multipliers=[0.8, 1.0, 1.3],
        regime_vol_multipliers=[0.8, 1.0, 1.5],
        seed=seed,
    ).generate()


def structural_break(n_periods: int = 300, seed: int | None = 4) -> GeneratedSeries:
    """Single, permanent, known-index level shift -- the simplest changepoint case."""
    return PriceRegimeGenerator(
        n_periods=n_periods,
        base_level=100.0,
        mean_reversion_speed=0.1,
        volatility=1.0,
        structural_break_at=150,
        structural_break_shift=40.0,
        seed=seed,
    ).generate()


PRICE_PRESETS: dict[str, Callable[[], GeneratedSeries]] = {
    "supply_shock_spike": supply_shock_spike,
    "seasonal_volatility_cluster": seasonal_volatility_cluster,
    "gradual_demand_trend": gradual_demand_trend,
    "regime_switch_markov": regime_switch_markov,
    "structural_break": structural_break,
}
