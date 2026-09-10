"""Unified registry of synthetic presets with known ground truth.

Basis for objective regime-detector validation (does the detector recover
`GeneratedSeries.changepoints` within a tolerance?) instead of eyeballing
plots. Economically-optimal-decision ground truth (for `decision/`) is not
yet included -- it lands once the decision models exist, to avoid
fabricating a "correct" decision ahead of that design.
"""

from __future__ import annotations

from collections.abc import Callable

from forecastlens.core.exceptions import ForecastLensError
from forecastlens.synthetic.demand import DEMAND_PRESETS
from forecastlens.synthetic.price import PRICE_PRESETS
from forecastlens.synthetic.series import GeneratedSeries

PRESETS: dict[str, Callable[[], GeneratedSeries]] = {**PRICE_PRESETS, **DEMAND_PRESETS}


def load_preset(name: str) -> GeneratedSeries:
    """Generate a preset series by name."""
    if name not in PRESETS:
        raise ForecastLensError(f"Unknown preset {name!r}. Available: {list_presets()}.")
    return PRESETS[name]()


def list_presets() -> list[str]:
    """Names of all registered presets (price and demand), sorted."""
    return sorted(PRESETS)
