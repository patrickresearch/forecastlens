"""Shared economic-value scoring for decision models.

`relative_value_score` is a deliberately simplified MVP metric, NOT the
academic Murphy value score (which normalizes against a perfect-foresight
forecast on a 0-1 scale, per Murphy 1973 / Gneiting & Raftery 2007). Ours
can go negative (model worse than naive) or exceed 1 (naive cost near
zero) -- see roadmap.md 1.5. A full Murphy-normalized score is a v0.2
opt-in, not part of this MVP.
"""

from __future__ import annotations

from forecastlens.core.exceptions import ZeroNaiveCostError


def relative_value_score(cost_naive: float, cost_model: float) -> float:
    """(cost_naive - cost_model) / cost_naive.

    Positive means the model beat the naive baseline; 1.0 means the model
    was free (cost_model == 0); negative means the model did worse than
    naive. Raises rather than returning NaN/inf when `cost_naive == 0`,
    since the ratio is genuinely undefined there, not just numerically
    awkward.
    """
    if cost_naive == 0:
        raise ZeroNaiveCostError(
            "relative_value_score is undefined when cost_naive == 0 "
            "(the naive baseline was free under this cost matrix)."
        )
    return (cost_naive - cost_model) / cost_naive
