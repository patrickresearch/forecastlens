"""Shared conceptual interface for decision models (roadmap.md 1.5).

`DecisionRelevantBucketAccuracy` and `ProcurementTimingModel` each expose a
richer, purpose-specific `evaluate()` method rather than this Protocol's
generic `decide`/`cost` pair -- their inputs differ too much (a cost matrix
over discrete buckets vs. a walk-forward buy/wait simulation over raw
prices) to share one call signature usefully. This Protocol documents the
shared idea -- turn a forecast into a decision, then price that decision
against what actually happened -- not a rigid interface both must satisfy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from forecastlens.core.result import ForecastResult


@dataclass(frozen=True)
class Decision:
    """A single decision output: a label plus any supporting detail."""

    label: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DecisionModel(Protocol):
    def decide(self, forecast: ForecastResult) -> Decision: ...
    def cost(self, decision: Decision, realized: float) -> float: ...
