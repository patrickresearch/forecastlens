"""Shared conceptual interface for evaluators (ROADMAP.md Section 2).

`DemandForecastEvaluator` and `RegimeAwareEvaluator` each return a
purpose-specific report dataclass rather than a common one -- a demand
evaluation (WMAPE, bias, service level) and a regime-conditional one
(per-regime CRPS/WQL/calibration) share almost no fields. This Protocol
documents the shared `evaluate(forecast, y_true) -> Report` shape, not a
rigid interface both must satisfy identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

    from forecastlens.core.result import ForecastResult


class Evaluator(Protocol):
    def evaluate(self, forecast: ForecastResult, y_true: np.ndarray) -> Any: ...
