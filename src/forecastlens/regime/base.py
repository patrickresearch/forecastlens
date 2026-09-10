"""Shared interface for regime/changepoint detectors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class RegimeDetectionResult:
    """Output of a regime detector.

    Parameters
    ----------
    changepoints : tuple[int, ...]
        Positional indices where the detector flags a regime change.
    regime_labels : np.ndarray
        Integer regime id per timestamp.
    metadata : Mapping[str, Any]
        Detector name and any parameters/statistics worth surfacing.
    """

    changepoints: tuple[int, ...]
    regime_labels: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RegimeDetector(Protocol):
    """Interface both `VolatilityRegimeDetector` and `CUSUMDetector` implement.

    Lets `RegimeAwareEvaluator` swap detectors without caring which one it holds.
    """

    def detect(self, values: np.ndarray) -> RegimeDetectionResult: ...
