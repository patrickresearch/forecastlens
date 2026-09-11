"""Merges a detector's raw segments by statistically similar mean level.

A detector like `CUSUMDetector` assigns a new, ever-increasing label to
every changepoint it finds -- even when the series returns to a level it
has already visited. Over a long series this can produce 20+ raw segments
where only a handful of genuinely distinct levels exist. This wraps any
`RegimeDetector` and collapses recurring levels back into one label via
agglomerative clustering on each segment's mean value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from forecastlens.core.exceptions import InvalidDetectorConfigError
from forecastlens.regime.base import RegimeDetectionResult

if TYPE_CHECKING:
    from forecastlens.regime.base import RegimeDetector


def _merge_similar_segments(
    values: np.ndarray, raw_labels: np.ndarray, distance_threshold: float
) -> np.ndarray:
    """Relabel `raw_labels` so segments with a similar mean share one label.

    Clustering runs on each raw segment's mean value (1-D agglomerative
    clustering, average linkage, cut at `distance_threshold * overall_std`)
    -- not on time, so two segments far apart in the series can still merge
    if their levels match. The returned labels are 0..K-1 in order of each
    cluster's first temporal appearance, so they stay readable.
    """
    raw_labels = np.asarray(raw_labels)
    unique_ids: list[int] = []
    seen: set[int] = set()
    for lbl in raw_labels.tolist():
        if lbl not in seen:
            seen.add(lbl)
            unique_ids.append(lbl)

    if len(unique_ids) <= 1:
        unchanged: np.ndarray = raw_labels.copy()
        return unchanged

    segment_means = np.array([values[raw_labels == uid].mean() for uid in unique_ids])
    overall_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    if overall_std == 0.0:
        cluster_of_id = dict.fromkeys(unique_ids, 1)
    else:
        linkage_matrix = linkage(segment_means.reshape(-1, 1), method="average")
        cluster_ids = fcluster(
            linkage_matrix, t=distance_threshold * overall_std, criterion="distance"
        )
        cluster_of_id = dict(zip(unique_ids, cluster_ids.tolist(), strict=True))

    remap: dict[int, int] = {}
    new_labels = np.empty_like(raw_labels)
    for i, lbl in enumerate(raw_labels.tolist()):
        cluster = cluster_of_id[lbl]
        if cluster not in remap:
            remap[cluster] = len(remap)
        new_labels[i] = remap[cluster]
    return new_labels


@dataclass(frozen=True)
class ClusteredRegimeDetector:
    """Wraps another `RegimeDetector`, merging segments with similar mean levels.

    Parameters
    ----------
    base_detector : RegimeDetector
        Any detector implementing the shared protocol (e.g. `CUSUMDetector`).
    distance_threshold : float
        Merge segments whose mean values differ by less than
        `distance_threshold * overall_std`. Larger values merge more
        aggressively (fewer, coarser clusters).
    """

    base_detector: RegimeDetector
    distance_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.distance_threshold <= 0:
            raise InvalidDetectorConfigError(
                f"distance_threshold must be > 0, got {self.distance_threshold}."
            )

    def detect(self, values: np.ndarray) -> RegimeDetectionResult:
        values_arr = np.asarray(values, dtype=float)
        base_result = self.base_detector.detect(values_arr)
        merged_labels = _merge_similar_segments(
            values_arr, base_result.regime_labels, self.distance_threshold
        )
        changepoints = tuple(
            int(t) for t in range(1, len(merged_labels)) if merged_labels[t] != merged_labels[t - 1]
        )
        n_raw = len(set(base_result.regime_labels.tolist()))
        n_clusters = len(set(merged_labels.tolist()))
        return RegimeDetectionResult(
            changepoints=changepoints,
            regime_labels=merged_labels,
            metadata={
                **base_result.metadata,
                "wrapped_by": "ClusteredRegimeDetector",
                "n_raw_segments": n_raw,
                "n_clusters": n_clusters,
            },
        )
