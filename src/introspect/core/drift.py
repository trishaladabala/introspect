"""
drift.py — Semantic Drift Detection in continuous embedding spaces.

Monitors embedding-space geometry across model versions, quantization
schemes, or checkpoint updates. If a new model configuration causes
embeddings to drift beyond a statistical threshold, the detector flags
a regression — preventing silent quality degradation before it reaches
production.

The detector uses cosine distance as its primary metric and z-score
based threshold detection for automated pass/fail decisions. It stores
baselines and supports both per-layer and aggregate drift analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LayerDrift:
    """Drift metrics for a single embedding layer or segment.

    Attributes:
        layer_name: Identifier for the layer/segment.
        mean_cosine_distance: Average cosine distance from baseline.
        max_cosine_distance: Worst-case cosine distance.
        std_cosine_distance: Standard deviation of cosine distances.
        z_score: How many standard deviations the mean drift is from
            the historical average drift for this layer.
        flagged: Whether this layer exceeds the z-score threshold.
        num_positions: Number of positions compared.
    """
    layer_name: str
    mean_cosine_distance: float
    max_cosine_distance: float
    std_cosine_distance: float
    z_score: float
    flagged: bool
    num_positions: int


@dataclass(frozen=True)
class DriftReport:
    """Complete drift analysis across all layers/segments.

    Attributes:
        aggregate_drift: Overall mean cosine distance across all layers.
        aggregate_z_score: Z-score of the aggregate drift.
        passed: Whether the aggregate drift is within acceptable bounds.
        threshold_z: The z-score threshold used for flagging.
        layer_drifts: Per-layer breakdown.
        timestamp: Unix timestamp of this report.
        baseline_id: Identifier for the baseline used.
        comparison_id: Identifier for the comparison embeddings.
        elapsed_ms: Computation time in milliseconds.
    """
    aggregate_drift: float
    aggregate_z_score: float
    passed: bool
    threshold_z: float
    layer_drifts: list[LayerDrift]
    timestamp: float
    baseline_id: str
    comparison_id: str
    elapsed_ms: float





class SemanticDriftDetector:
    """Detects semantic drift in continuous embedding spaces.

    Compares incoming embeddings against stored baselines using cosine
    distance. Maintains a drift history to compute z-scores, enabling
    statistically grounded anomaly detection.

    Usage:
        detector = SemanticDriftDetector(threshold_z=2.0)
        detector.set_baseline("v1.0", baseline_embeddings)
        report = detector.compare("v1.0", "v1.1-quantized", new_embeddings)
        assert report.passed, f"Drift z-score {report.aggregate_z_score:.2f} exceeds threshold"
    """

    def __init__(
        self,
        store: Any | None = None,
        threshold_z: float = 2.0,
        min_history: int = 3,
    ) -> None:
        """Initialize the drift detector.

        Args:
            store: MetricsStore instance to fetch historical drift scores.
            threshold_z: Z-score threshold for flagging drift. Values
                above this are considered anomalous.
            min_history: Minimum number of historical comparisons before
                z-score based detection activates. Before this, uses a
                fixed cosine distance threshold of 0.1.
        """
        if threshold_z <= 0:
            raise ValueError(f"threshold_z must be positive, got {threshold_z}")

        self._store = store
        self._threshold_z = threshold_z
        self._min_history = min_history

    @property
    def threshold_z(self) -> float:
        """Current z-score threshold."""
        return self._threshold_z

    def compare(
        self,
        baseline_id: str,
        comparison_id: str,
        baseline_embeddings: NDArray[np.float32],
        comparison_embeddings: NDArray[np.float32],
        layer_segments: dict[str, tuple[int, int]] | None = None,
    ) -> DriftReport:
        """Compare embeddings against a baseline in a stateless manner.

        Args:
            baseline_id: Identifier of the baseline.
            comparison_id: Identifier for the incoming embeddings.
            baseline_embeddings: The baseline embedding matrix.
            comparison_embeddings: New embedding matrix (shape: [seq_len, embed_dim]).
            layer_segments: Optional dict mapping layer names to (start, end)
                index ranges for per-layer analysis. If None, treats the
                entire matrix as a single segment called "full_sequence".

        Returns:
            A DriftReport with aggregate and per-layer metrics.

        Raises:
            ValueError: If embedding shapes are incompatible.
        """
        start_time = time.perf_counter()

        if baseline_embeddings.ndim != 2:
            raise ValueError(
                f"baseline_embeddings must be 2D, got shape {baseline_embeddings.shape}"
            )
        if comparison_embeddings.ndim != 2:
            raise ValueError(
                f"comparison_embeddings must be 2D, got shape {comparison_embeddings.shape}"
            )
        if comparison_embeddings.shape != baseline_embeddings.shape:
            raise ValueError(
                f"Shape mismatch: baseline={baseline_embeddings.shape}, "
                f"comparison={comparison_embeddings.shape}"
            )
            
        drift_history = []
        if self._store is not None:
            # Fetch last N aggregate drifts for this baseline specifically to compute z-scores.
            reports = self._store.get_drift_history(limit=100)
            drift_history = [
                r["aggregate_drift"] for r in reversed(reports) 
                if r["baseline_id"] == baseline_id
            ]

        # Default: treat entire sequence as one segment.
        if layer_segments is None:
            layer_segments = {"full_sequence": (0, baseline_embeddings.shape[0])}

        # Compute per-layer drift.
        layer_drifts: list[LayerDrift] = []
        all_distances: list[float] = []

        for layer_name, (start, end) in layer_segments.items():
            baseline_slice = baseline_embeddings[start:end]
            comparison_slice = comparison_embeddings[start:end]

            distances = self._cosine_distances(baseline_slice, comparison_slice)
            all_distances.extend(distances.tolist())

            mean_dist = float(distances.mean())
            max_dist = float(distances.max())
            std_dist = float(distances.std())

            z = self._compute_z_score(mean_dist, drift_history)
            flagged = self._is_flagged(mean_dist, z, drift_history)

            layer_drifts.append(LayerDrift(
                layer_name=layer_name,
                mean_cosine_distance=mean_dist,
                max_cosine_distance=max_dist,
                std_cosine_distance=std_dist,
                z_score=z,
                flagged=flagged,
                num_positions=end - start,
            ))

        # Aggregate metrics.
        all_dist_arr = np.array(all_distances)
        aggregate_drift = float(all_dist_arr.mean())
        aggregate_z = self._compute_z_score(aggregate_drift, drift_history)
        passed = not self._is_flagged(aggregate_drift, aggregate_z, drift_history)

        elapsed = (time.perf_counter() - start_time) * 1000

        return DriftReport(
            aggregate_drift=aggregate_drift,
            aggregate_z_score=aggregate_z,
            passed=passed,
            threshold_z=self._threshold_z,
            layer_drifts=layer_drifts,
            timestamp=time.time(),
            baseline_id=baseline_id,
            comparison_id=comparison_id,
            elapsed_ms=elapsed,
        )

    def _cosine_distances(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute row-wise cosine distance: 1 - cosine_similarity.

        Both arrays must have shape [N, D].
        """
        # Normalize rows.
        a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-10)
        b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-10)

        # Cosine similarity per row.
        cos_sim = np.sum(a_norm * b_norm, axis=1)

        # Clamp to [-1, 1] for numerical safety.
        cos_sim = np.clip(cos_sim, -1.0, 1.0)

        return (1.0 - cos_sim).astype(np.float32)

    def _compute_z_score(
        self,
        value: float,
        history: list[float],
    ) -> float:
        """Compute z-score of a value against its history."""
        if len(history) < self._min_history:
            return 0.0  # Not enough data for statistical detection.

        hist = np.array(history)
        mean = float(hist.mean())
        std = float(hist.std())

        if std < 1e-10:
            return 0.0 if abs(value - mean) < 1e-10 else float("inf")

        return (value - mean) / std

    def _is_flagged(
        self,
        distance: float,
        z_score: float,
        history: list[float],
    ) -> bool:
        """Determine if drift should be flagged."""
        if len(history) < self._min_history:
            # Fallback: use fixed threshold when history is sparse.
            return distance > 0.1

        return abs(z_score) > self._threshold_z
