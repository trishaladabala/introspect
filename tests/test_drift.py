"""
test_drift.py — Tests for the Semantic Drift Detector.

Validates drift detection with synthetic embedding pairs, threshold
flagging accuracy, baseline management, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from introspect.core.drift import SemanticDriftDetector, DriftReport


class TestSemanticDriftDetector:
    """Tests for SemanticDriftDetector."""

    def test_zero_drift_identical_embeddings(self) -> None:
        """Identical embeddings should produce zero drift."""
        detector = SemanticDriftDetector(threshold_z=2.0)
        embeddings = np.random.randn(32, 64).astype(np.float32)

        detector.set_baseline("v1", embeddings)
        report = detector.compare("v1", "v1-copy", embeddings)

        assert report.aggregate_drift == pytest.approx(0.0, abs=1e-6)
        assert report.passed is True

    def test_high_drift_random_embeddings(self) -> None:
        """Completely different embeddings should produce high drift."""
        detector = SemanticDriftDetector(threshold_z=2.0)
        rng = np.random.default_rng(42)

        baseline = rng.standard_normal((32, 64)).astype(np.float32)
        comparison = rng.standard_normal((32, 64)).astype(np.float32)

        detector.set_baseline("v1", baseline)
        report = detector.compare("v1", "v2-random", comparison)

        # Cosine distance between random vectors should be ~1.0.
        assert report.aggregate_drift > 0.5
        assert report.passed is False  # High drift → flagged

    def test_small_perturbation_passes(self) -> None:
        """Small perturbations should not trigger drift flags."""
        detector = SemanticDriftDetector(threshold_z=2.0)
        rng = np.random.default_rng(42)

        baseline = rng.standard_normal((32, 64)).astype(np.float32)
        # Small perturbation.
        noise = rng.standard_normal((32, 64)).astype(np.float32) * 0.01
        comparison = baseline + noise

        detector.set_baseline("v1", baseline)
        report = detector.compare("v1", "v1.1", comparison)

        assert report.aggregate_drift < 0.1
        assert report.passed is True

    def test_per_layer_segmentation(self) -> None:
        """Layer segments should produce independent drift scores."""
        detector = SemanticDriftDetector(threshold_z=2.0)
        rng = np.random.default_rng(42)

        baseline = rng.standard_normal((64, 32)).astype(np.float32)

        # Perturb only the second half.
        comparison = baseline.copy()
        comparison[32:] += rng.standard_normal((32, 32)).astype(np.float32) * 2.0

        detector.set_baseline("v1", baseline)
        report = detector.compare(
            "v1", "v2",
            comparison,
            layer_segments={
                "layer_0": (0, 32),
                "layer_1": (32, 64),
            },
        )

        assert len(report.layer_drifts) == 2
        # First layer should have low drift.
        assert report.layer_drifts[0].mean_cosine_distance < 0.01
        # Second layer should have high drift.
        assert report.layer_drifts[1].mean_cosine_distance > report.layer_drifts[0].mean_cosine_distance

    def test_z_score_detection_with_history(self) -> None:
        """Z-score detection should activate after sufficient history."""
        detector = SemanticDriftDetector(threshold_z=2.0, min_history=3)
        rng = np.random.default_rng(42)

        baseline = rng.standard_normal((16, 32)).astype(np.float32)
        detector.set_baseline("v1", baseline)

        # Build history with small perturbations.
        for i in range(4):
            noise = rng.standard_normal((16, 32)).astype(np.float32) * 0.01
            detector.compare("v1", f"v1.{i}", baseline + noise)

        # Now a large perturbation should be flagged.
        big_noise = rng.standard_normal((16, 32)).astype(np.float32) * 3.0
        report = detector.compare("v1", "v1-anomaly", baseline + big_noise)

        assert report.aggregate_z_score > 2.0
        assert report.passed is False

    def test_baseline_not_found_raises(self) -> None:
        """Comparing against a nonexistent baseline should raise KeyError."""
        detector = SemanticDriftDetector()

        comparison = np.random.randn(10, 32).astype(np.float32)
        with pytest.raises(KeyError, match="not found"):
            detector.compare("nonexistent", "v2", comparison)

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched embedding shapes should raise ValueError."""
        detector = SemanticDriftDetector()

        baseline = np.random.randn(10, 32).astype(np.float32)
        comparison = np.random.randn(10, 64).astype(np.float32)

        detector.set_baseline("v1", baseline)
        with pytest.raises(ValueError, match="Shape mismatch"):
            detector.compare("v1", "v2", comparison)

    def test_non_2d_embeddings_raises(self) -> None:
        """Non-2D embeddings should raise ValueError."""
        detector = SemanticDriftDetector()

        with pytest.raises(ValueError, match="must be 2D"):
            detector.set_baseline("v1", np.ones(10, dtype=np.float32))

    def test_baseline_ids_property(self) -> None:
        """baseline_ids should list all stored baselines."""
        detector = SemanticDriftDetector()
        assert detector.baseline_ids == []

        detector.set_baseline("v1", np.random.randn(5, 8).astype(np.float32))
        detector.set_baseline("v2", np.random.randn(5, 8).astype(np.float32))

        assert sorted(detector.baseline_ids) == ["v1", "v2"]

    def test_report_timestamps_and_ids(self) -> None:
        """DriftReport should have correct baseline and comparison IDs."""
        detector = SemanticDriftDetector()
        emb = np.random.randn(8, 16).astype(np.float32)

        detector.set_baseline("baseline-A", emb)
        report = detector.compare("baseline-A", "comparison-B", emb)

        assert report.baseline_id == "baseline-A"
        assert report.comparison_id == "comparison-B"
        assert report.timestamp > 0
        assert report.elapsed_ms >= 0

    def test_positive_threshold_validation(self) -> None:
        """threshold_z must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            SemanticDriftDetector(threshold_z=-1.0)
