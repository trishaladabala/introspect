"""
test_drift.py — Tests for the Semantic Drift Detector.

Validates drift detection with synthetic embedding pairs, threshold
flagging accuracy, z-score history, and edge cases.

All tests use the stateless ``compare()`` API — both baseline and
comparison embeddings are passed directly to the detector.
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

        report = detector.compare(
            "v1", "v1-copy",
            baseline_embeddings=embeddings,
            comparison_embeddings=embeddings,
        )

        assert report.aggregate_drift == pytest.approx(0.0, abs=1e-6)
        assert report.passed is True

    def test_high_drift_random_embeddings(self) -> None:
        """Completely different embeddings should produce high drift."""
        detector = SemanticDriftDetector(threshold_z=2.0)
        rng = np.random.default_rng(42)

        baseline = rng.standard_normal((32, 64)).astype(np.float32)
        comparison = rng.standard_normal((32, 64)).astype(np.float32)

        report = detector.compare(
            "v1", "v2-random",
            baseline_embeddings=baseline,
            comparison_embeddings=comparison,
        )

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

        report = detector.compare(
            "v1", "v1.1",
            baseline_embeddings=baseline,
            comparison_embeddings=comparison,
        )

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

        report = detector.compare(
            "v1", "v2",
            baseline_embeddings=baseline,
            comparison_embeddings=comparison,
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
        """Z-score detection should activate after sufficient history.

        We wire up a MetricsStore so the detector can fetch historical
        drift scores and compute z-scores for anomaly flagging.
        """
        from introspect.storage.timeseries import MetricsStore
        import tempfile, os

        db_path = os.path.join(tempfile.mkdtemp(), "test_drift_history.db")
        store = MetricsStore(db_path)

        detector = SemanticDriftDetector(store=store, threshold_z=2.0, min_history=3)
        rng = np.random.default_rng(42)

        baseline = rng.standard_normal((16, 32)).astype(np.float32)

        # Build history with small perturbations.
        for i in range(4):
            noise = rng.standard_normal((16, 32)).astype(np.float32) * 0.01
            report = detector.compare(
                "v1", f"v1.{i}",
                baseline_embeddings=baseline,
                comparison_embeddings=baseline + noise,
            )
            # Persist the drift report so the store has history.
            run_id = store.create_run(model_config={"test": True})
            store.record_drift(
                run_id=run_id,
                aggregate_drift=report.aggregate_drift,
                aggregate_z_score=report.aggregate_z_score,
                passed=report.passed,
                threshold_z=report.threshold_z,
                baseline_id=report.baseline_id,
                comparison_id=report.comparison_id,
                elapsed_ms=report.elapsed_ms,
            )

        # Now a large perturbation should be flagged.
        big_noise = rng.standard_normal((16, 32)).astype(np.float32) * 3.0
        report = detector.compare(
            "v1", "v1-anomaly",
            baseline_embeddings=baseline,
            comparison_embeddings=baseline + big_noise,
        )

        assert report.aggregate_z_score > 2.0
        assert report.passed is False

        store.close()

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched embedding shapes should raise ValueError."""
        detector = SemanticDriftDetector()

        baseline = np.random.randn(10, 32).astype(np.float32)
        comparison = np.random.randn(10, 64).astype(np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            detector.compare(
                "v1", "v2",
                baseline_embeddings=baseline,
                comparison_embeddings=comparison,
            )

    def test_non_2d_baseline_raises(self) -> None:
        """Non-2D baseline embeddings should raise ValueError."""
        detector = SemanticDriftDetector()

        with pytest.raises(ValueError, match="must be 2D"):
            detector.compare(
                "v1", "v2",
                baseline_embeddings=np.ones(10, dtype=np.float32),
                comparison_embeddings=np.ones((10, 8), dtype=np.float32),
            )

    def test_non_2d_comparison_raises(self) -> None:
        """Non-2D comparison embeddings should raise ValueError."""
        detector = SemanticDriftDetector()

        with pytest.raises(ValueError, match="must be 2D"):
            detector.compare(
                "v1", "v2",
                baseline_embeddings=np.ones((10, 8), dtype=np.float32),
                comparison_embeddings=np.ones(10, dtype=np.float32),
            )

    def test_report_timestamps_and_ids(self) -> None:
        """DriftReport should have correct baseline and comparison IDs."""
        detector = SemanticDriftDetector()
        emb = np.random.randn(8, 16).astype(np.float32)

        report = detector.compare(
            "baseline-A", "comparison-B",
            baseline_embeddings=emb,
            comparison_embeddings=emb,
        )

        assert report.baseline_id == "baseline-A"
        assert report.comparison_id == "comparison-B"
        assert report.timestamp > 0
        assert report.elapsed_ms >= 0

    def test_positive_threshold_validation(self) -> None:
        """threshold_z must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            SemanticDriftDetector(threshold_z=-1.0)

    def test_drift_monotonically_increases_with_noise(self) -> None:
        """Increasing perturbation magnitude should increase drift score."""
        detector = SemanticDriftDetector(threshold_z=2.0)
        rng = np.random.default_rng(99)

        baseline = rng.standard_normal((32, 64)).astype(np.float32)
        drift_scores = []

        for scale in [0.01, 0.1, 0.5, 1.0, 2.0]:
            noise = rng.standard_normal((32, 64)).astype(np.float32) * scale
            report = detector.compare(
                "base", f"perturbed-{scale}",
                baseline_embeddings=baseline,
                comparison_embeddings=baseline + noise,
            )
            drift_scores.append(report.aggregate_drift)

        # Drift should generally increase (allow minor non-monotonicity
        # due to random direction alignment).
        assert drift_scores[-1] > drift_scores[0], (
            f"Expected drift to increase with noise, got {drift_scores}"
        )
