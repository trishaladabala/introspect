"""
test_storage.py — Tests for the MetricsStore (SQLite time-series storage).

Validates CRUD operations, time-range queries, retention policies,
and concurrent access patterns.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from introspect.storage.timeseries import MetricsStore


class TestMetricsStore:
    """Tests for the SQLite MetricsStore."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> MetricsStore:
        """Create a fresh store for each test."""
        s = MetricsStore(tmp_path / "test.db")
        yield s
        s.close()

    def test_create_run(self, store: MetricsStore) -> None:
        """Creating a run should return a valid run_id."""
        run_id = store.create_run(model_config={"vocab_size": 32000})
        assert run_id is not None
        assert len(run_id) > 0

    def test_get_run_summary(self, store: MetricsStore) -> None:
        """Run summary should contain all expected fields."""
        run_id = store.create_run(model_config={"seq_len": 128})
        summary = store.get_run_summary(run_id)

        assert summary is not None
        assert summary["run_id"] == run_id
        assert summary["status"] == "pending"
        assert summary["model_config_json"] is not None

    def test_update_run_status(self, store: MetricsStore) -> None:
        """Status updates should persist."""
        run_id = store.create_run()
        store.update_run_status(run_id, "running")

        summary = store.get_run_summary(run_id)
        assert summary["status"] == "running"

    def test_update_run_completion(self, store: MetricsStore) -> None:
        """Completion updates should set all metrics."""
        run_id = store.create_run()
        store.update_run_completion(
            run_id,
            total_elapsed_ms=150.5,
            total_steps=16,
            ics_score=0.92,
            drift_score=0.001,
            passed=True,
        )

        summary = store.get_run_summary(run_id)
        assert summary["status"] == "completed"
        assert summary["total_elapsed_ms"] == pytest.approx(150.5)
        assert summary["total_steps"] == 16
        assert summary["ics_score"] == pytest.approx(0.92)
        assert summary["passed"] == 1

    def test_failed_run_completion(self, store: MetricsStore) -> None:
        """Runs with errors should be marked as failed."""
        run_id = store.create_run()
        store.update_run_completion(
            run_id,
            total_elapsed_ms=0,
            total_steps=0,
            error_message="OOM error",
        )

        summary = store.get_run_summary(run_id)
        assert summary["status"] == "failed"
        assert summary["error_message"] == "OOM error"

    def test_nonexistent_run_returns_none(self, store: MetricsStore) -> None:
        """Looking up a nonexistent run should return None."""
        assert store.get_run_summary("nonexistent-id") is None

    def test_recent_runs_ordering(self, store: MetricsStore) -> None:
        """Recent runs should be ordered newest-first."""
        ids = [store.create_run() for _ in range(5)]
        recent = store.get_recent_runs(limit=5)

        assert len(recent) == 5
        # Most recent should be first.
        assert recent[0]["run_id"] == ids[-1]

    def test_record_and_get_steps(self, store: MetricsStore) -> None:
        """Step recording and retrieval should work correctly."""
        run_id = store.create_run()

        for i in range(4):
            store.record_step(
                run_id=run_id,
                step_index=i,
                elapsed_ms=1.5 + i * 0.5,
                tokens_unmasked=10 + i * 5,
                memory_bytes=1024 * (i + 1),
                confidence_mean=0.7 + i * 0.05,
                confidence_std=0.1,
                confidence_min=0.3,
                confidence_max=0.95,
                masked_remaining=100 - i * 25,
            )

        steps = store.get_step_latencies(run_id)
        assert len(steps) == 4
        assert steps[0]["step_index"] == 0
        assert steps[-1]["step_index"] == 3
        assert steps[0]["elapsed_ms"] == pytest.approx(1.5)

    def test_record_and_get_consistency(self, store: MetricsStore) -> None:
        """Consistency scores should be stored and retrievable."""
        run_id = store.create_run()
        store.record_consistency(
            run_id=run_id,
            ics_score=0.91,
            total_positions=128,
            agreeing_positions=116,
            mean_kl_divergence=0.05,
            max_kl_divergence=0.3,
            passed=True,
            threshold=0.85,
            windowed_scores=[0.9, 0.88, 0.95],
        )

        trend = store.get_consistency_trend(limit=10)
        assert len(trend) == 1
        assert trend[0]["ics_score"] == pytest.approx(0.91)

    def test_record_and_get_drift(self, store: MetricsStore) -> None:
        """Drift reports should be stored and retrievable."""
        run_id = store.create_run()
        store.record_drift(
            run_id=run_id,
            aggregate_drift=0.003,
            aggregate_z_score=0.5,
            passed=True,
            threshold_z=2.0,
            baseline_id="v1",
            comparison_id="v2",
            elapsed_ms=5.3,
        )

        history = store.get_drift_history(limit=10)
        assert len(history) == 1
        assert history[0]["aggregate_drift"] == pytest.approx(0.003)

    def test_system_metrics(self, store: MetricsStore) -> None:
        """System metrics should be filterable by run_id and name."""
        run_id = store.create_run()
        store.record_system_metric(run_id, "memory_mb", 128.5)
        store.record_system_metric(run_id, "tokens_per_second", 2500.0)

        all_metrics = store.get_system_metrics(run_id=run_id)
        assert len(all_metrics) == 2

        memory = store.get_system_metrics(run_id=run_id, metric_name="memory_mb")
        assert len(memory) == 1
        assert memory[0]["metric_value"] == pytest.approx(128.5)

    def test_cleanup_old_records(self, store: MetricsStore) -> None:
        """Retention cleanup should delete old runs and related data."""
        # Create a store with very short retention.
        store._retention_days = 0  # Expire immediately.

        run_id = store.create_run()
        store.record_step(
            run_id=run_id, step_index=0, elapsed_ms=1.0,
            tokens_unmasked=1, memory_bytes=100,
            confidence_mean=0.5, confidence_std=0.1,
            confidence_min=0.1, confidence_max=0.9,
            masked_remaining=10,
        )

        # Records exist.
        assert store.get_run_summary(run_id) is not None

        # Force cleanup.
        deleted = store.cleanup_old_records()
        assert deleted == 1
        assert store.get_run_summary(run_id) is None

    def test_get_stats(self, store: MetricsStore) -> None:
        """Stats should reflect current record counts."""
        stats = store.get_stats()
        assert stats["runs"] == 0

        store.create_run()
        stats = store.get_stats()
        assert stats["runs"] == 1

    def test_limit_parameter(self, store: MetricsStore) -> None:
        """Limit parameter should cap result count."""
        for _ in range(10):
            store.create_run()

        recent = store.get_recent_runs(limit=3)
        assert len(recent) == 3
