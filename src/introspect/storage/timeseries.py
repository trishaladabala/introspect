"""
timeseries.py — SQLite-backed time-series metrics storage.

Provides a structured store for evaluation runs, per-step telemetry,
consistency scores, drift reports, and system metrics. Optimized for
time-range queries with proper indexing and WAL journal mode.

The schema supports the full lifecycle of an evaluation:
  run → steps → consistency scores → drift reports → system metrics
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator

import numpy as np


@dataclass
class RunRecord:
    """A single evaluation run."""
    run_id: str
    status: str  # "pending", "running", "completed", "failed"
    model_config_json: str
    started_at: float
    completed_at: float | None = None
    total_elapsed_ms: float | None = None
    total_steps: int | None = None
    ics_score: float | None = None
    drift_score: float | None = None
    passed: bool | None = None
    error_message: str | None = None


@dataclass
class StepRecord:
    """Per-step telemetry for a run."""
    run_id: str
    step_index: int
    elapsed_ms: float
    tokens_unmasked: int
    memory_bytes: int
    confidence_mean: float
    confidence_std: float
    confidence_min: float
    confidence_max: float
    masked_remaining: int
    timestamp: float


@dataclass
class SystemMetricRecord:
    """System-level metrics snapshot."""
    run_id: str
    timestamp: float
    metric_name: str
    metric_value: float


class MetricsStore:
    """SQLite-backed storage for evaluation telemetry.

    Thread-safe via WAL journal mode. Supports time-range queries,
    aggregation, and configurable retention policies.

    Usage:
        store = MetricsStore("metrics.db")
        run_id = store.create_run(model_config={...})
        store.record_step(run_id, step_index=0, ...)
        store.update_run_completion(run_id, ics_score=0.92, ...)
        summary = store.get_run_summary(run_id)
    """

    def __init__(
        self,
        db_path: str | Path,
        retention_days: int = 30,
    ) -> None:
        """Initialize the metrics store.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:"
                for in-memory databases (useful for testing).
            retention_days: Number of days to retain records before
                automatic cleanup.
        """
        self._db_path = str(db_path)
        self._retention_days = retention_days
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        """Create all tables and indices."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                model_config_json TEXT,
                started_at REAL NOT NULL,
                completed_at REAL,
                total_elapsed_ms REAL,
                total_steps INTEGER,
                ics_score REAL,
                drift_score REAL,
                passed INTEGER,
                error_message TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_runs_started_at
                ON runs(started_at);
            CREATE INDEX IF NOT EXISTS idx_runs_status
                ON runs(status);

            CREATE TABLE IF NOT EXISTS steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                step_index INTEGER NOT NULL,
                elapsed_ms REAL NOT NULL,
                tokens_unmasked INTEGER NOT NULL,
                memory_bytes INTEGER NOT NULL,
                confidence_mean REAL NOT NULL,
                confidence_std REAL NOT NULL,
                confidence_min REAL NOT NULL,
                confidence_max REAL NOT NULL,
                masked_remaining INTEGER NOT NULL,
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_steps_run_id
                ON steps(run_id);

            CREATE TABLE IF NOT EXISTS consistency_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                ics_score REAL NOT NULL,
                total_positions INTEGER NOT NULL,
                agreeing_positions INTEGER NOT NULL,
                mean_kl_divergence REAL NOT NULL,
                max_kl_divergence REAL NOT NULL,
                passed INTEGER NOT NULL,
                threshold REAL NOT NULL,
                windowed_scores_json TEXT,
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_consistency_run_id
                ON consistency_scores(run_id);

            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                aggregate_drift REAL NOT NULL,
                aggregate_z_score REAL NOT NULL,
                passed INTEGER NOT NULL,
                threshold_z REAL NOT NULL,
                baseline_id TEXT NOT NULL,
                comparison_id TEXT NOT NULL,
                layer_drifts_json TEXT,
                elapsed_ms REAL NOT NULL,
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_drift_run_id
                ON drift_reports(run_id);

            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                timestamp REAL NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sysmetrics_run_id
                ON system_metrics(run_id);
            CREATE INDEX IF NOT EXISTS idx_sysmetrics_timestamp
                ON system_metrics(timestamp);
        """)
        self._conn.commit()

    # ── Run management ──────────────────────────────────────────────────────

    def create_run(self, model_config: dict[str, Any] | None = None) -> str:
        """Create a new evaluation run.

        Args:
            model_config: Optional model configuration dictionary.

        Returns:
            The new run_id.
        """
        run_id = str(uuid.uuid4())[:12]
        config_json = json.dumps(model_config, default=str) if model_config else "{}"

        self._conn.execute(
            "INSERT INTO runs (run_id, status, model_config_json, started_at) "
            "VALUES (?, 'pending', ?, ?)",
            (run_id, config_json, time.time()),
        )
        self._conn.commit()
        return run_id

    def update_run_status(self, run_id: str, status: str) -> None:
        """Update a run's status."""
        self._conn.execute(
            "UPDATE runs SET status = ? WHERE run_id = ?",
            (status, run_id),
        )
        self._conn.commit()

    def update_run_completion(
        self,
        run_id: str,
        total_elapsed_ms: float,
        total_steps: int,
        ics_score: float | None = None,
        drift_score: float | None = None,
        passed: bool | None = None,
        error_message: str | None = None,
    ) -> None:
        """Mark a run as completed with final metrics."""
        status = "completed" if error_message is None else "failed"
        self._conn.execute(
            """UPDATE runs SET
                status = ?, completed_at = ?, total_elapsed_ms = ?,
                total_steps = ?, ics_score = ?, drift_score = ?,
                passed = ?, error_message = ?
               WHERE run_id = ?""",
            (
                status,
                time.time(),
                total_elapsed_ms,
                total_steps,
                ics_score,
                drift_score,
                1 if passed else (0 if passed is not None else None),
                error_message,
                run_id,
            ),
        )
        self._conn.commit()

    def get_run_summary(self, run_id: str) -> dict[str, Any] | None:
        """Get a run's summary as a dictionary."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recent runs."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Step recording ──────────────────────────────────────────────────────

    def record_step(
        self,
        run_id: str,
        step_index: int,
        elapsed_ms: float,
        tokens_unmasked: int,
        memory_bytes: int,
        confidence_mean: float,
        confidence_std: float,
        confidence_min: float,
        confidence_max: float,
        masked_remaining: int,
    ) -> None:
        """Record telemetry for a single denoising step."""
        self._conn.execute(
            """INSERT INTO steps
               (run_id, step_index, elapsed_ms, tokens_unmasked, memory_bytes,
                confidence_mean, confidence_std, confidence_min, confidence_max,
                masked_remaining, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                step_index,
                elapsed_ms,
                tokens_unmasked,
                memory_bytes,
                confidence_mean,
                confidence_std,
                confidence_min,
                confidence_max,
                masked_remaining,
                time.time(),
            ),
        )
        self._conn.commit()

    def get_step_latencies(self, run_id: str) -> list[dict[str, Any]]:
        """Get step-by-step latency data for a run."""
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_index",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Consistency recording ───────────────────────────────────────────────

    def record_consistency(
        self,
        run_id: str,
        ics_score: float,
        total_positions: int,
        agreeing_positions: int,
        mean_kl_divergence: float,
        max_kl_divergence: float,
        passed: bool,
        threshold: float,
        windowed_scores: list[float] | None = None,
    ) -> None:
        """Record a consistency evaluation result."""
        self._conn.execute(
            """INSERT INTO consistency_scores
               (run_id, ics_score, total_positions, agreeing_positions,
                mean_kl_divergence, max_kl_divergence, passed, threshold,
                windowed_scores_json, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                ics_score,
                total_positions,
                agreeing_positions,
                mean_kl_divergence,
                max_kl_divergence,
                1 if passed else 0,
                threshold,
                json.dumps(windowed_scores) if windowed_scores else "[]",
                time.time(),
            ),
        )
        self._conn.commit()

    def get_consistency_trend(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent consistency scores for trend analysis."""
        rows = self._conn.execute(
            """SELECT c.*, r.run_id
               FROM consistency_scores c
               JOIN runs r ON c.run_id = r.run_id
               ORDER BY c.timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Drift recording ─────────────────────────────────────────────────────

    def record_drift(
        self,
        run_id: str,
        aggregate_drift: float,
        aggregate_z_score: float,
        passed: bool,
        threshold_z: float,
        baseline_id: str,
        comparison_id: str,
        layer_drifts: list[dict[str, Any]] | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        """Record a drift detection result."""
        self._conn.execute(
            """INSERT INTO drift_reports
               (run_id, aggregate_drift, aggregate_z_score, passed,
                threshold_z, baseline_id, comparison_id, layer_drifts_json,
                elapsed_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                aggregate_drift,
                aggregate_z_score,
                1 if passed else 0,
                threshold_z,
                baseline_id,
                comparison_id,
                json.dumps(layer_drifts, default=str) if layer_drifts else "[]",
                elapsed_ms,
                time.time(),
            ),
        )
        self._conn.commit()

    def get_drift_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent drift reports."""
        rows = self._conn.execute(
            "SELECT * FROM drift_reports ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── System metrics ──────────────────────────────────────────────────────

    def record_system_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Record a system-level metric (memory, CPU, etc.)."""
        self._conn.execute(
            "INSERT INTO system_metrics (run_id, timestamp, metric_name, metric_value) "
            "VALUES (?, ?, ?, ?)",
            (run_id, time.time(), metric_name, metric_value),
        )
        self._conn.commit()

    def get_system_metrics(
        self,
        run_id: str | None = None,
        metric_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query system metrics with optional filters."""
        query = "SELECT * FROM system_metrics WHERE 1=1"
        params: list[Any] = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Maintenance ─────────────────────────────────────────────────────────

    def cleanup_old_records(self) -> int:
        """Delete records older than the retention period.

        Returns:
            Number of runs deleted.
        """
        cutoff = time.time() - (self._retention_days * 86400)

        # Get runs to delete.
        old_runs = self._conn.execute(
            "SELECT run_id FROM runs WHERE started_at < ?", (cutoff,)
        ).fetchall()

        if not old_runs:
            return 0

        run_ids = [r["run_id"] for r in old_runs]
        placeholders = ",".join("?" for _ in run_ids)

        # Cascade delete related records.
        for table in ("steps", "consistency_scores", "drift_reports", "system_metrics"):
            self._conn.execute(
                f"DELETE FROM {table} WHERE run_id IN ({placeholders})",
                run_ids,
            )

        self._conn.execute(
            f"DELETE FROM runs WHERE run_id IN ({placeholders})",
            run_ids,
        )
        self._conn.commit()
        return len(run_ids)

    def get_stats(self) -> dict[str, int]:
        """Get storage statistics."""
        tables = ["runs", "steps", "consistency_scores", "drift_reports", "system_metrics"]
        stats: dict[str, int] = {}
        for table in tables:
            row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            stats[table] = row["cnt"] if row else 0
        return stats

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
