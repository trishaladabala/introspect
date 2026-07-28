"""
test_api.py — Integration tests for the FastAPI server.

Uses FastAPI's TestClient for synchronous endpoint testing and
validates all REST API responses and WebSocket connections.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from introspect.api.server import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client with proper lifespan handling."""
    with TestClient(app) as c:
        yield c


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for REST API endpoints."""

    def test_root_returns_html(self, client: TestClient) -> None:
        """Root path should serve the dashboard or a fallback."""
        res = client.get("/")
        assert res.status_code == 200

    def test_list_runs_empty(self, client: TestClient) -> None:
        """Runs list should start empty."""
        res = client.get("/api/runs")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

    def test_get_nonexistent_run_404(self, client: TestClient) -> None:
        """Nonexistent run should return 404."""
        res = client.get("/api/runs/nonexistent-id")
        assert res.status_code == 404

    def test_trigger_evaluation(self, client: TestClient) -> None:
        """Triggering an evaluation should return results."""
        res = client.post(
            "/api/evaluate?seq_len=16&num_steps=4&inconsistency_rate=0.1&vocab_size=500"
        )
        assert res.status_code == 200

        data = res.json()
        assert "run_id" in data
        assert "ics_score" in data
        assert "drift_score" in data
        assert "passed" in data
        assert data["status"] == "completed"

    def test_evaluation_populates_runs(self, client: TestClient) -> None:
        """After evaluation, runs list should contain the new run."""
        # Trigger an evaluation.
        eval_res = client.post(
            "/api/evaluate?seq_len=16&num_steps=4&vocab_size=500"
        )
        run_id = eval_res.json()["run_id"]

        # Check runs list.
        runs = client.get("/api/runs").json()
        run_ids = [r["run_id"] for r in runs]
        assert run_id in run_ids

    def test_get_run_steps(self, client: TestClient) -> None:
        """Steps endpoint should return per-step telemetry."""
        eval_res = client.post(
            "/api/evaluate?seq_len=16&num_steps=4&vocab_size=500"
        )
        run_id = eval_res.json()["run_id"]

        steps = client.get(f"/api/runs/{run_id}/steps").json()
        assert isinstance(steps, list)
        assert len(steps) > 0
        assert "step_index" in steps[0]
        assert "elapsed_ms" in steps[0]

    def test_consistency_trend(self, client: TestClient) -> None:
        """Consistency trend should contain entries after evaluation."""
        client.post("/api/evaluate?seq_len=16&num_steps=4&vocab_size=500")

        trend = client.get("/api/consistency/trend").json()
        assert isinstance(trend, list)
        assert len(trend) > 0

    def test_drift_history(self, client: TestClient) -> None:
        """Drift history should contain entries after evaluation."""
        client.post("/api/evaluate?seq_len=16&num_steps=4&vocab_size=500")

        history = client.get("/api/drift/history").json()
        assert isinstance(history, list)
        assert len(history) > 0

    def test_system_metrics(self, client: TestClient) -> None:
        """System metrics should be queryable."""
        client.post("/api/evaluate?seq_len=16&num_steps=4&vocab_size=500")

        metrics = client.get("/api/metrics/system").json()
        assert isinstance(metrics, list)

    def test_storage_stats(self, client: TestClient) -> None:
        """Stats endpoint should return table counts."""
        stats = client.get("/api/stats").json()
        assert "runs" in stats
        assert "steps" in stats

    def test_run_consistency_endpoint(self, client: TestClient) -> None:
        """Run-specific consistency scores should be retrievable."""
        eval_res = client.post(
            "/api/evaluate?seq_len=16&num_steps=4&vocab_size=500"
        )
        run_id = eval_res.json()["run_id"]

        consistency = client.get(f"/api/runs/{run_id}/consistency").json()
        assert isinstance(consistency, list)
        assert len(consistency) > 0
        assert "ics_score" in consistency[0]

    def test_websocket_connection(self, client: TestClient) -> None:
        """WebSocket should accept connections and respond to pings."""
        with client.websocket_connect("/ws/live") as ws:
            ws.send_text("ping")
            data = ws.receive_json()
            assert data["type"] == "pong"
