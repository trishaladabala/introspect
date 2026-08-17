"""
server.py — FastAPI application exposing evaluation metrics and WebSocket updates.

Provides REST endpoints for querying evaluation runs, consistency scores,
drift reports, and system metrics. Includes a WebSocket endpoint for
real-time streaming of active evaluation steps.

Also serves the static dashboard files for the monitoring UI.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from introspect.core.consistency import IntrospectiveScorer, ConsistencyReport
from introspect.core.drift import SemanticDriftDetector
from introspect.core.models import (
    MockDiffusionModel,
    MockAutoregressiveModel,
    ModelConfig,
    GenerationResult,
    GenerationStep,
)
from introspect.core.hf_adapter import (
    BD3Adapter,
    HuggingFaceAutoregressiveAdapter,
)
from starlette.concurrency import run_in_threadpool
from introspect.storage.timeseries import MetricsStore
from introspect.tracing.instrumentor import DiffusionInstrumentor
from introspect.tracing.exporters import SQLiteSpanExporter, PrettyConsoleSpanExporter


# ════════════════════════════════════════════════════════════════════════════════
# Application state
# ════════════════════════════════════════════════════════════════════════════════

class AppState:
    """Shared application state for dependency injection."""

    def __init__(self, db_path: str = "introspect_metrics.db") -> None:
        self.store = MetricsStore(db_path)
        self.scorer = IntrospectiveScorer(threshold=0.85)
        self.drift_detector = SemanticDriftDetector(store=self.store, threshold_z=2.0)
        
        self.dlm_adapter: BD3Adapter | None = None
        self.ar_adapter: HuggingFaceAutoregressiveAdapter | None = None

        # Set up tracing with SQLite export.
        sqlite_exporter = SQLiteSpanExporter(db_path)
        console_exporter = PrettyConsoleSpanExporter()
        self.instrumentor = DiffusionInstrumentor(
            service_name="introspect",
            exporters=[sqlite_exporter, console_exporter],
        )

        # WebSocket connection manager.
        self.ws_connections: list[WebSocket] = []

    def close(self) -> None:
        """Cleanup resources."""
        self.instrumentor.shutdown()
        self.store.close()


# Global state instance (initialized in lifespan).
_state: AppState | None = None


def get_state() -> AppState:
    """Get the global application state."""
    assert _state is not None, "Application not initialized"
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — initialize and cleanup resources."""
    global _state
    import os
    _state = AppState()
    
    # Only pre-load real models when explicitly requested (e.g. production).
    # Tests and lightweight starts skip this to avoid blocking on model downloads.
    if os.environ.get("INTROSPECT_LOAD_MODELS", "").strip() == "1":
        _state.dlm_adapter = BD3Adapter(
            model_name="kuleshov-group/bd3lm-owt-block_size4",
            max_new_tokens=32,
            num_steps=16,
        )
        _state.ar_adapter = HuggingFaceAutoregressiveAdapter(
            model_name="distilgpt2",
            max_new_tokens=32,
        )
    
    yield
    _state.close()
    _state = None


# ════════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="introspect",
    description="ML Model Consistency Evaluator & Observability Harness",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the dashboard static files.
DASHBOARD_DIR = Path(__file__).parent.parent.parent.parent / "dashboard"


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> FileResponse:
    """Serve the main dashboard page."""
    index_path = DASHBOARD_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>introspect</h1><p>Dashboard not found. Run from project root.</p>")


@app.get("/dashboard/{filename}")
async def serve_dashboard_file(filename: str) -> FileResponse:
    """Serve dashboard static assets."""
    filepath = DASHBOARD_DIR / filename
    if filepath.exists() and filepath.is_file():
        return FileResponse(str(filepath))
    raise HTTPException(status_code=404, detail=f"File {filename} not found")


# ── Runs ────────────────────────────────────────────────────────────────────

@app.get("/api/runs")
async def list_runs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    """List recent evaluation runs."""
    return get_state().store.get_recent_runs(limit)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Get details for a specific run."""
    summary = get_state().store.get_run_summary(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return summary


@app.get("/api/runs/{run_id}/steps")
async def get_run_steps(run_id: str) -> list[dict[str, Any]]:
    """Get per-step telemetry for a run."""
    return get_state().store.get_step_latencies(run_id)


@app.get("/api/runs/{run_id}/consistency")
async def get_run_consistency(run_id: str) -> list[dict[str, Any]]:
    """Get consistency scores for a run."""
    state = get_state()
    rows = state.store._conn.execute(
        "SELECT * FROM consistency_scores WHERE run_id = ? ORDER BY timestamp",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Drift ───────────────────────────────────────────────────────────────────

@app.get("/api/drift/history")
async def get_drift_history(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Get drift score timeline."""
    return get_state().store.get_drift_history(limit)


# ── Consistency trend ───────────────────────────────────────────────────────

@app.get("/api/consistency/trend")
async def get_consistency_trend(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Get consistency score timeline for trend visualization."""
    return get_state().store.get_consistency_trend(limit)


# ── System metrics ──────────────────────────────────────────────────────────

@app.get("/api/metrics/system")
async def get_system_metrics(
    run_id: str | None = None,
    metric_name: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Get system-level metrics with optional filters."""
    return get_state().store.get_system_metrics(run_id, metric_name, limit)


# ── Storage stats ──────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats() -> dict[str, Any]:
    """Get storage statistics."""
    return get_state().store.get_stats()


# ── Evaluation trigger ─────────────────────────────────────────────────────

@app.post("/api/evaluate")
async def trigger_evaluation(
    vocab_size: int = Query(default=32000, ge=100),
    seq_len: int = Query(default=128, ge=8, le=1024),
    num_steps: int = Query(default=16, ge=2, le=64),
    inconsistency_rate: float = Query(default=0.1, ge=0.0, le=1.0),
    seed: int | None = Query(default=None),
) -> dict[str, Any]:
    """Trigger a new evaluation run with mock models.

    Runs the full pipeline: DLM generation → AR generation →
    consistency scoring → drift detection → results stored.
    """
    state = get_state()

    config = ModelConfig(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_steps=num_steps,
        inconsistency_rate=inconsistency_rate,
        seed=seed,
    )

    # Create run.
    run_id = state.store.create_run(model_config={
        "vocab_size": config.vocab_size,
        "seq_len": config.seq_len,
        "num_steps": config.num_steps,
        "inconsistency_rate": config.inconsistency_rate,
    })
    state.store.update_run_status(run_id, "running")

    try:
        use_real_models = state.dlm_adapter is not None and state.ar_adapter is not None

        if use_real_models:
            # Generate with real DLM (traced) in a separate thread.
            def run_dlm() -> GenerationResult:
                return state.instrumentor.traced_generate(
                    model=state.dlm_adapter,
                    run_id=run_id,
                    max_new_tokens=config.seq_len,
                    num_steps=config.num_steps,
                )

            dlm_result = await run_in_threadpool(run_dlm)

            # Generate with real AR model (reference) in a separate thread.
            def run_ar() -> GenerationResult:
                return state.ar_adapter.generate(max_new_tokens=config.seq_len)

            ar_result = await run_in_threadpool(run_ar)
        else:
            # Fall back to mock models (tests, lightweight starts).
            dlm = MockDiffusionModel(config)
            ar = MockAutoregressiveModel(config)
            dlm_result = state.instrumentor.traced_generate(dlm, run_id=run_id)
            ar_result = ar.generate()

        # Record per-step telemetry.
        for step in dlm_result.steps:
            state.store.record_step(
                run_id=run_id,
                step_index=step.step_index,
                elapsed_ms=step.elapsed_ms,
                tokens_unmasked=step.tokens_unmasked,
                memory_bytes=step.memory_bytes,
                confidence_mean=float(step.confidence.mean()),
                confidence_std=float(step.confidence.std()),
                confidence_min=float(step.confidence.min()),
                confidence_max=float(step.confidence.max()),
                masked_remaining=sum(1 for s in step.states if s.value == "masked"),
            )

            # Broadcast step to WebSocket clients.
            await _broadcast_step(run_id, step)

        # Align sequences for scoring (models may produce different lengths).
        min_len = min(len(dlm_result.token_ids), len(ar_result.token_ids))
        dlm_tokens = dlm_result.token_ids[:min_len]
        ar_tokens = ar_result.token_ids[:min_len]
        dlm_logits_for_score = dlm_result.logits[:min_len]
        ar_logits_for_score = ar_result.logits[:min_len]

        # Consistency scoring (align vocabularies for real models)
        min_vocab = min(dlm_logits_for_score.shape[-1], ar_logits_for_score.shape[-1])
        dlm_logits_aligned = dlm_logits_for_score[..., :min_vocab]
        ar_logits_aligned = ar_logits_for_score[..., :min_vocab]
        
        with state.instrumentor.trace_operation("consistency.score", {"run_id": run_id}):
            consistency_report = state.scorer.score(
                dlm_logits=dlm_logits_aligned,
                ar_logits=ar_logits_aligned,
                dlm_tokens=dlm_tokens,
                ar_tokens=ar_tokens,
            )

        state.store.record_consistency(
            run_id=run_id,
            ics_score=consistency_report.ics_score,
            total_positions=consistency_report.total_positions,
            agreeing_positions=consistency_report.agreeing_positions,
            mean_kl_divergence=consistency_report.mean_kl_divergence,
            max_kl_divergence=consistency_report.max_kl_divergence,
            passed=consistency_report.passed,
            threshold=consistency_report.threshold,
            windowed_scores=consistency_report.windowed_scores,
        )

        # Drift detection (use length-aligned embeddings).
        dlm_emb = dlm_result.embeddings[:min_len]
        ar_emb = ar_result.embeddings[:min_len]
        with state.instrumentor.trace_operation("drift.detect", {"run_id": run_id}):
            drift_report = state.drift_detector.compare(
                baseline_id="reference",
                comparison_id=f"dlm-{run_id}",
                baseline_embeddings=ar_emb,
                comparison_embeddings=dlm_emb,
            )

        state.store.record_drift(
            run_id=run_id,
            aggregate_drift=drift_report.aggregate_drift,
            aggregate_z_score=drift_report.aggregate_z_score,
            passed=drift_report.passed,
            threshold_z=drift_report.threshold_z,
            baseline_id=drift_report.baseline_id,
            comparison_id=drift_report.comparison_id,
            layer_drifts=[
                {
                    "layer_name": ld.layer_name,
                    "mean_cosine_distance": ld.mean_cosine_distance,
                    "max_cosine_distance": ld.max_cosine_distance,
                    "z_score": ld.z_score,
                    "flagged": ld.flagged,
                }
                for ld in drift_report.layer_drifts
            ],
            elapsed_ms=drift_report.elapsed_ms,
        )

        # System metrics.
        state.store.record_system_metric(run_id, "total_elapsed_ms", dlm_result.total_elapsed_ms)
        state.store.record_system_metric(
            run_id,
            "tokens_per_second",
            config.seq_len / (dlm_result.total_elapsed_ms / 1000)
            if dlm_result.total_elapsed_ms > 0 else 0.0,
        )

        # Finalize run.
        overall_passed = consistency_report.passed and drift_report.passed
        state.store.update_run_completion(
            run_id=run_id,
            total_elapsed_ms=dlm_result.total_elapsed_ms,
            total_steps=len(dlm_result.steps),
            ics_score=consistency_report.ics_score,
            drift_score=drift_report.aggregate_drift,
            passed=overall_passed,
        )

        return {
            "run_id": run_id,
            "status": "completed",
            "passed": overall_passed,
            "ics_score": round(consistency_report.ics_score, 4),
            "drift_score": round(drift_report.aggregate_drift, 6),
            "total_steps": len(dlm_result.steps),
            "total_elapsed_ms": round(dlm_result.total_elapsed_ms, 2),
        }

    except Exception as exc:
        state.store.update_run_completion(
            run_id=run_id,
            total_elapsed_ms=0,
            total_steps=0,
            error_message=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc))


# ── WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time evaluation updates."""
    state = get_state()
    await websocket.accept()
    state.ws_connections.append(websocket)

    try:
        while True:
            # Keep connection alive; client can send ping messages.
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        state.ws_connections.remove(websocket)


async def _broadcast_step(run_id: str, step: GenerationStep) -> None:
    """Broadcast a denoising step to all connected WebSocket clients."""
    state = get_state()
    message = json.dumps({
        "type": "step",
        "run_id": run_id,
        "step_index": step.step_index,
        "total_steps": step.total_steps,
        "tokens_unmasked": step.tokens_unmasked,
        "elapsed_ms": round(step.elapsed_ms, 3),
        "confidence_mean": round(float(step.confidence.mean()), 4),
        "masked_remaining": sum(1 for s in step.states if s.value == "masked"),
    })

    disconnected: list[WebSocket] = []
    for ws in state.ws_connections:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        state.ws_connections.remove(ws)
