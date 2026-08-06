#!/usr/bin/env python3
"""
e2e_real_model_eval.py — End-to-end evaluation with real HuggingFace models.

Proves the introspect observability platform works with actual model outputs.
Uses distilgpt2 (a lightweight 82M parameter model) to run the complete
pipeline: generation → consistency scoring → drift detection → storage →
tracing.

This script produces verifiable evidence of real model integration.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import tracemalloc

# Add src to path for direct execution.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from introspect.core.hf_adapter import (
    HuggingFaceAutoregressiveAdapter,
    BD3Adapter,
)
from introspect.core.consistency import IntrospectiveScorer
from introspect.core.drift import SemanticDriftDetector
from introspect.storage.timeseries import MetricsStore
from introspect.tracing.instrumentor import DiffusionInstrumentor
from introspect.tracing.exporters import SQLiteSpanExporter, PrettyConsoleSpanExporter


# ANSI colors.
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def main() -> None:
    """Run the complete end-to-end evaluation with real models."""

    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   introspect — Real Model End-to-End Evaluation      ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}\n")

    # Configuration.
    BD3_MODEL_NAME = "kuleshov-group/bd3lm-owt-block_size4"
    AR_MODEL_NAME = "distilgpt2"
    PROMPT = "The meaning of life is"
    MAX_NEW_TOKENS = 32
    NUM_STEPS = 8
    DB_PATH = "introspect_metrics_real.db"

    # Start memory tracking.
    tracemalloc.start()

    # ── Initialize storage & tracing ──────────────────────────────────────────
    store = MetricsStore(DB_PATH)
    sqlite_exporter = SQLiteSpanExporter(DB_PATH)
    console_exporter = PrettyConsoleSpanExporter()
    instrumentor = DiffusionInstrumentor(
        service_name="introspect-real-eval",
        exporters=[sqlite_exporter, console_exporter],
    )

    run_id = store.create_run(model_config={
        "model_name": BD3_MODEL_NAME,
        "prompt": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_steps": NUM_STEPS,
        "type": "real_model_e2e",
    })
    store.update_run_status(run_id, "running")

    print(f"  {DIM}Run ID:{RESET}              {BOLD}{run_id}{RESET}")
    print(f"  {DIM}Model:{RESET}               {BOLD}{BD3_MODEL_NAME}{RESET}")
    print(f"  {DIM}Prompt:{RESET}              \"{PROMPT}\"")
    print(f"  {DIM}Max new tokens:{RESET}      {MAX_NEW_TOKENS}")
    print(f"  {DIM}Denoising steps:{RESET}     {NUM_STEPS}")
    print(f"  {DIM}Database:{RESET}            {DB_PATH}")
    print()

    try:
        # ── Load models ──────────────────────────────────────────────────────
        print(f"  {YELLOW}▸ Loading {BD3_MODEL_NAME} for BD3...{RESET}")
        load_start = time.perf_counter()
        dlm = BD3Adapter(
            model_name=BD3_MODEL_NAME,
            max_new_tokens=MAX_NEW_TOKENS,
            num_steps=NUM_STEPS,
            device="cpu",
            seed=42,
        )
        load_elapsed = (time.perf_counter() - load_start) * 1000
        print(f"  {GREEN}✓ BD3 model loaded{RESET} ({load_elapsed:.0f}ms)\n")

        print(f"  {YELLOW}▸ Loading {AR_MODEL_NAME} for autoregressive anchor...{RESET}")
        load_start = time.perf_counter()
        ar = HuggingFaceAutoregressiveAdapter(
            model_name=AR_MODEL_NAME,
            max_new_tokens=MAX_NEW_TOKENS,
            device="cpu",
            seed=42,
        )
        load_elapsed = (time.perf_counter() - load_start) * 1000
        print(f"  {GREEN}✓ AR anchor model loaded{RESET} ({load_elapsed:.0f}ms)\n")

        # ── DLM generation (traced) ──────────────────────────────────────────
        print(f"  {YELLOW}▸ Running BD3 generation (traced)...{RESET}")
        dlm_result = instrumentor.traced_generate(dlm, run_id=run_id)

        # Decode and display generated text.
        dlm_text = dlm.tokenizer.decode(dlm_result.token_ids, skip_special_tokens=True)
        print(f"  {GREEN}✓ DLM generation complete{RESET} ({dlm_result.total_elapsed_ms:.1f}ms)")
        print(f"  {DIM}DLM output:{RESET} \"{dlm_text[:200]}\"")
        print()

        # Record per-step telemetry.
        for step in dlm_result.steps:
            masked = sum(1 for s in step.states if s.value == "masked")
            bar_len = 30
            progress = 1.0 - (masked / len(step.states)) if len(step.states) > 0 else 1.0
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"    Step {step.step_index:>2}/{step.total_steps}  "
                f"[{bar}] {progress:>5.1%}  "
                f"{DIM}+{step.tokens_unmasked} tokens  "
                f"{step.elapsed_ms:.1f}ms{RESET}"
            )

            store.record_step(
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
        print()

        # ── AR generation ────────────────────────────────────────────────────
        print(f"  {YELLOW}▸ Running AR reference generation...{RESET}")

        # Use the same prompt tokens.
        prompt_token_ids = dlm.tokenizer.encode(PROMPT)
        ar_result = ar.generate(prompt_ids=np.array(prompt_token_ids, dtype=np.int64))

        ar_text = ar.tokenizer.decode(ar_result.token_ids, skip_special_tokens=True)
        print(f"  {GREEN}✓ AR reference complete{RESET} ({ar_result.total_elapsed_ms:.1f}ms)")
        print(f"  {DIM}AR output:{RESET} \"{ar_text[:200]}\"")
        print()

        # ── Consistency scoring ──────────────────────────────────────────────
        print(f"  {YELLOW}▸ Computing introspective consistency...{RESET}")

        # Align sequences for comparison.
        # Both models may produce different-length sequences.
        min_len = min(len(dlm_result.token_ids), len(ar_result.token_ids))
        dlm_ids = dlm_result.token_ids[:min_len]
        ar_ids = ar_result.token_ids[:min_len]
        dlm_logits = dlm_result.logits[:min_len]
        ar_logits = ar_result.logits[:min_len]

        scorer = IntrospectiveScorer(threshold=0.85)
        with instrumentor.trace_operation("consistency.score", {"run_id": run_id}):
            min_vocab = min(dlm_logits.shape[-1], ar_logits.shape[-1])
            dlm_logits = dlm_logits[..., :min_vocab]
            ar_logits = ar_logits[..., :min_vocab]
            
            report = scorer.score(
                dlm_logits=dlm_logits,
                ar_logits=ar_logits,
                dlm_tokens=dlm_ids,
                ar_tokens=ar_ids,
            )

        store.record_consistency(
            run_id=run_id,
            ics_score=report.ics_score,
            total_positions=report.total_positions,
            agreeing_positions=report.agreeing_positions,
            mean_kl_divergence=report.mean_kl_divergence,
            max_kl_divergence=report.max_kl_divergence,
            passed=report.passed,
            threshold=report.threshold,
            windowed_scores=report.windowed_scores,
        )

        status = f"{GREEN}PASS{RESET}" if report.passed else f"{RED}FAIL{RESET}"
        print(f"  {BOLD}ICS Score: {report.ics_score:.4f}{RESET}  [{status}]")
        print(f"  {DIM}Agreeing: {report.agreeing_positions}/{report.total_positions}  "
              f"Mean KL: {report.mean_kl_divergence:.4f}  "
              f"Max KL: {report.max_kl_divergence:.4f}{RESET}\n")

        # ── Drift detection ──────────────────────────────────────────────────
        print(f"  {YELLOW}▸ Detecting semantic drift...{RESET}")

        # Align embeddings.
        dlm_emb = dlm_result.embeddings[:min_len]
        ar_emb = ar_result.embeddings[:min_len]

        detector = SemanticDriftDetector(threshold_z=2.0)
        with instrumentor.trace_operation("drift.detect", {"run_id": run_id}):
            detector.set_baseline("ar-reference", ar_emb)
            drift = detector.compare(
                baseline_id="ar-reference",
                comparison_id=f"dlm-{run_id}",
                comparison_embeddings=dlm_emb,
            )

        store.record_drift(
            run_id=run_id,
            aggregate_drift=drift.aggregate_drift,
            aggregate_z_score=drift.aggregate_z_score,
            passed=drift.passed,
            threshold_z=drift.threshold_z,
            baseline_id=drift.baseline_id,
            comparison_id=drift.comparison_id,
            elapsed_ms=drift.elapsed_ms,
        )

        drift_status = f"{GREEN}PASS{RESET}" if drift.passed else f"{RED}FAIL{RESET}"
        print(f"  {BOLD}Drift Score: {drift.aggregate_drift:.6f}{RESET}  [{drift_status}]")
        print(f"  {DIM}Z-Score: {drift.aggregate_z_score:.4f}  "
              f"Threshold: {drift.threshold_z}{RESET}\n")

        # ── Memory & system metrics ──────────────────────────────────────────
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        store.record_system_metric(run_id, "total_elapsed_ms", dlm_result.total_elapsed_ms)
        store.record_system_metric(run_id, "memory_current_mb", current / 1024 / 1024)
        store.record_system_metric(run_id, "memory_peak_mb", peak / 1024 / 1024)
        store.record_system_metric(
            run_id, "tokens_per_second",
            len(dlm_result.token_ids) / (dlm_result.total_elapsed_ms / 1000)
            if dlm_result.total_elapsed_ms > 0 else 0.0,
        )

        # ── Finalize ─────────────────────────────────────────────────────────
        overall_passed = report.passed and drift.passed
        store.update_run_completion(
            run_id=run_id,
            total_elapsed_ms=dlm_result.total_elapsed_ms,
            total_steps=len(dlm_result.steps),
            ics_score=report.ics_score,
            drift_score=drift.aggregate_drift,
            passed=overall_passed,
        )

        overall_status = f"{GREEN}PASSED{RESET}" if overall_passed else f"{RED}FAILED{RESET}"
        print(f"  {BOLD}{'═' * 54}{RESET}")
        print(f"  {BOLD}Overall: {overall_status}{RESET}")
        print()

        # ── Detailed results summary ─────────────────────────────────────────
        print(f"  {BOLD}{CYAN}── Experimental Results ──{RESET}")
        print(f"  {DIM}Prompt:{RESET}              \"{PROMPT}\"")
        print(f"  {DIM}Model:{RESET}               {BD3_MODEL_NAME}")
        print(f"  {DIM}DLM output:{RESET}          \"{dlm_text[:120]}\"")
        print(f"  {DIM}AR output:{RESET}           \"{ar_text[:120]}\"")
        print(f"  {DIM}ICS Score:{RESET}           {report.ics_score:.4f}")
        print(f"  {DIM}KL Divergence (mean):{RESET} {report.mean_kl_divergence:.4f}")
        print(f"  {DIM}KL Divergence (max):{RESET}  {report.max_kl_divergence:.4f}")
        print(f"  {DIM}Drift Score:{RESET}         {drift.aggregate_drift:.6f}")
        print(f"  {DIM}DLM Latency:{RESET}         {dlm_result.total_elapsed_ms:.1f}ms")
        print(f"  {DIM}AR Latency:{RESET}          {ar_result.total_elapsed_ms:.1f}ms")
        print(f"  {DIM}Memory (current):{RESET}    {current / 1024 / 1024:.1f} MB")
        print(f"  {DIM}Memory (peak):{RESET}       {peak / 1024 / 1024:.1f} MB")
        print(f"  {DIM}Denoising steps:{RESET}     {len(dlm_result.steps)}")
        print(f"  {DIM}Results stored in:{RESET}   {DB_PATH}")
        print()

        # ── Verify stored data ───────────────────────────────────────────────
        print(f"  {BOLD}{CYAN}── Storage Verification ──{RESET}")
        summary = store.get_run_summary(run_id)
        steps_data = store.get_step_latencies(run_id)
        consistency_data = store.get_consistency_trend(limit=1)
        drift_data = store.get_drift_history(limit=1)
        sys_metrics = store.get_system_metrics(run_id=run_id)
        stats = store.get_stats()

        print(f"  {DIM}Run summary:{RESET}        {'✓ stored' if summary else '✗ missing'}")
        print(f"  {DIM}Steps recorded:{RESET}     {len(steps_data)}")
        print(f"  {DIM}Consistency scores:{RESET} {len(consistency_data)}")
        print(f"  {DIM}Drift reports:{RESET}      {len(drift_data)}")
        print(f"  {DIM}System metrics:{RESET}     {len(sys_metrics)}")
        print(f"  {DIM}Total DB stats:{RESET}     {json.dumps(stats)}")
        print()

    except Exception as exc:
        store.update_run_completion(
            run_id=run_id,
            total_elapsed_ms=0,
            total_steps=0,
            error_message=str(exc),
        )
        print(f"\n  {RED}✗ Evaluation failed: {exc}{RESET}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        instrumentor.shutdown()
        store.close()


if __name__ == "__main__":
    main()
