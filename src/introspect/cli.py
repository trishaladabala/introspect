"""
cli.py — Command-line interface for introspect.

Provides commands for running evaluations, starting the dashboard server,
and viewing results from the terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from introspect.core.consistency import IntrospectiveScorer
from introspect.core.drift import SemanticDriftDetector
from introspect.core.models import MockDiffusionModel, MockAutoregressiveModel, ModelConfig
from introspect.storage.timeseries import MetricsStore
from introspect.tracing.instrumentor import DiffusionInstrumentor
from introspect.tracing.exporters import PrettyConsoleSpanExporter, SQLiteSpanExporter


# ANSI colors for terminal output.
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"


def main() -> None:
    """Entry point for the introspect CLI."""
    parser = argparse.ArgumentParser(
        prog="introspect",
        description="ML Model Consistency Evaluator & Observability Harness",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── evaluate ────────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Run a consistency evaluation")
    eval_parser.add_argument("--vocab-size", type=int, default=32000)
    eval_parser.add_argument("--seq-len", type=int, default=128)
    eval_parser.add_argument("--num-steps", type=int, default=16)
    eval_parser.add_argument("--inconsistency-rate", type=float, default=0.1)
    eval_parser.add_argument("--threshold", type=float, default=0.85)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--db", type=str, default="introspect_metrics.db")
    eval_parser.add_argument("--verbose", "-v", action="store_true")

    # ── serve ───────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the dashboard server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8710)
    serve_parser.add_argument("--db", type=str, default="introspect_metrics.db")

    # ── stats ───────────────────────────────────────────────────────────────
    stats_parser = subparsers.add_parser("stats", help="Show storage statistics")
    stats_parser.add_argument("--db", type=str, default="introspect_metrics.db")

    args = parser.parse_args()

    if args.command == "evaluate":
        _cmd_evaluate(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "stats":
        _cmd_stats(args)
    else:
        parser.print_help()


def _cmd_evaluate(args: argparse.Namespace) -> None:
    """Run a full consistency + drift evaluation."""
    config = ModelConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        inconsistency_rate=args.inconsistency_rate,
        seed=args.seed,
    )

    store = MetricsStore(args.db)
    run_id = store.create_run(model_config={
        "vocab_size": config.vocab_size,
        "seq_len": config.seq_len,
        "num_steps": config.num_steps,
        "inconsistency_rate": config.inconsistency_rate,
    })

    print(f"\n{_BOLD}{_CYAN}╔══════════════════════════════════════════╗{_RESET}")
    print(f"{_BOLD}{_CYAN}║   introspect — Consistency Evaluator     ║{_RESET}")
    print(f"{_BOLD}{_CYAN}╚══════════════════════════════════════════╝{_RESET}\n")
    print(f"  {_DIM}Run ID:{_RESET}             {_BOLD}{run_id}{_RESET}")
    print(f"  {_DIM}Sequence length:{_RESET}     {config.seq_len}")
    print(f"  {_DIM}Denoising steps:{_RESET}     {config.num_steps}")
    print(f"  {_DIM}Inconsistency rate:{_RESET}  {config.inconsistency_rate}")
    print(f"  {_DIM}Threshold:{_RESET}           {args.threshold}")
    print()

    # Set up tracing.
    exporters = [SQLiteSpanExporter(args.db)]
    if args.verbose:
        exporters.append(PrettyConsoleSpanExporter())

    instrumentor = DiffusionInstrumentor(
        service_name="introspect-cli",
        exporters=exporters,
    )

    store.update_run_status(run_id, "running")

    # ── DLM generation ──────────────────────────────────────────────────────
    print(f"  {_YELLOW}▸ Running diffusion model generation...{_RESET}")
    dlm = MockDiffusionModel(config)
    dlm_result = instrumentor.traced_generate(dlm, run_id=run_id)

    for step in dlm_result.steps:
        masked = sum(1 for s in step.states if s.value == "masked")
        bar_len = 30
        progress = 1.0 - (masked / config.seq_len)
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"    Step {step.step_index:>2}/{step.total_steps}  "
            f"[{bar}] {progress:>5.1%}  "
            f"{_DIM}+{step.tokens_unmasked} tokens  "
            f"{step.elapsed_ms:.1f}ms{_RESET}"
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
            masked_remaining=masked,
        )

    print(f"  {_GREEN}✓ DLM generation complete{_RESET} "
          f"({dlm_result.total_elapsed_ms:.1f}ms)\n")

    # ── AR generation ───────────────────────────────────────────────────────
    print(f"  {_YELLOW}▸ Running autoregressive reference...{_RESET}")
    ar = MockAutoregressiveModel(config)
    ar_result = ar.generate()
    print(f"  {_GREEN}✓ AR reference complete{_RESET} "
          f"({ar_result.total_elapsed_ms:.1f}ms)\n")

    # ── Consistency scoring ─────────────────────────────────────────────────
    print(f"  {_YELLOW}▸ Computing introspective consistency...{_RESET}")
    scorer = IntrospectiveScorer(threshold=args.threshold)
    report = scorer.score(
        dlm_logits=dlm_result.logits,
        ar_logits=ar_result.logits,
        dlm_tokens=dlm_result.token_ids,
        ar_tokens=ar_result.token_ids,
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

    status = f"{_GREEN}PASS{_RESET}" if report.passed else f"{_RED}FAIL{_RESET}"
    print(f"  {_BOLD}ICS Score: {report.ics_score:.4f}{_RESET}  [{status}]")
    print(f"  {_DIM}Agreeing: {report.agreeing_positions}/{report.total_positions}  "
          f"Mean KL: {report.mean_kl_divergence:.4f}{_RESET}\n")

    # ── Drift detection ─────────────────────────────────────────────────────
    print(f"  {_YELLOW}▸ Detecting semantic drift...{_RESET}")
    detector = SemanticDriftDetector(threshold_z=2.0)
    drift = detector.compare(
        baseline_id="ar-reference",
        comparison_id=f"dlm-{run_id}",
        baseline_embeddings=ar_result.embeddings,
        comparison_embeddings=dlm_result.embeddings,
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

    drift_status = f"{_GREEN}PASS{_RESET}" if drift.passed else f"{_RED}FAIL{_RESET}"
    print(f"  {_BOLD}Drift Score: {drift.aggregate_drift:.6f}{_RESET}  [{drift_status}]")
    print(f"  {_DIM}Z-Score: {drift.aggregate_z_score:.4f}  "
          f"Threshold: {drift.threshold_z}{_RESET}\n")

    # ── Finalize ────────────────────────────────────────────────────────────
    overall_passed = report.passed and drift.passed
    store.update_run_completion(
        run_id=run_id,
        total_elapsed_ms=dlm_result.total_elapsed_ms,
        total_steps=len(dlm_result.steps),
        ics_score=report.ics_score,
        drift_score=drift.aggregate_drift,
        passed=overall_passed,
    )

    overall_status = f"{_GREEN}PASSED{_RESET}" if overall_passed else f"{_RED}FAILED{_RESET}"
    print(f"  {_BOLD}{'═' * 42}{_RESET}")
    print(f"  {_BOLD}Overall: {overall_status}{_RESET}")
    print(f"  {_DIM}Results stored in {args.db}{_RESET}\n")

    instrumentor.shutdown()
    store.close()

    # Exit with non-zero code on failure (for CI integration).
    if not overall_passed:
        sys.exit(1)


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI dashboard server."""
    import uvicorn
    print(f"\n{_BOLD}{_CYAN}Starting introspect dashboard...{_RESET}")
    print(f"  {_DIM}Database:{_RESET} {args.db}")
    print(f"  {_DIM}URL:{_RESET}      http://{args.host}:{args.port}\n")
    uvicorn.run(
        "introspect.api.server:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def _cmd_stats(args: argparse.Namespace) -> None:
    """Show storage statistics."""
    store = MetricsStore(args.db)
    stats = store.get_stats()

    print(f"\n{_BOLD}{_CYAN}introspect — Storage Statistics{_RESET}\n")
    for table, count in stats.items():
        print(f"  {_DIM}{table}:{_RESET} {count}")
    print()

    store.close()


if __name__ == "__main__":
    main()
