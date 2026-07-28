"""
instrumentor.py — OpenTelemetry instrumentation for diffusion model inference.

Wraps model inference to emit OpenTelemetry spans for each diffusion
denoising step (T₁, T₂, ... Tₙ). Each span captures per-step telemetry:
tokens unmasked, confidence distribution, memory usage, and step latency.

This provides the kind of granular, step-by-step latency insights that
SRE teams need to debug inference performance in production.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SimpleSpanProcessor
from opentelemetry.trace import StatusCode

import numpy as np

from introspect.core.models import (
    GenerationResult,
    GenerationStep,
    ModelAdapter,
    ModelConfig,
)


class DiffusionInstrumentor:
    """Instruments model inference with OpenTelemetry distributed tracing.

    Wraps any ModelAdapter to automatically emit spans for:
    - The complete generation run (root span)
    - Each individual denoising step (child spans)
    - Post-processing operations (consistency scoring, drift detection)

    Usage:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        instrumentor = DiffusionInstrumentor(
            service_name="introspect-eval",
            exporters=[ConsoleSpanExporter()],
        )

        result = instrumentor.traced_generate(model, prompt_ids=None)
    """

    def __init__(
        self,
        service_name: str = "introspect",
        exporters: list[SpanExporter] | None = None,
    ) -> None:
        """Initialize the instrumentor with OpenTelemetry configuration.

        Args:
            service_name: Name of the service for span attribution.
            exporters: List of SpanExporters. If None, spans are created
                but not exported (useful for testing).
        """
        self._provider = TracerProvider()

        if exporters:
            for exporter in exporters:
                self._provider.add_span_processor(SimpleSpanProcessor(exporter))

        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(
            instrumenting_module_name="introspect.tracing",
            tracer_provider=self._provider,
        )
        self._service_name = service_name

    @property
    def tracer(self) -> trace.Tracer:
        """The underlying OpenTelemetry tracer."""
        return self._tracer

    def traced_generate(
        self,
        model: ModelAdapter,
        prompt_ids: np.ndarray | None = None,
        run_id: str | None = None,
    ) -> GenerationResult:
        """Execute model generation with full tracing instrumentation.

        Creates a root span for the entire generation, with child spans
        for each denoising step. All spans include detailed attributes.

        Args:
            model: The model adapter to generate with.
            prompt_ids: Optional prompt token IDs.
            run_id: Optional run identifier for correlation.

        Returns:
            The GenerationResult from the model, unchanged.
        """
        span_attrs: dict[str, Any] = {
            "introspect.service": self._service_name,
            "model.vocab_size": model.config.vocab_size,
            "model.embed_dim": model.config.embed_dim,
            "model.seq_len": model.config.seq_len,
            "model.num_steps": model.config.num_steps,
        }
        if run_id:
            span_attrs["introspect.run_id"] = run_id

        with self._tracer.start_as_current_span(
            "generation.full",
            attributes=span_attrs,
        ) as root_span:
            try:
                result = model.generate(prompt_ids)

                # Annotate root span with aggregate metrics.
                root_span.set_attribute(
                    "generation.total_elapsed_ms",
                    round(result.total_elapsed_ms, 3),
                )
                root_span.set_attribute(
                    "generation.total_steps",
                    len(result.steps),
                )
                root_span.set_attribute(
                    "generation.tokens_per_second",
                    round(
                        model.config.seq_len / (result.total_elapsed_ms / 1000)
                        if result.total_elapsed_ms > 0 else 0.0,
                        2,
                    ),
                )

                # Create child spans for each denoising step.
                for step in result.steps:
                    self._record_step_span(step)

                root_span.set_status(StatusCode.OK)
                return result

            except Exception as exc:
                root_span.set_status(StatusCode.ERROR, str(exc))
                root_span.record_exception(exc)
                raise

    def _record_step_span(self, step: GenerationStep) -> None:
        """Create a child span for a single denoising step."""
        with self._tracer.start_as_current_span(
            f"generation.step.{step.step_index}",
            attributes={
                "step.index": step.step_index,
                "step.total_steps": step.total_steps,
                "step.tokens_unmasked": step.tokens_unmasked,
                "step.elapsed_ms": round(step.elapsed_ms, 3),
                "step.memory_bytes": step.memory_bytes,
                "step.confidence_mean": round(float(step.confidence.mean()), 4),
                "step.confidence_std": round(float(step.confidence.std()), 4),
                "step.confidence_min": round(float(step.confidence.min()), 4),
                "step.confidence_max": round(float(step.confidence.max()), 4),
                "step.masked_remaining": sum(
                    1 for s in step.states if s.value == "masked"
                ),
            },
        ):
            pass  # Span auto-closes with context manager.

    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[trace.Span, None, None]:
        """Context manager for tracing arbitrary operations.

        Useful for instrumenting consistency scoring, drift detection,
        and other post-processing steps.

        Args:
            operation_name: Name for the span (e.g., "consistency.score").
            attributes: Optional span attributes.
        """
        with self._tracer.start_as_current_span(
            operation_name,
            attributes=attributes or {},
        ) as span:
            try:
                yield span
                span.set_status(StatusCode.OK)
            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
                raise

    def shutdown(self) -> None:
        """Flush and shut down the tracer provider."""
        self._provider.shutdown()
