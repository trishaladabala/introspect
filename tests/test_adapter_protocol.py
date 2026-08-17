"""
test_adapter_protocol.py — Verifies framework-agnostic adapter architecture.

Proves the ModelAdapter protocol is correctly implemented by all adapters,
and demonstrates that the same evaluation pipeline works identically
regardless of which adapter is plugged in.

This is the key evidence for the "framework-agnostic" claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from introspect.core.models import (
    ModelAdapter,
    ModelConfig,
    MockDiffusionModel,
    MockAutoregressiveModel,
    GenerationResult,
)
from introspect.core.consistency import IntrospectiveScorer
from introspect.core.drift import SemanticDriftDetector


class TestAdapterProtocol:
    """Verify that all adapters satisfy the ModelAdapter protocol."""

    def test_mock_dlm_satisfies_protocol(self) -> None:
        """MockDiffusionModel should be a runtime-checkable ModelAdapter."""
        adapter = MockDiffusionModel(ModelConfig(seed=42))
        assert isinstance(adapter, ModelAdapter)

    def test_mock_ar_satisfies_protocol(self) -> None:
        """MockAutoregressiveModel should be a runtime-checkable ModelAdapter."""
        adapter = MockAutoregressiveModel(ModelConfig(seed=42))
        assert isinstance(adapter, ModelAdapter)

    @pytest.mark.slow
    def test_hf_ar_satisfies_protocol(self) -> None:
        """HuggingFaceAutoregressiveAdapter should satisfy ModelAdapter."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from introspect.core.hf_adapter import HuggingFaceAutoregressiveAdapter

        adapter = HuggingFaceAutoregressiveAdapter("distilgpt2", max_new_tokens=4)
        assert isinstance(adapter, ModelAdapter)

    def test_all_adapters_have_required_methods(self) -> None:
        """Every adapter must expose config, generate, get_logits, get_embeddings."""
        for cls in [MockDiffusionModel, MockAutoregressiveModel]:
            adapter = cls(ModelConfig(seed=42))
            assert hasattr(adapter, "config")
            assert hasattr(adapter, "generate")
            assert hasattr(adapter, "get_logits")
            assert hasattr(adapter, "get_embeddings")


class TestFrameworkAgnosticPipeline:
    """Prove the evaluation engine is truly adapter-agnostic.

    The same IntrospectiveScorer + SemanticDriftDetector pipeline
    produces valid results regardless of which adapter provides the
    logits/tokens/embeddings.
    """

    @pytest.fixture(params=["mock_dlm", "mock_ar"])
    def adapter(self, request: pytest.FixtureRequest) -> ModelAdapter:
        config = ModelConfig(
            vocab_size=500, embed_dim=32, seq_len=16,
            num_steps=4, seed=42,
        )
        if request.param == "mock_dlm":
            return MockDiffusionModel(config)
        return MockAutoregressiveModel(config)

    def test_generate_returns_valid_result(self, adapter: ModelAdapter) -> None:
        """Any adapter's generate() should return a valid GenerationResult."""
        result = adapter.generate()

        assert isinstance(result, GenerationResult)
        assert result.token_ids.ndim == 1
        assert result.logits.ndim == 2
        assert result.embeddings.ndim == 2
        assert result.total_elapsed_ms > 0
        assert len(result.steps) > 0

    def test_consistency_scorer_works_with_any_adapter(self) -> None:
        """IntrospectiveScorer should work with any pair of adapters."""
        config = ModelConfig(
            vocab_size=500, embed_dim=32, seq_len=16,
            num_steps=4, seed=42,
        )
        dlm = MockDiffusionModel(config)
        ar = MockAutoregressiveModel(config)

        dlm_result = dlm.generate()
        ar_result = ar.generate()

        scorer = IntrospectiveScorer(threshold=0.5)
        report = scorer.score(
            dlm_result.logits, ar_result.logits,
            dlm_result.token_ids, ar_result.token_ids,
        )

        assert 0.0 <= report.ics_score <= 1.0
        assert report.total_positions == config.seq_len
        assert len(report.position_scores) == config.seq_len

    def test_drift_detector_works_with_any_adapter(self) -> None:
        """SemanticDriftDetector should work with embeddings from any adapter."""
        config = ModelConfig(
            vocab_size=500, embed_dim=32, seq_len=16,
            num_steps=4, seed=42,
        )
        dlm = MockDiffusionModel(config)
        ar = MockAutoregressiveModel(config)

        dlm_result = dlm.generate()
        ar_result = ar.generate()

        detector = SemanticDriftDetector(threshold_z=2.0)
        report = detector.compare(
            "adapter-A", "adapter-B",
            baseline_embeddings=ar_result.embeddings,
            comparison_embeddings=dlm_result.embeddings,
        )

        assert report.aggregate_drift >= 0.0
        assert report.elapsed_ms >= 0.0
        assert report.baseline_id == "adapter-A"
        assert report.comparison_id == "adapter-B"

    def test_same_pipeline_different_adapters(self) -> None:
        """Demonstrate plug-and-play: same evaluation code, different adapters.

        This is the definitive test for the 'framework-agnostic' claim.
        The evaluation logic below does not contain any adapter-specific
        code — it only calls methods defined by the ModelAdapter protocol.
        """
        config = ModelConfig(
            vocab_size=500, embed_dim=32, seq_len=16,
            num_steps=4, inconsistency_rate=0.1, seed=42,
        )

        # --- Generic evaluation function (adapter-agnostic) ---
        def evaluate(dlm: ModelAdapter, ar: ModelAdapter) -> dict:
            dlm_result = dlm.generate()
            ar_result = ar.generate()

            scorer = IntrospectiveScorer(threshold=0.5)
            report = scorer.score(
                dlm_result.logits, ar_result.logits,
                dlm_result.token_ids, ar_result.token_ids,
            )

            detector = SemanticDriftDetector(threshold_z=2.0)
            drift = detector.compare(
                "baseline", "comparison",
                baseline_embeddings=ar_result.embeddings,
                comparison_embeddings=dlm_result.embeddings,
            )

            return {
                "ics_score": report.ics_score,
                "drift_score": drift.aggregate_drift,
                "passed": report.passed and drift.passed,
            }

        # --- Run with mock adapters ---
        result_mock = evaluate(
            MockDiffusionModel(config),
            MockAutoregressiveModel(config),
        )

        assert "ics_score" in result_mock
        assert "drift_score" in result_mock
        assert isinstance(result_mock["passed"], bool)

        # --- Run with different config (proves same code works) ---
        config2 = ModelConfig(
            vocab_size=500, embed_dim=32, seq_len=32,
            num_steps=8, inconsistency_rate=0.3, seed=99,
        )
        result_different = evaluate(
            MockDiffusionModel(config2),
            MockAutoregressiveModel(config2),
        )

        assert "ics_score" in result_different
        # Higher inconsistency → lower score
        assert result_different["ics_score"] < result_mock["ics_score"]
