"""
test_consistency.py — Tests for the Introspective Consistency Scorer.

Validates ICS computation with known-good and known-bad distributions,
parametrized tests across different inconsistency rates, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from introspect.core.consistency import IntrospectiveScorer, ConsistencyReport
from introspect.core.models import (
    MockDiffusionModel,
    MockAutoregressiveModel,
    ModelConfig,
)


class TestIntrospectiveScorer:
    """Tests for IntrospectiveScorer."""

    def test_perfect_consistency(self, consistent_config: ModelConfig) -> None:
        """When inconsistency_rate=0, DLM tokens should perfectly match AR tokens."""
        dlm = MockDiffusionModel(consistent_config)
        ar = MockAutoregressiveModel(consistent_config)

        dlm_result = dlm.generate()
        ar_result = ar.generate()

        scorer = IntrospectiveScorer(threshold=0.85)
        report = scorer.score(
            dlm_result.logits, ar_result.logits,
            dlm_result.token_ids, ar_result.token_ids,
        )

        assert report.ics_score == 1.0
        assert report.agreeing_positions == report.total_positions
        assert report.passed is True

    def test_high_inconsistency_fails(self, inconsistent_config: ModelConfig) -> None:
        """When inconsistency_rate=0.5, ICS should be below threshold."""
        dlm = MockDiffusionModel(inconsistent_config)
        ar = MockAutoregressiveModel(inconsistent_config)

        dlm_result = dlm.generate()
        ar_result = ar.generate()

        scorer = IntrospectiveScorer(threshold=0.85)
        report = scorer.score(
            dlm_result.logits, ar_result.logits,
            dlm_result.token_ids, ar_result.token_ids,
        )

        # With 50% inconsistency, we expect significant disagreement.
        assert report.ics_score < 0.85, (
            f"Expected ICS < 0.85 with 50% inconsistency, got {report.ics_score:.3f}"
        )
        assert report.passed is False

    @pytest.mark.parametrize("inconsistency_rate,expected_pass", [
        (0.0, True),
        (0.05, True),
        (0.1, True),
        (0.3, False),
        (0.5, False),
        (0.8, False),
    ])
    def test_parametrized_inconsistency_rates(
        self,
        inconsistency_rate: float,
        expected_pass: bool,
    ) -> None:
        """ICS pass/fail should correlate with inconsistency rate."""
        config = ModelConfig(
            vocab_size=1000, embed_dim=64, seq_len=64,
            num_steps=8, inconsistency_rate=inconsistency_rate, seed=42,
        )
        dlm = MockDiffusionModel(config)
        ar = MockAutoregressiveModel(config)

        # Important: generate once per model so logits and tokens are consistent.
        dlm_result = dlm.generate()
        ar_result = ar.generate()

        scorer = IntrospectiveScorer(threshold=0.85)
        report = scorer.score(
            dlm_result.logits, ar_result.logits,
            dlm_result.token_ids, ar_result.token_ids,
        )

        assert report.passed == expected_pass, (
            f"Rate={inconsistency_rate}: ICS={report.ics_score:.3f}, "
            f"expected_pass={expected_pass}"
        )

    def test_windowed_scoring(self) -> None:
        """Windowed scores should localize inconsistency clusters."""
        config = ModelConfig(
            vocab_size=500, embed_dim=32, seq_len=64,
            num_steps=8, inconsistency_rate=0.1, seed=42,
        )
        dlm = MockDiffusionModel(config)
        ar = MockAutoregressiveModel(config)

        scorer = IntrospectiveScorer(threshold=0.85, window_size=16)
        report = scorer.score(
            dlm.generate().logits, ar.generate().logits,
            dlm.generate().token_ids, ar.generate().token_ids,
        )

        assert len(report.windowed_scores) > 0
        assert all(0.0 <= s <= 1.0 for s in report.windowed_scores)

    def test_kl_divergence_is_non_negative(self) -> None:
        """KL divergence should always be non-negative."""
        config = ModelConfig(
            vocab_size=500, embed_dim=32, seq_len=16,
            num_steps=4, inconsistency_rate=0.2, seed=123,
        )
        dlm = MockDiffusionModel(config)
        ar = MockAutoregressiveModel(config)

        scorer = IntrospectiveScorer(threshold=0.5)
        report = scorer.score(
            dlm.generate().logits, ar.generate().logits,
            dlm.generate().token_ids, ar.generate().token_ids,
        )

        assert report.mean_kl_divergence >= 0
        assert report.max_kl_divergence >= 0
        for ps in report.position_scores:
            assert ps.kl_divergence >= 0

    def test_empty_sequence_handling(self) -> None:
        """Scorer should handle zero-length sequences gracefully."""
        scorer = IntrospectiveScorer(threshold=0.85)

        # This should not raise — degenerate but valid.
        logits = np.zeros((0, 100), dtype=np.float32)
        tokens = np.array([], dtype=np.int64)

        report = scorer.score(logits, logits, tokens, tokens)
        assert report.ics_score == 0.0
        assert report.total_positions == 0

    def test_single_token_sequence(self) -> None:
        """Scorer should work with a single token."""
        scorer = IntrospectiveScorer(threshold=0.85)

        dlm_logits = np.random.randn(1, 100).astype(np.float32)
        ar_logits = dlm_logits.copy()
        tokens = np.array([42], dtype=np.int64)

        report = scorer.score(dlm_logits, ar_logits, tokens, tokens)
        assert report.ics_score == 1.0
        assert report.total_positions == 1

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched logit shapes should raise ValueError."""
        scorer = IntrospectiveScorer()
        dlm_logits = np.zeros((10, 100), dtype=np.float32)
        ar_logits = np.zeros((10, 200), dtype=np.float32)
        tokens = np.zeros(10, dtype=np.int64)

        with pytest.raises(ValueError, match="Shape mismatch"):
            scorer.score(dlm_logits, ar_logits, tokens, tokens)

    def test_token_count_mismatch_raises(self) -> None:
        """Mismatched token/logit counts should raise ValueError."""
        scorer = IntrospectiveScorer()
        logits = np.zeros((10, 100), dtype=np.float32)
        tokens = np.zeros(5, dtype=np.int64)

        with pytest.raises(ValueError, match="Token count mismatch"):
            scorer.score(logits, logits, tokens, tokens)

    def test_threshold_validation(self) -> None:
        """Invalid thresholds should raise ValueError."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            IntrospectiveScorer(threshold=1.5)
        with pytest.raises(ValueError, match="Threshold must be in"):
            IntrospectiveScorer(threshold=-0.1)

    def test_report_position_scores_length(self, model_config: ModelConfig) -> None:
        """Position scores list should match sequence length."""
        dlm = MockDiffusionModel(model_config)
        ar = MockAutoregressiveModel(model_config)

        scorer = IntrospectiveScorer()
        report = scorer.score(
            dlm.generate().logits, ar.generate().logits,
            dlm.generate().token_ids, ar.generate().token_ids,
        )

        assert len(report.position_scores) == model_config.seq_len
