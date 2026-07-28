"""
consistency.py — Introspective Consistency Scoring engine.

Implements the Introspective Consistency Score (ICS) as described in
recent I-DLM literature. The scorer takes a diffusion model's parallel
output and re-evaluates it against a causal (autoregressive) anchor
distribution to quantify how often the parallel predictions agree with
what a sequential model would have generated.

ICS = (number of agreeing positions) / (total evaluated positions)

A score of 1.0 means perfect consistency; scores below the configured
threshold trigger CI failures, flagging potential hallucination or
logical incoherence in the model's parallel decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PositionScore:
    """Consistency analysis for a single token position.

    Attributes:
        position: Index in the sequence.
        dlm_token_id: Token predicted by the diffusion model.
        ar_token_id: Token predicted by the autoregressive anchor.
        agrees: Whether the two predictions match.
        dlm_confidence: DLM's softmax confidence at this position.
        ar_confidence: AR model's softmax confidence at this position.
        kl_divergence: KL(AR || DLM) at this position — measures how
            different the DLM's distribution is from the AR reference.
    """
    position: int
    dlm_token_id: int
    ar_token_id: int
    agrees: bool
    dlm_confidence: float
    ar_confidence: float
    kl_divergence: float


@dataclass(frozen=True)
class ConsistencyReport:
    """Full consistency evaluation report for a single generation.

    Attributes:
        ics_score: Aggregate Introspective Consistency Score ∈ [0, 1].
        total_positions: Total positions evaluated.
        agreeing_positions: Positions where DLM and AR agree.
        mean_kl_divergence: Average KL divergence across all positions.
        max_kl_divergence: Worst-case KL divergence.
        position_scores: Per-position breakdown.
        passed: Whether the score meets the configured threshold.
        threshold: The threshold used for pass/fail determination.
        windowed_scores: ICS computed over sliding windows of the sequence,
            useful for identifying localized inconsistency clusters.
    """
    ics_score: float
    total_positions: int
    agreeing_positions: int
    mean_kl_divergence: float
    max_kl_divergence: float
    position_scores: list[PositionScore]
    passed: bool
    threshold: float
    windowed_scores: list[float] = field(default_factory=list)


class IntrospectiveScorer:
    """Evaluates introspective consistency between parallel and sequential outputs.

    The scorer is model-agnostic — it operates solely on logit arrays (NumPy),
    making it usable with any model backend that implements the ModelAdapter
    protocol.

    Usage:
        scorer = IntrospectiveScorer(threshold=0.85)
        report = scorer.score(dlm_logits, ar_logits, dlm_tokens, ar_tokens)
        assert report.passed, f"ICS {report.ics_score:.3f} below threshold"
    """

    def __init__(
        self,
        threshold: float = 0.85,
        window_size: int = 16,
        epsilon: float = 1e-10,
    ) -> None:
        """Initialize the scorer.

        Args:
            threshold: Minimum ICS score to pass evaluation (∈ [0, 1]).
            window_size: Size of sliding window for windowed scoring.
            epsilon: Small constant for numerical stability in KL computation.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        if window_size < 1:
            raise ValueError(f"Window size must be >= 1, got {window_size}")

        self._threshold = threshold
        self._window_size = window_size
        self._epsilon = epsilon

    @property
    def threshold(self) -> float:
        """Current pass/fail threshold."""
        return self._threshold

    def score(
        self,
        dlm_logits: NDArray[np.float32],
        ar_logits: NDArray[np.float32],
        dlm_tokens: NDArray[np.int64],
        ar_tokens: NDArray[np.int64],
    ) -> ConsistencyReport:
        """Compute the full Introspective Consistency Score.

        Args:
            dlm_logits: Logits from the diffusion model (shape: [seq_len, vocab_size]).
            ar_logits: Logits from the autoregressive anchor (shape: [seq_len, vocab_size]).
            dlm_tokens: Token IDs sampled by the DLM (shape: [seq_len]).
            ar_tokens: Token IDs sampled by the AR model (shape: [seq_len]).

        Returns:
            A ConsistencyReport with aggregate and per-position metrics.

        Raises:
            ValueError: If input shapes are mismatched.
        """
        self._validate_inputs(dlm_logits, ar_logits, dlm_tokens, ar_tokens)

        seq_len = len(dlm_tokens)

        # Compute softmax distributions.
        dlm_probs = self._softmax(dlm_logits)
        ar_probs = self._softmax(ar_logits)

        # Per-position analysis.
        position_scores: list[PositionScore] = []
        agreement_mask = np.zeros(seq_len, dtype=bool)

        for pos in range(seq_len):
            agrees = bool(dlm_tokens[pos] == ar_tokens[pos])
            agreement_mask[pos] = agrees

            kl_div = self._kl_divergence(ar_probs[pos], dlm_probs[pos])

            position_scores.append(PositionScore(
                position=pos,
                dlm_token_id=int(dlm_tokens[pos]),
                ar_token_id=int(ar_tokens[pos]),
                agrees=agrees,
                dlm_confidence=float(dlm_probs[pos].max()),
                ar_confidence=float(ar_probs[pos].max()),
                kl_divergence=float(kl_div),
            ))

        # Aggregate metrics.
        agreeing = int(agreement_mask.sum())
        ics = agreeing / seq_len if seq_len > 0 else 0.0

        kl_values = np.array([ps.kl_divergence for ps in position_scores])
        mean_kl = float(kl_values.mean()) if len(kl_values) > 0 else 0.0
        max_kl = float(kl_values.max()) if len(kl_values) > 0 else 0.0

        # Windowed scoring.
        windowed = self._compute_windowed_scores(agreement_mask)

        return ConsistencyReport(
            ics_score=ics,
            total_positions=seq_len,
            agreeing_positions=agreeing,
            mean_kl_divergence=mean_kl,
            max_kl_divergence=max_kl,
            position_scores=position_scores,
            passed=ics >= self._threshold,
            threshold=self._threshold,
            windowed_scores=windowed,
        )

    def _softmax(self, logits: NDArray[np.float32]) -> NDArray[np.float32]:
        """Numerically stable softmax over the last axis."""
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _kl_divergence(
        self,
        p: NDArray[np.float32],
        q: NDArray[np.float32],
    ) -> float:
        """KL(P || Q) — measures how Q diverges from the reference P.

        Uses epsilon-smoothing to avoid log(0).
        """
        p_safe = np.clip(p, self._epsilon, None)
        q_safe = np.clip(q, self._epsilon, None)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))

    def _compute_windowed_scores(
        self,
        agreement_mask: NDArray[np.bool_],
    ) -> list[float]:
        """Compute ICS over sliding windows to localize inconsistency clusters."""
        seq_len = len(agreement_mask)
        if seq_len < self._window_size:
            return [float(agreement_mask.mean())] if seq_len > 0 else []

        scores: list[float] = []
        for start in range(0, seq_len - self._window_size + 1, self._window_size // 2):
            end = min(start + self._window_size, seq_len)
            window = agreement_mask[start:end]
            scores.append(float(window.mean()))

        return scores

    def _validate_inputs(
        self,
        dlm_logits: NDArray[np.float32],
        ar_logits: NDArray[np.float32],
        dlm_tokens: NDArray[np.int64],
        ar_tokens: NDArray[np.int64],
    ) -> None:
        """Validate input shapes and types."""
        if dlm_logits.ndim != 2:
            raise ValueError(f"dlm_logits must be 2D, got shape {dlm_logits.shape}")
        if ar_logits.ndim != 2:
            raise ValueError(f"ar_logits must be 2D, got shape {ar_logits.shape}")
        if dlm_logits.shape != ar_logits.shape:
            raise ValueError(
                f"Shape mismatch: dlm_logits={dlm_logits.shape}, ar_logits={ar_logits.shape}"
            )
        if len(dlm_tokens) != dlm_logits.shape[0]:
            raise ValueError(
                f"Token count mismatch: {len(dlm_tokens)} tokens vs "
                f"{dlm_logits.shape[0]} logit rows"
            )
        if len(ar_tokens) != ar_logits.shape[0]:
            raise ValueError(
                f"Token count mismatch: {len(ar_tokens)} tokens vs "
                f"{ar_logits.shape[0]} logit rows"
            )
