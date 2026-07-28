"""
models.py — Model adapter protocol and mock simulators.

Defines the abstract interface that all model backends must implement,
plus fully functional mock simulators for diffusion and autoregressive
models. The mocks produce statistically realistic outputs (proper softmax
distributions, controllable confidence levels, tunable inconsistency
rates) so the entire project is demonstrable without GPU or model weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# ════════════════════════════════════════════════════════════════════════════════
# Data types
# ════════════════════════════════════════════════════════════════════════════════

class TokenState(Enum):
    """State of a single token position during diffusion generation."""
    MASKED = "masked"
    UNMASKED = "unmasked"


@dataclass(frozen=True)
class GenerationStep:
    """Snapshot of a single diffusion denoising step.

    Attributes:
        step_index: The current step number (0-indexed).
        total_steps: Total denoising steps T.
        token_ids: Array of token IDs at this step (shape: [seq_len]).
        states: Array of TokenState for each position.
        logits: Raw logit array (shape: [seq_len, vocab_size]).
        confidence: Per-position confidence scores (shape: [seq_len]).
        elapsed_ms: Wall-clock time for this step in milliseconds.
        memory_bytes: Estimated memory consumption in bytes.
        tokens_unmasked: Count of tokens newly unmasked in this step.
    """
    step_index: int
    total_steps: int
    token_ids: NDArray[np.int64]
    states: list[TokenState]
    logits: NDArray[np.float32]
    confidence: NDArray[np.float32]
    elapsed_ms: float
    memory_bytes: int
    tokens_unmasked: int


@dataclass(frozen=True)
class GenerationResult:
    """Complete output of a generation run.

    Attributes:
        token_ids: Final token sequence (shape: [seq_len]).
        logits: Final logit array (shape: [seq_len, vocab_size]).
        embeddings: Continuous embedding matrix (shape: [seq_len, embed_dim]).
        steps: Per-step snapshots for the full denoising trajectory.
        total_elapsed_ms: Total wall-clock generation time.
    """
    token_ids: NDArray[np.int64]
    logits: NDArray[np.float32]
    embeddings: NDArray[np.float32]
    steps: list[GenerationStep]
    total_elapsed_ms: float


@dataclass
class ModelConfig:
    """Configuration for mock model behaviour.

    Attributes:
        vocab_size: Vocabulary size for token sampling.
        embed_dim: Embedding dimensionality.
        seq_len: Default sequence length for generation.
        num_steps: Number of diffusion denoising steps T.
        confidence_mean: Mean confidence for unmasking decisions.
        confidence_std: Standard deviation of confidence distribution.
        inconsistency_rate: Fraction of positions where the DLM will
            intentionally produce tokens that disagree with a sequential
            (autoregressive) evaluation. Controls how "inconsistent" the
            mock model is — 0.0 = perfectly consistent, 1.0 = fully random.
        seed: Random seed for reproducibility.
    """
    vocab_size: int = 32000
    embed_dim: int = 512
    seq_len: int = 128
    num_steps: int = 16
    confidence_mean: float = 0.7
    confidence_std: float = 0.15
    inconsistency_rate: float = 0.1
    seed: int | None = 42


# ════════════════════════════════════════════════════════════════════════════════
# Protocol
# ════════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ModelAdapter(Protocol):
    """Abstract interface that all model backends must implement.

    This protocol decouples the evaluation engine from any specific ML
    framework (PyTorch, MLX, llama.cpp), allowing seamless substitution
    of mock models during testing and real models in production.
    """

    @property
    def config(self) -> ModelConfig: ...

    def generate(self, prompt_ids: NDArray[np.int64] | None = None) -> GenerationResult:
        """Run full generation (diffusion denoising or autoregressive).

        Args:
            prompt_ids: Optional prompt token IDs to condition on.

        Returns:
            Complete generation result with per-step telemetry.
        """
        ...

    def get_logits(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Compute logits for a given token sequence.

        Args:
            token_ids: Input token IDs (shape: [seq_len]).

        Returns:
            Logit array (shape: [seq_len, vocab_size]).
        """
        ...

    def get_embeddings(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Extract continuous embeddings for a given token sequence.

        Args:
            token_ids: Input token IDs (shape: [seq_len]).

        Returns:
            Embedding matrix (shape: [seq_len, embed_dim]).
        """
        ...


# ════════════════════════════════════════════════════════════════════════════════
# Mock Diffusion Language Model
# ════════════════════════════════════════════════════════════════════════════════

class MockDiffusionModel:
    """Simulates masked diffusion denoising with realistic statistics.

    Mimics the generation process of models like LLaDA / Dream:
    1. Starts with a fully masked sequence.
    2. At each step, predicts logits for all positions, computes confidence.
    3. Unmasks positions exceeding a dynamic threshold.
    4. Repeats until all positions are unmasked or T steps are exhausted.

    The simulator produces proper softmax distributions with controllable
    temperature, and injects tunable inconsistency to test the evaluator.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()
        self._rng = np.random.default_rng(self._config.seed)
        # Pre-compute a stable "ground truth" token sequence that the
        # autoregressive evaluator would produce. Inconsistent positions
        # will deviate from this ground truth.
        self._ground_truth = self._rng.integers(
            0, self._config.vocab_size, size=self._config.seq_len, dtype=np.int64
        )
        # Pre-compute stable embedding vectors per token for consistency
        # across calls to get_embeddings().
        self._embedding_table = self._rng.standard_normal(
            (self._config.vocab_size, self._config.embed_dim)
        ).astype(np.float32)
        # Normalize rows for cosine-similarity stability.
        norms = np.linalg.norm(self._embedding_table, axis=1, keepdims=True)
        self._embedding_table /= np.maximum(norms, 1e-8)

    @property
    def config(self) -> ModelConfig:
        return self._config

    def generate(self, prompt_ids: NDArray[np.int64] | None = None) -> GenerationResult:
        """Simulate masked diffusion denoising.

        The process mirrors real DLM inference:
        - All positions start MASKED.
        - Each step: compute logits → sample confidence → unmask high-confidence positions.
        - Positions flagged as "inconsistent" get a different token than ground truth.
        """
        seq_len = self._config.seq_len
        vocab_size = self._config.vocab_size
        num_steps = self._config.num_steps

        # Initialize fully masked sequence (token_id = 0 represents MASK).
        current_ids = np.zeros(seq_len, dtype=np.int64)
        mask = np.ones(seq_len, dtype=bool)  # True = masked

        # Pre-select which positions will be inconsistent.
        num_inconsistent = int(seq_len * self._config.inconsistency_rate)
        inconsistent_positions = set(
            self._rng.choice(seq_len, size=num_inconsistent, replace=False).tolist()
        ) if num_inconsistent > 0 else set()

        # Pre-generate wrong tokens for inconsistent positions.
        wrong_tokens: dict[int, int] = {}
        for pos in inconsistent_positions:
            wrong_token = self._rng.integers(0, vocab_size, dtype=np.int64)
            while wrong_token == self._ground_truth[pos]:
                wrong_token = self._rng.integers(0, vocab_size, dtype=np.int64)
            wrong_tokens[pos] = int(wrong_token)

        steps: list[GenerationStep] = []
        total_start = time.perf_counter()

        for step in range(num_steps):
            step_start = time.perf_counter()

            # Generate logits for ALL positions (mimics full-sequence forward pass).
            logits = self._generate_logits(current_ids, mask)

            # Compute per-position confidence (max softmax probability).
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            confidence = softmax.max(axis=1).astype(np.float32)

            # Dynamic threshold: starts moderate, drops each step to ensure
            # most tokens unmask during denoising (not at force-unmask).
            progress = (step + 1) / num_steps
            threshold = max(0.01, self._config.confidence_mean * (1.0 - 0.7 * progress))

            # Determine which masked positions to unmask.
            candidates = mask & (confidence > threshold)
            newly_unmasked = int(candidates.sum())

            # Assign tokens to newly unmasked positions.
            for pos in np.where(candidates)[0]:
                if pos in inconsistent_positions:
                    # Intentionally deviate from ground truth.
                    current_ids[pos] = wrong_tokens[pos]
                else:
                    current_ids[pos] = self._ground_truth[pos]
                mask[pos] = False

            step_elapsed = (time.perf_counter() - step_start) * 1000

            states = [
                TokenState.MASKED if m else TokenState.UNMASKED
                for m in mask
            ]

            steps.append(GenerationStep(
                step_index=step,
                total_steps=num_steps,
                token_ids=current_ids.copy(),
                states=states,
                logits=logits,
                confidence=confidence,
                elapsed_ms=step_elapsed,
                memory_bytes=logits.nbytes + current_ids.nbytes,
                tokens_unmasked=newly_unmasked,
            ))

            # If everything is unmasked, we're done early.
            if not mask.any():
                break

        # Force-unmask any remaining positions at the final step,
        # preserving inconsistency for flagged positions.
        for pos in np.where(mask)[0]:
            if pos in inconsistent_positions:
                current_ids[pos] = wrong_tokens[pos]
            else:
                current_ids[pos] = self._ground_truth[pos]

        total_elapsed = (time.perf_counter() - total_start) * 1000
        final_logits = self._generate_logits(current_ids, np.zeros(seq_len, dtype=bool))
        embeddings = self.get_embeddings(current_ids)

        return GenerationResult(
            token_ids=current_ids.copy(),
            logits=final_logits,
            embeddings=embeddings,
            steps=steps,
            total_elapsed_ms=total_elapsed,
        )

    def get_logits(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Compute logits for a concrete (fully unmasked) sequence."""
        return self._generate_logits(token_ids, np.zeros(len(token_ids), dtype=bool))

    def get_embeddings(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Look up pre-computed normalized embeddings."""
        return self._embedding_table[token_ids]

    def _generate_logits(
        self,
        token_ids: NDArray[np.int64],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.float32]:
        """Generate realistic logit distributions.

        For unmasked positions: high logit at the assigned token (peaked distribution).
        For masked positions: spread distribution with moderate peaks at ground truth.
        """
        seq_len = len(token_ids)
        vocab_size = self._config.vocab_size

        # Base: low-magnitude noise across the vocabulary.
        logits = self._rng.standard_normal((seq_len, vocab_size)).astype(np.float32) * 0.3

        for pos in range(seq_len):
            if mask[pos]:
                # Masked: strong peak at ground truth to ensure unmasking happens.
                gt_token = self._ground_truth[pos]
                confidence = self._rng.normal(
                    self._config.confidence_mean, self._config.confidence_std
                )
                logits[pos, gt_token] += float(np.clip(confidence, 0.3, 1.0)) * 8.0
            else:
                # Unmasked: sharp peak at the committed token.
                logits[pos, token_ids[pos]] += 10.0

        return logits


# ════════════════════════════════════════════════════════════════════════════════
# Mock Autoregressive Model
# ════════════════════════════════════════════════════════════════════════════════

class MockAutoregressiveModel:
    """Simulates left-to-right autoregressive generation.

    Used as the "causal anchor" for introspective consistency evaluation.
    Generates tokens sequentially, conditioning each position only on
    preceding context — mirroring how a standard GPT-style model works.

    Shares the same ground truth and embedding table as a paired
    MockDiffusionModel when given the same config, ensuring meaningful
    consistency comparisons.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or ModelConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._ground_truth = self._rng.integers(
            0, self._config.vocab_size, size=self._config.seq_len, dtype=np.int64
        )
        self._embedding_table = self._rng.standard_normal(
            (self._config.vocab_size, self._config.embed_dim)
        ).astype(np.float32)
        norms = np.linalg.norm(self._embedding_table, axis=1, keepdims=True)
        self._embedding_table /= np.maximum(norms, 1e-8)

    @property
    def config(self) -> ModelConfig:
        return self._config

    def generate(self, prompt_ids: NDArray[np.int64] | None = None) -> GenerationResult:
        """Simulate sequential autoregressive generation."""
        seq_len = self._config.seq_len
        current_ids = np.zeros(seq_len, dtype=np.int64)
        steps: list[GenerationStep] = []
        total_start = time.perf_counter()

        for pos in range(seq_len):
            step_start = time.perf_counter()

            # AR model always produces ground truth (it is the reference).
            current_ids[pos] = self._ground_truth[pos]
            logits = self._generate_logits_at(current_ids, pos)

            step_elapsed = (time.perf_counter() - step_start) * 1000
            states = [
                TokenState.UNMASKED if i <= pos else TokenState.MASKED
                for i in range(seq_len)
            ]

            steps.append(GenerationStep(
                step_index=pos,
                total_steps=seq_len,
                token_ids=current_ids.copy(),
                states=states,
                logits=logits,
                confidence=np.ones(seq_len, dtype=np.float32),
                elapsed_ms=step_elapsed,
                memory_bytes=logits.nbytes + current_ids.nbytes,
                tokens_unmasked=1,
            ))

        total_elapsed = (time.perf_counter() - total_start) * 1000
        final_logits = self.get_logits(current_ids)
        embeddings = self.get_embeddings(current_ids)

        return GenerationResult(
            token_ids=current_ids.copy(),
            logits=final_logits,
            embeddings=embeddings,
            steps=steps,
            total_elapsed_ms=total_elapsed,
        )

    def get_logits(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Compute logits for a complete sequence (teacher forcing)."""
        seq_len = len(token_ids)
        vocab_size = self._config.vocab_size
        logits = self._rng.standard_normal((seq_len, vocab_size)).astype(np.float32) * 0.3
        for pos in range(seq_len):
            logits[pos, self._ground_truth[pos]] += 8.0
        return logits

    def get_embeddings(self, token_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Look up pre-computed normalized embeddings."""
        return self._embedding_table[token_ids]

    def _generate_logits_at(
        self,
        token_ids: NDArray[np.int64],
        current_pos: int,
    ) -> NDArray[np.float32]:
        """Generate logits with causal masking (only attend to positions <= current_pos)."""
        seq_len = len(token_ids)
        vocab_size = self._config.vocab_size
        logits = self._rng.standard_normal((seq_len, vocab_size)).astype(np.float32) * 0.3

        for pos in range(current_pos + 1):
            logits[pos, self._ground_truth[pos]] += 8.0

        return logits
