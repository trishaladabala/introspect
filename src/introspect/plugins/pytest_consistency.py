"""
pytest_consistency.py — pytest plugin for CI-integrated consistency testing.

Registers as a pytest plugin via pyproject.toml entry points. Provides:
- Custom markers: @pytest.mark.consistency, @pytest.mark.drift
- Fixtures: mock_dlm, mock_ar, scorer, drift_detector, metrics_store
- Auto-generates JUnit XML compatible reports
- CI exit codes: non-zero on consistency/drift failures

Usage in tests:
    @pytest.mark.consistency
    def test_model_consistency(mock_dlm, mock_ar, scorer):
        dlm_result = mock_dlm.generate()
        ar_result = mock_ar.generate()
        report = scorer.score(
            dlm_result.logits, ar_result.logits,
            dlm_result.token_ids, ar_result.token_ids,
        )
        assert report.passed, f"ICS {report.ics_score:.3f} below threshold"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from introspect.core.consistency import IntrospectiveScorer
from introspect.core.drift import SemanticDriftDetector
from introspect.core.models import (
    MockDiffusionModel,
    MockAutoregressiveModel,
    ModelConfig,
)
from introspect.storage.timeseries import MetricsStore


# ════════════════════════════════════════════════════════════════════════════════
# Plugin registration hooks
# ════════════════════════════════════════════════════════════════════════════════

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "consistency: mark test as an introspective consistency evaluation",
    )
    config.addinivalue_line(
        "markers",
        "drift: mark test as a semantic drift detection evaluation",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring API server",
    )


# ════════════════════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def model_config() -> ModelConfig:
    """Default model configuration for testing.

    Uses small dimensions for fast test execution.
    """
    return ModelConfig(
        vocab_size=1000,
        embed_dim=64,
        seq_len=32,
        num_steps=8,
        confidence_mean=0.7,
        confidence_std=0.15,
        inconsistency_rate=0.1,
        seed=42,
    )


@pytest.fixture
def mock_dlm(model_config: ModelConfig) -> MockDiffusionModel:
    """A mock diffusion language model."""
    return MockDiffusionModel(model_config)


@pytest.fixture
def mock_ar(model_config: ModelConfig) -> MockAutoregressiveModel:
    """A mock autoregressive model (causal anchor)."""
    return MockAutoregressiveModel(model_config)


@pytest.fixture
def scorer() -> IntrospectiveScorer:
    """An introspective consistency scorer with default threshold."""
    return IntrospectiveScorer(threshold=0.85)


@pytest.fixture
def lenient_scorer() -> IntrospectiveScorer:
    """A consistency scorer with a relaxed threshold for edge-case testing."""
    return IntrospectiveScorer(threshold=0.5)


@pytest.fixture
def drift_detector() -> SemanticDriftDetector:
    """A semantic drift detector with default settings."""
    return SemanticDriftDetector(threshold_z=2.0)


@pytest.fixture
def metrics_store(tmp_path: Path) -> MetricsStore:
    """An in-test metrics store using a temporary database."""
    db_path = tmp_path / "test_metrics.db"
    store = MetricsStore(str(db_path))
    yield store
    store.close()


@pytest.fixture
def consistent_config() -> ModelConfig:
    """A model config that produces perfectly consistent output."""
    return ModelConfig(
        vocab_size=1000,
        embed_dim=64,
        seq_len=32,
        num_steps=8,
        inconsistency_rate=0.0,
        seed=42,
    )


@pytest.fixture
def inconsistent_config() -> ModelConfig:
    """A model config that produces highly inconsistent output."""
    return ModelConfig(
        vocab_size=1000,
        embed_dim=64,
        seq_len=32,
        num_steps=8,
        inconsistency_rate=0.5,
        seed=42,
    )
