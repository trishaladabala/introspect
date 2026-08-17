# introspect

**ML Model Consistency Evaluator & Observability Harness**

A framework-agnostic evaluation platform that measures output consistency between autoregressive and diffusion language models, detects semantic drift in embedding spaces, and provides real-time observability through OpenTelemetry instrumentation and a time-series metrics dashboard.

[![CI](https://github.com/trishaladabala/introspect/actions/workflows/ci.yml/badge.svg)](https://github.com/trishaladabala/introspect/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI / API                              │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────────────────┐ │
│  │  pytest  │  │  FastAPI    │  │  CLI (evaluate/serve)    │ │
│  │  Plugin  │  │  + WebSocket│  │                          │ │
│  └────┬─────┘  └──────┬──────┘  └────────────┬─────────────┘ │
│       │               │                      │               │
│  ┌────▼───────────────▼──────────────────────▼─────────────┐ │
│  │              Core Evaluation Engine                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │ Consistency  │  │    Drift     │  │    Model     │  │ │
│  │  │   Scorer     │  │  Detector    │  │  Adapters    │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  └─────────────────────────┬───────────────────────────────┘ │
│                            │                                 │
│  ┌─────────────────────────▼───────────────────────────────┐ │
│  │                  Instrumentation                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │ OpenTelemetry│  │   SQLite     │  │   Console    │  │ │
│  │  │ Instrumentor │  │  Exporter    │  │  Exporter    │  │ │
│  │  └──────────────┘  └──────┬───────┘  └──────────────┘  │ │
│  └───────────────────────────┼─────────────────────────────┘ │
│                              │                               │
│  ┌───────────────────────────▼─────────────────────────────┐ │
│  │              SQLite Time-Series Store                    │ │
│  │  runs │ steps │ consistency │ drift │ system_metrics     │ │
│  └───────────────────────────┬─────────────────────────────┘ │
│                              │                               │
│  ┌───────────────────────────▼─────────────────────────────┐ │
│  │              Dashboard (Canvas 2D Charts)                │ │
│  │  Consistency Timeline │ Step Waterfall │ Drift Monitor   │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

- **Introspective Consistency Scoring (ICS)** — Compares diffusion model parallel output against an autoregressive causal anchor to quantify token-level agreement, with KL divergence and windowed analysis
- **Semantic Drift Detection** — Monitors cosine distance in continuous embedding spaces with z-score anomaly detection and per-layer segmentation
- **Framework-Agnostic Model Adapter** — Protocol-based adapter system supporting mock models, HuggingFace autoregressive models (distilgpt2/GPT-2), and a real discrete diffusion language model (BD3-LM)
- **OpenTelemetry Instrumentation** — Per-step tracing for every denoising iteration with latency, confidence, and memory attributes
- **pytest Plugin** — CI-integrated test markers (`@pytest.mark.consistency`, `@pytest.mark.drift`) with automated pass/fail gates
- **Real-time Dashboard** — WebSocket-powered monitoring with custom Canvas 2D charts (no external charting libraries)
- **CLI** — Full evaluation pipeline with formatted terminal output and non-zero exit on failure

## Supported Models

| Adapter | Model | Type | Status |
|---------|-------|------|--------|
| `HuggingFaceAutoregressiveAdapter` | distilgpt2 / GPT-2 | Autoregressive | ✅ Verified |
| `BD3Adapter` | kuleshov-group/bd3lm-owt-block_size4 | Discrete Diffusion (BD3-LM) | ✅ Verified |
| `HuggingFacePseudoDiffusionAdapter` | Any HF causal LM | Simulated DLM (real logits, simulated denoising) | ✅ Verified |
| `MockDiffusionModel` / `MockAutoregressiveModel` | N/A | Synthetic (for testing) | ✅ Verified |

> **Note on BD3-LM**: This is a *real* discrete diffusion language model from [Arriaga et al.](https://github.com/kuleshov-group/bd3lm), not a simulated proxy. The adapter implements actual masked-token denoising with confidence-based unmasking across T steps. A `torch.compile` compatibility patch is applied for Python 3.12 + PyTorch < 2.5 environments.

## Quickstart

### Install

```bash
git clone https://github.com/trishaladabala/introspect.git
cd introspect
pip install -e ".[dev]"

# For real model evaluation (optional)
pip install torch transformers einops
```

### Run an Evaluation

```bash
# Full consistency + drift evaluation with mock models (no GPU required)
python -m introspect.cli evaluate --seq-len 128 --num-steps 16 --inconsistency-rate 0.1

# With verbose tracing output
python -m introspect.cli evaluate -v
```

### Run Tests

```bash
# Full test suite (69 tests, ~1s)
pytest tests/ -v

# Skip slow real-model tests
pytest tests/ -v -k "not slow"
```

### Start the Dashboard

```bash
python -m introspect.cli serve --port 8710
# Open http://localhost:8710

# With real models loaded
INTROSPECT_LOAD_MODELS=1 python -m introspect.cli serve --port 8710
```

### Run Real-Model End-to-End Evaluation

```bash
# Runs distilgpt2 (AR) vs BD3-LM (DLM) through the full pipeline
python e2e_real_model_eval.py
```

## Project Structure

```
introspect/
├── src/introspect/
│   ├── core/
│   │   ├── consistency.py     # Introspective Consistency Scorer (ICS + KL + windowed)
│   │   ├── drift.py           # Semantic Drift Detector (cosine + z-score)
│   │   ├── hf_adapter.py      # HuggingFace model adapters (AR, pseudo-DLM, BD3)
│   │   └── models.py          # ModelAdapter protocol + mock implementations
│   ├── tracing/
│   │   ├── instrumentor.py    # OpenTelemetry diffusion step tracing
│   │   └── exporters.py       # SQLite + console span exporters
│   ├── storage/
│   │   └── timeseries.py      # SQLite time-series metrics store
│   ├── api/
│   │   └── server.py          # FastAPI + WebSocket server
│   ├── plugins/
│   │   └── pytest_consistency.py  # pytest plugin with fixtures
│   └── cli.py                 # CLI entry point
├── dashboard/
│   ├── index.html             # Dashboard markup
│   ├── index.css              # Design system (dark mode, glassmorphism)
│   └── app.js                 # Charts (pure Canvas 2D) + WebSocket client
├── tests/
│   ├── test_consistency.py           # 15 consistency scorer tests
│   ├── test_consistency_validation.py # 8 ICS mathematical property tests
│   ├── test_drift.py                 # 11 drift detector tests
│   ├── test_adapter_protocol.py      # 9 adapter protocol + pipeline tests
│   ├── test_storage.py               # 15 storage CRUD tests
│   └── test_api.py                   # 12 API integration tests
├── e2e_real_model_eval.py     # End-to-end real model evaluation script
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
└── pyproject.toml             # Packaging + tool configuration
```

## How It Works

### Introspective Consistency Score (ICS)

Diffusion Language Models generate all tokens in parallel, which can produce tokens that disagree with what a sequential (autoregressive) model would generate. The ICS measures this disagreement:

```
ICS = (agreeing positions) / (total positions)
```

A score of **1.0** means perfect consistency. The CI pipeline fails when ICS drops below the configured threshold (default: 0.85).

Auxiliary metrics include per-position KL divergence (`KL(DLM || AR)`) and sliding-window scores for localizing inconsistency clusters.

### Semantic Drift Detection

When model weights are updated or quantization schemes change, the continuous embeddings may drift. The detector is **stateless** — both baseline and comparison embeddings are passed directly:

1. Computes **cosine distance** between baseline and comparison embeddings
2. Uses **z-score anomaly detection** (via historical drift data in the MetricsStore) to flag statistically significant drift
3. Supports **per-layer segmentation** for localized analysis
4. Reports aggregate drift score, per-position distances, and pass/fail status

### OpenTelemetry Tracing

Every denoising step emits an OpenTelemetry span with attributes:

| Attribute | Description |
|---|---|
| `step.index` | Current denoising step |
| `step.tokens_unmasked` | Tokens committed in this step |
| `step.elapsed_ms` | Wall-clock step latency |
| `step.confidence_mean` | Average confidence across positions |
| `step.memory_bytes` | Estimated memory consumption |

## CI/CD Integration

The GitHub Actions workflow runs on every push and PR:

1. **Lint** — `ruff check` for style enforcement
2. **Type Check** — `mypy` for static type analysis
3. **Test Suite** — 69 tests across 6 modules
4. **Consistency Gate** — Runs a live evaluation; fails the build if ICS < 0.85

## Honest Limitations

- **Not production infrastructure**: This is a single-machine evaluation harness, not a multi-tenant platform. No auth, no horizontal scaling, SQLite storage.
- **DLM ecosystem is nascent**: Real pre-trained diffusion LMs (LLaDA, MDLM, Dream) are not available as lightweight HuggingFace checkpoints. BD3-LM is the most accessible real DLM and requires `trust_remote_code=True`.
- **ICS measures token-level agreement, not semantic equivalence**: Two outputs can be semantically identical but score low if they use different tokens (e.g., "big" vs. "large").
- **Drift detection uses static embeddings**: The cosine distance is computed on input embeddings (from the model's embedding layer), not on contextual hidden states. This is a proxy for representation shift, not a direct measure of behavioral change.
- **CPU-only inference**: Both distilgpt2 and BD3-LM run on CPU. This is intentional for accessibility but means generation is slower (~1-2s for 16 tokens).

## License

MIT
