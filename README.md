# introspect

**ML Model Consistency Evaluator & Observability Harness**

A production-quality CI/CD-integrated testing framework that evaluates generative model output consistency, detects semantic drift in embedding spaces, and provides real-time observability through OpenTelemetry instrumentation and a time-series metrics dashboard.

[![CI](https://github.com/your-username/introspect/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/introspect/actions)
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

- **Introspective Consistency Scoring (ICS)** — Compares diffusion model parallel output against an autoregressive causal anchor to quantify token-level agreement
- **Semantic Drift Detection** — Monitors cosine distance in continuous embedding spaces with z-score anomaly detection
- **OpenTelemetry Instrumentation** — Per-step tracing for every denoising iteration with latency, confidence, and memory attributes
- **pytest Plugin** — CI-integrated test markers (`@pytest.mark.consistency`, `@pytest.mark.drift`) with automated pass/fail gates
- **Real-time Dashboard** — WebSocket-powered monitoring with custom Canvas 2D charts
- **CLI** — Full evaluation pipeline with formatted terminal output and non-zero exit on failure

## Quickstart

### Install

```bash
git clone https://github.com/your-username/introspect.git
cd introspect
pip install -e ".[dev]"
```

### Run an Evaluation

```bash
# Full consistency + drift evaluation with CLI output
python -m introspect.cli evaluate --seq-len 128 --num-steps 16 --inconsistency-rate 0.1

# With verbose tracing output
python -m introspect.cli evaluate -v
```

### Run Tests

```bash
# Full test suite
pytest tests/ -v

# Only consistency tests
pytest tests/ -v -m consistency

# Only drift tests
pytest tests/ -v -m drift
```

### Start the Dashboard

```bash
python -m introspect.cli serve --port 8710
# Open http://localhost:8710
```

## Project Structure

```
introspect/
├── src/introspect/
│   ├── core/
│   │   ├── consistency.py     # Introspective Consistency Scorer
│   │   ├── drift.py           # Semantic Drift Detector
│   │   └── models.py          # Model adapter protocol + mock simulators
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
│   ├── test_consistency.py    # 11 consistency scorer tests
│   ├── test_drift.py          # 11 drift detector tests
│   ├── test_storage.py        # 12 storage CRUD tests
│   └── test_api.py            # 12 API integration tests
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

### Semantic Drift Detection

When model weights are updated or quantization schemes change, the continuous embeddings may drift. The detector:

1. Stores a **baseline** embedding matrix
2. Computes **cosine distance** between baseline and new embeddings
3. Uses **z-score anomaly detection** to flag statistically significant drift
4. Supports **per-layer segmentation** for localized analysis

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
3. **Test Suite** — 46 tests across 4 modules
4. **Consistency Gate** — Runs a live evaluation; fails the build if ICS < 0.85

## License

MIT
