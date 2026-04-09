# Provenance AI - Remaining Work Handoff

**Date:** 2026-04-04
**Branch:** main
**Last Commit:** `b117ff1` — "fix: address remaining issues - thread safety, import guards, docker"

---

## Context

This is a modular AI text detection library with detectors across statistical, stylometric, transformer, and LLM categories. Two rounds of bug fixes have been completed (23 issues total). The codebase is now structurally sound — imports are guarded, thread-safe, dockerized with uv, and API respects request parameters.

### What Was Fixed (Already Committed)

**Commit `bab2565` — 17 fixes:**
1. Added `click` to core dependencies
2. Created `py.typed` marker file
3. Fixed perplexity window bug (skipped first 512 tokens)
4. Deduplicated `PerplexityDetector`/`PerplexityDetectorNeo` via inheritance
5. Cached GPT-2 model in `FeatureExtractor` (was loading per call)
6. Fixed API to respect `detectors`/`ensemble_strategy` params
7. Fixed `RADARDetector` → `coai/roberta-ai-detector-v2` (was broken Vicuna-7B)
8. Fixed `HuggingFaceClassifierDetector` redundant model loading
9. Fixed `LightGBMDetector` import guard
10. Registered `hc3` + `m4` datasets in `DatasetRegistry`
11. Added `seed` param to `CurvatureDetector`
12. Fixed `BurstinessDetector` latency_tier → `slow`
13. Renamed `_compute_log_probs` → `_compute_ai_scores`
14. Fixed leading space in `TRANSITION_PHRASES`
15. Replaced unicode chars in benchmark script

**Commit `b117ff1` — 6 fixes:**
1. Thread-safe `DetectorRegistry` with double-checked locking
2. Dockerfile migrated from pip to uv
3. Added build artifacts to `.gitignore`
4. Fixed nltk unconditional import in `entropy.py`
5. Added import guards to `transformer/__init__.py`
6. Added import guards to `llm/__init__.py`

---

## Remaining Work (Priority Order)

### 1. Replace Heuristic Scoring with Calibrated Models (P0)

**Problem:** Nearly every detector uses hand-tuned `if/else` thresholds instead of learned models. These will produce unreliable results on real data.

**Affected files and the specific heuristic blocks:**

| Detector | File | Lines | Current Logic |
|----------|------|-------|---------------|
| `EntropyDetector` | `detectors/statistical/entropy.py` | 80-91 | `if kl_div > 2.0: score = 0.8` |
| `BurstinessDetector` | `detectors/statistical/burstiness.py` | 71-78 | `if cv < 0.2: score = 0.8` |
| `RepetitionDetector` | `detectors/statistical/repetition.py` | 128-129 | `if ngram_entropy < 0.5: score = 0.8` |
| `CurvatureDetector` | `detectors/statistical/curvature.py` | 106-134 | `if curvature > 0.5: score = 0.8` |
| `SurprisalDetector` | `detectors/statistical/surprisal.py` | 153-187 | `if surprisal_cv > 0.8: score = 0.8` |
| `FeatureExtractor` (stylometric) | `detectors/stylometric/feature_extractor.py` | 470-481 | `if flesch_kincaid > 14: score += 0.1` |
| `CognitiveDetector` | `detectors/stylometric/cognitive.py` | 176-208 | `if paragraph_length_cv < 0.3: ai_score += 0.2` |
| `AcademicDetector` | `domains/academic.py` | 147-162 | `if hallucination_risk > 0.3: score = 0.75` |
| `CodeDetector` | `domains/code.py` | 134-145 | `if low_entropy_score > 0.3: score = 0.7` |
| `MultilingualDetector` | `domains/multilingual.py` | 151-162 | `if burstiness_cv < threshold * 0.5: score = 0.8` |

**Recommended approach:**

Option A — **Isotonic/Platt calibration per detector** (fastest path):
- Keep feature extraction as-is
- Add a `calibrate(texts, labels)` method to each detector
- Use `sklearn.calibration.CalibratedClassifierCV` to map raw features → calibrated probabilities
- Store calibration model as instance variable
- `detect()` uses calibration model if available, falls back to heuristic if not

Option B — **Train a single meta-classifier** (cleaner architecture):
- Each detector outputs a raw score (0-1)
- Train a single logistic regression / LightGBM on top of all detector scores
- This is essentially what the `Ensemble` stacking strategy does, but needs calibrated training data
- Requires benchmark dataset with known labels (HC3, RAID, MAGE)

**Suggested starting point:** Option A for `EntropyDetector` and `BurstinessDetector` as proofs of concept, then generalize.

**Training data sources:**
- `Hello-SimpleAI/HC3` — human vs ChatGPT responses
- `liamdugan/raid` — diverse AI-generated text
- `NickyNicky/M4` — multi-model AI text
- Already loadable via `provenance.benchmarks.workflow.DatasetRegistry`

---

### 2. Add Integration Tests with Real Detectors (P1)

**Problem:** Existing tests use `DummyDetector` that returns fixed scores. No tests exercise actual detector logic, model loading, or end-to-end pipelines.

**What's needed:**

```
tests/
├── integration/
│   ├── test_statistical_detectors.py    # Entropy, Burstiness, Repetition with real text
│   ├── test_stylometric_detectors.py    # FeatureExtractor with known human/AI samples
│   ├── test_transformer_detectors.py    # HF classifiers (skip if no GPU/models)
│   ├── test_ensemble_integration.py     # Full Provenance().detect() pipeline
│   └── test_api_integration.py          # FastAPI TestClient with real requests
└── fixtures/
    ├── sample_human.txt                  # Real human-written text (1000+ words)
    ├── sample_ai.txt                     # Known AI-generated text
    └── sample_code.py                    # Real code snippet
```

**Key test scenarios:**
- Detector returns score in [0, 1] range
- Detector returns confidence in [0, 1] range
- Detector handles empty/short text gracefully
- Detector handles unicode/special characters
- Ensemble produces different scores with different strategies
- API endpoints return correct HTTP status codes
- Batch endpoint processes all items

**Mark slow tests:** Use `@pytest.mark.slow` for tests that download models or make network calls. CI should run `pytest -m "not slow"`.

---

### 3. Fix conftest.py sys.path → Editable Install (P2)

**Problem:** `tests/conftest.py` manipulates `sys.path` directly:
```python
provenance_path = Path(__file__).parent.parent
sys.path.insert(0, str(provenance_path))
```

**Fix:**
```bash
pip install -e ".[dev]"
```

Then remove the `sys.path` manipulation from `conftest.py`. The `PROVENANCE_SKIP_ENTRY_POINTS` env var should stay (prevents loading real detectors during unit tests).

---

### 4. Additional Improvements (Future)

| Item | Effort | Notes |
|------|--------|-------|
| Add `domain` filtering to API `/detect` endpoint | 30 min | Request param exists but not used in `_get_provenance()` |
| Make `FeatureExtractor._surprisal_model` thread-safe | 1 hr | Class-level cache needs lock for multi-threaded serving |
| Add rate limiting to API | 2 hr | For production deployment |
| Add request validation (max text length) | 15 min | Prevent OOM on huge inputs |
| Docker multi-stage build | 1 hr | Reduce image size |
| Add `pyproject.toml` scripts for common tasks | 30 min | `uv run provenance benchmark`, etc. |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `provenance/__init__.py` | Public API exports |
| `provenance/sentinel.py` | `Provenance` main facade class |
| `provenance/core/base.py` | `BaseDetector`, `DetectorResult`, `SentinelResult` |
| `provenance/core/ensemble.py` | `Ensemble` class with 3 strategies |
| `provenance/core/registry.py` | Thread-safe `DetectorRegistry` singleton |
| `provenance/api.py` | FastAPI REST server |
| `provenance/cli.py` | Click CLI interface |
| `provenance/benchmarks/workflow.py` | `BenchmarkRunner`, `DatasetRegistry` |
| `provenance/benchmarks/run_comprehensive.py` | Script to run all detectors on datasets |
| `pyproject.toml` | Dependencies, entry points, project config |

## Dependency Groups

| Group | Purpose |
|-------|---------|
| (core) | spacy, langdetect, numpy, scipy, click |
| `statistical` | nltk, torch, transformers |
| `stylometric` | spacy (already in core), scikit-learn |
| `ml` | lightgbm, shap, optuna |
| `transformer` | torch, transformers, captum |
| `llm` | ollama, litellm |
| `api` | fastapi, uvicorn, pydantic |
| `dev` | pytest, pytest-cov, ruff, black, mypy |
| `all` | Everything above |

## Running Tests

```bash
# Unit tests only (fast, no model downloads)
PROVENANCE_SKIP_ENTRY_POINTS=1 pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=provenance --cov-report=html

# Lint
ruff check provenance/ tests/
black --check provenance/ tests/

# Type check
mypy provenance/
```
