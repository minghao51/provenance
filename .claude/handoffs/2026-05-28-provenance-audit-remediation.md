# Session Handoff Plan: Provenance Audit & Remediation

**Slug:** `provenance-audit-remediation`
**Readable Summary:** Comprehensive Codebase Audit & Phased Remediation Plan
**Created:** 2026-05-28

---

## 1. Primary Request and Intent

The user requested a full codebase review and audit of the **provenance** project — a modular, explainable, composable Python library for detecting AI-generated text. The audit covered:

- **Architecture & design patterns** — base classes, registry, ensemble, config, errors
- **Detector implementations** — statistical, stylometric, transformer, LLM detectors
- **Test suite quality** — coverage, assertions, isolation, gaps
- **Infrastructure** — Docker, CI/CD, dependencies, API, CLI, benchmarks
- **Security & performance** — vulnerabilities, memory leaks, DoS vectors
- **Domain modules & explainability** — multilingual, academic, code, heatmaps

After the audit, the user requested:
1. A **phased plan** accounting for all identified issues, starting with the suggested next steps
2. **Google AI research** for best practices on low-confidence tasks
3. A **handoff document** for implementation/fix leveraging sub-agents and phases

---

## 2. Key Technical Concepts

- **Pickle/Joblib deserialization security** — HMAC signing for model integrity, `safetensors` as zero-risk alternative
- **LLM Prompt Injection Prevention** — Input sanitization, XML delimiters, chat template separation, guardrail classifiers, Pydantic output validation
- **FastAPI Security** — API key auth via `HTTPBearer`, rate limiting via `slowapi`, Pydantic `Field(max_length=)`, CORS, request size middleware
- **DetectGPT Correct Implementation** — Log-probability curvature: `D(x) = log p_M(x) - (1/N) * Σ log p_M(x̃_i)`, normalized z-score, T5 mask-filling perturbation model
- **Sliding Window Perplexity** — Stride-based NLL accumulation, target masking with `-100`, proper normalization by evaluated token count: `PPL = exp(L_total / |T_eval|)`
- **lru_cache Memory Leak Fix** — Use `weakref.WeakKeyDictionary` descriptor pattern or `methodtools.lru_cache` for instance methods; `functools.cached_property` for zero-arg methods
- **Thread-safe Singleton** — Lock around full initialization, not just instance check
- **Registry caching** — Cache detector instances to avoid re-loading ML models on every call

---

## 3. Audit Findings Summary

### Total Issues: 160 (15 Critical, 45 High, 67 Medium, 33 Low)

### Top 10 Critical Findings

| # | Finding | Location | Phase |
|---|---------|----------|-------|
| C-01 | RCE via `joblib.load()` pickle deserialization | `calibration.py:133` | P1 |
| C-02 | Prompt injection in LLM detectors | `llm_detectors.py:95,127,231` | P1 |
| C-03 | Label/feature misalignment in `ensemble.calibrate()` | `ensemble.py:282-292` | P1 |
| C-04 | DetectGPT fundamentally wrong implementation | `llm_detectors.py:117-181` | P2 |
| C-05 | Windowed perplexity math incorrect | `perplexity.py:41-73` | P2 |
| C-06 | Non-deterministic feature ordering breaks models | `feature_extractor.py:456-459` | P2 |
| C-07 | Shared mutable state in `CalibratedDetectorMixin` | `calibration.py:39-42` | P1 |
| C-08 | LightGBM default model trained on dummy data | `lightgbm_detector.py:49-68` | P2 |
| C-09 | XSS in heatmap HTML rendering | `heatmaps.py:59-69` | P1 |
| C-10 | `lru_cache` on instance methods = memory leak | `entropy.py:61,78`, `repetition.py:36` | P2 |

### Security Issues (3 Critical, 6 High)

- No authentication or rate limiting on API endpoints
- No input size limits — DoS via multi-GB payloads
- Container runs as root, no `USER` directive
- Secrets exposed in docker-compose environment variables
- Arbitrary model loading — `model_id` accepts any HF model ID
- SSRF via configurable LLM/Ollama hosts
- curl-pipe-sh in Dockerfile (supply chain risk)
- XSS in heatmap HTML rendering
- Error messages leak internal details

### Architecture Issues

- Split exception hierarchy — `DetectionError` not under `DetectorError`
- Singleton registry not thread-safe
- New detector instance created on every `registry.get()` — ML models reloaded
- Sentinel calls private `ensemble._determine_label()`
- Inconsistent length metrics (words vs characters)
- Silent metadata overwrite between chunks

### Test Suite Issues

- **10 source files have ZERO test coverage** (all transformer, LLM, LightGBM, cognitive detectors)
- Singleton registry mutated by 6 test files — parallel execution breaks
- Vacuous assertions (`assert len(x) >= 0` appears 3 times)
- Tests that can never fail (`except Exception: pass`)
- Permanently skipped CurvatureDetector tests
- No differential tests (AI vs human text scoring)

---

## 4. Phased Remediation Plan

### Phase 1: Critical Security & Correctness Fixes (2-3 sub-agents)

**Sub-agent 1A: Security Hardening**
Scope: API auth, rate limiting, input validation, Docker security

Files to modify:
- `provenance/api.py` — Add API key auth, rate limiting (slowapi), input size limits, CORS, request body middleware
- `Dockerfile` — Add non-root user, pin base image, remove curl-pipe-sh, add `.dockerignore`
- `docker-compose.yml` — Use Docker secrets or env_file with strict permissions
- `provenance/explainability/heatmaps.py` — HTML-escape tokens with `html.escape()`
- `provenance/detectors/llm/llm_detectors.py:95-98,127,231` — Sanitize user text before prompt interpolation; use XML delimiters; add Pydantic output validation
- `provenance/detectors/llm/llm_detectors.py:27` — Validate Ollama host against allowlist
- `provenance/detectors/transformer/hf_classifier.py:58` — Restrict `model_id` to allowlist

**Best Practice Reference (from Google AI research):**
- FastAPI security: API key via `HTTPBearer` + `secrets.compare_digest`, slowapi rate limiting, Pydantic `Field(max_length=1_000_000)`, `ContentLengthLimitMiddleware`
- Prompt injection prevention: Input sanitization (regex for injection keywords, `html.escape()`, length limits), XML delimiter isolation (`<user_input>` tags with tag-closing prevention), chat template separation (system/user roles), Pydantic output validation
- Docker: Non-root user, multi-stage build, pin image by digest

**Verification:** Run `pytest tests/test_api.py` after changes; manual test with `curl` for auth/limits

---

**Sub-agent 1B: Pickle Deserialization Security + Calibration Fix**
Scope: HMAC signing, calibration mixin fix, error hierarchy

Files to modify:
- `provenance/core/calibration.py` — Add HMAC signing for `joblib.load()`, fix shared mutable state (make `_calibrator`, `_feature_names`, `_loaded_calibration_path` instance attributes set in `__init__`), add dimension validation
- `provenance/core/errors.py` — Unify exception hierarchy: `DetectionError` should extend `DetectorError` (or both extend a common `ProvenanceError`)
- `provenance/core/base.py` — Add `score`/`confidence` range validation in `DetectorResult.__post_init__`
- `provenance/sentinel.py` — Add logging when detector fails to load, fix `IndexError` on empty chunks, remove private method access to `_determine_label`

**Best Practice Reference (from Google AI research):**
- HMAC signing pattern:
  ```python
  import hmac, hashlib, joblib, secrets
  
  HMAC_SECRET_KEY = os.environ.get("PROVENANCE_HMAC_KEY")
  
  def save_secure_joblib(model, file_path, secret_key):
      model_data = joblib.dumps(model)
      signature = hmac.new(secret_key, model_data, hashlib.sha256).digest()
      with open(file_path, "wb") as f:
          f.write(signature + model_data)
  
  def load_secure_joblib(file_path, secret_key):
      with open(file_path, "rb") as f:
          file_content = f.read()
      extracted_signature = file_content[:32]
      model_data = file_content[32:]
      expected_signature = hmac.new(secret_key, model_data, hashlib.sha256).digest()
      if not hmac.compare_digest(extracted_signature, expected_signature):
          raise PermissionError("Model tampering detected!")
      return joblib.loads(model_data)
  ```

**Verification:** Run `pytest tests/test_calibration.py tests/test_base.py`

---

**Sub-agent 1C: Ensemble Correctness Fix**
Scope: Label/feature alignment bug, metadata merge, dead code

Files to modify:
- `provenance/core/ensemble.py:282-292` — Fix `calibrate()`: when `_build_stacking_features` returns `None`, also skip the corresponding label. Track indices of skipped items.
- `provenance/core/ensemble.py:331-332` — Move `from sklearn.metrics import brier_score_loss` to top of function (not inside hot loop)
- `provenance/core/ensemble.py:372` — Replace hardcoded z-values with `scipy.stats.norm.ppf(1 - (1 - confidence_level) / 2)`
- `provenance/core/ensemble.py:326` — Add zero-division guard: `if weights.sum() == 0: weights = np.ones_like(weights) / len(weights)`
- `provenance/sentinel.py:101-103` — Deep-merge metadata instead of `dict.update()` (or prefix keys with chunk index)
- `provenance/core/ensemble.py:169-177` — Replace `isinstance` runtime type checking with explicit heatmap type contract

**Verification:** Run `pytest tests/test_ensemble.py`

---

### Phase 2: Mathematical Correctness & Memory Fixes (2-3 sub-agents)

**Sub-agent 2A: Perplexity & Surprisal Math Fixes**
Scope: Correct sliding-window PPL, fix surprisal entropy

Files to modify:
- `provenance/detectors/statistical/perplexity.py` — Rewrite `_extract_features` using proper sliding-window PPL with stride-based NLL accumulation and target masking (`-100`). Normalize by total evaluated tokens, not window count.
- `provenance/detectors/statistical/surprisal.py:126-135` — Remove `_compute_surprisal_entropy` (treating surprisal values as probabilities is invalid). Replace with meaningful statistics: coefficient of variation of surprisal, or use proper Shannon entropy of the token probability distribution.
- `provenance/detectors/statistical/surprisal.py:94-112` — Fix autocorrelation normalization (use same denominator for both variance and autocovariance)

**Best Practice Reference (from Google AI research):**
- Correct sliding-window PPL implementation:
  ```python
  total_nll = 0.0
  total_evaluated_tokens = 0
  prev_end_loc = 0
  for begin_loc in range(0, seq_len, stride):
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc
      target_ids = input_ids[:, begin_loc:end_loc].clone()
      target_ids[:, :-trg_len] = -100  # mask already-evaluated tokens
      with torch.no_grad():
          outputs = model(input_ids[:, begin_loc:end_loc])
          # compute loss only on unmasked tokens
      valid_losses = loss_per_token[shift_labels.view(-1) != -100]
      total_nll += valid_losses.sum().item()
      total_evaluated_tokens += valid_losses.numel()
      prev_end_loc = end_loc
  ppl = exp(total_nll / total_evaluated_tokens)
  ```

**Verification:** Run `pytest tests/test_statistical_detectors.py`

---

**Sub-agent 2B: DetectGPT Rewrite + LLM Detector Fixes**
Scope: Correct DetectGPT implementation, fix LLM detectors

Files to modify:
- `provenance/detectors/llm/llm_detectors.py` — Rewrite `DetectGPTDetector` to use proper log-probability curvature:
  1. Compute `log p_M(x)` using model's cross-entropy loss
  2. Generate perturbations via T5 mask-filling (not LLM prompt)
  3. Compute `log p_M(x̃_i)` for each perturbation under the same model
  4. Calculate discrepancy `D(x) = log p_M(x) - mean(log p_M(x̃_i))`
  5. Normalize: `z(x) = (log p_M(x) - μ_x̃) / σ_x̃`
  6. Classify: `z > threshold` → AI-generated
- `provenance/detectors/llm/llm_detectors.py:45-46` — Handle missing `logprobs` in ollama response gracefully
- `provenance/detectors/llm/llm_detectors.py:56` — Make normalization scale configurable, not hardcoded to `-10.0`
- `provenance/detectors/llm/llm_detectors.py:247-250` — Use robust JSON extraction (handle markdown code fences)
- `provenance/detectors/llm/llm_detectors.py:223-230` — Add fallback for missing kwargs in `LLMMetaReasoningDetector.detect()`

**Best Practice Reference (from Google AI research):**
- DetectGPT implementation with T5 mask-filling + GPT-Neo evaluation:
  ```python
  def detect_gpt(text, num_perturbations=15):
      original_ll = get_log_likelihood(text)  # using eval model
      perturbed = generate_perturbations(text, num_perturbations)  # using T5
      perturbed_lls = [get_log_likelihood(p) for p in perturbed]
      mean_p = np.mean(perturbed_lls)
      std_p = np.std(perturbed_lls) if np.std(perturbed_lls) > 1e-6 else 1e-6
      discrepancy = original_ll - mean_p
      z_score = (original_ll - mean_p) / std_p
      return {"discrepancy": discrepancy, "z_score": z_score, "is_ai": z_score > 1.5}
  ```

**Verification:** Run `pytest tests/test_integration.py`

---

**Sub-agent 2C: Memory Leak & Performance Fixes**
Scope: lru_cache fixes, registry caching, eager loading

Files to modify:
- `provenance/detectors/statistical/entropy.py:61,78` — Replace `@lru_cache` on instance methods with `weak_lru_cache` descriptor pattern (or `methodtools.lru_cache`)
- `provenance/detectors/statistical/repetition.py:36-37` — Same fix
- `provenance/core/preprocessor.py:13-17` — Use `functools.cached_property` for `_get_default_nlp` singleton
- `provenance/core/registry.py:41-42` — Cache detector instances in registry (add `_instances: dict[str, BaseDetector]` with lazy instantiation)
- `provenance/detectors/statistical/perplexity.py:30-38` — Make GPT-2 loading lazy (load on first `detect()` call, not in `__init__`)
- `provenance/detectors/statistical/entropy.py:42-59` — Make Brown corpus loading lazy (load on first use)
- `provenance/core/pytorch_utils.py:5` — Add `try/except ImportError` guard for `torch`
- `provenance/detectors/statistical/entropy.py:85-87` — Precompute `ref_probs` dict in `__init__`, not on every call
- `provenance/detectors/statistical/curvature.py:115-248` — Deduplicate computation between `_extract_features()` and `detect()`

**Best Practice Reference (from Google AI research):**
- Weak reference cache descriptor:
  ```python
  class weak_lru_cache:
      def __init__(self, maxsize=128, typed=False):
          self.maxsize = maxsize
          self.typed = typed
          self.caches = weakref.WeakKeyDictionary()
      
      def __get__(self, instance, objtype=None):
          if instance is None:
              return self
          if instance not in self.caches:
              @lru_cache(maxsize=self.maxsize, typed=self.typed)
              def _bound_cache(*args, **kwargs):
                  return self.func(instance, *args, **kwargs)
              self.caches[instance] = _bound_cache
          return self.caches[instance]
      
      def __call__(self, func):
          self.func = func
          return self
  ```

**Verification:** Run `pytest tests/ -x`

---

### Phase 3: Architecture & Code Quality (2 sub-agents)

**Sub-agent 3A: Core Architecture Fixes**
Scope: Registry thread safety, config consistency, input validation

Files to modify:
- `provenance/core/registry.py:23-30` — Fix singleton: publish instance only after full initialization inside the lock
- `provenance/core/registry.py:70-99` — Move entry point loading inside the lock
- `provenance/core/config.py:20` — Clarify `min_text_length` as word count, or change sentinel to use character count
- `provenance/core/preprocessor.py:68` — Unify length metric with config
- `provenance/core/base.py:36` — Make `domains` a frozen set or tuple, not mutable list
- `provenance/core/statistics.py:23` — Document population vs sample variance; add `ddof` parameter
- All detector `detect()` methods — Add `None`/empty string validation at the top

**Verification:** Run `pytest tests/test_registry.py tests/test_config.py tests/test_base.py`

---

**Sub-agent 3B: Domain Module Refactoring**
Scope: Extract shared patterns, remove dead code, fix inconsistencies

Files to modify:
- Create `provenance/core/text_utils.py` — Extract shared `split_sentences()`, `compute_word_statistics()`, `tokenize_words()` used by all detectors
- `provenance/domains/multilingual.py:49-51` — Remove dead `self.model = None`, `self.tokenizer = None`
- `provenance/domains/academic.py:32-33` — Remove dead `self.citation_model = None`
- `provenance/domains/code.py:12-17` — Remove dead `tree_sitter`/`Parser` import; `self.nlp = None`
- `provenance/domains/code.py:43` — Fix `_compute_ast_features` to respect `language` parameter (or implement tree-sitter for multi-language)
- All domain detectors — Extract shared `detect()` pattern (extract features → check calibration → fallback heuristic) into a template method in `CalibratedDetectorMixin`
- All domain detectors — Fix double computation: have `detect()` call `_extract_features()` instead of duplicating the work
- Unify short-text handling thresholds across all domain detectors
- Make all domain detectors use `build_error_result` from `BaseDetector`

**Verification:** Run `pytest tests/test_domains.py`

---

### Phase 4: Test Suite Overhaul (2 sub-agents)

**Sub-agent 4A: Test Infrastructure & Critical Test Gaps**
Scope: Fix test isolation, add tests for untested modules

Files to modify:
- `tests/conftest.py` — Add `registry_isolation` fixture that saves/restores registry state; add shared fixtures for long text, multilingual text, mixed text
- `tests/test_registry.py` — Fix singleton mutation: use `registry_isolation` fixture; add `>=` → `==` assertions; add duplicate registration test
- `tests/test_stylometric.py` — Fix vacuous assertions (`assert len(x) >= 0` → `assert len(x) > 0` with specific expected keys)
- `tests/test_sentinel.py` — Remove `except Exception: pass`; add actual detection test with registered detectors
- `tests/test_ensemble.py` — Fix tautological label assertions; add edge case tests (single detector, zero weights)
- Create `tests/test_llm_detectors.py` — Test prompt sanitization, output validation, missing logprobs, rate limiting
- Create `tests/test_hf_classifier.py` — Test label mapping for different model conventions, RoBERTa-specific code with model mocking
- Create `tests/test_lightgbm.py` — Test default model warning, feature dimension mismatch, SHAP optional
- Create `tests/test_cognitive.py` — Test transition word detection, advanced word list

**Verification:** Run `pytest tests/ -v --cov=provenance --cov-report=term-missing`

---

**Sub-agent 4B: Differential & Integration Tests**
Scope: Behavioral tests that verify AI text scores higher than human text

Files to create:
- `tests/test_differential.py` — For each detector, verify that known AI-generated text scores significantly higher than known human text
- `tests/test_api_security.py` — Test auth rejection, rate limiting, input size limits, XSS prevention
- Update `tests/test_integration.py` — Add full pipeline test with real statistical detectors; test calibration → detection → reporting pipeline

**Verification:** Run `pytest tests/ -v`

---

### Phase 5: Infrastructure & Documentation (1-2 sub-agents)

**Sub-agent 5A: CI/CD, Docker, Dependencies**
Scope: Fix CI pipeline, Docker hardening, pin dependencies

Files to modify:
- `pyproject.toml` — Pin minimum versions for `numpy>=1.24,<2`, `langdetect>=1.0.9`, `scipy>=1.10`; remove E501 from ruff ignore since black handles it; add `"B019"` to ruff select (catches lru_cache on instance methods)
- `.github/workflows/ci.yml` — Use `uv` instead of `pip` for consistency; add codeql/trivy scanning; pin action SHAs; add `dependabot.yml`
- `Dockerfile` — Multi-stage build, non-root user, pin base image by digest, replace curl-pipe-sh with `pip install uv`
- `docker-compose.yml` — Use `env_file` instead of inline environment; add healthcheck; add `restart: unless-stopped`
- Create `.dockerignore` — Exclude `.git`, `.venv`, `__pycache__`, `.env`, `benchmark_results/`, `*.egg-info`
- `.gitignore` — Remove duplicate sections, add `uv.lock` if needed
- `provenance/benchmarks/loaders.py:137` — Replace global `np.random.seed()` with `numpy.random.Generator` for thread safety
- `provenance/benchmarks/metrics.py:53` — Replace deprecated `np.trapz` with `np.trapezoid`

**Verification:** Run `ruff check . && mypy provenance/ && pytest tests/`

---

**Sub-agent 5B: Documentation Alignment**
Scope: Update roadmap, fix naming, document calibration

Files to modify:
- `technical_roadmap.md` — Replace all "sentinel" references with "provenance"; update directory structure; align phase numbering with actual implementation status; mark completed phases
- `docs/implementation-plan.md` — Update phase alignment with roadmap; remove MAGE performance claims if no results exist
- `provenance/calibrate.py` — Add docstrings explaining HMAC signing requirement, env vars
- `config/default_thresholds.yml` — Add comments for curvature bands `float("inf")` entry
- `provenance/benchmarks/run_comprehensive.py` — Fix docstring (RAID only, not "RAID and MAGE"); or add MAGE support

**Verification:** Manual review

---

## 5. Dependency Between Phases

```
Phase 1 (Security + Correctness) ←── MUST complete first
    │
    ├── Phase 2 (Math + Memory) ←── Can run after P1
    │       │
    │       └── Phase 3 (Architecture) ←── Can run after P2
    │               │
    │               └── Phase 4 (Tests) ←── Should run after P3
    │                       │
    │                       └── Phase 5 (Infra + Docs) ←── Can run in parallel with P4
    │
```

Phases 1-2 are sequential (security first, then correctness).
Phase 3 can start after Phase 2 completes.
Phase 4 (tests) should follow Phase 3 since test targets will change.
Phase 5 can run in parallel with Phase 4.

---

## 6. Sub-Agent Assignment Summary

| Phase | Sub-Agent | Files Modified | Estimated Scope |
|-------|-----------|---------------|-----------------|
| P1 | 1A: Security Hardening | `api.py`, `Dockerfile`, `docker-compose.yml`, `heatmaps.py`, `llm_detectors.py`, `hf_classifier.py` | ~6 files |
| P1 | 1B: Pickle Security + Errors | `calibration.py`, `errors.py`, `base.py`, `sentinel.py` | ~4 files |
| P1 | 1C: Ensemble Correctness | `ensemble.py`, `sentinel.py` | ~2 files |
| P2 | 2A: Perplexity/Surprisal Math | `perplexity.py`, `surprisal.py` | ~2 files |
| P2 | 2B: DetectGPT Rewrite | `llm_detectors.py` | ~1 file (major rewrite) |
| P2 | 2C: Memory/Performance | `entropy.py`, `repetition.py`, `preprocessor.py`, `registry.py`, `pytorch_utils.py`, `curvature.py` | ~6 files |
| P3 | 3A: Core Architecture | `registry.py`, `config.py`, `base.py`, `statistics.py`, all detectors | ~8 files |
| P3 | 3B: Domain Refactoring | New `text_utils.py`, `multilingual.py`, `academic.py`, `code.py`, `calibration.py` | ~5 files |
| P4 | 4A: Test Infrastructure | `conftest.py`, 3 existing test files, 4 new test files | ~7 files |
| P4 | 4B: Differential Tests | 2 new test files, 1 updated | ~3 files |
| P5 | 5A: CI/Infra/Dependencies | `pyproject.toml`, `ci.yml`, `Dockerfile`, `docker-compose.yml`, new `.dockerignore`, `loaders.py`, `metrics.py` | ~7 files |
| P5 | 5B: Documentation | `technical_roadmap.md`, `implementation-plan.md`, `calibrate.py`, `default_thresholds.yml` | ~4 files |

---

## 7. Research References

All Google AI research results saved to:
- `/Users/minghao/.claude/skills/google-ai-mode/results/2026-05-28_01-31-24_Secure_ML_model_serialization_Python_202.md`
- `/Users/minghao/.claude/skills/google-ai-mode/results/2026-05-28_01-31-43_LLM_prompt_injection_prevention_Python_2.md`
- `/Users/minghao/.claude/skills/google-ai-mode/results/2026-05-28_01-31-57_FastAPI_security_best_practices_2026__au.md`
- `/Users/minghao/.claude/skills/google-ai-mode/results/2026-05-28_01-32-15_DetectGPT_implementation_Python_2026__lo.md`
- `/Users/minghao/.claude/skills/google-ai-mode/results/2026-05-28_01-32-31_Python_lru_cache_memory_leak_instance_me.md`
- `/Users/minghao/.claude/skills/google-ai-mode/results/2026-05-28_01-30-49_Perplexity_calculation_sliding_window_NL.md`

---

## 8. Current Work

The audit has been completed. All findings have been consolidated. The phased remediation plan above is ready for implementation. No code changes have been made yet — this is purely a planning and research artifact.

---

## 9. Next Step

**Pick up this handoff and begin Phase 1, Sub-agent 1A (Security Hardening)** — this is the highest-priority work. Spawn a sub-agent to implement API authentication, rate limiting, input validation, Docker security, XSS fix, and prompt injection prevention across the 6 identified files. Use the FastAPI security and prompt injection best practices from the Google AI research results.

Use `/pickup 2026-05-28-provenance-audit-remediation` to continue.
