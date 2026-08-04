# Next Agent Handoff

Date: 2026-04-09

## Purpose

This note is a continuation handoff for the next agent working on the post-calibration roadmap in Provenance.

Read this together with:

- [`20260409-followup-agent-plan.md`](./20260409-followup-agent-plan.md)
- [`20260409-detection-landscape-findings.md`](./20260409-detection-landscape-findings.md)
- [`20260408-raid-calibration.md`](./20260408-raid-calibration.md)
- [`20260409-ensemble-and-transformer-followup.md`](./20260409-ensemble-and-transformer-followup.md)

## What Has Been Completed

### 1. Calibration promotion is now reproducible

A policy-driven promotion command was added to the calibration workflow.

Relevant files:

- [`../provenance/calibrate.py`](../provenance/calibrate.py)
- [`../provenance.calibrated.raid.yaml`](../provenance.calibrated.raid.yaml)
- [`../calibration_models/raid_promotion_summary.json`](../calibration_models/raid_promotion_summary.json)

Behavior now in repo:

- reads a saved calibration summary JSON
- applies explicit promotion gates
- writes a curated config file
- writes rejected models and rejection reasons to a machine-readable summary

Current RAID promotion result:

- promoted: `repetition`, `burstiness`, `surprisal_diveye`, `curvature_detectgpt`
- rejected: `entropy`

Current rejection reason for `entropy`:

- AUROC delta was negative

### 2. Low-FPR metrics are now first-class benchmark outputs

Relevant files:

- [`../provenance/benchmarks/metrics.py`](../provenance/benchmarks/metrics.py)
- [`../provenance/benchmarks/evaluator.py`](../provenance/benchmarks/evaluator.py)
- [`../provenance/benchmarks/models.py`](../provenance/benchmarks/models.py)
- [`../provenance/benchmarks/reporting.py`](../provenance/benchmarks/reporting.py)
- [`../provenance/benchmarks/evaluation.py`](../provenance/benchmarks/evaluation.py)
- [`../docs/benchmark.md`](./benchmark.md)
- [`../docs/benchmark-results.md`](./benchmark-results.md)

Added benchmark metrics:

- `TPR@1%FPR`
- `TPR@5%FPR`

These now appear in:

- markdown reports
- JSON reports
- CSV reports
- benchmark guide docs
- CLI comparison summaries

### 3. There is now a benchmarkable learned stacker comparison path

Relevant files:

- [`../provenance/benchmarks/ensemble_workflow.py`](../provenance/benchmarks/ensemble_workflow.py)
- [`../provenance/core/ensemble.py`](../provenance/core/ensemble.py)
- [`../provenance/cli.py`](../provenance/cli.py)
- [`../benchmark_results/ensemble_comparison_raid.json`](../benchmark_results/ensemble_comparison_raid.json)

What was added:

- a held-out ensemble benchmarking workflow
- a CLI command:
  `benchmark-ensemble-compare`
- comparison across:
  - `calibrated_weighted_average`
  - `uncertainty_aware_ensemble`
  - `learned_stacker`

Important implementation detail:

- the stacking path in [`../provenance/core/ensemble.py`](../provenance/core/ensemble.py) was fixed to use proper per-detector score/confidence feature vectors and to pass a single feature row to `predict_proba`

### 4. The first real held-out comparison has already been run

Command used:

```bash
uv run python -m provenance.cli benchmark-ensemble-compare \
  -d repetition \
  -d burstiness \
  -ds raid \
  -l 100 \
  --seed 43 \
  --config provenance.calibrated.raid.yaml \
  -o benchmark_results \
  -f json
```

Artifact:

- [`../benchmark_results/ensemble_comparison_raid.json`](../benchmark_results/ensemble_comparison_raid.json)

Observed result on that small RAID slice:

- `uncertainty_aware_ensemble`: AUROC `0.9722`, F1 `0.9730`, TPR@1%FPR `0.9444`
- `calibrated_weighted_average`: AUROC `0.9444`, F1 `0.9730`, TPR@1%FPR `0.8889`
- `learned_stacker`: AUROC `0.9167`, F1 `0.9189`, TPR@1%FPR `0.8889`

Interpretation:

- the learned stacker did **not** beat the current uncertainty-aware ensemble on this run
- the repo now has the benchmark table path the plan asked for
- the current evidence does not justify claiming the stacker is better yet

### 5. Ensemble comparison is now repeatable across seeds

Relevant files:

- [`../provenance/benchmarks/ensemble_workflow.py`](../provenance/benchmarks/ensemble_workflow.py)
- [`../provenance/benchmarks/reporting.py`](../provenance/benchmarks/reporting.py)
- [`../provenance/cli.py`](../provenance/cli.py)
- [`../docs/benchmark.md`](./benchmark.md)

Behavior now in repo:

- `benchmark-ensemble-compare` accepts `--repeat`
- `benchmark-ensemble-compare` accepts `--force-refresh`
- each run records its seed in result metadata
- reports include aggregate summaries across runs

This reduces the chance of over-reading one lucky or skewed sampled split.

### 6. The learned stacker feature set is slightly stronger but still interpretable

Relevant file:

- [`../provenance/core/ensemble.py`](../provenance/core/ensemble.py)

The stacker now uses:

- detector score
- detector confidence
- normalized word count
- normalized character count
- short-text indicator
- normalized average word length
- log-scaled length

This is still a small, debuggable feature vector rather than a large opaque metadata dump.

### 7. The RAID transformer baseline is now structured as reusable training helpers

Relevant files:

- [`../provenance/detectors/transformer/train_raid.py`](../provenance/detectors/transformer/train_raid.py)
- [`../tests/test_transformer_training.py`](../tests/test_transformer_training.py)

What changed:

- label normalization helper
- reusable split-preparation helper
- lightweight augmentation option
- class-weight computation
- weighted trainer loss
- evaluation metrics helper
- persisted `training_summary.json`

This still needs a real benchmark artifact, but it is no longer just a thin one-off script.

## Tests Already Added / Passing

Relevant files:

- [`../tests/test_benchmarks.py`](../tests/test_benchmarks.py)
- [`../tests/test_cli.py`](../tests/test_cli.py)
- [`../tests/test_ensemble.py`](../tests/test_ensemble.py)
- [`../tests/test_ensemble_workflow.py`](../tests/test_ensemble_workflow.py)
- [`../tests/test_transformer_training.py`](../tests/test_transformer_training.py)

Verified during this turn:

```bash
uv run pytest tests/test_benchmarks.py tests/test_config.py
uv run pytest tests/test_cli.py
uv run pytest tests/test_ensemble_workflow.py tests/test_cli.py tests/test_ensemble.py
uv run pytest tests/test_transformer_training.py tests/test_ensemble.py tests/test_ensemble_workflow.py tests/test_cli.py tests/test_benchmarks.py
```

## Important Caveats Discovered

### 1. Small sampled RAID slices can produce invalid held-out splits

The held-out ensemble benchmarking workflow now explicitly rejects splits where either:

- training data contains only one class
- held-out data contains only one class

This was necessary because tiny or skewed samples can otherwise generate meaningless AUROC / low-FPR numbers.

### 2. Cached dataset samples can preserve a bad earlier sample

The HuggingFace loader caches sampled records keyed by:

- repo
- config
- sample limit
- seed

If a bad sampled slice already exists in cache, rerunning with the same `sample_limit` and `seed` may replay it.

Practical workaround:

- rerun with a different `--seed`
- or use `--force-refresh`

### 3. The current learned comparison artifact is still small and narrow

The existing saved comparison only uses:

- RAID
- a 100-sample slice
- `repetition` + `burstiness`

This is enough to prove the workflow, but not enough to support a broad conclusion.

### 4. The new repeat support does not replace broader evaluation

Repeated seeds are better than a single run, but they still do not solve:

- dataset shift
- unseen-model shift
- prompt-style shift

The next agent should still prioritize at least one shifted benchmark such as `mage`.

## Recommended Next Steps

These are the highest-value next tasks for the next agent.

### Priority 1: Run repeated comparisons with the full calibrated detector set

Use:

- `repetition`
- `burstiness`
- `surprisal_diveye`
- `curvature_detectgpt`

Goal:

- generate a stronger held-out RAID comparison between:
  - `calibrated_weighted_average`
  - `uncertainty_aware_ensemble`
  - `learned_stacker`

Suggested command shape:

```bash
uv run python -m provenance.cli benchmark-ensemble-compare \
  -d repetition \
  -d burstiness \
  -d surprisal_diveye \
  -d curvature_detectgpt \
  -ds raid \
  -l <larger_limit> \
  --seed <seed> \
  --repeat <count> \
  --force-refresh \
  --config provenance.calibrated.raid.yaml \
  -o benchmark_results \
  -f all
```

### Priority 2: Run the same comparison on a harder or shifted dataset

If feasible from current registry/data setup, run the learned comparison on:

- `mage`

Goal:

- check whether the learned stacker helps under harder distribution shift
- avoid overfitting conclusions to RAID only

### Priority 3: Benchmark the refactored transformer baseline

There is now a cleaner baseline path in:

- [`../provenance/detectors/transformer/train_raid.py`](../provenance/detectors/transformer/train_raid.py)

The next step is to produce an actual artifact and compare it fairly against:

- `uncertainty_aware_ensemble`
- `calibrated_weighted_average`
- `learned_stacker`

### Priority 4: Decide whether the stacker should remain the main learned baseline

If repeated runs continue to show:

- no gain over `uncertainty_aware_ensemble`
- worse low-FPR performance

then the repo should not spend too long polishing the stacker.

In that case, the next major step should shift to Workstream 4 from the plan:

- a transformer-based robust learned detector baseline

The repo now has a better training scaffold, but it still is not a benchmarked robust baseline until that comparison is run.

## What Has Not Been Completed Yet

The following plan items remain open:

- larger learned stacker experiments across broader detector sets
- evaluation on harder or more shifted benchmarks
- benchmarked robust transformer detector baseline
- frontier-model refresh benchmark using newer generators

## Suggested Guardrails For The Next Agent

Before making large changes:

1. verify the calibrated config still loads:
   [`../provenance.calibrated.raid.yaml`](../provenance.calibrated.raid.yaml)
2. prefer `uv run ...` for commands
3. keep benchmark claims tied to exact datasets, detector sets, sample limits, and seeds
4. do not claim the learned stacker is better unless it clearly wins on low-FPR metrics, not just AUROC
5. do not claim the transformer baseline is better until it has a saved benchmark artifact

## Minimum Useful Continuation

If time is limited, the best next continuation is:

1. rerun `benchmark-ensemble-compare` with the full calibrated detector set on RAID
2. rerun it with `--repeat` and `--force-refresh`
3. run it again on one harder dataset if available
4. write a short comparison note summarizing whether the learned stacker actually beats the current calibrated ensemble

## Context Snapshot

Useful artifacts already present in repo:

- curated calibrated config:
  [`../provenance.calibrated.raid.yaml`](../provenance.calibrated.raid.yaml)
- promotion summary:
  [`../calibration_models/raid_promotion_summary.json`](../calibration_models/raid_promotion_summary.json)
- first learned comparison result:
  [`../benchmark_results/ensemble_comparison_raid.json`](../benchmark_results/ensemble_comparison_raid.json)
- follow-up workflow note:
  [`20260409-ensemble-and-transformer-followup.md`](./20260409-ensemble-and-transformer-followup.md)
- research guidance:
  [`20260409-detection-landscape-findings.md`](./20260409-detection-landscape-findings.md)
- execution plan:
  [`20260409-followup-agent-plan.md`](./20260409-followup-agent-plan.md)
