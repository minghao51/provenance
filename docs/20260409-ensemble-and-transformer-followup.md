# Ensemble And Transformer Follow-Up

Date: 2026-04-09

## Purpose

This note records the next round of post-calibration improvements after the initial learned-stacker benchmark path was added.

## What Changed

### 1. Ensemble comparison is now more repeatable

Relevant files:

- [`../provenance/benchmarks/ensemble_workflow.py`](../provenance/benchmarks/ensemble_workflow.py)
- [`../provenance/benchmarks/reporting.py`](../provenance/benchmarks/reporting.py)
- [`../provenance/cli.py`](../provenance/cli.py)

The ensemble comparison workflow now supports:

- repeated runs via `--repeat`
- cache-busting sampled dataset reloads via `--force-refresh`
- aggregate summaries across runs

Aggregate output currently reports:

- mean AUROC
- AUROC standard deviation
- mean F1
- mean `TPR@1%FPR`
- mean `TPR@5%FPR`
- run count

This makes it easier to compare:

- `calibrated_weighted_average`
- `uncertainty_aware_ensemble`
- `learned_stacker`

without over-interpreting a single sampled split.

### 2. The learned stacker now gets lightweight text features

Relevant file:

- [`../provenance/core/ensemble.py`](../provenance/core/ensemble.py)

The stacker feature vector now includes the existing per-detector:

- score
- confidence

plus stable text-level features:

- normalized word count
- normalized character count
- short-text indicator
- normalized average word length
- log-scaled length feature

The goal is still interpretability, not a large opaque feature dump.

### 3. The RAID transformer baseline is more benchmark-ready

Relevant files:

- [`../provenance/detectors/transformer/train_raid.py`](../provenance/detectors/transformer/train_raid.py)
- [`../tests/test_transformer_training.py`](../tests/test_transformer_training.py)

The previous RAID transformer script has been refactored into reusable helpers for:

- label normalization
- train/validation split preparation
- lightweight augmentation
- class-weight computation
- trainer evaluation metrics
- weighted classification loss

The default base model was also shifted to `distilroberta-base` for a faster first robust baseline.

Training now writes:

- `training_metrics.json`
- `training_summary.json`

so benchmark-aligned metadata is preserved with the model artifact.

## Validation

Verified locally:

```bash
uv run pytest tests/test_transformer_training.py tests/test_ensemble.py tests/test_ensemble_workflow.py tests/test_cli.py tests/test_benchmarks.py
```

## Remaining Gap

The repo now has a stronger workflow for repeated ensemble comparisons and a cleaner transformer baseline scaffold, but the larger empirical question is still open:

- does the learned stacker beat the uncertainty-aware ensemble on broader RAID runs?
- does it help on MAGE or other shifted data?
- does the transformer baseline outperform both?

Those need direct benchmark artifacts before making any stronger claim.
