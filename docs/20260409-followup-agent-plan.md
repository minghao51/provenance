# Follow-Up Agent Plan

Date: 2026-04-09

## Goal

Continue the post-calibration work from the current repository state without repeating completed work, and move Provenance from a calibrated statistical baseline toward a stronger, better-documented, and better-evaluated detection system.

## Current State

Completed inputs available to the next agent:

- RAID-calibrated config:
  [`provenance.calibrated.raid.yaml`](../provenance.calibrated.raid.yaml)
- Calibration summary:
  [`calibration_models/raid_calibration_summary.json`](../calibration_models/raid_calibration_summary.json)
- RAID calibration note:
  [`docs/20260408-raid-calibration.md`](./20260408-raid-calibration.md)
- Research findings note:
  [`docs/20260409-detection-landscape-findings.md`](./20260409-detection-landscape-findings.md)

Important context:

- `repetition`, `burstiness`, `surprisal_diveye`, and `curvature_detectgpt` have curated RAID calibration artifacts.
- `entropy` was intentionally excluded because its calibration did not improve held-out performance.
- Config loading now supports explicit per-detector calibration paths.
- Calibration jobs were fixed to avoid auto-loading prior artifacts during training.

## Primary Objectives

1. Make calibration promotion reproducible and policy-driven.
2. Improve benchmark quality to reflect practical detector deployment.
3. Evaluate whether a learned stacker or learned detector beats the current calibrated ensemble.
4. Extend evaluation toward unseen and newer model families.

## Workstream 1: Calibration Promotion Rules

### Objective

Replace manual selection of calibrated artifacts with a deterministic promotion rule.

### Tasks

1. Add a script that reads calibration outputs and materializes a curated config file.
2. Encode an explicit promotion policy such as:
   - require AUROC improvement above a configurable threshold
   - require no unacceptable regression in F1
   - optionally require better low-FPR performance
3. Store rejected models and rejection reasons in a machine-readable summary.
4. Add tests for:
   - selected models
   - rejected models
   - output config structure

### Suggested files

- `provenance/calibrate.py`
- `provenance/core/config.py`
- new script or helper under `provenance/`
- new tests under `tests/`

### Done when

- A single command can regenerate the curated calibration config from saved evaluation outputs.

## Workstream 2: Benchmark Metrics and Reporting

### Objective

Improve benchmark outputs so they reflect practical deployment constraints.

### Tasks

1. Add low-FPR metrics as first-class benchmark outputs.
2. Surface metrics such as:
   - TPR@1% FPR
   - TPR@5% FPR
   - precision at target recall if available
3. Update markdown and JSON reporting to include these metrics.
4. Document why these metrics matter, referencing the practical detector evaluation literature.

### Suggested files

- benchmark evaluation and reporting modules under `provenance/benchmarks/`
- relevant tests in `tests/test_benchmarks.py`
- docs under `docs/`

### Done when

- Benchmark reports can distinguish models that look good on AUROC but fail at practical operating points.

## Workstream 3: Learned Stacker Over Current Detectors

### Objective

Test whether a learned stacker improves materially over the current calibrated weighted ensemble.

### Tasks

1. Build a training path for a stacker that consumes:
   - calibrated detector scores
   - detector confidences
   - selected metadata features if stable
2. Compare:
   - current weighted ensemble
   - uncertainty-aware ensemble
   - learned stacker
3. Evaluate on RAID and at least one harder or more distribution-shifted dataset.
4. Keep the output interpretable enough for debugging and explainability.

### Risks

- Overfitting to a single benchmark.
- Hidden leakage if calibration and stacking use the same validation slice improperly.

### Done when

- There is a clean benchmark table showing whether the stacker is actually better than the current calibrated ensemble.

## Workstream 4: Strong Learned Detector Baseline

### Objective

Introduce a transformer-based detection baseline trained for robustness, not just in-distribution accuracy.

### Tasks

1. Pick one initial architecture:
   - DistilRoBERTa-style baseline for speed
   - larger encoder only if justified by evaluation
2. Train on RAID-style data with:
   - class balancing
   - augmentation
   - domain diversity
3. Evaluate on:
   - RAID held-out split
   - MAGE if feasible
   - unseen-model or cross-domain slice if available
4. Compare directly against the calibrated ensemble and learned stacker.

### External guidance

Use the 2025 papers in the findings doc as inspiration, especially the weight-balanced and active-learning approaches.

### Done when

- Provenance has a robust learned baseline that can be compared fairly against the current detector stack.

## Workstream 5: Frontier-Model Refresh Evaluation

### Objective

Check whether current conclusions hold on newer model families.

### Tasks

1. Define a small refresh benchmark using contemporary models.
2. Include both:
   - frontier closed models
   - newer open models
3. Keep prompts and domains broad enough to expose distribution shift.
4. Run the current calibrated ensemble and any new learned baselines on that set.
5. Document which conclusions transfer and which do not.

### Constraints

- Avoid claiming transfer to current SOTA models without direct evaluation.
- Use exact model names and dates in documentation.

### Done when

- There is a written refresh report saying whether the RAID-calibrated setup still holds up on newer generators.

## Recommended Order

1. Calibration promotion rule
2. Benchmark metric upgrades
3. Learned stacker experiment
4. Learned robust detector baseline
5. Frontier-model refresh benchmark

## Suggested Handoff Checks

Before making large changes, the next agent should verify:

1. The curated RAID config still loads and runs end-to-end.
2. Calibration tests and integration tests still pass.
3. Benchmark modules are not already being refactored elsewhere in the worktree.

## Deliverables Expected From The Next Agent

Minimum acceptable continuation:

- reproducible calibration promotion
- improved benchmark metrics
- one benchmark comparison between current calibrated ensemble and a learned alternative

Preferred continuation:

- all of the above plus a refresh benchmark on newer model outputs
