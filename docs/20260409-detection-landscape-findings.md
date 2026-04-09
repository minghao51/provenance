# Detection Landscape Findings

Date: 2026-04-09

## Purpose

This note captures the current evidence-backed assessment of where Provenance stands after the RAID calibration work, whether ensemble methods are likely to be the best path forward, and how well the current findings are expected to transfer to newer frontier models.

## Executive Summary

The current calibrated statistical ensemble is a meaningful improvement over the previous heuristic-only setup, but it should not be treated as the likely end-state or state-of-the-art solution.

The literature as of 2024-2025 points to two consistent conclusions:

1. Robust AI-text detection remains brittle under distribution shift, adversarial rewriting, and unseen generators.
2. The strongest recent results are increasingly coming from supervised, robustly trained transformer-based detectors and detector frameworks that explicitly optimize for cross-domain and unseen-model generalization.

The practical implication for this repository is:

- Keep the calibrated statistical detectors.
- Treat them as a strong feature layer and fallback layer.
- Move the main roadmap toward a learned detector or learned stacker trained and validated on harder benchmarks.

## Current Repo Position

Recent local work produced a curated RAID-calibrated config:

- [`provenance.calibrated.raid.yaml`](../provenance.calibrated.raid.yaml)

Selected calibrated detectors:

- `repetition`
- `burstiness`
- `surprisal_diveye`
- `curvature_detectgpt`

Excluded detector:

- `entropy`

RAID summary artifact:

- [`calibration_models/raid_calibration_summary.json`](../calibration_models/raid_calibration_summary.json)

These results are real improvements on held-out RAID splits for several detectors, but they do not by themselves establish that the current ensemble will generalize best to newer frontier models.

## Evidence From Recent Literature

### 1. Robustness remains the main failure mode

RAID (ACL 2024) evaluates detectors across a much harder benchmark than many older evaluations. Its abstract reports that detectors are easily fooled by:

- adversarial attacks
- decoding strategy changes
- repetition penalties
- unseen generative models

This is highly relevant to Provenance because the current calibration work also used RAID, which means the local benchmark choice is aligned with a strong contemporary robustness benchmark rather than a softer in-distribution test.

### 2. Popular detectors still struggle in practical deployment settings

The NAACL 2025 paper "A Practical Examination of AI-Generated Text Detectors for Large Language Models" evaluates several widely used detectors on unseen models, domains, and prompt attacks. The authors report that both trained and zero-shot detectors can perform very poorly at low false-positive operating points, including TPR@1% FPR dropping to 0% in some settings.

This matters because AUROC and F1 alone can hide operational weakness. Provenance should continue reporting AUROC and F1, but should also treat low-FPR metrics as first-class evaluation targets.

### 3. The frontier is shifting toward robust supervised detectors

Several 2025 papers suggest that better results are coming from models trained with:

- diverse generator coverage
- cross-domain training
- augmentation
- adversarial examples
- active learning or hard-example mining
- explicit robustness objectives

Examples:

- AIDER (COLING 2025) emphasizes topic transfer and adversarial robustness.
- Leidos at GenAI Detection Task 3 (COLING Workshop 2025) uses weight-balanced transformer training on RAID and evaluates on additional unseen-model data.
- Pangram at GenAI Detection Task 3 (COLING Workshop 2025) combines broad pretraining with RAID hard-example mining and reports the top score on the adversarial portion of the RAID leaderboard.
- OpenTuringBench (EMNLP 2025) introduces a benchmark and detector oriented around open-model texts, out-of-domain texts, manipulated texts, and unseen models.

The common pattern is not "ensembles always win." It is "robust learned detectors win more often when trained on broader and harder data."

## Should Ensemble Methods Perform the Best?

Short answer: not necessarily.

### Where ensembles help

Ensembles are still valuable because they:

- combine complementary signals
- reduce dependence on a single brittle heuristic
- provide interpretable detector breakdowns
- offer a natural place to plug in calibration

For Provenance specifically, the statistical ensemble remains useful because it gives:

- transparent features
- fast fallbacks
- partial resilience when one detector fails

### Where ensembles are likely insufficient

A simple weighted-average or heuristic ensemble is unlikely to be the strongest approach on its own against:

- paraphrase attacks
- new generator families
- topic shift
- prompt-style shift
- distribution changes in contemporary frontier models

The more likely high-performing design is:

- a learned robust detector as the main classifier
- plus the current calibrated statistical detectors as input features or auxiliary signals

That means the most promising "ensemble" direction is not a hand-tuned ensemble. It is a learned stacker or hybrid system.

## Are the Current Findings Applicable to Current SOTA Models?

Only partially.

### What likely still transfers

The following conclusions are likely stable:

- calibration can materially improve some detectors
- some detectors should be excluded if calibration does not improve them
- RAID-style evaluation is more trustworthy than narrow in-domain evaluation
- robustness to unseen generators is a central requirement

### What does not safely transfer without new evaluation

The measured before/after numbers from the local RAID calibration should not be assumed to hold for:

- current GPT-4-class systems
- current Claude-class systems
- current Gemini-class systems
- newer open models not represented in the local benchmark slice

This is an inference from the cited literature, especially RAID and the NAACL 2025 practical evaluation. The papers do not test every current frontier model, but they repeatedly show that unseen-model generalization is one of the main failure points of existing detectors.

Therefore:

- the direction of the local findings is useful
- the absolute performance numbers are not portable without refresh benchmarking

## Implications For Provenance

### Recommended product position

Provenance should present the current calibrated ensemble as:

- a materially improved baseline
- a transparent and explainable system
- not yet a demonstrated SOTA detector for current frontier models

### Recommended technical direction

The next serious performance step should be one of:

1. a learned stacker over calibrated detector outputs and metadata
2. a transformer-based detector trained on RAID-style data with augmentation
3. a hybrid model that uses both learned text representations and the current detector feature set

## Concrete Next Steps

Recommended next steps, in order:

1. Add low-FPR metrics to the benchmark workflow.
2. Add a promotion rule so calibrated models are only adopted when they improve a chosen validation objective.
3. Benchmark a learned transformer detector against the current calibrated ensemble on RAID and MAGE.
4. Build a refresh benchmark set with texts from newer frontier and open models.
5. Evaluate a learned stacker that consumes the calibrated statistical detector outputs.

## Sources

- RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors
  [ACL Anthology](https://aclanthology.org/2024.acl-long.674/)
- A Practical Examination of AI-Generated Text Detectors for Large Language Models
  [ACL Anthology](https://aclanthology.org/2025.findings-naacl.271/)
- AIDER: a Robust and Topic-Independent Framework for Detecting AI-Generated Text
  [ACL Anthology](https://aclanthology.org/2025.coling-main.625/)
- Leidos at GenAI Detection Task 3: A Weight-Balanced Transformer Approach for AI Generated Text Detection Across Domains
  [ACL Anthology PDF](https://aclanthology.org/2025.genaidetect-1.39.pdf)
- Pangram at GenAI Detection Task 3: An Active Learning Approach to Machine-Generated Text Detection
  [ACL Anthology](https://aclanthology.org/2025.genaidetect-1.40/)
- OpenTuringBench: An Open-Model-based Benchmark and Framework for Machine-Generated Text Detection and Attribution
  [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1354/)
