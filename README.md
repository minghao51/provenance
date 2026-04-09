# Provenance AI

A modular, explainable, composable library for detecting AI-generated text. Built for data scientists and developers — not just end-users. Prioritizes transparency, extensibility, and domain-awareness over black-box accuracy.

## Installation

```bash
pip install provenance
```

## Quick Start

```python
from provenance import Provenance

provenance = Provenance(detectors=["perplexity_gpt2", "lgbm_stylometric"])
result = provenance.detect("Your text here...")

print(f"Score: {result.score}")  # 0.0 (human) → 1.0 (AI)
print(f"Label: {result.label}")  # "human", "ai", "mixed", or "uncertain"
```

## Calibration

RAID-calibrated detector artifacts and usage notes are documented in [20260408-raid-calibration](docs/20260408-raid-calibration.md).

## Architecture

```
Input Text
    │
    ▼
Preprocessor Layer (tokenization, sentence split, language detection, chunking)
    │
    ▼
Detector Registry (statistical, stylometric, transformer, LLM scorers)
    │
    ▼
Ensemble Layer (weighted voting, calibration, confidence intervals)
    │
    ▼
Explainability Layer (token-level heatmap, feature importance, sentence scores)
```

## License

MIT
