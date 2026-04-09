# Benchmarking Guide

This document explains how to evaluate AI text detectors using Provenance's benchmarking framework.

## Quick Start

```bash
# List available datasets
provenance benchmark-datasets

# Compare detectors on a dataset
provenance benchmark-compare \
  -d perplexity_gpt2 \
  -d hf_chatgpt_detector \
  -d lgbm_stylometric \
  -ds raid \
  -l 100
```

## Available Datasets

| Dataset | Repo | Description | Strength |
|---------|------|-------------|----------|
| `raid` | liamdugan/raid | Adversarial AI-human pairs from 10+ generators | Best overall |
| `mage` | yaful/MAGE | AI-edited human text (mixed) | Hardest to detect |

### Dataset Details

#### RAID (Recommended)
- **Source**: [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid)
- **Size**: ~10,000+ samples
- **Generators**: GPT-J, GPT-Neo, GPT-3, davinci, llama-chat, and more
- **Domains**: prose, academic, code, news, reviews, etc.
- **Best for**: General AI text detection evaluation

#### MAGE
- **Source**: [yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE)
- **Size**: ~5,000 samples
- **Content**: Mixed AI-edited human text
- **Best for**: Detecting AI-edited content (harder task)

## Available Detectors

| Detector | Technique | Speed | Best For |
|---------|-----------|-------|----------|
| `perplexity_gpt2` | GPT-2 perplexity scoring | Medium | General detection |
| `hf_chatgpt_detector` | Fine-tuned RoBERTa | Slow | ChatGPT-specific |
| `hf_openai_detector` | RoBERTa (GPT-2 era) | Slow | GPT-2 detection |
| `hf_radar_detector` | Vicuna-7B based | Slow | Paraphrase-robust |
| `lgbm_stylometric` | LightGBM + SHAP | Fast | Fast baseline, explainable |

## Running Benchmarks

### CLI Commands

```bash
# List all available datasets
provenance benchmark-datasets

# Single detector benchmark
provenance benchmark -d perplexity_gpt2 -ds raid -l 100

# Compare multiple detectors
provenance benchmark-compare \
  -d perplexity_gpt2 \
  -d hf_chatgpt_detector \
  -d lgbm_stylometric \
  -ds raid \
  -l 100 \
  --format all

# Full evaluation (no sample limit)
provenance benchmark-compare \
  -d perplexity_gpt2 \
  -d hf_chatgpt_detector \
  -ds raid
```

### Python API

```python
from provenance.benchmarks.workflow import (
    DatasetRegistry,
    HuggingFaceDatasetLoader,
    BenchmarkEvaluator,
    BenchmarkRunner,
)

# Load dataset
loader = HuggingFaceDatasetLoader()
config = DatasetRegistry.get('raid')
texts, labels, metadata = loader.load(config, sample_limit=200)

# Evaluate single detector
from provenance.detectors.statistical.perplexity import PerplexityDetector

evaluator = BenchmarkEvaluator()
detector = PerplexityDetector()
result = evaluator.evaluate_detector(
    detector,
    texts,
    labels,
    threshold=0.5,
    dataset_name='raid'
)

print(f"AUROC: {result.auroc:.4f}")
print(f"F1: {result.f1:.4f}")
print(f"TPR@1%FPR: {result.tpr_at_1fpr:.4f}")
print(f"TPR@5%FPR: {result.tpr_at_5fpr:.4f}")
print(f"FPR@10%TPR: {result.fpr_at_10tpr:.4f}")
```

### Compare Multiple Detectors

```python
from provenance.core.registry import get_registry

registry = get_registry()
registry.load_entry_points()

detectors = [
    registry.get('perplexity_gpt2'),
    registry.get('hf_chatgpt_detector'),
]

runner = BenchmarkRunner(output_dir="benchmark_results")

for det in detectors:
    suite = runner.run_benchmark(
        detector=det,
        datasets=['raid', 'mage'],
        sample_limit=100,
        stratified=True,
    )

# Generate report
runner.generate_report(suite, output_format='all')
```

## Understanding Metrics

### Key Metrics

| Metric | Description | Perfect | Random | Good |
|--------|-------------|---------|--------|------|
| **AUROC** | Area Under ROC Curve | 1.0 | 0.5 | >0.80 |
| **F1** | Harmonic mean precision/recall | 1.0 | varies | >0.70 |
| **TPR@1%FPR** | Recall when only 1% of human text may be falsely flagged | 1.0 | often near 0.0 | higher is better |
| **TPR@5%FPR** | Recall when only 5% of human text may be falsely flagged | 1.0 | often low | higher is better |
| **FPR@10%TPR** | False Positive Rate at 90% recall | 0.0 | ~0.9 | <0.10 |
| **Precision** | Of texts labeled AI, how many truly are | 1.0 | varies | >0.80 |
| **Recall** | Of actual AI, how many caught | 1.0 | varies | >0.80 |

### Metric Interpretation

- **AUROC** (0.0-1.0): Threshold-independent measure of separation ability. 1.0 = perfect, 0.5 = random guessing.

- **TPR@1%FPR / TPR@5%FPR**: Practical deployment metrics. They answer: if we cap false accusations against human text at 1% or 5%, how much AI text do we still catch? Higher is better.

- **FPR@10%TPR**: Stress metric for aggressive operating points. At 90% recall (catching 90% of AI text), what % of human text is falsely flagged? Lower is better.

- **Precision vs Recall Trade-off**: 
  - High precision = fewer false accusations of human text
  - High recall = fewer missed AI texts
  - F1 balances both

### Why Low-FPR Metrics Matter

The detector literature is increasingly clear that AUROC and headline F1 can look strong while deployment-relevant operating points collapse. In particular, the NAACL 2025 practical evaluation cited in [20260409-detection-landscape-findings](./20260409-detection-landscape-findings.md) reports settings where `TPR@1%FPR` falls to 0 on unseen models and prompt attacks. That means a detector can look healthy in aggregate while still failing once the false-positive budget is constrained to something a real product or moderation workflow could tolerate.

For Provenance, that makes `TPR@1%FPR` and `TPR@5%FPR` first-class comparison metrics alongside AUROC and F1. Use them to separate detectors that merely rank examples well from detectors that remain usable when false accusations must stay rare.

### Example Results

```
=== RAID Dataset (50 samples) ===
PerplexityDetector:   AUROC=0.9754  F1=0.7733  TPR@1%FPR=0.7931  TPR@5%FPR=0.9310  FPR@10%TPR=0.0476
ChatGPTDetector:      AUROC=1.0000  F1=0.8163  TPR@1%FPR=0.9655  TPR@5%FPR=1.0000  FPR@10%TPR=0.0000

=== MAGE Dataset (AI-edited, 50 samples) ===
PerplexityDetector:   AUROC=0.4057  F1=0.7532  TPR@1%FPR=0.0000  TPR@5%FPR=0.0690  FPR@10%TPR=1.0000
ChatGPTDetector:      AUROC=0.5395  F1=0.4528  TPR@1%FPR=0.0345  TPR@5%FPR=0.1034  FPR@10%TPR=0.7500
```

**Note**: MAGE is harder because AI-edited text is partially human.

## Adding Custom Datasets

```python
from provenance.benchmarks.workflow import DatasetConfig, DatasetRegistry

# Register a custom dataset
DatasetRegistry.register(DatasetConfig(
    name="my_dataset",
    repo_id="path/to/my/dataset",
    split="train",
    text_field="text",           # field containing the text
    label_field="label",          # field containing the label
    label_map={"human": 0, "ai": 1},  # map string labels to 0/1
    meta_fields={"domain": "category"},  # for stratified evaluation
))
```

## Stratified Evaluation

Run evaluation broken down by metadata categories:

```bash
provenance benchmark-compare \
  -d perplexity_gpt2 \
  -ds raid \
  -l 200 \
  --stratify
```

This produces additional metrics for:
- Text length (<150 words = short text, known to degrade performance)
- Domain/category (if available in metadata)
- Language (for multilingual datasets)

## Output Formats

The `--format` flag produces:

- `markdown`: Human-readable report (`.md`)
- `json`: Machine-parseable results (`.json`)
- `csv`: Tabular format for analysis (`.csv`)
- `all`: All three formats

## Performance Tips

1. **Use sample limits for quick iteration**: `-l 50` or `-l 100`
2. **Skip slow detectors for quick tests**: Only use `perplexity_gpt2` and `lgbm_stylometric`
3. **Cache datasets**: HuggingFace caches automatically; first load may be slow
4. **GPU acceleration**: Transformer classifiers (RoBERTa) run faster on CUDA

## Benchmarking New Detectors

To benchmark a new detector:

```python
from provenance.core.base import BaseDetector, DetectorResult

class MyDetector(BaseDetector):
    name = "my_detector"
    latency_tier = "fast"  # fast (<50ms), medium (<2s), slow (>2s)
    domains = ["prose", "academic"]

    def detect(self, text: str) -> DetectorResult:
        # Your detection logic
        score = compute_ai_probability(text)
        return DetectorResult(
            score=score,
            confidence=0.8,
            metadata={}
        )

# Register and use
from provenance.core.registry import get_registry
registry = get_registry()
registry.register(MyDetector)

# Now available in CLI
# provenance benchmark -d my_detector -ds raid
```

## Limitations

- **Dataset availability**: Many AI detection datasets are private or require authentication
- **Class imbalance**: Datasets often have more AI than human text
- **Temporal drift**: Detectors trained on GPT-2 era text may not detect GPT-4
- **Short text**: Most detectors fail on texts <150 words
- **Humanizers**: Tools like Undetectable.ai can defeat statistical detectors

## References

- [RAID Benchmark Paper](https://arxiv.org/abs/XXXX)
- [HC3 Dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- [MAGE Dataset](https://huggingface.co/datasets/yaful/MAGE)
- [DetectGPT Paper](https://arxiv.org/abs/2301.11305)
