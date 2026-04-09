"""Benchmarks module for evaluating AI text detectors."""

from provenance.benchmarks.evaluation import (
    BenchmarkHarness,
    run_audit,
)
from provenance.benchmarks.models import BenchmarkResult
from provenance.benchmarks.workflow import (
    BenchmarkEvaluator,
    BenchmarkRunner,
    BenchmarkSuite,
    DatasetConfig,
    DatasetRegistry,
    HuggingFaceDatasetLoader,
)

__all__ = [
    "BenchmarkHarness",
    "BenchmarkResult",
    "run_audit",
    "BenchmarkEvaluator",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "DatasetConfig",
    "DatasetRegistry",
    "HuggingFaceDatasetLoader",
]
