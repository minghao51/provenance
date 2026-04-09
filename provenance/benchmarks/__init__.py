"""Benchmarks module for evaluating AI text detectors."""

from provenance.benchmarks.evaluation import (
    BenchmarkHarness,
    run_audit,
)
from provenance.benchmarks.ensemble_workflow import (
    EnsembleBenchmarkDetector,
    benchmark_ensemble_strategies,
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
    "EnsembleBenchmarkDetector",
    "benchmark_ensemble_strategies",
    "run_audit",
    "BenchmarkEvaluator",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "DatasetConfig",
    "DatasetRegistry",
    "HuggingFaceDatasetLoader",
]
