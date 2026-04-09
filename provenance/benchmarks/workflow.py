"""Compatibility facade for benchmark workflow APIs."""

from provenance.benchmarks.evaluator import BenchmarkEvaluator
from provenance.benchmarks.loaders import HuggingFaceDatasetLoader
from provenance.benchmarks.models import BenchmarkResult, BenchmarkSuite, DatasetConfig
from provenance.benchmarks.registry import (
    DatasetRegistry,
    register_default_datasets,
)
from provenance.benchmarks.runner import BenchmarkRunner


def _register_datasets() -> None:
    register_default_datasets()


_register_datasets()

__all__ = [
    "BenchmarkEvaluator",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSuite",
    "DatasetConfig",
    "DatasetRegistry",
    "HuggingFaceDatasetLoader",
]
