"""Tests for ensemble benchmarking workflow."""

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.registry import get_registry


class LowSignalDetector(BaseDetector):
    name = "low_signal_detector"
    latency_tier = "fast"
    domains = ["prose"]

    def detect(self, text: str) -> DetectorResult:
        score = 0.25 if "human" in text else 0.75
        confidence = 0.55 if "maybe" in text else 0.8
        return DetectorResult(score=score, confidence=confidence)


class ConfidenceDetector(BaseDetector):
    name = "confidence_detector"
    latency_tier = "fast"
    domains = ["prose"]

    def detect(self, text: str) -> DetectorResult:
        score = 0.35 if "human" in text else 0.85
        confidence = 0.3 if "noisy" in text else 0.9
        return DetectorResult(score=score, confidence=confidence)


class TestEnsembleBenchmarkWorkflow:
    def setup_method(self):
        registry = get_registry()
        registry.clear()
        registry.register(LowSignalDetector)
        registry.register(ConfidenceDetector)

    def teardown_method(self):
        get_registry().clear()

    def test_stratified_train_test_split_preserves_alignment(self):
        from provenance.benchmarks.ensemble_workflow import stratified_train_test_split

        split = stratified_train_test_split(
            ["human a", "ai a", "human b", "ai b"],
            [0, 1, 0, 1],
            test_size=0.5,
            seed=7,
        )

        assert len(split.train_texts) == len(split.train_labels)
        assert len(split.test_texts) == len(split.test_labels)
        assert sorted(split.train_labels + split.test_labels) == [0, 0, 1, 1]

    def test_benchmark_ensemble_strategies_returns_comparison_suite(self, monkeypatch):
        from provenance.benchmarks.ensemble_workflow import benchmark_ensemble_strategies

        class DummyLoader:
            def load(self, config, sample_limit=None, seed=42):
                texts = [
                    "human baseline sample",
                    "ai generated sample",
                    "human maybe noisy sample",
                    "ai maybe noisy sample",
                    "human reference text",
                    "ai reference text",
                    "human final text",
                    "ai final text",
                ]
                labels = [0, 1, 0, 1, 0, 1, 0, 1]
                return texts, labels, [{} for _ in texts]

        monkeypatch.setattr(
            "provenance.benchmarks.evaluator.HuggingFaceDatasetLoader",
            lambda cache_dir=None: DummyLoader(),
        )

        suite = benchmark_ensemble_strategies(
            detector_names=["low_signal_detector", "confidence_detector"],
            dataset_name="raid",
            sample_limit=8,
            test_size=0.25,
        )

        assert len(suite.results) == 3
        assert {result.detector_name for result in suite.results} == {
            "calibrated_weighted_average",
            "uncertainty_aware_ensemble",
            "learned_stacker",
        }
        assert all(0.0 <= result.tpr_at_1fpr <= 1.0 for result in suite.results)

    def test_benchmark_ensemble_strategies_rejects_single_class_holdout(self, monkeypatch):
        import pytest

        from provenance.benchmarks.ensemble_workflow import benchmark_ensemble_strategies

        class DummyLoader:
            def load(self, config, sample_limit=None, seed=42):
                texts = [
                    "human ordered 1",
                    "human ordered 2",
                    "human ordered 3",
                    "human ordered 4",
                    "ai ordered 1",
                ]
                labels = [0, 0, 0, 0, 1]
                return texts, labels, [{} for _ in texts]

        monkeypatch.setattr(
            "provenance.benchmarks.evaluator.HuggingFaceDatasetLoader",
            lambda cache_dir=None: DummyLoader(),
        )

        with pytest.raises(ValueError, match="must contain both classes"):
            benchmark_ensemble_strategies(
                detector_names=["low_signal_detector", "confidence_detector"],
                dataset_name="raid",
                sample_limit=5,
                test_size=0.2,
            )
