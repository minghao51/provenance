"""Tests for benchmark harness and evaluation."""

import pytest

from provenance.benchmarks.evaluation import (
    BenchmarkHarness,
    BenchmarkResult,
    run_audit,
)
from provenance.benchmarks.workflow import (
    BenchmarkEvaluator,
    BenchmarkRunner,
    BenchmarkSuite,
    DatasetConfig,
    DatasetRegistry,
)


class TestDatasetRegistry:
    def test_register_and_get(self):
        config = DatasetConfig(
            name="test_ds",
            repo_id="test/repo",
            split="train",
        )
        DatasetRegistry.register(config)
        assert DatasetRegistry.get("test_ds") is not None
        assert DatasetRegistry.get("test_ds").repo_id == "test/repo"

    def test_list_datasets(self):
        datasets = DatasetRegistry.list_datasets()
        assert len(datasets) > 0
        assert "raid" in datasets

    def test_predefined_datasets(self):
        for name in ["raid", "mage"]:
            config = DatasetRegistry.get(name)
            assert config is not None
            assert config.repo_id is not None


class TestBenchmarkEvaluator:
    def setup_method(self):
        self.evaluator = BenchmarkEvaluator()

    def test_compute_auroc_perfect(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        auroc = self.evaluator.compute_auroc(y_true, y_score)
        assert auroc == 1.0

    def test_compute_auroc_random(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.5, 0.5, 0.5, 0.5]
        auroc = self.evaluator.compute_auroc(y_true, y_score)
        assert auroc == 0.5

    def test_compute_fpr_at_tpr(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.2, 0.8, 0.9]
        fpr = self.evaluator.compute_fpr_at_tpr(y_true, y_score, 0.9)
        assert 0.0 <= fpr <= 1.0

    def test_compute_confusion_matrix(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        cm = self.evaluator.compute_confusion_matrix(y_true, y_pred)
        assert cm["tn"] == 1
        assert cm["fp"] == 1
        assert cm["fn"] == 0
        assert cm["tp"] == 2

    def test_compute_metrics(self):
        metrics = self.evaluator.compute_metrics(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            y_score=[0.1, 0.7, 0.8, 0.9],
        )
        assert metrics["accuracy"] == 0.75
        assert metrics["precision"] == pytest.approx(2 / 3)
        assert metrics["recall"] == 1.0
        assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


class TestBenchmarkResult:
    def test_benchmark_result_creation(self):
        result = BenchmarkResult(
            detector_name="test",
            dataset="hc3",
            auroc=0.85,
            f1=0.80,
            fpr_at_10tpr=0.15,
            precision=0.82,
            recall=0.78,
            accuracy=0.81,
            num_samples=100,
            metadata={},
        )
        assert result.detector_name == "test"
        assert result.auroc == 0.85


class TestBenchmarkHarness:
    def setup_method(self):
        self.harness = BenchmarkHarness(sample_limit=10)

    def test_init(self):
        assert self.harness.sample_limit == 10

    def test_load_unknown_dataset(self):
        with pytest.raises(ValueError):
            self.harness.load_dataset("unknown_dataset")

    def test_generate_report(self):
        result = BenchmarkResult(
            detector_name="test_detector",
            dataset="hc3",
            auroc=0.85,
            f1=0.80,
            fpr_at_10tpr=0.15,
            precision=0.82,
            recall=0.78,
            accuracy=0.81,
            num_samples=100,
            metadata={},
        )
        report = self.harness.generate_report([result])
        assert "test_detector" in report
        assert "0.8500" in report

    def test_audit_fpr_empty(self):
        results = self.harness.audit_fpr([], [])
        assert results == {}

    def test_audit_fpr_with_demographics(self):
        texts = ["text1", "text2", "text3"]
        labels = [0, 0, 1]
        demographics = ["native", "non-native", "native"]
        results = self.harness.audit_fpr(texts, labels, demographics)
        assert isinstance(results, dict)


class TestBenchmarkRunner:
    def setup_method(self):
        self.runner = BenchmarkRunner(output_dir="test_output")

    def test_init(self):
        assert self.runner.output_dir.name == "test_output"

    def test_generate_json_report(self):
        from provenance.benchmarks.workflow import BenchmarkResult as WorkflowResult

        result = WorkflowResult(
            detector_name="test",
            dataset="hc3",
            auroc=0.85,
            f1=0.80,
            fpr_at_10tpr=0.15,
            precision=0.82,
            recall=0.78,
            accuracy=0.81,
            num_samples=100,
            num_positives=50,
            num_negatives=50,
            eval_time_seconds=1.5,
            metadata={},
        )
        suite = BenchmarkSuite(name="test_suite", results=[result])
        json_str = self.runner._generate_json(suite)
        assert '"detector_name": "test"' in json_str
        assert '"auroc": 0.85' in json_str

    def test_generate_markdown_report_includes_detector_column(self):
        from provenance.benchmarks.workflow import BenchmarkResult as WorkflowResult

        result = WorkflowResult(
            detector_name="test",
            dataset="raid",
            auroc=0.85,
            f1=0.80,
            fpr_at_10tpr=0.15,
            precision=0.82,
            recall=0.78,
            accuracy=0.81,
            num_samples=100,
            num_positives=50,
            num_negatives=50,
            eval_time_seconds=1.5,
            metadata={"confusion_matrix": {"tp": 40, "tn": 41, "fp": 9, "fn": 10}},
        )
        suite = BenchmarkSuite(
            name="test_suite",
            results=[result],
            config={"detectors": ["test"], "datasets": ["raid"]},
        )
        markdown = self.runner._generate_markdown(suite)
        assert "| Detector | Dataset |" in markdown
        assert "### test on raid" in markdown
        assert "Confusion Matrix" in markdown


class TestRunAudit:
    def test_run_audit_empty(self):
        result = run_audit(detector=None, texts=[], labels=[])
        assert isinstance(result, dict)
