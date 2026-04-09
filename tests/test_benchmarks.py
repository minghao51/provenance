"""Tests for benchmark harness and evaluation."""

import json

import pytest
from click.testing import CliRunner

from provenance.benchmarks import (
    BenchmarkResult as PublicBenchmarkResult,
)
from provenance.benchmarks import (
    BenchmarkRunner as PublicBenchmarkRunner,
)
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
from provenance.calibrate import cli as calibrate_cli


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

    def test_registry_bootstrap_is_idempotent(self):
        from provenance.benchmarks.registry import register_default_datasets

        baseline = DatasetRegistry.available_datasets()
        register_default_datasets()
        register_default_datasets()

        assert DatasetRegistry.available_datasets().keys() == baseline.keys()
        assert len(DatasetRegistry.list_datasets()) == len(set(DatasetRegistry.list_datasets()))


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
        assert 0.0 <= metrics["tpr_at_1fpr"] <= 1.0
        assert 0.0 <= metrics["tpr_at_5fpr"] <= 1.0
        assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}

    def test_compute_tpr_at_fpr(self):
        tpr = self.evaluator.compute_tpr_at_fpr(
            y_true=[0, 0, 1, 1],
            y_score=[0.1, 0.2, 0.8, 0.9],
            target_fpr=0.05,
        )
        assert tpr == 1.0


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

    def test_benchmark_result_backward_compatible_defaults(self):
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
        assert result.num_positives == 0
        assert result.num_negatives == 0
        assert result.eval_time_seconds == 0.0
        assert result.tpr_at_1fpr == 0.0
        assert result.tpr_at_5fpr == 0.0


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
        assert "TPR@1%FPR" in report

    def test_public_imports_stay_stable(self):
        assert PublicBenchmarkRunner is not None
        assert PublicBenchmarkResult is BenchmarkResult

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
        assert "TPR@1%FPR" in markdown
        assert "### test on raid" in markdown
        assert "Confusion Matrix" in markdown

    def test_load_previous_results(self, tmp_path):
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
            num_positives=50,
            num_negatives=50,
            eval_time_seconds=1.5,
            metadata={"confusion_matrix": {"tp": 1, "tn": 2, "fp": 3, "fn": 4}},
            stratified_results={"overall": {"accuracy": 0.81}},
        )
        payload = {
            "name": "saved_suite",
            "created_at": "2024-01-01T00:00:00",
            "config": {"datasets": ["hc3"]},
            "results": [result.__dict__],
        }
        path = tmp_path / "suite.json"
        path.write_text(json.dumps(payload))

        loaded = self.runner.load_previous_results(str(path))

        assert loaded is not None
        assert loaded.name == "saved_suite"
        assert loaded.results[0].detector_name == "test"
        assert loaded.results[0].metadata["confusion_matrix"]["tp"] == 1

    def test_generate_json_report_includes_low_fpr_metrics(self):
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
            metadata={},
            tpr_at_1fpr=0.42,
            tpr_at_5fpr=0.73,
        )
        suite = BenchmarkSuite(name="test_suite", results=[result])

        json_str = self.runner._generate_json(suite)

        assert '"tpr_at_1fpr": 0.42' in json_str
        assert '"tpr_at_5fpr": 0.73' in json_str


class TestCalibrationPromotion:
    def test_promote_writes_curated_config_and_rejections(self, tmp_path, monkeypatch):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "dataset": "raid",
                    "selected_models": {
                        "repetition": "calibration_models/repetition_raid.pkl",
                        "entropy": "calibration_models/entropy_raid.pkl",
                    },
                    "results": {
                        "repetition": {
                            "before": {"auroc": 0.60, "f1": 0.40},
                            "after": {"auroc": 0.75, "f1": 0.45},
                            "delta": {"auroc": 0.15, "f1": 0.05},
                        },
                        "entropy": {
                            "before": {"auroc": 0.60, "f1": 0.90},
                            "after": {"auroc": 0.58, "f1": 0.85},
                            "delta": {"auroc": -0.02, "f1": -0.05},
                        },
                    },
                }
            )
        )

        class DummyDetector:
            def __init__(self, aliases):
                self.calibration_aliases = aliases

        def fake_import_detector(name, autoload_calibration=True):
            alias_map = {
                "repetition": ("repetition",),
                "entropy": ("entropy",),
            }
            return DummyDetector(alias_map[name])

        monkeypatch.setattr("provenance.calibrate._import_detector", fake_import_detector)

        output_config = tmp_path / "provenance.calibrated.raid.yaml"
        output_summary = tmp_path / "promotion_summary.json"
        runner = CliRunner()

        result = runner.invoke(
            calibrate_cli,
            [
                "promote",
                "--summary-path",
                str(summary_path),
                "--output-config",
                str(output_config),
                "--output-summary",
                str(output_summary),
                "--min-auroc-improvement",
                "0.01",
                "--max-f1-regression",
                "0.0",
            ],
        )

        assert result.exit_code == 0
        config_text = output_config.read_text()
        assert "detector_calibration_paths" in config_text
        assert "repetition: calibration_models/repetition_raid.pkl" in config_text
        assert "entropy:" not in config_text

        promotion_summary = json.loads(output_summary.read_text())
        assert promotion_summary["selected_models"] == {
            "repetition": "calibration_models/repetition_raid.pkl"
        }
        assert "entropy" in promotion_summary["rejected_models"]
        assert promotion_summary["rejected_models"]["entropy"]["reasons"]


class TestRunAudit:
    def test_run_audit_empty(self):
        result = run_audit(detector=None, texts=[], labels=[])
        assert isinstance(result, dict)
