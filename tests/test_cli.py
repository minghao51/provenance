"""Tests for CLI commands."""

from dataclasses import dataclass, field

from click.testing import CliRunner

from provenance.cli import main


class TestCliDetect:
    def setup_method(self):
        self.runner = CliRunner()

    def test_detect_with_text(self):
        result = self.runner.invoke(
            main,
            [
                "detect",
                "This is a test sentence that should be long enough to process properly by the detection system.",
            ],
        )
        assert result.exit_code == 0

    def test_detect_json_output(self):
        result = self.runner.invoke(
            main,
            [
                "detect",
                "This is a test sentence that should be long enough to process properly by the detection system.",
                "--output",
                "json",
            ],
        )
        assert result.exit_code == 0
        assert "score" in result.output
        assert "label" in result.output

    def test_detect_no_text_error(self):
        result = self.runner.invoke(main, ["detect"])
        assert result.exit_code != 0

    def test_detect_with_ensemble_strategy(self):
        result = self.runner.invoke(
            main,
            [
                "detect",
                "This is a test sentence that should be long enough to process properly by the detection system.",
                "--ensemble",
                "uncertainty_aware",
            ],
        )
        assert result.exit_code == 0


class TestCliListDetectors:
    def setup_method(self):
        self.runner = CliRunner()

    def test_list_detectors(self):
        result = self.runner.invoke(main, ["list-detectors"])
        assert result.exit_code == 0


class TestCliServe:
    def setup_method(self):
        self.runner = CliRunner()

    def test_serve_help(self):
        result = self.runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "host" in result.output
        assert "port" in result.output


class TestCliTrain:
    def setup_method(self):
        self.runner = CliRunner()

    def test_train_help(self):
        result = self.runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output


class TestCliBenchmark:
    def setup_method(self):
        self.runner = CliRunner()

    def test_benchmark_help(self):
        result = self.runner.invoke(main, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "detector" in result.output

    def test_benchmark_compare_summary_includes_low_fpr_metrics(self, monkeypatch):
        @dataclass
        class DummyResult:
            detector_name: str = "demo_detector"
            dataset: str = "raid"
            auroc: float = 0.9
            f1: float = 0.8
            fpr_at_10tpr: float = 0.2
            precision: float = 0.85
            recall: float = 0.76
            accuracy: float = 0.81
            num_samples: int = 10
            metadata: dict = field(default_factory=dict)
            num_positives: int = 5
            num_negatives: int = 5
            eval_time_seconds: float = 0.1
            stratified_results: dict | None = None
            tpr_at_1fpr: float = 0.33
            tpr_at_5fpr: float = 0.66

        @dataclass
        class DummySuite:
            name: str = "suite"
            results: list = field(default_factory=lambda: [DummyResult()])
            config: dict = field(default_factory=dict)

        class DummyDetector:
            def __init__(self, name: str):
                self.name = name

        class DummyRegistry:
            def load_entry_points(self):
                return None

            def get(self, name: str):
                return DummyDetector(name)

        class DummyRunner:
            def __init__(self, output_dir=None):
                self.output_dir = output_dir

            def run_benchmark(
                self,
                detector,
                datasets,
                sample_limit=None,
                stratified=True,
                show_progress=False,
            ):
                return DummySuite()

            def generate_report(self, suite, output_format="all"):
                return "report"

        monkeypatch.setattr("provenance.core.registry.get_registry", lambda: DummyRegistry())
        monkeypatch.setattr(
            "provenance.benchmarks.workflow.BenchmarkRunner", DummyRunner
        )

        result = self.runner.invoke(
            main,
            ["benchmark-compare", "-d", "demo_detector", "-ds", "raid"],
        )

        assert result.exit_code == 0
        assert "TPR@1%FPR=0.3300" in result.output
        assert "TPR@5%FPR=0.6600" in result.output

    def test_benchmark_ensemble_compare_summary(self, monkeypatch):
        @dataclass
        class DummyResult:
            detector_name: str
            dataset: str = "raid"
            auroc: float = 0.91
            f1: float = 0.81
            fpr_at_10tpr: float = 0.2
            precision: float = 0.8
            recall: float = 0.82
            accuracy: float = 0.84
            num_samples: int = 10
            metadata: dict = field(default_factory=dict)
            num_positives: int = 5
            num_negatives: int = 5
            eval_time_seconds: float = 0.2
            stratified_results: dict | None = None
            tpr_at_1fpr: float = 0.41
            tpr_at_5fpr: float = 0.72

        @dataclass
        class DummySuite:
            name: str = "suite"
            results: list = field(
                default_factory=lambda: [
                    DummyResult(detector_name="calibrated_weighted_average"),
                    DummyResult(detector_name="learned_stacker"),
                ]
            )
            config: dict = field(default_factory=dict)

        class DummyRunner:
            def __init__(self, output_dir=None):
                self.output_dir = output_dir

            def generate_report(self, suite, output_format="all"):
                return "report"

        monkeypatch.setattr(
            "provenance.benchmarks.ensemble_workflow.benchmark_ensemble_strategies",
            lambda **kwargs: DummySuite(),
        )
        monkeypatch.setattr(
            "provenance.benchmarks.workflow.BenchmarkRunner", DummyRunner
        )

        result = self.runner.invoke(
            main,
            ["benchmark-ensemble-compare", "-d", "demo_detector", "-ds", "raid"],
        )

        assert result.exit_code == 0
        assert "Held-out ensemble comparison" in result.output
        assert "learned_stacker" in result.output
        assert "TPR@1%FPR=0.4100" in result.output

    def test_benchmark_ensemble_compare_reports_split_errors(self, monkeypatch):
        monkeypatch.setattr(
            "provenance.benchmarks.ensemble_workflow.benchmark_ensemble_strategies",
            lambda **kwargs: (_ for _ in ()).throw(
                ValueError("Held-out split must contain both classes")
            ),
        )

        result = self.runner.invoke(
            main,
            ["benchmark-ensemble-compare", "-d", "demo_detector", "-ds", "raid"],
        )

        assert result.exit_code != 0
        assert "Benchmark setup error" in result.output


class TestCliAudit:
    def setup_method(self):
        self.runner = CliRunner()

    def test_audit_help(self):
        result = self.runner.invoke(main, ["audit", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output
