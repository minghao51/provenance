"""Tests for CLI commands."""

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


class TestCliAudit:
    def setup_method(self):
        self.runner = CliRunner()

    def test_audit_help(self):
        result = self.runner.invoke(main, ["audit", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output
