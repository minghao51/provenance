"""Integration tests for provenance workflows."""

from pathlib import Path

from click.testing import CliRunner
import pytest

from provenance import Provenance
from provenance.cli import main
from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.config import ProvenanceConfig
from provenance.core.registry import get_registry
from provenance.core.ensemble import Ensemble, EnsembleConfig
from provenance.detectors.statistical.burstiness import BurstinessDetector
from provenance.detectors.statistical.entropy import EntropyDetector
from provenance.detectors.statistical.repetition import RepetitionDetector


class StableDetector(BaseDetector):
    name = "stable_detector"
    latency_tier = "fast"
    domains = ["prose"]

    def detect(self, text: str) -> DetectorResult:
        return DetectorResult(score=0.8, confidence=0.9)


class ExplodingDetector(BaseDetector):
    name = "exploding_detector"
    latency_tier = "fast"
    domains = ["prose"]

    def detect(self, text: str) -> DetectorResult:
        raise RuntimeError("unexpected failure")


class CalibratedPathDetector(BaseDetector):
    name = "calibrated_path_detector"
    latency_tier = "fast"
    domains = ["prose"]

    def __init__(self):
        self.loaded_path = None

    def load_calibration(self, path: str):
        self.loaded_path = path

    def detect(self, text: str) -> DetectorResult:
        return DetectorResult(score=0.4, confidence=0.8)


class TestProvenanceIntegration:
    def setup_method(self):
        self.registry = get_registry()
        self.registry.clear()
        self.registry.register(StableDetector)
        self.registry.register(ExplodingDetector)
        self.registry.register(CalibratedPathDetector)

    def teardown_method(self):
        self.registry.clear()

    def test_provenance_survives_detector_failures(self):
        provenance = Provenance(
            detectors=["stable_detector", "exploding_detector"],
            config=ProvenanceConfig(min_text_length=1),
        )

        result = provenance.detect(
            "This is a deliberately long enough sentence to exercise detector execution."
        )

        assert result.score > 0.0
        assert result.detector_scores["stable_detector"].score == 0.8
        assert result.detector_scores["exploding_detector"].metadata["error"] == (
            "Detector execution failed"
        )

    def test_cli_detect_accepts_config_file(self, tmp_path: Path):
        runner = CliRunner()
        config_path = tmp_path / "config.json"
        config_path.write_text('{"provenance": {"min_text_length": 1}}')

        result = runner.invoke(
            main,
            [
                "detect",
                "This sentence is long enough for the configured minimum.",
                "--detectors",
                "stable_detector",
                "--config",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        assert "stable_detector" in result.output

    def test_provenance_loads_explicit_detector_calibration_path(self):
        provenance = Provenance(
            detectors=["calibrated_path_detector"],
            config=ProvenanceConfig(
                min_text_length=1,
                detector_calibration_paths={
                    "calibrated_path_detector": "calibration_models/custom.pkl"
                },
            ),
        )

        detector = provenance.ensemble.detectors[0]
        assert detector.loaded_path == "calibration_models/custom.pkl"


class TestEnsembleIntegration:
    """Tests for ensemble integration with statistical detectors."""

    def test_ensemble_with_statistical_detectors(self):
        """Test that ensemble correctly combines multiple statistical detectors."""
        from provenance.core.ensemble import EnsembleConfig

        config = EnsembleConfig(strategy="weighted_average")
        ensemble = Ensemble(config=config)

        detectors = [
            EntropyDetector(),
            BurstinessDetector(),
            RepetitionDetector(),
        ]
        for detector in detectors:
            ensemble.add_detector(detector)

        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the alphabet."
        )
        result = ensemble.ensemble_detect(text)

        assert result.score >= 0.0
        assert result.score <= 1.0
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    def test_ensemble_all_strategies(self):
        """Test ensemble with different combining strategies."""
        from provenance.core.ensemble import EnsembleConfig

        detectors = [EntropyDetector(), RepetitionDetector()]
        text = "Sample text for ensemble testing with multiple strategies."

        for strategy in ["weighted_average", "stacking", "uncertainty_aware"]:
            config = EnsembleConfig(strategy=strategy)
            ensemble = Ensemble(config=config)
            for detector in detectors:
                ensemble.add_detector(detector)

            result = ensemble.ensemble_detect(text)
            assert 0.0 <= result.score <= 1.0, f"Failed for strategy: {strategy}"
            assert 0.0 <= result.confidence <= 1.0


class TestCalibrationWorkflow:
    """Tests for calibration training and evaluation workflow."""

    def test_detector_has_calibration_methods(self):
        """Test that calibratable detectors have calibration methods."""
        detector = EntropyDetector()

        assert hasattr(detector, "calibrate")
        assert callable(detector.calibrate)
        assert hasattr(detector, "save_calibration")
        assert hasattr(detector, "load_calibration")

    def test_calibration_feature_extraction(self):
        """Test that all calibratable detectors can extract features."""
        detectors = [
            EntropyDetector(),
            BurstinessDetector(),
            RepetitionDetector(),
        ]
        text = "Sample text for feature extraction testing."

        for detector in detectors:
            features = detector._extract_features(text)
            assert isinstance(features, list)
            assert all(isinstance(f, float) for f in features)
            assert len(features) > 0

            names = detector._extract_feature_names()
            assert isinstance(names, list)
            assert all(isinstance(n, str) for n in names)
            assert len(names) > 0


class TestConfigurationIntegration:
    """Tests for configuration system integration."""

    def test_entropy_detector_uses_thresholds(self):
        """Test that EntropyDetector uses its threshold configuration."""
        from provenance.core.config import EntropyThresholds

        custom_thresholds = EntropyThresholds(
            kl_div_high=3.0,
            kl_div_high_score=0.9,
        )

        detector = EntropyDetector(thresholds=custom_thresholds)

        assert detector.thresholds.kl_div_high == 3.0
        assert detector.thresholds.kl_div_high_score == 0.9

    def test_detector_default_thresholds(self):
        """Test that detectors have reasonable default thresholds."""
        detector = EntropyDetector()

        assert detector.thresholds is not None
        assert detector.thresholds.kl_div_high > 0
        assert 0.0 <= detector.thresholds.kl_div_high_score <= 1.0

    def test_config_from_dict(self):
        """Test loading configuration from dictionary."""
        from provenance.core.config import EntropyThresholds

        config = {
            "kl_div_high": 2.5,
            "kl_div_high_score": 0.85,
        }

        thresholds = EntropyThresholds(**config)

        assert thresholds.kl_div_high == 2.5
        assert thresholds.kl_div_high_score == 0.85


class TestCrossDetectorConsistency:
    """Tests for consistency across different detectors."""

    def test_all_detectors_accept_text_input(self):
        """Test that all detectors can accept text input."""
        detectors = [
            EntropyDetector(),
            BurstinessDetector(),
            RepetitionDetector(),
        ]

        text = "This is a sample text for testing."

        for detector in detectors:
            result = detector.detect(text)
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    def test_score_range_consistency(self):
        """Test that detectors produce scores in consistent ranges."""
        detectors = [
            EntropyDetector(),
            BurstinessDetector(),
            RepetitionDetector(),
        ]

        text = (
            "The utilization of canids in agricultural contexts represents a historical "
            "practice predating modern industrialization."
        )

        for detector in detectors:
            result = detector.detect(text)
            assert 0.0 <= result.score <= 1.0, f"{detector.name} produced out-of-range score"
            assert 0.0 <= result.confidence <= 1.0

    def test_edge_case_handling_consistency(self):
        """Test that multiple detectors handle edge cases consistently."""
        edge_cases = ["", "a", "Hi!"]

        for text in edge_cases:
            for detector in [EntropyDetector(), RepetitionDetector()]:
                result = detector.detect(text)
                # All results should be valid even for edge cases
                assert 0.0 <= result.score <= 1.0
                assert 0.0 <= result.confidence <= 1.0


class TestProvenanceFacade:
    """Tests for the main Provenance facade."""

    def test_provenance_with_statistical_detectors(self):
        """Test Provenance facade with statistical detectors."""
        provenance = Provenance()

        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a sample text for testing the Provenance facade."
        )

        result = provenance.detect(text)

        assert result is not None
        assert hasattr(result, "score")
        assert hasattr(result, "label")
        assert 0.0 <= result.score <= 1.0
