"""Integration tests for provenance workflows."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from provenance import Provenance
from provenance.cli import main
from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.config import ProvenanceConfig
from provenance.core.ensemble import Ensemble, EnsembleConfig
from provenance.core.registry import get_registry
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


class TestCalibrationDetectionPipeline:
    """Full pipeline: calibration → detection → reporting."""

    def test_pipeline_with_entropy_detector(self, sample_human_text):
        detector = EntropyDetector()
        features = detector._extract_features(sample_human_text)
        assert len(features) == 2
        assert all(isinstance(f, float) for f in features)

        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert "text_entropy" in result.metadata
        assert "kl_divergence" in result.metadata
        assert "calibrated" in result.metadata

    def test_pipeline_with_repetition_detector(self, sample_human_text):
        detector = RepetitionDetector()
        features = detector._extract_features(sample_human_text)
        assert len(features) > 0

        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_pipeline_with_burstiness_detector(self, sample_human_text):
        detector = BurstinessDetector()
        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert "burstiness_cv" in result.metadata

    def test_pipeline_feature_extraction_then_detection(self, sample_ai_text):
        detector = EntropyDetector()
        features = detector._extract_features(sample_ai_text)
        names = detector._extract_feature_names()
        assert len(features) == len(names)

        result = detector.detect(sample_ai_text)
        assert "text_entropy" in result.metadata
        assert "kl_divergence" in result.metadata

    def test_pipeline_all_detectors_produce_consistent_results(self, sample_human_text):
        detectors = [EntropyDetector(), RepetitionDetector(), BurstinessDetector()]
        for detector in detectors:
            result = detector.detect(sample_human_text)
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.metadata, dict)


class TestEnsembleFullPipeline:
    """Test ensemble with multiple detectors end-to-end."""

    def test_ensemble_combines_entropy_and_repetition(self):
        config = EnsembleConfig(strategy="weighted_average")
        ensemble = Ensemble(config=config)
        ensemble.add_detector(EntropyDetector())
        ensemble.add_detector(RepetitionDetector())

        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the alphabet. "
            "Scientists have studied this phenomenon extensively."
        )
        result = ensemble.ensemble_detect(text)

        assert 0.0 <= result.score <= 1.0
        assert result.label in ("human", "ai", "mixed", "uncertain")
        assert "entropy" in result.detector_scores
        assert "repetition" in result.detector_scores

    def test_ensemble_three_detectors_weighted_average(self):
        config = EnsembleConfig(strategy="weighted_average")
        ensemble = Ensemble(config=config)
        ensemble.add_detector(EntropyDetector())
        ensemble.add_detector(RepetitionDetector())
        ensemble.add_detector(BurstinessDetector())

        text = (
            "Artificial intelligence has transformed many industries. "
            "Machine learning models can generate human-like text. "
            "Detection systems help identify AI-generated content. "
            "These tools are becoming increasingly important."
        )
        result = ensemble.ensemble_detect(text)

        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.detector_scores) == 3

    def test_ensemble_uncertainty_aware_with_statistical(self):
        config = EnsembleConfig(strategy="uncertainty_aware", confidence_threshold=0.5)
        ensemble = Ensemble(config=config)
        ensemble.add_detector(EntropyDetector())
        ensemble.add_detector(RepetitionDetector())

        text = (
            "The implementation of neural networks requires careful tuning. "
            "Hyperparameter optimization is crucial for model performance."
        )
        result = ensemble.ensemble_detect(text)

        assert 0.0 <= result.score <= 1.0
        assert result.label in ("human", "ai", "mixed", "uncertain")

    def test_ensemble_detector_failure_isolation(self):
        from provenance.core.base import BaseDetector, DetectorResult

        class FailingStatistical(BaseDetector):
            name = "failing_stat"
            latency_tier = "fast"
            domains = ["prose"]

            def detect(self, text: str) -> DetectorResult:
                raise RuntimeError("statistical detector crash")

        config = EnsembleConfig(strategy="weighted_average")
        ensemble = Ensemble(config=config)
        ensemble.add_detector(EntropyDetector())
        ensemble.add_detector(FailingStatistical())

        text = "A reasonably long sentence for ensemble testing purposes."
        result = ensemble.ensemble_detect(text)

        assert 0.0 <= result.score <= 1.0
        assert "failing_stat" in result.detector_scores
        assert "error" in result.detector_scores["failing_stat"].metadata


class TestSentinelEndToEnd:
    """End-to-end tests: Provenance facade with registered statistical detectors."""

    def setup_method(self):
        self.registry = get_registry()
        self.registry.clear()

    def teardown_method(self):
        self.registry.clear()

    def test_sentinel_with_explicit_statistical_detectors(self):
        provenance = Provenance(
            detectors=["entropy", "repetition"],
            config=ProvenanceConfig(min_text_length=1),
        )
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Scientists have studied animal behavior for decades. "
            "Recent advances in machine learning have enabled new discoveries."
        )
        result = provenance.detect(text)

        assert hasattr(result, "score")
        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert 0.0 <= result.score <= 1.0
        assert result.label in ("human", "ai", "mixed", "uncertain")

    def test_sentinel_heatmap_populated(self):
        provenance = Provenance(
            detectors=["entropy"],
            config=ProvenanceConfig(min_text_length=1),
        )
        long_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a longer text that should produce heatmap data. "
            "Multiple sentences help ensure proper processing. "
            "The detection system analyzes each token individually. "
            "Heatmaps provide visual explanations of the results. "
        ) * 3
        result = provenance.detect(long_text)

        assert isinstance(result.heatmap, list)

    def test_sentinel_sentence_scores_populated(self):
        provenance = Provenance(
            detectors=["entropy"],
            config=ProvenanceConfig(min_text_length=1),
        )
        text = (
            "First sentence about something interesting. "
            "Second sentence continues the thought with more detail. "
            "Third sentence provides additional context and information. "
            "Fourth sentence wraps up the paragraph nicely. "
            "Fifth sentence adds one more thought. "
        ) * 5
        result = provenance.detect(text)

        assert isinstance(result.sentence_scores, list)

    def test_sentinel_short_text_returns_uncertain(self):
        provenance = Provenance(
            detectors=["entropy"],
            config=ProvenanceConfig(min_text_length=150),
        )
        result = provenance.detect("This is short.")
        assert result.label == "uncertain"
        assert result.confidence <= 0.5

    def test_sentinel_audit_with_statistical(self):
        provenance = Provenance(
            detectors=["entropy"],
            config=ProvenanceConfig(min_text_length=1),
        )
        texts = [
            "Human written text with varied structure and colloquial language.",
            "The implementation demonstrates optimal configurations for parameters.",
        ]
        labels = [0, 1]
        result = provenance.audit(texts=texts, labels=labels)
        assert isinstance(result, dict)
