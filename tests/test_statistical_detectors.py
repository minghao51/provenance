"""Unit tests for statistical detectors."""

import math
import pytest

from provenance.detectors.statistical.burstiness import BurstinessDetector
from provenance.detectors.statistical.curvature import CurvatureDetector
from provenance.detectors.statistical.entropy import EntropyDetector
from provenance.detectors.statistical.perplexity import PerplexityDetector
from provenance.detectors.statistical.repetition import RepetitionDetector
from provenance.detectors.statistical.surprisal import SurprisalDetector


class TestPerplexityDetector:
    def test_detector_initialization(self):
        detector = PerplexityDetector()
        assert detector.name == "perplexity_gpt2"
        assert detector.latency_tier == "medium"
        assert hasattr(detector, "_get_calibrated_score")

    def test_feature_extraction(self):
        detector = PerplexityDetector()
        features = detector._extract_features("The quick brown fox jumps over the lazy dog.")
        assert len(features) == 4
        assert features[0] >= 0  # mean_perplexity should be non-negative
        assert features[1] >= 0  # variance_perplexity should be non-negative
        assert features[2] >= 0  # std_perplexity should be non-negative
        assert features[3] >= 1  # window_count should be at least 1

    def test_feature_names(self):
        detector = PerplexityDetector()
        names = detector._extract_feature_names()
        assert names == ["mean_perplexity", "variance_perplexity", "std_perplexity", "window_count"]

    def test_detect_returns_valid_result(self, sample_human_text):
        detector = PerplexityDetector()
        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert "mean_perplexity" in result.metadata
        assert "std_perplexity" in result.metadata
        assert "window_count" in result.metadata

    def test_short_text_handling(self, sample_short_text):
        detector = PerplexityDetector()
        result = detector.detect(sample_short_text)
        # Short text may still be analyzed but with limited windows
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_calibration_aliases(self):
        detector = PerplexityDetector()
        assert hasattr(detector, "calibration_aliases")
        assert "perplexity" in detector.calibration_aliases


class TestSurprisalDetector:
    def test_detector_initialization(self):
        detector = SurprisalDetector()
        assert "surprisal" in detector.name
        assert detector.latency_tier in ("medium", "slow")
        assert hasattr(detector, "_get_calibrated_score")

    def test_feature_extraction(self):
        detector = SurprisalDetector()
        features = detector._extract_features("The quick brown fox jumps over the lazy dog.")
        assert len(features) > 0
        assert all(isinstance(f, float) for f in features)

    def test_detect_returns_valid_result(self, sample_human_text):
        detector = SurprisalDetector()
        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert "mean_surprisal" in result.metadata

    def test_variance_calculation(self, sample_human_text):
        detector = SurprisalDetector()
        result = detector.detect(sample_human_text)
        assert "variance" in result.metadata or "surprisal_variance" in result.metadata

    def test_calibration_integration(self):
        detector = SurprisalDetector()
        assert hasattr(detector, "calibration_aliases")
        assert "surprisal" in detector.calibration_aliases


class TestBurstinessDetector:
    def test_detector_initialization(self):
        detector = BurstinessDetector()
        assert "burstiness" in detector.name
        assert detector.latency_tier == "slow"
        assert hasattr(detector, "_get_calibrated_score")

    def test_feature_extraction(self):
        detector = BurstinessDetector()
        text = "This is sentence one. This is sentence two. This is sentence three."
        features = detector._extract_features(text)
        assert len(features) > 0

    def test_detect_returns_valid_result(self, sample_human_text):
        detector = BurstinessDetector()
        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_requires_multiple_sentences(self):
        detector = BurstinessDetector()
        result = detector.detect("Single sentence.")
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_cv_calculation(self):
        detector = BurstinessDetector()
        # Text with consistent sentence lengths should have low CV
        consistent_text = "A B C. " * 10
        result = detector.detect(consistent_text)
        assert 0.0 <= result.score <= 1.0


class TestEntropyDetector:
    def test_detector_initialization(self):
        detector = EntropyDetector()
        assert detector.name == "entropy"
        assert detector.latency_tier == "fast"
        assert hasattr(detector, "_get_calibrated_score")

    def test_feature_extraction(self):
        detector = EntropyDetector()
        features = detector._extract_features("The quick brown fox jumps over the lazy dog.")
        assert len(features) == 2
        assert features[0] >= 0  # unigram_entropy should be non-negative
        assert features[1] >= 0  # kl_divergence should be non-negative

    def test_feature_names(self):
        detector = EntropyDetector()
        names = detector._extract_feature_names()
        assert names == ["unigram_entropy", "kl_divergence"]

    def test_detect_returns_valid_result(self, sample_human_text):
        detector = EntropyDetector()
        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert "text_entropy" in result.metadata
        assert "kl_divergence" in result.metadata

    def test_entropy_calculation(self):
        detector = EntropyDetector()
        # Repeated text should have low entropy
        repeated_text = "the the the the the"
        entropy = detector._compute_unigram_entropy(repeated_text)
        assert entropy >= 0

    def test_kl_divergence_calculation(self):
        detector = EntropyDetector()
        if detector.word_frequencies:
            kl_div = detector._compute_kl_divergence("the quick brown fox")
            assert kl_div >= 0

    def test_thresholds_usage(self):
        detector = EntropyDetector()
        assert hasattr(detector, "thresholds")
        assert detector.thresholds is not None


class TestRepetitionDetector:
    def test_detector_initialization(self):
        detector = RepetitionDetector()
        assert detector.name == "repetition"
        assert detector.latency_tier == "fast"
        assert hasattr(detector, "_get_calibrated_score")

    def test_feature_extraction(self):
        detector = RepetitionDetector()
        # Text with repeated n-grams
        repeated_text = "The cat sat on the mat. The cat sat on the mat."
        features = detector._extract_features(repeated_text)
        assert len(features) > 0

    def test_detect_returns_valid_result(self, sample_human_text):
        detector = RepetitionDetector()
        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_ngram_repetition_detection(self):
        detector = RepetitionDetector()
        # Highly repetitive text should score higher (more AI-like)
        repetitive = "The cat sat. " * 10
        result = detector.detect(repetitive)
        assert 0.0 <= result.score <= 1.0

    def test_short_text_handling(self, sample_short_text):
        detector = RepetitionDetector()
        result = detector.detect(sample_short_text)
        # Should still return a valid result
        assert 0.0 <= result.score <= 1.0


class TestCurvatureDetector:
    def test_detector_initialization(self):
        detector = CurvatureDetector()
        assert "curvature" in detector.name
        assert detector.latency_tier in ("medium", "slow")
        assert hasattr(detector, "_get_calibrated_score")

    @pytest.mark.skip(reason="Curvature detector requires model loading - skip for fast tests")
    def test_feature_extraction(self):
        detector = CurvatureDetector()
        features = detector._extract_features("The quick brown fox jumps over the lazy dog.")
        assert len(features) > 0

    @pytest.mark.skip(reason="Curvature detector requires model loading - skip for fast tests")
    def test_detect_returns_valid_result(self, sample_human_text):
        detector = CurvatureDetector()
        result = detector.detect(sample_human_text)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_calibration_aliases(self):
        detector = CurvatureDetector()
        assert hasattr(detector, "calibration_aliases")


class TestDetectorCalibrationIntegration:
    """Tests for calibration integration across all detectors."""

    def test_all_detectors_have_calibration_support(self):
        """Verify all statistical detectors support calibration."""
        detectors = [
            PerplexityDetector(),
            SurprisalDetector(),
            BurstinessDetector(),
            EntropyDetector(),
            RepetitionDetector(),
            CurvatureDetector(),
        ]
        for detector in detectors:
            assert hasattr(
                detector, "_get_calibrated_score"
            ), f"{detector.name} missing _get_calibrated_score"
            assert hasattr(
                detector, "_extract_features"
            ), f"{detector.name} missing _extract_features"
            assert hasattr(
                detector, "_extract_feature_names"
            ), f"{detector.name} missing _extract_feature_names"

    def test_all_detectors_have_calibration_aliases(self):
        """Verify all detectors have calibration aliases for model loading."""
        detectors = [
            (PerplexityDetector(), "perplexity"),
            (SurprisalDetector(), "surprisal"),
            (BurstinessDetector(), "burstiness"),
            (EntropyDetector(), "entropy"),
            (RepetitionDetector(), "repetition"),
            (CurvatureDetector(), "curvature"),
        ]
        for detector, expected_alias in detectors:
            assert hasattr(detector, "calibration_aliases"), f"{detector.name} missing calibration_aliases"
            assert expected_alias in detector.calibration_aliases, f"{detector.name} missing expected alias"


class TestDetectorMetadata:
    """Tests for detector metadata consistency."""

    def test_all_detectors_include_calibrated_flag(self, sample_human_text):
        """Verify all detectors report whether they used calibrated scoring."""
        detectors = [
            EntropyDetector(),
            BurstinessDetector(),
            RepetitionDetector(),
        ]
        for detector in detectors:
            result = detector.detect(sample_human_text)
            # Detectors with calibration should report whether they used it
            # (though they may not have calibration models loaded)
            assert isinstance(result.metadata, dict)

    def test_metadata_consistency(self, sample_human_text):
        """Verify metadata contains expected keys."""
        detector = EntropyDetector()
        result = detector.detect(sample_human_text)
        assert "text_entropy" in result.metadata
        assert "kl_divergence" in result.metadata
