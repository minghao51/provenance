"""Tests for provenance.Provenance facade."""

from unittest.mock import MagicMock, patch

import pytest

from provenance import Provenance
from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.registry import DetectorRegistry


class StubDetector(BaseDetector):
    name = "stub_detector"
    latency_tier = "fast"
    domains = ["prose"]

    def detect(self, text: str) -> DetectorResult:
        return DetectorResult(score=0.7, confidence=0.8)


class TestProvenance:
    def setup_method(self):
        self.registry = DetectorRegistry()
        self.registry.clear()

    def teardown_method(self):
        self.registry.clear()

    def test_provenance_init_default_detectors(self):
        provenance = Provenance()
        assert provenance.preprocessor is not None
        assert provenance.ensemble is not None

    def test_provenance_init_empty_detectors(self):
        provenance = Provenance(detectors=[])
        assert provenance.ensemble is not None
        assert len(provenance.ensemble.detectors) == 0

    def test_provenance_init_with_strategy(self):
        provenance = Provenance(ensemble_strategy="uncertainty_aware")
        assert provenance.ensemble.config.strategy == "uncertainty_aware"

    def test_provenance_init_with_weights(self):
        provenance = Provenance(weights={"det1": 0.5, "det2": 0.5})
        assert provenance.ensemble.config.weights == {"det1": 0.5, "det2": 0.5}

    def test_provenance_min_text_length_constant_preserved(self):
        assert Provenance.MIN_TEXT_LENGTH == 150

    def test_provenance_detect_short_text(self):
        provenance = Provenance(detectors=[])
        result = provenance.detect("Short text.")
        assert result.label == "uncertain"
        assert result.confidence <= 0.5

    def test_provenance_detect_returns_sentinel_result(self):
        from provenance.core.base import SentinelResult

        provenance = Provenance(detectors=[])
        result = provenance.detect(
            "This is a much longer piece of text that should be processed correctly by the provenance system. "
            * 5
        )
        assert isinstance(result, SentinelResult)
        assert hasattr(result, "score")
        assert hasattr(result, "label")
        assert hasattr(result, "confidence")
        assert hasattr(result, "detector_scores")
        assert hasattr(result, "heatmap")
        assert hasattr(result, "sentence_scores")

    def test_provenance_detect_long_text_chunks(self):
        provenance = Provenance(detectors=[])
        long_text = "This is a test sentence. " * 100
        result = provenance.detect(long_text)
        assert result.score is not None
        assert 0.0 <= result.score <= 1.0

    def test_provenance_detect_with_registered_detector(self):
        self.registry.register(StubDetector)
        provenance = Provenance(detectors=["stub_detector"])
        assert len(provenance.ensemble.detectors) == 1
        words = ["word"] * 200
        long_text = " ".join(words)
        result = provenance.detect(long_text)
        assert result.score == pytest.approx(0.7)
        assert "stub_detector" in result.detector_scores

    def test_provenance_audit_method_exists(self):
        provenance = Provenance(detectors=[])
        assert hasattr(provenance, "audit")
        assert callable(provenance.audit)

    def test_provenance_audit_empty_data(self):
        provenance = Provenance(detectors=[])
        result = provenance.audit(texts=[], labels=[])
        assert isinstance(result, dict)
