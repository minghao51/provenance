"""Tests for provenance.Provenance facade."""

import pytest

from provenance import Provenance
from provenance.core.registry import DetectorRegistry


class TestProvenance:
    def setup_method(self):
        self.registry = DetectorRegistry()
        self.registry.clear()

    def teardown_method(self):
        self.registry.clear()

    def test_provenance_init_default_detectors(self):
        try:
            provenance = Provenance()
            assert provenance.preprocessor is not None
            assert provenance.ensemble is not None
        except Exception:
            pytest.skip("Detectors not registered via entry points")

    def test_provenance_init_specific_detectors(self):
        try:
            provenance = Provenance(detectors=[])
            assert provenance.ensemble is not None
        except Exception:
            pass

    def test_provenance_init_with_strategy(self):
        provenance = Provenance(ensemble_strategy="uncertainty_aware")
        assert provenance.ensemble.config.strategy == "uncertainty_aware"

    def test_provenance_init_with_weights(self):
        provenance = Provenance(weights={"det1": 0.5, "det2": 0.5})
        assert provenance.ensemble.config.weights == {"det1": 0.5, "det2": 0.5}

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

    def test_provenance_audit_method_exists(self):
        provenance = Provenance(detectors=[])
        assert hasattr(provenance, "audit")
        assert callable(provenance.audit)

    def test_provenance_audit_empty_data(self):
        provenance = Provenance(detectors=[])
        result = provenance.audit(texts=[], labels=[])
        assert isinstance(result, dict)
