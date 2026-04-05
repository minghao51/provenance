"""Tests for detector calibration helpers."""

import pytest

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import SKLEARN_AVAILABLE, CalibratedDetectorMixin


class DummyCalibratedDetector(CalibratedDetectorMixin, BaseDetector):
    name = "dummy_calibrated"
    latency_tier = "fast"
    domains = ["prose"]

    def _extract_features(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("!"))]

    def _extract_feature_names(self) -> list[str]:
        return ["length", "exclamation_count"]

    def detect(self, text: str) -> DetectorResult:
        calibrated = self._get_calibrated_score(text)
        score, confidence = calibrated or (0.5, 0.0)
        return DetectorResult(score=score, confidence=confidence)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn is not installed")
class TestCalibratedDetectorMixin:
    def test_calibrate_reduces_cv_for_small_balanced_dataset(self):
        detector = DummyCalibratedDetector()
        texts = ["short", "tiny", "a much longer sample", "another long example"]
        labels = [0, 0, 1, 1]

        detector.calibrate(texts, labels, cv=5)

        result = detector.detect("medium sized text")
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
