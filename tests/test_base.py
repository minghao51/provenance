"""Tests for sentinel.core.base module."""

import pytest

from provenance.core.base import (
    BaseDetector,
    DetectorResult,
    SentinelResult,
    TokenScore,
)


class DummyDetector(BaseDetector):
    name = "dummy_detector"
    latency_tier = "fast"
    domains = ["prose"]

    def detect(self, text: str) -> DetectorResult:
        return DetectorResult(
            score=0.7,
            confidence=0.8,
            metadata={"text_length": len(text)},
        )


class TestDetectorResult:
    def test_detector_result_creation(self):
        result = DetectorResult(score=0.5, confidence=0.9)
        assert result.score == 0.5
        assert result.confidence == 0.9
        assert result.metadata == {}

    def test_detector_result_with_metadata(self):
        result = DetectorResult(
            score=0.8,
            confidence=0.95,
            metadata={"perplexity": 15.3, "burstiness": 0.42},
        )
        assert result.score == 0.8
        assert result.metadata["perplexity"] == 15.3


class TestTokenScore:
    def test_token_score_creation(self):
        ts = TokenScore(token="hello", score=0.75)
        assert ts.token == "hello"
        assert ts.score == 0.75


class TestSentinelResult:
    def test_sentinel_result_defaults(self):
        result = SentinelResult(
            score=0.6,
            label="mixed",
            confidence=0.75,
        )
        assert result.score == 0.6
        assert result.label == "mixed"
        assert result.detector_scores == {}
        assert result.heatmap == []
        assert result.sentence_scores == []
        assert result.feature_vector == {}
        assert result.top_features == []

    def test_sentinel_result_full(self):
        ts = TokenScore(token="test", score=0.5)
        dr = DetectorResult(score=0.7, confidence=0.8)
        result = SentinelResult(
            score=0.65,
            label="ai",
            confidence=0.85,
            detector_scores={"dummy": dr},
            heatmap=[ts],
            sentence_scores=[0.6, 0.7],
            feature_vector={"flesch_kincaid": 12.5},
            top_features=[("flesch_kincaid", 0.15)],
        )
        assert result.label == "ai"
        assert len(result.heatmap) == 1
        assert len(result.sentence_scores) == 2
        assert result.feature_vector["flesch_kincaid"] == 12.5


class TestBaseDetector:
    def test_base_detector_is_abstract(self):
        with pytest.raises(TypeError):
            BaseDetector()

    def test_dummy_detector_detect(self):
        detector = DummyDetector()
        result = detector.detect("Some test text here.")
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_dummy_detector_batch_detect(self):
        detector = DummyDetector()
        texts = ["Text one.", "Text two.", "Text three."]
        results = detector.batch_detect(texts)
        assert len(results) == 3
        assert all(isinstance(r, DetectorResult) for r in results)

    def test_detector_name_and_tiers(self):
        detector = DummyDetector()
        assert detector.name == "dummy_detector"
        assert detector.latency_tier in ["fast", "medium", "slow"]
        assert isinstance(detector.domains, list)
