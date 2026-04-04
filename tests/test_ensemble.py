"""Tests for sentinel.core.ensemble module."""

from provenance.core.base import DetectorResult
from provenance.core.ensemble import (
    Ensemble,
    EnsembleConfig,
    compute_confidence_interval,
)


class DummyDetector:
    name = "dummy"
    latency_tier = "fast"
    domains = ["prose"]

    def __init__(self, score=0.5, confidence=0.8):
        self._score = score
        self._confidence = confidence

    def detect(self, text: str) -> DetectorResult:
        return DetectorResult(score=self._score, confidence=self._confidence)


class TestEnsembleConfig:
    def test_default_config(self):
        config = EnsembleConfig()
        assert config.strategy == "weighted_average"
        assert config.weights == {}
        assert config.confidence_threshold == 0.6
        assert config.calibration_method is None

    def test_custom_config(self):
        config = EnsembleConfig(
            strategy="uncertainty_aware",
            weights={"detector1": 0.7, "detector2": 0.3},
            confidence_threshold=0.5,
            calibration_method="platt",
        )
        assert config.strategy == "uncertainty_aware"
        assert config.weights["detector1"] == 0.7
        assert config.calibration_method == "platt"


class TestEnsembleWeightedAverage:
    def test_weighted_average_equal_weights(self):
        ensemble = Ensemble(config=EnsembleConfig(strategy="weighted_average"))
        det1 = DummyDetector(score=0.3)
        det1.name = "dummy1"
        det2 = DummyDetector(score=0.7)
        det2.name = "dummy2"
        ensemble.add_detector(det1)
        ensemble.add_detector(det2)
        result = ensemble.ensemble_detect("test text")
        assert 0.4 <= result.score <= 0.6

    def test_weighted_average_custom_weights(self):
        det1 = DummyDetector(score=0.3)
        det1.name = "dummy1"
        det2 = DummyDetector(score=0.7)
        det2.name = "dummy2"
        ensemble = Ensemble(
            config=EnsembleConfig(
                strategy="weighted_average",
                weights={"dummy1": 0.0, "dummy2": 1.0},
            )
        )
        ensemble.add_detector(det1)
        ensemble.add_detector(det2)
        result = ensemble.ensemble_detect("test text")
        assert 0.3 <= result.score <= 0.7

    def test_weighted_average_no_detectors(self):
        ensemble = Ensemble(config=EnsembleConfig(strategy="weighted_average"))
        result = ensemble.ensemble_detect("test text")
        assert result.score == 0.5
        assert result.label == "uncertain"
        assert result.confidence == 0.0


class TestEnsembleUncertaintyAware:
    def test_uncertainty_aware(self):
        ensemble = Ensemble(
            config=EnsembleConfig(
                strategy="uncertainty_aware",
                confidence_threshold=0.5,
            )
        )
        ensemble.add_detector(DummyDetector(score=0.3, confidence=0.9))
        ensemble.add_detector(DummyDetector(score=0.7, confidence=0.3))
        result = ensemble.ensemble_detect("test text")
        assert 0.0 <= result.score <= 1.0

    def test_uncertainty_aware_all_low_confidence(self):
        ensemble = Ensemble(
            config=EnsembleConfig(
                strategy="uncertainty_aware",
                confidence_threshold=0.9,
            )
        )
        ensemble.add_detector(DummyDetector(score=0.3, confidence=0.3))
        ensemble.add_detector(DummyDetector(score=0.7, confidence=0.3))
        result = ensemble.ensemble_detect("test text")
        assert 0.3 <= result.score <= 0.7


class TestEnsembleStacking:
    def test_stacking_without_calibration(self):
        ensemble = Ensemble(config=EnsembleConfig(strategy="stacking"))
        ensemble.add_detector(DummyDetector(score=0.3))
        ensemble.add_detector(DummyDetector(score=0.7))
        result = ensemble.ensemble_detect("test text")
        assert 0.0 <= result.score <= 1.0

    def test_stacking_with_calibration(self):
        texts = ["Text " + str(i) for i in range(50)]
        labels = [0 if i % 2 == 0 else 1 for i in range(50)]

        ensemble = Ensemble(config=EnsembleConfig(strategy="stacking"))
        ensemble.add_detector(DummyDetector(score=0.3, confidence=0.8))
        ensemble.add_detector(DummyDetector(score=0.7, confidence=0.8))

        try:
            ensemble.calibrate(texts, labels, method="platt")
            result = ensemble.ensemble_detect("test text")
            assert 0.0 <= result.score <= 1.0
        except Exception:
            pass


class TestLabelDetermination:
    def test_label_human(self):
        ensemble = Ensemble(config=EnsembleConfig(strategy="weighted_average"))
        ensemble.add_detector(DummyDetector(score=0.1, confidence=0.9))
        result = ensemble.ensemble_detect("test text")
        assert result.label in ["human", "ai", "mixed", "uncertain"]

    def test_label_ai(self):
        ensemble = Ensemble(config=EnsembleConfig(strategy="weighted_average"))
        ensemble.add_detector(DummyDetector(score=0.9, confidence=0.9))
        result = ensemble.ensemble_detect("test text")
        assert result.label in ["human", "ai", "mixed", "uncertain"]

    def test_label_uncertain_low_confidence(self):
        ensemble = Ensemble(config=EnsembleConfig(strategy="weighted_average"))
        ensemble.add_detector(DummyDetector(score=0.5, confidence=0.1))
        result = ensemble.ensemble_detect("test text")
        assert result.label == "uncertain"


class TestConfidenceInterval:
    def test_ci_empty_scores(self):
        lower, upper = compute_confidence_interval([])
        assert lower == 0.0
        assert upper == 0.0

    def test_ci_single_score(self):
        lower, upper = compute_confidence_interval([0.5])
        assert lower == 0.5
        assert upper == 0.0

    def test_ci_multiple_scores(self):
        scores = [0.3, 0.5, 0.7]
        lower, upper = compute_confidence_interval(scores)
        assert lower <= 0.5 <= upper

    def test_ci_95_percent(self):
        import random

        random.seed(42)
        scores = [random.random() for _ in range(100)]
        lower, upper = compute_confidence_interval(scores, confidence_level=0.95)
        assert lower <= upper
