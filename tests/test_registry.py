"""Tests for sentinel.core.registry module."""

import pytest

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.registry import DetectorRegistry, get_registry


class TestDetectorRegistry:
    def setup_method(self):
        self.registry = DetectorRegistry()
        self.registry.clear()

    def teardown_method(self):
        self.registry.clear()

    def test_singleton_pattern(self):
        reg1 = DetectorRegistry()
        reg2 = DetectorRegistry()
        assert reg1 is reg2

    def test_register_and_get(self):
        class TestDetector(BaseDetector):
            name = "test_detector"
            latency_tier = "fast"
            domains = ["prose"]

            def detect(self, text: str) -> DetectorResult:
                return DetectorResult(score=0.5, confidence=0.8)

        self.registry.register(TestDetector)
        detector = self.registry.get("test_detector")
        assert detector is not None
        assert detector.name == "test_detector"

    def test_get_nonexistent(self):
        detector = self.registry.get("nonexistent")
        assert detector is None

    def test_list_detectors_all(self):
        class Detector1(BaseDetector):
            name = "detector1"
            latency_tier = "fast"
            domains = ["prose"]

            def detect(self, text):
                return DetectorResult(0.5, 0.8)

        class Detector2(BaseDetector):
            name = "detector2"
            latency_tier = "slow"
            domains = ["academic"]

            def detect(self, text):
                return DetectorResult(0.5, 0.8)

        self.registry.register(Detector1)
        self.registry.register(Detector2)
        detectors = self.registry.list_detectors()
        assert len(detectors) >= 2

    def test_list_detectors_by_latency(self):
        class FastDetector(BaseDetector):
            name = "fast_detector"
            latency_tier = "fast"
            domains = ["prose"]

            def detect(self, text):
                return DetectorResult(0.5, 0.8)

        class SlowDetector(BaseDetector):
            name = "slow_detector"
            latency_tier = "slow"
            domains = ["prose"]

            def detect(self, text):
                return DetectorResult(0.5, 0.8)

        self.registry.register(FastDetector)
        self.registry.register(SlowDetector)

        fast = self.registry.list_detectors(latency_tier="fast")
        assert all(d.latency_tier == "fast" for d in fast)

        slow = self.registry.list_detectors(latency_tier="slow")
        assert all(d.latency_tier == "slow" for d in slow)

    def test_list_detectors_by_domain(self):
        class ProseDetector(BaseDetector):
            name = "prose_detector"
            latency_tier = "fast"
            domains = ["prose"]

            def detect(self, text):
                return DetectorResult(0.5, 0.8)

        class AcademicDetector(BaseDetector):
            name = "academic_detector"
            latency_tier = "fast"
            domains = ["academic"]

            def detect(self, text):
                return DetectorResult(0.5, 0.8)

        self.registry.register(ProseDetector)
        self.registry.register(AcademicDetector)

        prose = self.registry.list_detectors(domain="prose")
        assert all("prose" in d.domains for d in prose)

    def test_clear(self):
        class TestDetector(BaseDetector):
            name = "test_clear"
            latency_tier = "fast"
            domains = ["prose"]

            def detect(self, text):
                return DetectorResult(0.5, 0.8)

        self.registry.register(TestDetector)
        assert len(self.registry.list_detectors()) >= 1
        self.registry.clear()
        assert len(self.registry.list_detectors()) == 0

    def test_get_registry_function(self):
        registry = get_registry()
        assert isinstance(registry, DetectorRegistry)


class TestEntryPointLoading:
    def test_load_entry_points_handles_errors(self):
        registry = DetectorRegistry()
        registry.clear()
        try:
            registry.load_entry_points()
        except Exception:
            pytest.fail("load_entry_points should not raise exceptions")
