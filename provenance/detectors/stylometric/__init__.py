"""Stylometric detectors for AI text detection."""

from provenance.detectors.stylometric.cognitive import CognitiveDetector
from provenance.detectors.stylometric.feature_extractor import (
    FeatureExtractor,
    StylometricDetector,
)

try:
    from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector
except ImportError:
    LightGBMDetector = None  # type: ignore[misc,assignment]

__all__ = [
    "FeatureExtractor",
    "StylometricDetector",
    "LightGBMDetector",
    "CognitiveDetector",
]


def register(registry) -> None:
    registry.register(StylometricDetector)
    if LightGBMDetector is not None:
        registry.register(LightGBMDetector)
    registry.register(CognitiveDetector)
