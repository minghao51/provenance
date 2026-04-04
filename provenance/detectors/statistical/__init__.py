"""Statistical detectors for AI text detection."""

from provenance.detectors.statistical.burstiness import BurstinessDetector
from provenance.detectors.statistical.entropy import EntropyDetector
from provenance.detectors.statistical.repetition import RepetitionDetector

try:
    from provenance.detectors.statistical.curvature import CurvatureDetector
except ImportError:
    CurvatureDetector = None  # type: ignore[misc,assignment]

try:
    from provenance.detectors.statistical.perplexity import (
        PerplexityDetector,
        PerplexityDetectorNeo,
    )
except ImportError:
    PerplexityDetector = None  # type: ignore[misc,assignment]
    PerplexityDetectorNeo = None  # type: ignore[misc,assignment]

try:
    from provenance.detectors.statistical.surprisal import SurprisalDetector
except ImportError:
    SurprisalDetector = None  # type: ignore[misc,assignment]

__all__ = [
    "PerplexityDetector",
    "PerplexityDetectorNeo",
    "BurstinessDetector",
    "CurvatureDetector",
    "EntropyDetector",
    "RepetitionDetector",
    "SurprisalDetector",
]


def register(registry) -> None:
    if PerplexityDetector is not None:
        registry.register(PerplexityDetector)
    if PerplexityDetectorNeo is not None:
        registry.register(PerplexityDetectorNeo)
    registry.register(BurstinessDetector)
    if CurvatureDetector is not None:
        registry.register(CurvatureDetector)
    registry.register(EntropyDetector)
    registry.register(RepetitionDetector)
    if SurprisalDetector is not None:
        registry.register(SurprisalDetector)
