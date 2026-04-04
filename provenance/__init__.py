"""Provenance AI - Modular AI Text Detection Library."""

__version__ = "0.1.0"

from provenance.core.base import (
    BaseDetector,
    DetectorResult,
    SentinelResult,
    TokenScore,
)
from provenance.core.ensemble import Ensemble, EnsembleConfig
from provenance.core.preprocessor import PreprocessedText, Preprocessor, TextChunk
from provenance.core.registry import DetectorRegistry, get_registry
from provenance.sentinel import Provenance

__all__ = [
    "__version__",
    "BaseDetector",
    "DetectorResult",
    "SentinelResult",
    "TokenScore",
    "Ensemble",
    "EnsembleConfig",
    "Preprocessor",
    "PreprocessedText",
    "TextChunk",
    "DetectorRegistry",
    "get_registry",
    "Provenance",
]
