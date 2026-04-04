"""LLM-based detectors for AI text detection."""

try:
    from provenance.detectors.llm.llm_detectors import (
        DetectGPTDetector,
        LLMMetaReasoningDetector,
        OllamaLogProbDetector,
    )

    LLM_AVAILABLE = True
except ImportError:
    DetectGPTDetector = None  # type: ignore[misc,assignment]
    LLMMetaReasoningDetector = None  # type: ignore[misc,assignment]
    OllamaLogProbDetector = None  # type: ignore[misc,assignment]
    LLM_AVAILABLE = False

__all__ = [
    "OllamaLogProbDetector",
    "DetectGPTDetector",
    "LLMMetaReasoningDetector",
]


def register(registry) -> None:
    if not LLM_AVAILABLE:
        return
    registry.register(OllamaLogProbDetector)
    registry.register(DetectGPTDetector)
    registry.register(LLMMetaReasoningDetector)
