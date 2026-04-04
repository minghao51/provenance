"""LLM-based detectors for AI text detection."""

from provenance.detectors.llm.llm_detectors import (
    DetectGPTDetector,
    LLMMetaReasoningDetector,
    OllamaLogProbDetector,
)

__all__ = [
    "OllamaLogProbDetector",
    "DetectGPTDetector",
    "LLMMetaReasoningDetector",
]


def register(registry) -> None:
    registry.register(OllamaLogProbDetector)
    registry.register(DetectGPTDetector)
    registry.register(LLMMetaReasoningDetector)
