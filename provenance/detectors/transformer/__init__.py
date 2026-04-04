"""Transformer-based detectors for AI text detection."""

from provenance.detectors.transformer.hf_classifier import (
    AttentionHeatmapDetector,
    ChatGPTDetector,
    HuggingFaceClassifierDetector,
    OpenAIDetector,
    RADARDetector,
    RAIDDetection,
)

__all__ = [
    "HuggingFaceClassifierDetector",
    "OpenAIDetector",
    "ChatGPTDetector",
    "RADARDetector",
    "RAIDDetection",
    "AttentionHeatmapDetector",
]


def register(registry) -> None:
    registry.register(HuggingFaceClassifierDetector)
    registry.register(OpenAIDetector)
    registry.register(ChatGPTDetector)
    registry.register(RADARDetector)
    registry.register(RAIDDetection)
    registry.register(AttentionHeatmapDetector)
