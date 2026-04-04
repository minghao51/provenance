"""Transformer-based detectors for AI text detection."""

try:
    from provenance.detectors.transformer.hf_classifier import (
        AttentionHeatmapDetector,
        ChatGPTDetector,
        HuggingFaceClassifierDetector,
        OpenAIDetector,
        RADARDetector,
        RAIDDetection,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AttentionHeatmapDetector = None  # type: ignore[misc,assignment]
    ChatGPTDetector = None  # type: ignore[misc,assignment]
    HuggingFaceClassifierDetector = None  # type: ignore[misc,assignment]
    OpenAIDetector = None  # type: ignore[misc,assignment]
    RADARDetector = None  # type: ignore[misc,assignment]
    RAIDDetection = None  # type: ignore[misc,assignment]
    TRANSFORMERS_AVAILABLE = False

__all__ = [
    "HuggingFaceClassifierDetector",
    "OpenAIDetector",
    "ChatGPTDetector",
    "RADARDetector",
    "RAIDDetection",
    "AttentionHeatmapDetector",
]


def register(registry) -> None:
    if not TRANSFORMERS_AVAILABLE:
        return
    registry.register(HuggingFaceClassifierDetector)
    registry.register(OpenAIDetector)
    registry.register(ChatGPTDetector)
    registry.register(RADARDetector)
    registry.register(RAIDDetection)
    registry.register(AttentionHeatmapDetector)
