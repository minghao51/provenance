from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DetectorResult:
    score: float
    confidence: float
    metadata: dict = field(default_factory=dict)


@dataclass
class TokenScore:
    token: str
    score: float


@dataclass
class SentinelResult:
    score: float
    label: Literal["human", "ai", "mixed", "uncertain"]
    confidence: float
    detector_scores: dict = field(default_factory=dict)
    heatmap: list[TokenScore] = field(default_factory=list)
    sentence_scores: list[float] = field(default_factory=list)
    feature_vector: dict = field(default_factory=dict)
    top_features: list[tuple[str, float]] = field(default_factory=list)


class BaseDetector(ABC):
    name: str
    latency_tier: Literal["fast", "medium", "slow"]
    domains: list[str]
    default_error_score: float = 0.5
    default_error_confidence: float = 0.0

    @abstractmethod
    def detect(self, text: str) -> DetectorResult: ...

    def batch_detect(self, texts: list[str]) -> list[DetectorResult]:
        return [self.detect(t) for t in texts]

    def build_error_result(
        self,
        message: str,
        *,
        exception: Exception | None = None,
        score: float | None = None,
        confidence: float | None = None,
        metadata: dict | None = None,
    ) -> DetectorResult:
        error_metadata = dict(metadata or {})
        error_metadata["error"] = message
        error_metadata["error_type"] = (
            type(exception).__name__ if exception is not None else "detector_error"
        )
        if exception is not None and str(exception):
            error_metadata["error_details"] = str(exception)

        return DetectorResult(
            score=self.default_error_score if score is None else score,
            confidence=(
                self.default_error_confidence if confidence is None else confidence
            ),
            metadata=error_metadata,
        )
