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

    @abstractmethod
    def detect(self, text: str) -> DetectorResult: ...

    def batch_detect(self, texts: list[str]) -> list[DetectorResult]:
        return [self.detect(t) for t in texts]
