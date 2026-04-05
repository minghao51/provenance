"""Burstiness detector - measures variation in sentence-level AI probability."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .perplexity import PerplexityDetector

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin


class BurstinessDetector(CalibratedDetectorMixin, BaseDetector):
    """Detects AI text by analyzing variation in per-sentence AI probability.

    Human writing has natural bursts of complexity - some sentences are simple,
    others complex. AI text tends to maintain consistent complexity throughout.

    This detector computes the AI probability for each sentence and measures
    how uniform those probabilities are.
    """

    name = "burstiness"
    latency_tier = "slow"
    domains = ["prose", "academic"]

    def __init__(self, perplexity_detector: PerplexityDetector | None = None):
        if perplexity_detector is None:
            from .perplexity import PerplexityDetector

            self.perplexity_detector = PerplexityDetector()
        else:
            self.perplexity_detector = perplexity_detector

    def _compute_sentence_scores(self, text: str) -> list[float]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if len(sentences) < 2:
            return []

        scores = []
        for sentence in sentences:
            result = self.perplexity_detector.detect(sentence)
            scores.append(result.score)

        return scores

    def _extract_features(self, text: str) -> list[float]:
        sentence_scores = self._compute_sentence_scores(text)
        if len(sentence_scores) < 2:
            return [0.0, 0.0, 0.0, 0.0]

        mean_score = sum(sentence_scores) / len(sentence_scores)
        variance = sum((s - mean_score) ** 2 for s in sentence_scores) / len(
            sentence_scores
        )
        std_score = variance**0.5
        cv = std_score / mean_score if mean_score > 0 else 0.0
        return [cv, mean_score, std_score, float(len(sentence_scores))]

    def _extract_feature_names(self) -> list[str]:
        return [
            "burstiness_cv",
            "mean_sentence_score",
            "std_sentence_score",
            "sentence_count",
        ]

    def detect(self, text: str) -> DetectorResult:
        sentence_scores = self._compute_sentence_scores(text)

        if len(sentence_scores) < 2:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Not enough sentences to compute burstiness"},
            )

        mean_score = sum(sentence_scores) / len(sentence_scores)
        variance = sum((s - mean_score) ** 2 for s in sentence_scores) / len(
            sentence_scores
        )
        std_score = variance**0.5

        cv = std_score / mean_score if mean_score > 0 else 0.0

        calibrated = self._get_calibrated_score(text)
        if calibrated is not None:
            score, confidence = calibrated
        else:
            ai_score = max(0.0, min(1.0, 1.0 - cv))
            score = ai_score
            if cv < 0.2:
                confidence = 0.7
            elif cv < 0.4:
                confidence = 0.6
            elif cv < 0.6:
                confidence = 0.5
            else:
                confidence = 0.4

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "burstiness_cv": cv,
                "mean_sentence_score": mean_score,
                "std_sentence_score": std_score,
                "sentence_count": len(sentence_scores),
                "sentence_scores": sentence_scores,
                "calibrated": calibrated is not None,
            },
        )


def register(registry=None) -> None:
    registry.register(BurstinessDetector)
