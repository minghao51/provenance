"""Burstiness detector - measures variation in sentence-level AI probability."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .perplexity import PerplexityDetector

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin
from provenance.core.config import BurstinessThresholds
from provenance.core.preprocessor import Preprocessor
from provenance.core.statistics import compute_cv, compute_mean_variance_std


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
    calibration_aliases = ("burstiness",)

    def __init__(
        self,
        perplexity_detector: PerplexityDetector | None = None,
        thresholds: BurstinessThresholds | None = None,
    ):
        self.thresholds = thresholds or BurstinessThresholds()
        if perplexity_detector is None:
            from .perplexity import PerplexityDetector

            self.perplexity_detector = PerplexityDetector()
        else:
            self.perplexity_detector = perplexity_detector

        self.preprocessor = Preprocessor()

    def _compute_sentence_scores(self, text: str) -> list[float]:
        sentences = self.preprocessor.split_sentences(text)
        sentences = [
            s.strip()
            for s in sentences
            if len(s.strip()) > self.thresholds.min_sentence_length
        ]

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

        mean_score, variance, std_score = compute_mean_variance_std(sentence_scores)
        cv = compute_cv(sentence_scores, mean_score, std_score)
        return [cv, mean_score, std_score, float(len(sentence_scores))]

    def _extract_feature_names(self) -> list[str]:
        return [
            "burstiness_cv",
            "mean_sentence_score",
            "std_sentence_score",
            "sentence_count",
        ]

    def detect(self, text: str) -> DetectorResult:
        try:
            sentence_scores = self._compute_sentence_scores(text)

            if len(sentence_scores) < 2:
                return self.build_error_result(
                    "Not enough sentences to compute burstiness"
                )

            mean_score, variance, std_score = compute_mean_variance_std(sentence_scores)
            cv = compute_cv(sentence_scores, mean_score, std_score)

            calibrated = self._get_calibrated_score(text)
            if calibrated is not None:
                score, confidence = calibrated
            else:
                ai_score = max(0.0, min(1.0, 1.0 - cv))
                score = ai_score
                if cv < self.thresholds.cv_high:
                    confidence = self.thresholds.cv_high_confidence
                elif cv < self.thresholds.cv_medium:
                    confidence = self.thresholds.cv_medium_confidence
                elif cv < self.thresholds.cv_low:
                    confidence = self.thresholds.cv_low_confidence
                else:
                    confidence = self.thresholds.cv_default_confidence

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
        except Exception as e:
            return self.build_error_result(
                "Burstiness analysis failed",
                exception=e,
            )


def register(registry=None) -> None:
    registry.register(BurstinessDetector)
