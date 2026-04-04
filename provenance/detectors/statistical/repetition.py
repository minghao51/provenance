"""Repetition detector using Self-BLEU and n-gram analysis."""

from __future__ import annotations

import re
from collections import Counter

from provenance.core.base import BaseDetector, DetectorResult


class RepetitionDetector(BaseDetector):
    name = "repetition"
    latency_tier = "fast"
    domains = ["prose", "academic"]

    def __init__(
        self,
        ngram_sizes: tuple[int, ...] = (3, 4),
        repetition_threshold: float = 0.3,
    ):
        self.ngram_sizes = ngram_sizes
        self.repetition_threshold = repetition_threshold

    def _get_ngrams(self, words: list[str], n: int) -> list[tuple[str, ...]]:
        if len(words) < n:
            return []
        return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    def _compute_ngram_repetition_ratio(self, words: list[str], n: int) -> float:
        ngrams = self._get_ngrams(words, n)
        if not ngrams:
            return 0.0

        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counts)

        if total_ngrams == 0:
            return 0.0

        repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)
        return repetition_ratio

    def _compute_self_bleu(self, paragraphs: list[str]) -> float:
        if len(paragraphs) < 2:
            return 0.0

        try:
            from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        except ImportError:
            return 0.0

        smoothing = SmoothingFunction().method1
        scores = []

        for i, para in enumerate(paragraphs):
            refs = [p for j, p in enumerate(paragraphs) if j != i]
            if not refs:
                continue

            words = para.split()
            if len(words) < 5:
                continue

            for ref in refs[:3]:
                ref_words = ref.split()
                if len(ref_words) < 5:
                    continue
                try:
                    score = sentence_bleu(
                        [ref_words],
                        words,
                        smoothing_function=smoothing,
                        weights=(0.25, 0.25, 0.25, 0.25),
                    )
                    scores.append(score)
                except Exception:
                    pass

        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def _detect_repeated_ngrams(self, words: list[str]) -> list[tuple[str, float]]:
        repeated = []
        for n in self.ngram_sizes:
            ngrams = self._get_ngrams(words, n)
            if not ngrams:
                continue

            ngram_counts = Counter(ngrams)
            threshold_count = 2

            for ngram, count in ngram_counts.items():
                if count >= threshold_count:
                    ngram_str = " ".join(ngram)
                    ratio = count / len(ngrams)
                    repeated.append((ngram_str, ratio))

        repeated.sort(key=lambda x: x[1], reverse=True)
        return repeated[:10]

    def detect(self, text: str) -> DetectorResult:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if len(words) < 10:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Text too short for repetition analysis"},
            )

        ngram_repetitions = {}
        for n in self.ngram_sizes:
            ratio = self._compute_ngram_repetition_ratio(words, n)
            ngram_repetitions[f"ngram_{n}_repetition"] = ratio

        self_bleu = self._compute_self_bleu(paragraphs) if len(paragraphs) >= 2 else 0.0
        repeated_ngrams = self._detect_repeated_ngrams(words)

        max_repetition = max(ngram_repetitions.values()) if ngram_repetitions else 0.0
        repetition_score = max_repetition

        if self_bleu > 0.5:
            repetition_score = max(repetition_score, 0.7)

        score = min(1.0, repetition_score)
        confidence = 0.8 if len(words) > 50 else 0.4

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                **ngram_repetitions,
                "self_bleu": self_bleu,
                "repeated_ngrams": repeated_ngrams[:5],
                "paragraph_count": len(paragraphs),
            },
        )


def register(registry=None) -> None:
    registry.register(RepetitionDetector)
