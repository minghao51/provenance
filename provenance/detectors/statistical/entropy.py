"""Entropy-based detector using word frequency distributions."""

from __future__ import annotations

import math
import re
from collections import Counter

import nltk

from provenance.core.base import BaseDetector, DetectorResult

BROWN_CORPUS_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"


class EntropyDetector(BaseDetector):
    name = "entropy"
    latency_tier = "fast"
    domains = ["prose", "academic"]

    def __init__(self):
        self.word_frequencies: Counter[str] | None = None
        self._load_brown_frequencies()

    def _load_brown_frequencies(self) -> None:
        try:
            from nltk.corpus import brown

            words = brown.words()
            self.word_frequencies = Counter(w.lower() for w in words if w.isalpha())
        except LookupError:
            nltk.download("brown", quiet=True)
            from nltk.corpus import brown

            words = brown.words()
            self.word_frequencies = Counter(w.lower() for w in words if w.isalpha())

    def _compute_unigram_entropy(self, text: str) -> float:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return 0.0

        word_counts = Counter(words)
        total = len(words)
        entropy = 0.0

        for count in word_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _compute_kl_divergence(self, text: str) -> float:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words or not self.word_frequencies:
            return 0.0

        total_ref = sum(self.word_frequencies.values())
        ref_probs = {
            w: c / total_ref for w, c in self.word_frequencies.items() if c > 0
        }

        word_counts = Counter(words)
        total = len(words)

        kl_div = 0.0
        for word, count in word_counts.items():
            p = count / total
            q = ref_probs.get(word, 1e-10)
            if p > 0:
                kl_div += p * math.log2(p / q)

        return kl_div

    def detect(self, text: str) -> DetectorResult:
        text_entropy = self._compute_unigram_entropy(text)
        kl_div = self._compute_kl_divergence(text)

        if kl_div > 2.0:
            score = 0.8
            confidence = 0.7
        elif kl_div > 1.0:
            score = 0.6
            confidence = 0.6
        elif kl_div > 0.5:
            score = 0.4
            confidence = 0.5
        else:
            score = 0.2
            confidence = 0.6

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "text_entropy": text_entropy,
                "kl_divergence": kl_div,
                "vocabulary_size": len(set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))),
            },
        )


def register(registry=None) -> None:
    registry.register(EntropyDetector)
