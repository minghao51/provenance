"""Multilingual domain adapter for cross-lingual AI detection."""

from __future__ import annotations

import re
from collections import Counter

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]


class MultilingualDetector(CalibratedDetectorMixin, BaseDetector):
    name = "multilingual_detector"
    latency_tier = "slow"
    domains = ["multilingual"]

    LANGUAGE_FAMILIES = {
        "en": "germanic",
        "de": "germanic",
        "nl": "germanic",
        "fr": "romance",
        "es": "romance",
        "it": "romance",
        "pt": "romance",
        "ro": "romance",
        "zh": "cjk",
        "ja": "cjk",
        "ko": "cjk",
        "ru": "cyrillic",
        "uk": "cyrillic",
        "ar": "semitic",
        "he": "semitic",
    }

    BURSTINESS_THRESHOLDS = {
        "germanic": 0.35,
        "romance": 0.40,
        "cjk": 0.25,
        "cyrillic": 0.35,
        "semitic": 0.30,
    }

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def _detect_language(self, text: str) -> tuple[str, float]:
        try:
            import langdetect

            lang = langdetect.detect(text)
            prob = 0.8
            return lang, prob
        except Exception:
            return "unknown", 0.0

    def _get_language_family(self, lang: str) -> str:
        return self.LANGUAGE_FAMILIES.get(lang, "other")

    def _compute_cross_lingual_features(self, text: str) -> dict[str, float]:
        features = {}

        char_ngrams = re.findall(r"\b\w+\b", text)
        if char_ngrams:
            bigrams = [
                "".join(pair)
                for pair in zip(char_ngrams[:-1], char_ngrams[1:], strict=False)
            ]
            bigram_counts = Counter(bigrams)
            total_bigrams = len(bigrams) if bigrams else 1
            unique_bigrams = len(bigram_counts)

            features["char_bigram_diversity"] = (
                unique_bigrams / total_bigrams if total_bigrams > 0 else 0
            )
        else:
            features["char_bigram_diversity"] = 0

        words = char_ngrams
        if words:
            word_lengths = [len(w) for w in words]
            features["avg_word_length"] = (
                sum(word_lengths) / len(word_lengths) if word_lengths else 0
            )
            features["word_length_variance"] = (
                sum(
                    (length - features["avg_word_length"]) ** 2
                    for length in word_lengths
                )
                / len(word_lengths)
                if word_lengths
                else 0
            )
        else:
            features["avg_word_length"] = 0
            features["word_length_variance"] = 0

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            mean_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
            features["sentence_length_variance"] = (
                sum((length - mean_sentence_length) ** 2 for length in sentence_lengths)
                / len(sentence_lengths)
                if sentence_lengths
                else 0
            )
        else:
            features["sentence_length_variance"] = 0

        return features

    def _estimate_burstiness_adapted(self, text: str, family: str) -> float:
        threshold = self.BURSTINESS_THRESHOLDS.get(family, 0.35)

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]

        if len(sentences) < 2:
            return threshold

        sentence_lengths = [len(s.split()) for s in sentences]
        mean_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((length - mean_len) ** 2 for length in sentence_lengths) / len(
            sentence_lengths
        )
        std = variance**0.5
        return float(std / mean_len) if mean_len > 0 else 0.0

    def _extract_features(self, text: str) -> list[float]:
        lang, lang_prob = self._detect_language(text)
        family = self._get_language_family(lang)
        features = self._compute_cross_lingual_features(text)
        burstiness_cv = self._estimate_burstiness_adapted(text, family)
        threshold = self.BURSTINESS_THRESHOLDS.get(family, 0.35)
        return [
            burstiness_cv,
            threshold,
            features.get("char_bigram_diversity", 0.0),
            features.get("avg_word_length", 0.0),
            features.get("word_length_variance", 0.0),
            features.get("sentence_length_variance", 0.0),
            lang_prob,
        ]

    def _extract_feature_names(self) -> list[str]:
        return [
            "burstiness_cv",
            "family_threshold",
            "char_bigram_diversity",
            "avg_word_length",
            "word_length_variance",
            "sentence_length_variance",
            "language_probability",
        ]

    def detect(self, text: str) -> DetectorResult:
        if len(text) < 50:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Text too short for multilingual analysis"},
            )

        lang, lang_prob = self._detect_language(text)
        family = self._get_language_family(lang)
        features = self._compute_cross_lingual_features(text)
        burstiness_cv = self._estimate_burstiness_adapted(text, family)

        threshold = self.BURSTINESS_THRESHOLDS.get(family, 0.35)

        calibrated = self._get_calibrated_score(text)
        if calibrated is not None:
            score, confidence = calibrated
        else:
            if burstiness_cv < threshold * 0.5:
                score = 0.8
                confidence = 0.7
            elif burstiness_cv < threshold:
                score = 0.6
                confidence = 0.6
            elif features.get("char_bigram_diversity", 1) < 0.3:
                score = 0.55
                confidence = 0.5
            else:
                score = 0.35
                confidence = 0.5

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "detected_language": lang,
                "language_probability": lang_prob,
                "language_family": family,
                "burstiness_cv": burstiness_cv,
                **features,
                "calibrated": calibrated is not None,
            },
        )


def register(registry) -> None:
    registry.register(MultilingualDetector)
