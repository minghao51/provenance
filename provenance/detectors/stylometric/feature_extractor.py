"""Comprehensive feature extraction for stylometric analysis."""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import cast

import nltk
import spacy
import textstat

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin


class FeatureExtractor:
    _surprisal_model = None
    _surprisal_tokenizer = None

    @classmethod
    def _get_surprisal_model(cls):
        if cls._surprisal_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            cls._surprisal_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            cls._surprisal_model = AutoModelForCausalLM.from_pretrained("gpt2")
            cls._surprisal_model.eval()
            if torch.cuda.is_available():
                cls._surprisal_model = cls._surprisal_model.to("cuda")
        return cls._surprisal_model, cls._surprisal_tokenizer

    TRANSITION_PHRASES = {
        "furthermore",
        "moreover",
        "additionally",
        "besides",
        "likewise",
        "consequently",
        "therefore",
        "thus",
        "hence",
        "accordingly",
        "in conclusion",
        "to conclude",
        "finally",
        "lastly",
        "in contrast",
        "however",
        "nevertheless",
        "nonetheless",
        "on the other hand",
        "conversely",
        "similarly",
        "for example",
        "for instance",
        "specifically",
        "in particular",
        "in addition",
        "equally",
        "meanwhile",
    }

    FUNCTION_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
    }

    def __init__(self, nlp=None):
        try:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        except Exception:
            self.nlp = nlp

        try:
            nltk.data.find("corpora/brown")
        except LookupError:
            nltk.download("brown", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def extract(self, text: str) -> dict[str, float]:
        features = {}

        features.update(self._extract_surface_features(text))
        features.update(self._extract_lexical_richness(text))
        features.update(self._extract_syntactic_features(text))
        features.update(self._extract_stylistic_features(text))
        features.update(self._extract_surprisal_features(text))

        return features

    def _extract_surface_features(self, text: str) -> dict[str, float]:
        features = {}

        try:
            features["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(text)
        except Exception:
            features["flesch_kincaid_grade"] = 0.0

        try:
            features["gunning_fog_index"] = textstat.gunning_fog(text)
        except Exception:
            features["gunning_fog_index"] = 0.0

        try:
            features["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
        except Exception:
            features["flesch_reading_ease"] = 0.0

        words = re.findall(r"\b[a-zA-Z]+\b", text)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if words:
            word_lengths = [len(w) for w in words]
            features["avg_word_length"] = statistics.mean(word_lengths)
            if len(word_lengths) > 1:
                features["std_word_length"] = statistics.stdev(word_lengths)
            else:
                features["std_word_length"] = 0.0
        else:
            features["avg_word_length"] = 0.0
            features["std_word_length"] = 0.0

        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            features["avg_sentence_length"] = statistics.mean(sentence_lengths)
            if len(sentence_lengths) > 1:
                features["std_sentence_length"] = statistics.stdev(sentence_lengths)
                features["max_sentence_length"] = max(sentence_lengths)
                features["min_sentence_length"] = min(sentence_lengths)
            else:
                features["std_sentence_length"] = 0.0
                features["max_sentence_length"] = sentence_lengths[0]
                features["min_sentence_length"] = sentence_lengths[0]
        else:
            features["avg_sentence_length"] = 0.0
            features["std_sentence_length"] = 0.0
            features["max_sentence_length"] = 0.0
            features["min_sentence_length"] = 0.0

        return features

    def _extract_lexical_richness(self, text: str) -> dict[str, float]:
        features = {}

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return {
                "ttr": 0.0,
                "yules_k": 0.0,
                "herdans_c": 0.0,
                "hapax_ratio": 0.0,
            }

        unique_words = set(words)
        total_words = len(words)

        ttr = len(unique_words) / total_words if total_words > 0 else 0.0
        features["ttr"] = ttr

        word_counts = Counter(words)
        freq_counts = Counter(word_counts.values())

        try:
            m1 = sum(f * c for f, c in freq_counts.items())
            m2 = sum(f * f * c for f, c in freq_counts.items())
            k = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0.0
            features["yules_k"] = k
        except Exception:
            features["yules_k"] = 0.0

        v = len(unique_words)
        n = total_words
        c = (v - 1) / n if n > 1 else 0.0
        features["herdans_c"] = c

        hapax = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax / total_words if total_words > 0 else 0.0
        features["hapax_ratio"] = hapax_ratio

        return features

    def _extract_syntactic_features(self, text: str) -> dict[str, float]:
        features = {}

        try:
            doc = self.nlp(text[:100000])
        except Exception:
            return {
                "pos_diversity": 0.0,
                "dep_depth_mean": 0.0,
                "dep_depth_max": 0.0,
                "passive_ratio": 0.0,
                "subordinate_density": 0.0,
            }

        pos_tags = [token.pos_ for token in doc if token.pos_]
        unique_pos = len(set(pos_tags))
        total_pos = len(pos_tags)
        features["pos_diversity"] = unique_pos / total_pos if total_pos > 0 else 0.0

        pos_counts = Counter(pos_tags)
        total_pos_tags = len(pos_tags)
        for pos, count in pos_counts.most_common(10):
            features[f"pos_{pos}_ratio"] = (
                count / total_pos_tags if total_pos_tags > 0 else 0.0
            )

        dep_depths = []
        branching_factors = []

        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            dep_depths.append(depth)

            children_count = len(list(token.children))
            branching_factors.append(children_count)

        features["dep_depth_mean"] = statistics.mean(dep_depths) if dep_depths else 0.0
        features["dep_depth_max"] = max(dep_depths) if dep_depths else 0.0
        features["branching_factor_mean"] = (
            statistics.mean(branching_factors) if branching_factors else 0.0
        )

        passive_count = sum(
            1 for token in doc if token.tag_ == "VBN" and token.dep_ == "nsubjpass"
        )
        active_count = sum(
            1 for token in doc if token.tag_ == "VBD" and token.dep_ == "nsubj"
        )
        total_verb_phrases = passive_count + active_count
        features["passive_ratio"] = (
            passive_count / total_verb_phrases if total_verb_phrases > 0 else 0.0
        )

        subordinate_clause_count = sum(
            1 for token in doc if token.dep_ in {"advcl", "acl", "ccomp", "xcomp"}
        )
        features["subordinate_density"] = (
            subordinate_clause_count / len(doc) if len(doc) > 0 else 0.0
        )

        return features

    def _extract_stylistic_features(self, text: str) -> dict[str, float]:
        features = {}

        text_lower = text.lower()
        words = re.findall(r"\b[a-zA-Z]+\b", text_lower)
        total_words = len(words) if words else 1

        transition_count = sum(
            1 for phrase in self.TRANSITION_PHRASES if phrase in text_lower
        )
        features["transition_phrase_freq"] = transition_count / total_words

        function_word_counts = Counter(w for w in words if w in self.FUNCTION_WORDS)
        features["function_word_ratio"] = (
            sum(function_word_counts.values()) / total_words
        )

        for fw in ["the", "and", "of", "to", "a", "in", "that", "is", "was", "it"]:
            features[f"function_word_{fw}_ratio"] = (
                function_word_counts.get(fw, 0) / total_words
            )

        punctuation_patterns = {
            "comma": r",",
            "semicolon": r";",
            "colon": r":",
            "exclamation": r"!",
            "question": r"\?",
            "dash": r"-",
            "quotes_double": r'"',
            "quotes_single": r"'",
            "parenthetical": r"\([^)]*\)",
        }

        for name, pattern in punctuation_patterns.items():
            count = len(re.findall(pattern, text))
            features[f"punctuation_{name}_ratio"] = count / total_words

        quote_count = len(re.findall(r'"[^"]*"|"[^"]*"', text))
        features["quote_density"] = quote_count / total_words

        parenthetical_count = len(re.findall(r"\([^)]*\)", text))
        features["parenthetical_density"] = parenthetical_count / total_words

        return features

    def _extract_surprisal_features(self, text: str) -> dict[str, float]:
        default_result = {
            "surprisal_mean": 0.0,
            "surprisal_std": 0.0,
            "surprisal_cv": 0.0,
            "surprisal_autocorr": 0.0,
            "surprisal_entropy": 0.0,
        }

        try:
            import torch
        except ImportError:
            return default_result

        try:
            model, tokenizer = self._get_surprisal_model()
        except Exception:
            return default_result

        try:
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )
            input_ids = encodings["input_ids"]
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            surprisals = token_losses.tolist()

            if len(surprisals) < 2:
                return default_result

            mean_s = sum(surprisals) / len(surprisals)
            std_s = (
                sum((s - mean_s) ** 2 for s in surprisals) / len(surprisals)
            ) ** 0.5
            cv_s = std_s / mean_s if mean_s > 0 else 0.0

            n = len(surprisals)
            autocorr = 0.0
            if n > 2 and std_s > 0:
                autocov = sum(
                    (surprisals[i] - mean_s) * (surprisals[i + 1] - mean_s)
                    for i in range(n - 1)
                ) / (n - 1)
                autocorr = autocov / (std_s**2)

            total = sum(surprisals)
            entropy = 0.0
            if total > 0:
                probs = [s / total for s in surprisals]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)

            return {
                "surprisal_mean": mean_s,
                "surprisal_std": std_s,
                "surprisal_cv": cv_s,
                "surprisal_autocorr": autocorr,
                "surprisal_entropy": entropy,
            }
        except Exception:
            return default_result

    def to_vector(self, features: dict[str, float]) -> list[float]:
        feature_names = sorted(features.keys())
        return [features[name] for name in feature_names]

    def get_feature_names(self) -> list[str]:
        sample_text = "The quick brown fox jumps over the lazy dog."
        features = self.extract(sample_text)
        return sorted(features.keys())


class StylometricDetector(CalibratedDetectorMixin, BaseDetector):
    name = "stylometric"
    latency_tier = "fast"
    domains = ["prose", "academic"]

    def __init__(self):
        self.extractor = FeatureExtractor()

    def _extract_features(self, text: str) -> list[float]:
        features = self.extractor.extract(text)
        return cast(list[float], self.extractor.to_vector(features))

    def _extract_feature_names(self) -> list[str]:
        return cast(list[str], self.extractor.get_feature_names())

    def detect(self, text: str) -> DetectorResult:
        features = self.extractor.extract(text)
        vector = self.extractor.to_vector(features)

        calibrated = self._get_calibrated_score(text)
        if calibrated is not None:
            score, confidence = calibrated
        else:
            if features.get("std_sentence_length", 0) < 3:
                score = 0.7
                confidence = 0.6
            elif features.get("ttr", 1) < 0.4:
                score = 0.6
                confidence = 0.5
            elif features.get("hapax_ratio", 0) < 0.2:
                score = 0.55
                confidence = 0.5
            else:
                score = 0.3
                confidence = 0.4

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "features": features,
                "vector": vector,
                "calibrated": calibrated is not None,
            },
        )


def register(registry) -> None:
    registry.register(StylometricDetector)
