"""Cognitive signature detection for humanizer-resistant AI text detection."""

from __future__ import annotations

import re
import statistics
from collections import Counter

import spacy

from provenance.core.base import BaseDetector, DetectorResult


class CognitiveDetector(BaseDetector):
    """Detect AI text using cognitive signatures.

    Analyzes structural logic patterns that persist after vocabulary swaps:
    - Paragraph structure regularity
    - Transition pattern uniformity
    - Argument flow predictability
    - "Too perfect" structural patterns
    """

    name = "cognitive_signature"
    latency_tier = "fast"
    domains = ["prose", "academic"]

    def __init__(self, nlp=None):
        try:
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
        except Exception:
            self.nlp = nlp

    def _extract_paragraph_structure(self, text: str) -> dict[str, float]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            return {"paragraph_count": 0, "paragraph_length_cv": 0.0}

        lengths = [len(p.split()) for p in paragraphs]
        mean_len = statistics.mean(lengths)
        std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
        cv = std_len / mean_len if mean_len > 0 else 0

        return {
            "paragraph_count": len(paragraphs),
            "paragraph_length_cv": cv,
        }

    def _extract_transition_patterns(self, text: str) -> dict[str, float]:
        transition_words = {
            "furthermore", "moreover", "additionally", "however",
            "therefore", "thus", "consequently", "nevertheless",
            "meanwhile", "similarly", "conversely", "specifically",
            "in conclusion", "finally", "lastly", "in contrast",
            "on the other hand", "for example", "for instance",
            "in addition", "as a result", "accordingly",
        }

        text_lower = text.lower()
        sentences = re.split(r"[.!?]+", text_lower)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return {"transition_density": 0.0, "transition_uniformity": 0.0}

        transitions_per_sentence = []
        for sent in sentences:
            count = sum(1 for tw in transition_words if tw in sent)
            transitions_per_sentence.append(count)

        total_transitions = sum(transitions_per_sentence)
        transition_density = total_transitions / len(sentences)

        sentences_with_transitions = sum(1 for c in transitions_per_sentence if c > 0)
        transition_uniformity = (
            sentences_with_transitions / len(sentences) if sentences else 0
        )

        return {
            "transition_density": transition_density,
            "transition_uniformity": transition_uniformity,
        }

    def _extract_argument_flow(self, text: str) -> dict[str, float]:
        if self.nlp is None:
            return {"clause_complexity": 0.0, "sentence_depth_cv": 0.0}

        try:
            doc = self.nlp(text[:10000])
        except Exception:
            return {"clause_complexity": 0.0, "sentence_depth_cv": 0.0}

        sentences = list(doc.sents)
        if len(sentences) < 2:
            return {"clause_complexity": 0.0, "sentence_depth_cv": 0.0}

        clause_counts = []
        for sent in sentences:
            n_clauses = sum(
                1 for token in sent
                if token.dep_ in {"advcl", "acl", "ccomp", "xcomp", "relcl"}
            )
            clause_counts.append(n_clauses + 1)

        mean_clauses = statistics.mean(clause_counts)
        std_clauses = statistics.stdev(clause_counts) if len(clause_counts) > 1 else 0
        cv = std_clauses / mean_clauses if mean_clauses > 0 else 0

        return {
            "clause_complexity": mean_clauses,
            "sentence_depth_cv": cv,
        }

    def _extract_structural_perfection(self, text: str) -> dict[str, float]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            return {"structure_regularity": 0.0, "pattern_repetition": 0.0}

        lengths = [len(p.split()) for p in paragraphs]
        mean_len = statistics.mean(lengths)
        std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
        cv = std_len / mean_len if mean_len > 0 else 0

        structure_regularity = 1.0 - min(1.0, cv)

        first_words = [p.split()[0].lower() if p.split() else "" for p in paragraphs]
        word_counts = Counter(first_words)
        max_repetition = max(word_counts.values()) if word_counts else 0
        pattern_repetition = max_repetition / len(paragraphs) if paragraphs else 0

        return {
            "structure_regularity": structure_regularity,
            "pattern_repetition": pattern_repetition,
        }

    def _extract_vocabulary_richness(self, text: str) -> dict[str, float]:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return {"ttr": 0.0, "advanced_word_ratio": 0.0}

        unique_words = set(words)
        ttr = len(unique_words) / len(words)

        advanced_words = {
            "furthermore", "moreover", "consequently", "nevertheless",
            "nonetheless", "accordingly", "subsequently", "predominantly",
            "predominant", "substantial", "significant", "comprehensive",
            "fundamental", "inherent", "intrinsic", "paradigm",
        }
        advanced_count = sum(1 for w in words if w in advanced_words)
        advanced_ratio = advanced_count / len(words)

        return {
            "ttr": ttr,
            "advanced_word_ratio": advanced_ratio,
        }

    def detect(self, text: str) -> DetectorResult:
        if len(text.split()) < 20:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Text too short for cognitive analysis"},
            )

        para_features = self._extract_paragraph_structure(text)
        trans_features = self._extract_transition_patterns(text)
        flow_features = self._extract_argument_flow(text)
        struct_features = self._extract_structural_perfection(text)
        vocab_features = self._extract_vocabulary_richness(text)

        all_features = {**para_features, **trans_features, **flow_features, **struct_features, **vocab_features}

        ai_score = 0.0
        n_signals = 0

        if para_features.get("paragraph_length_cv", 1) < 0.3:
            ai_score += 0.2
            n_signals += 1

        if trans_features.get("transition_uniformity", 0) > 0.6:
            ai_score += 0.15
            n_signals += 1

        if flow_features.get("sentence_depth_cv", 1) < 0.3:
            ai_score += 0.15
            n_signals += 1

        if struct_features.get("structure_regularity", 0) > 0.7:
            ai_score += 0.2
            n_signals += 1

        if struct_features.get("pattern_repetition", 0) > 0.3:
            ai_score += 0.1
            n_signals += 1

        if vocab_features.get("advanced_word_ratio", 0) > 0.02:
            ai_score += 0.1
            n_signals += 1

        if n_signals > 0:
            score = min(1.0, ai_score)
        else:
            score = 0.5

        confidence = min(0.9, 0.4 + 0.1 * n_signals)

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata=all_features,
        )


def register(registry=None) -> None:
    registry.register(CognitiveDetector)
