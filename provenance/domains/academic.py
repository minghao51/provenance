"""Academic domain adapter for scientific text detection."""

from __future__ import annotations

import re

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin

try:
    from transformers import pipeline
except ImportError:
    pipeline = None  # type: ignore[assignment,misc]


class AcademicDetector(CalibratedDetectorMixin, BaseDetector):
    name = "academic_detector"
    latency_tier = "medium"
    domains = ["academic"]

    CITATION_PATTERNS = [
        r"\([A-Z][a-z]+,?\s*\d{4}[a-z]?\)",
        r"\[[A-Z][a-z]+,?\s*\d{4}[a-z]?\]",
        r"\d+\s*\(\d{4}\)",
    ]

    HALLUCINATED_CITATION_INDICATORS = [
        r"\d{4}\)",
        r"[A-Z]\.\s*[A-Z]\.",
    ]

    def __init__(self):
        self.citation_model = None

    def _extract_citations(self, text: str) -> list[str]:
        citations = []
        for pattern in self.CITATION_PATTERNS:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        return citations

    def _check_citation_formatting(self, citations: list[str]) -> dict[str, float]:
        if not citations:
            return {"citation_format_score": 0.5, "potential_hallucinations": 0}

        valid_count = 0
        suspicious_count = 0

        for cite in citations:
            is_valid = True
            for susp_pattern in self.HALLUCINATED_CITATION_INDICATORS:
                if re.search(susp_pattern, cite):
                    is_valid = False
                    suspicious_count += 1
                    break
            if is_valid:
                valid_count += 1

        total = len(citations)
        format_score = valid_count / total if total > 0 else 0.5

        return {
            "citation_format_score": format_score,
            "potential_hallucinations": suspicious_count / total if total > 0 else 0,
        }

    def _analyze_language_complexity(self, text: str) -> dict[str, float]:
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        if not words:
            return {"avg_word_length": 0, "latin_abbreviation_ratio": 0}

        avg_word_length = sum(len(w) for w in words) / len(words)

        latin_patterns = [
            r"\be\.g\.",
            r"\bi\.e\.",
            r"\bet\s+al\.",
            r"\bvs\.",
            r"\bapprox\.",
        ]
        latin_count = sum(
            len(re.findall(p, text, re.IGNORECASE)) for p in latin_patterns
        )
        latin_ratio = latin_count / len(words)

        passive_pattern = r"\b(was|were|been|being)\s+\w+ed\b"
        passive_count = len(re.findall(passive_pattern, text, re.IGNORECASE))
        passive_ratio = passive_count / len(words)

        return {
            "avg_word_length": avg_word_length,
            "latin_abbreviation_ratio": latin_ratio,
            "passive_ratio": passive_ratio,
        }

    def _analyze_claim_language(self, text: str) -> dict[str, float]:
        hedging_words = [
            "might",
            "may",
            "could",
            "would",
            "possibly",
            "probably",
            "suggest",
            "indicate",
            "appear",
            "seem",
            "potentially",
            "approximately",
            "roughly",
            "generally",
            "typically",
        ]
        certainty_words = [
            "definitely",
            "certainly",
            "obviously",
            "clearly",
            "prove",
            "demonstrate",
            "establish",
            "confirm",
            "always",
            "never",
        ]

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return {"hedging_ratio": 0, "certainty_ratio": 0}

        word_set = set(words)
        hedging_count = sum(1 for w in word_set if w in hedging_words)
        certainty_count = sum(1 for w in word_set if w in certainty_words)

        return {
            "hedging_ratio": hedging_count / len(word_set),
            "certainty_ratio": certainty_count / len(word_set),
        }

    def _extract_features(self, text: str) -> list[float]:
        citations = self._extract_citations(text)
        citation_analysis = self._check_citation_formatting(citations)
        complexity = self._analyze_language_complexity(text)
        claims = self._analyze_claim_language(text)
        features = {**citation_analysis, **complexity, **claims}
        return [features.get(k, 0.0) for k in self._extract_feature_names()]

    def _extract_feature_names(self) -> list[str]:
        return [
            "citation_format_score",
            "potential_hallucinations",
            "avg_word_length",
            "latin_abbreviation_ratio",
            "passive_ratio",
            "hedging_ratio",
            "certainty_ratio",
        ]

    def detect(self, text: str) -> DetectorResult:
        citations = self._extract_citations(text)
        citation_analysis = self._check_citation_formatting(citations)
        complexity = self._analyze_language_complexity(text)
        claims = self._analyze_claim_language(text)

        features = {**citation_analysis, **complexity, **claims}

        calibrated = self._get_calibrated_score(text)
        if calibrated is not None:
            score, confidence = calibrated
        else:
            hallucination_risk = citation_analysis.get("potential_hallucinations", 0)
            if hallucination_risk > 0.3:
                score = 0.75
                confidence = 0.7
            elif (
                citation_analysis.get("citation_format_score", 0.5) < 0.5
                and len(citations) > 3
            ):
                score = 0.6
                confidence = 0.6
            elif claims.get("hedging_ratio", 0) < 0.02 and len(text) > 500:
                score = 0.55
                confidence = 0.5
            else:
                score = 0.35
                confidence = 0.5

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "citations_found": len(citations),
                **features,
                "calibrated": calibrated is not None,
            },
        )


def register(registry) -> None:
    registry.register(AcademicDetector)
