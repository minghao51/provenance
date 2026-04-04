"""Tests for domain adapters."""

from provenance.domains.academic import AcademicDetector
from provenance.domains.code import CodeDetector
from provenance.domains.multilingual import MultilingualDetector


class TestCodeDetector:
    def setup_method(self):
        self.detector = CodeDetector()

    def test_valid_python_code(self, sample_code_text):
        result = self.detector.detect(sample_code_text)
        assert 0.0 <= result.score <= 1.0
        assert result.confidence >= 0.0

    def test_non_code_text(self):
        result = self.detector.detect(
            "This is just a regular paragraph with no code at all."
        )
        assert result.score == 0.5
        assert result.confidence == 0.3

    def test_code_with_functions(self):
        code = """
def add(a, b):
    return a + b

def multiply(a, b):
    result = a * b
    return result
"""
        result = self.detector.detect(code)
        assert "function_count" in result.metadata
        assert result.metadata["function_count"] == 2

    def test_code_comment_ratio(self):
        code = """
# This function adds two numbers
# It's very useful
def add(a, b):
    return a + b
"""
        result = self.detector.detect(code)
        assert "comment_to_code_ratio" in result.metadata


class TestAcademicDetector:
    def setup_method(self):
        self.detector = AcademicDetector()

    def test_academic_text_with_citations(self):
        text = """
        Recent studies have shown significant progress in natural language processing (Smith, 2023).
        Furthermore, Johnson et al. (2022) demonstrated that transformer architectures outperform
        previous approaches. As noted by [Williams, 2021], this trend is expected to continue.
        """
        result = self.detector.detect(text)
        assert result.metadata["citations_found"] > 0

    def test_text_without_citations(self):
        text = (
            "This is a simple paragraph without any academic citations or references."
        )
        result = self.detector.detect(text)
        assert result.metadata["citations_found"] == 0

    def test_hedging_language(self):
        text = """
        The results might suggest that this approach could potentially improve performance.
        It appears that the method seems to work in most cases, although further research
        would be needed to confirm these findings.
        """
        result = self.detector.detect(text)
        assert result.metadata.get("hedging_ratio", 0) > 0

    def test_certainty_language(self):
        text = """
        This study clearly demonstrates that the proposed method definitely outperforms
        all existing approaches. We prove that this is always the best solution and
        never fails under any circumstances.
        """
        result = self.detector.detect(text)
        assert result.metadata.get("certainty_ratio", 0) > 0


class TestMultilingualDetector:
    def setup_method(self):
        self.detector = MultilingualDetector()

    def test_english_text(self):
        text = "This is a simple English text for testing the multilingual detector capabilities."
        result = self.detector.detect(text)
        assert result.metadata["detected_language"] == "en"
        assert result.metadata["language_family"] == "germanic"

    def test_short_text(self):
        result = self.detector.detect("Short.")
        assert result.confidence == 0.0

    def test_language_family_thresholds(self):
        assert self.detector.BURSTINESS_THRESHOLDS["germanic"] == 0.35
        assert self.detector.BURSTINESS_THRESHOLDS["romance"] == 0.40
        assert self.detector.BURSTINESS_THRESHOLDS["cjk"] == 0.25
