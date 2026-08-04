"""Tests for cognitive signature detector (transitions, word lists, short/empty text)."""

import pytest

from provenance.core.base import DetectorResult


class TestTransitionWordDetection:
    @pytest.fixture
    def detector(self):
        from provenance.detectors.stylometric.cognitive import CognitiveDetector

        return CognitiveDetector()

    def test_detects_transition_words(self, detector):
        text = (
            "Furthermore, the results show improvement. "
            "Moreover, the approach is scalable. "
            "However, there are limitations. "
            "Therefore, we propose a new method. "
            "Consequently, the system performs better."
        )
        patterns = detector._extract_transition_patterns(text)
        assert patterns["transition_density"] > 0.0
        assert patterns["transition_uniformity"] > 0.0

    def test_no_transitions_in_natural_text(self, detector):
        text = (
            "I went to the store. I bought some milk. "
            "The weather was nice. My friend called me."
        )
        patterns = detector._extract_transition_patterns(text)
        assert patterns["transition_density"] == 0.0

    def test_transition_uniformity_high_for_uniform_text(self, detector):
        sentences = []
        transition_words = [
            "Furthermore", "Moreover", "Additionally",
            "However", "Therefore", "Consequently",
            "Nevertheless", "Meanwhile", "Similarly",
            "Specifically",
        ]
        for tw in transition_words:
            sentences.append(f"{tw}, the data supports this conclusion.")
        text = " ".join(sentences)

        patterns = detector._extract_transition_patterns(text)
        assert patterns["transition_uniformity"] > 0.8

    def test_transition_density_empty_text(self, detector):
        patterns = detector._extract_transition_patterns("")
        assert patterns["transition_density"] == 0.0
        assert patterns["transition_uniformity"] == 0.0


class TestAdvancedWordList:
    @pytest.fixture
    def detector(self):
        from provenance.detectors.stylometric.cognitive import CognitiveDetector

        return CognitiveDetector()

    def test_detects_advanced_vocabulary(self, detector):
        text = (
            "The predominant paradigm inherently requires substantial fundamental analysis. "
            "Furthermore, the comprehensive intrinsic properties demonstrate significant patterns. "
            "Subsequently, the observations predominantly indicate an inherent systematic bias."
        )
        richness = detector._extract_vocabulary_richness(text)
        assert richness["advanced_word_ratio"] > 0.02

    def test_natural_text_low_advanced_ratio(self, detector):
        text = (
            "I went to the park with my dog. We played fetch for a while. "
            "Then we sat on the bench and watched the sunset. It was a nice day."
        )
        richness = detector._extract_vocabulary_richness(text)
        assert richness["advanced_word_ratio"] < 0.02

    def test_ttr_computed_correctly(self, detector):
        text = "cat dog bird fish cat dog bird fish cat dog bird fish"
        richness = detector._extract_vocabulary_richness(text)
        assert richness["ttr"] == pytest.approx(4 / 12, abs=0.01)

    def test_empty_text_returns_zeros(self, detector):
        richness = detector._extract_vocabulary_richness("")
        assert richness["ttr"] == 0.0
        assert richness["advanced_word_ratio"] == 0.0


class TestShortEmptyTextHandling:
    @pytest.fixture
    def detector(self):
        from provenance.detectors.stylometric.cognitive import CognitiveDetector

        return CognitiveDetector()

    def test_short_text_returns_low_confidence(self, detector):
        result = detector.detect("Short text here.")
        assert result.score == 0.5
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_empty_text_returns_low_confidence(self, detector):
        result = detector.detect("")
        assert result.score == 0.5
        assert result.confidence == 0.0

    def test_minimal_text_returns_low_confidence(self, detector):
        result = detector.detect("One two three four five six seven eight nine ten eleven twelve")
        assert result.score == 0.5
        assert result.confidence == 0.0

    def test_long_text_produces_real_score(self, detector):
        text = (
            "The implementation of advanced machine learning algorithms has transformed "
            "the field of natural language processing. Furthermore, the utilization of "
            "transformer architectures has enabled remarkable capabilities in text generation. "
            "Moreover, these models demonstrate substantial improvements in downstream tasks. "
            "Consequently, organizations are increasingly adopting these technologies. "
            "Additionally, the research community continues to explore novel approaches. "
            "Therefore, the field is evolving rapidly with significant implications."
        )
        result = detector.detect(text)
        assert isinstance(result, DetectorResult)
        assert result.score != 0.5 or result.confidence > 0.0


class TestParagraphStructure:
    @pytest.fixture
    def detector(self):
        from provenance.detectors.stylometric.cognitive import CognitiveDetector

        return CognitiveDetector()

    def test_regular_paragraphs_flagged(self, detector):
        paragraphs = []
        for _ in range(5):
            paragraphs.append(
                "This is a paragraph with exactly twenty words in it to test regularity. "
                "Each paragraph has the same number of words as all others."
            )
        text = "\n\n".join(paragraphs)
        structure = detector._extract_paragraph_structure(text)
        assert structure["paragraph_length_cv"] < 0.1

    def test_single_paragraph_returns_zero(self, detector):
        text = "Just one paragraph with no line breaks at all."
        structure = detector._extract_paragraph_structure(text)
        assert structure["paragraph_count"] == 0

    def test_irregular_paragraphs_high_cv(self, detector):
        paragraphs = [
            "Short.",
            "This is a much longer paragraph with many more words to increase the variance in length significantly.",
            "Medium length paragraph.",
        ]
        text = "\n\n".join(paragraphs)
        structure = detector._extract_paragraph_structure(text)
        assert structure["paragraph_length_cv"] > 0.3


class TestStructuralPerfection:
    @pytest.fixture
    def detector(self):
        from provenance.detectors.stylometric.cognitive import CognitiveDetector

        return CognitiveDetector()

    def test_repeated_first_words_detected(self, detector):
        paragraphs = []
        for _ in range(4):
            paragraphs.append(
                "Furthermore, this paragraph starts with the same word. "
                "It continues with more text to make it longer."
            )
        text = "\n\n".join(paragraphs)
        result = detector._extract_structural_perfection(text)
        assert result["pattern_repetition"] == 1.0

    def test_varied_first_words_low_repetition(self, detector):
        paragraphs = [
            "First paragraph starts here with some content.",
            "Second paragraph begins differently than others.",
            "Third paragraph has its own unique beginning.",
            "Finally, the last paragraph wraps things up.",
        ]
        text = "\n\n".join(paragraphs)
        result = detector._extract_structural_perfection(text)
        assert result["pattern_repetition"] == 0.25
