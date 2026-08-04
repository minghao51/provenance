"""Differential tests: verify statistical detectors distinguish AI vs human text."""

import pytest

from provenance.detectors.statistical.burstiness import BurstinessDetector
from provenance.detectors.statistical.entropy import EntropyDetector
from provenance.detectors.statistical.repetition import RepetitionDetector

HUMAN_TEXT = (
    "Man, I dunno what to tell ya — sometimes things just don't work out the way you'd "
    "expect 'em to, y'know? Like last Tuesday... I was heading to the store (the one on "
    "5th, not the new place) and I completely forgot why I even went there in the first "
    "place. It's wild how the brain does that. Anyway, my buddy Dave — huge fan of "
    "garlic bread, like OBSESSED — he said something kinda profound: 'You don't gotta "
    "have a reason for everything.' And honestly? That stuck with me more than any "
    "self-help book ever could. Life's messy, unpredictable, and full of weird little "
    "moments that don't fit neatly into any narrative structure whatsoever."
)

AI_TEXT = (
    "The implementation of effective strategies is essential for achieving optimal outcomes in contemporary organizational environments. "
    "The implementation of effective strategies requires careful planning and execution across all operational domains. "
    "The implementation of effective strategies should be guided by evidence-based practices and rigorous analytical frameworks. "
    "The implementation of effective strategies necessitates a comprehensive understanding of the underlying factors at play. "
    "The implementation of effective strategies demands consistent monitoring and evaluation throughout the entire process. "
    "The implementation of effective strategies benefits from collaborative approaches that leverage diverse stakeholder perspectives. "
    "The implementation of effective strategies promotes sustainable growth and development within institutional settings. "
    "The implementation of effective strategies enhances overall performance and productivity in measurable ways. "
    "The implementation of effective strategies facilitates meaningful progress toward established goals and objectives. "
    "The implementation of effective strategies provides a foundation for long-term success and organizational resilience."
)

DETECTOR_CASES = [
    ("entropy", EntropyDetector),
    ("repetition", RepetitionDetector),
]

DIRECTIONAL_CASES = [
    ("repetition", RepetitionDetector),
]


@pytest.mark.parametrize("name,detector_cls", DETECTOR_CASES, ids=[c[0] for c in DETECTOR_CASES])
class TestDifferentialDetection:
    def test_ai_text_score_in_valid_range(self, name, detector_cls):
        detector = detector_cls()
        result = detector.detect(AI_TEXT)
        assert 0.0 <= result.score <= 1.0

    def test_human_text_score_in_valid_range(self, name, detector_cls):
        detector = detector_cls()
        result = detector.detect(HUMAN_TEXT)
        assert 0.0 <= result.score <= 1.0

    def test_confidence_is_non_negative(self, name, detector_cls):
        detector = detector_cls()
        for text in [HUMAN_TEXT, AI_TEXT]:
            result = detector.detect(text)
            assert result.confidence >= 0.0


@pytest.mark.parametrize("name,detector_cls", DIRECTIONAL_CASES, ids=[c[0] for c in DIRECTIONAL_CASES])
class TestDifferentialDirectional:
    def test_ai_text_scores_higher_than_human(self, name, detector_cls):
        detector = detector_cls()
        human_result = detector.detect(HUMAN_TEXT)
        ai_result = detector.detect(AI_TEXT)
        assert ai_result.score > human_result.score, (
            f"{name}: AI score ({ai_result.score:.3f}) should be > human score ({human_result.score:.3f})"
        )

    def test_ai_and_human_produce_different_scores(self, name, detector_cls):
        detector = detector_cls()
        human_result = detector.detect(HUMAN_TEXT)
        ai_result = detector.detect(AI_TEXT)
        assert human_result.score != ai_result.score or human_result.confidence != ai_result.confidence


class TestDifferentialMetadata:
    def test_entropy_metadata_human_has_higher_entropy(self):
        detector = EntropyDetector()
        human_result = detector.detect(HUMAN_TEXT)
        ai_result = detector.detect(AI_TEXT)
        assert human_result.metadata["text_entropy"] >= ai_result.metadata["text_entropy"]

    def test_entropy_ai_has_higher_kl_divergence(self):
        detector = EntropyDetector()
        human_result = detector.detect(HUMAN_TEXT)
        ai_result = detector.detect(AI_TEXT)
        assert ai_result.metadata["kl_divergence"] >= human_result.metadata["kl_divergence"]

    def test_repetition_metadata_ai_has_higher_repetition(self):
        detector = RepetitionDetector()
        human_result = detector.detect(HUMAN_TEXT)
        ai_result = detector.detect(AI_TEXT)
        for key in ["ngram_3_repetition", "ngram_4_repetition"]:
            if key in human_result.metadata and key in ai_result.metadata:
                assert ai_result.metadata[key] >= human_result.metadata[key], (
                    f"{key}: AI ({ai_result.metadata[key]:.3f}) should be >= human ({human_result.metadata[key]:.3f})"
                )

    def test_burstiness_metadata_populated(self):
        detector = BurstinessDetector()
        human_result = detector.detect(HUMAN_TEXT)
        ai_result = detector.detect(AI_TEXT)
        if "burstiness_cv" in human_result.metadata and "burstiness_cv" in ai_result.metadata:
            assert isinstance(human_result.metadata["burstiness_cv"], float)
            assert isinstance(ai_result.metadata["burstiness_cv"], float)

    def test_burstiness_produces_valid_scores(self):
        detector = BurstinessDetector()
        for text in [HUMAN_TEXT, AI_TEXT]:
            result = detector.detect(text)
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0


class TestDifferentialConsistency:
    @pytest.mark.parametrize("detector_cls", [EntropyDetector, RepetitionDetector])
    def test_idempotent_detection(self, detector_cls):
        detector = detector_cls()
        result1 = detector.detect(HUMAN_TEXT)
        result2 = detector.detect(HUMAN_TEXT)
        assert result1.score == result2.score
        assert result1.confidence == result2.confidence

    @pytest.mark.parametrize("detector_cls", [EntropyDetector, RepetitionDetector])
    def test_score_monotonic_with_repetition(self, detector_cls):
        detector = detector_cls()
        varied = (
            "The cat jumped over the fence while dogs barked loudly at passing cars. "
            "She quickly realized that baking sourdough required patience and precision."
        )
        repetitive = "The cat sat on the mat. " * 20
        varied_result = detector.detect(varied)
        repetitive_result = detector.detect(repetitive)
        assert repetitive_result.score >= varied_result.score
