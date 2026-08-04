"""Tests for HuggingFace classifier detectors (label mapping, allowlist, mocked models)."""

from unittest.mock import MagicMock, patch

import pytest

from provenance.core.base import DetectorResult
from provenance.core.errors import DetectorInitError


class MockPipelineResult:
    def __init__(self, label, score):
        self.label = label
        self.score = score

    def __getitem__(self, key):
        return [{"label": self.label, "score": self.score}][key]


def _make_mock_pipeline(label="FAKE", score=0.85):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": label, "score": score}]
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pipe.model = mock_model
    mock_pipe.tokenizer = mock_tokenizer
    return mock_pipe


@pytest.fixture
def mock_transformers():
    mock_transformers_mod = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    mock_pipeline_fn = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_tokenizer = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "transformers": mock_transformers_mod,
            "transformers.AutoModelForSequenceClassification": mock_auto_model,
            "transformers.AutoTokenizer": mock_auto_tokenizer,
            "torch": mock_torch,
        },
    ):
        yield {
            "transformers": mock_transformers_mod,
            "torch": mock_torch,
            "pipeline": mock_pipeline_fn,
            "auto_model": mock_auto_model,
        }


class TestHuggingFaceLabelMapping:
    @patch("provenance.detectors.transformer.hf_classifier.AutoModelForSequenceClassification", True)
    @patch("provenance.detectors.transformer.hf_classifier.AutoTokenizer", True)
    @patch("provenance.detectors.transformer.hf_classifier.pipeline")
    def test_human_label_inverts_score(self, mock_pipeline_fn):
        mock_pipe = _make_mock_pipeline(label="HUMAN", score=0.9)
        mock_pipeline_fn.return_value = mock_pipe

        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        det = HuggingFaceClassifierDetector.__new__(HuggingFaceClassifierDetector)
        det.model_id = "test"
        det.classifier = mock_pipe
        result = det.detect("Some text")
        assert result.score == pytest.approx(0.1, abs=0.01)

    @patch("provenance.detectors.transformer.hf_classifier.AutoModelForSequenceClassification", True)
    @patch("provenance.detectors.transformer.hf_classifier.AutoTokenizer", True)
    @patch("provenance.detectors.transformer.hf_classifier.pipeline")
    def test_fake_label_keeps_score(self, mock_pipeline_fn):
        mock_pipe = _make_mock_pipeline(label="FAKE", score=0.85)
        mock_pipeline_fn.return_value = mock_pipe

        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        det = HuggingFaceClassifierDetector.__new__(HuggingFaceClassifierDetector)
        det.model_id = "test"
        det.classifier = mock_pipe
        result = det.detect("Some text")
        assert result.score == pytest.approx(0.85, abs=0.01)

    @patch("provenance.detectors.transformer.hf_classifier.AutoModelForSequenceClassification", True)
    @patch("provenance.detectors.transformer.hf_classifier.AutoTokenizer", True)
    @patch("provenance.detectors.transformer.hf_classifier.pipeline")
    def test_real_label_inverts_score(self, mock_pipeline_fn):
        mock_pipe = _make_mock_pipeline(label="REAL", score=0.75)
        mock_pipeline_fn.return_value = mock_pipe

        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        det = HuggingFaceClassifierDetector.__new__(HuggingFaceClassifierDetector)
        det.model_id = "test"
        det.classifier = mock_pipe
        result = det.detect("Some text")
        assert result.score == pytest.approx(0.25, abs=0.01)

    @patch("provenance.detectors.transformer.hf_classifier.AutoModelForSequenceClassification", True)
    @patch("provenance.detectors.transformer.hf_classifier.AutoTokenizer", True)
    @patch("provenance.detectors.transformer.hf_classifier.pipeline")
    def test_gpt_label_keeps_score(self, mock_pipeline_fn):
        mock_pipe = _make_mock_pipeline(label="ChatGPT", score=0.7)
        mock_pipeline_fn.return_value = mock_pipe

        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        det = HuggingFaceClassifierDetector.__new__(HuggingFaceClassifierDetector)
        det.model_id = "test"
        det.classifier = mock_pipe
        result = det.detect("Some text")
        assert result.score == pytest.approx(0.7, abs=0.01)

    @patch("provenance.detectors.transformer.hf_classifier.AutoModelForSequenceClassification", True)
    @patch("provenance.detectors.transformer.hf_classifier.AutoTokenizer", True)
    @patch("provenance.detectors.transformer.hf_classifier.pipeline")
    def test_unknown_label_uses_raw_score(self, mock_pipeline_fn):
        mock_pipe = _make_mock_pipeline(label="POSITIVE", score=0.6)
        mock_pipeline_fn.return_value = mock_pipe

        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        det = HuggingFaceClassifierDetector.__new__(HuggingFaceClassifierDetector)
        det.model_id = "test"
        det.classifier = mock_pipe
        result = det.detect("Some text")
        assert result.score == pytest.approx(0.6, abs=0.01)


class TestModelAllowlist:
    def test_model_registry_contains_approved_models(self):
        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        registry = HuggingFaceClassifierDetector.MODEL_REGISTRY
        assert "openai_detector" in registry
        assert "chatgpt_detector" in registry
        assert "radar" in registry

    def test_model_registry_values_are_hf_ids(self):
        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        registry = HuggingFaceClassifierDetector.MODEL_REGISTRY
        for key, model_id in registry.items():
            assert "/" in model_id, f"Model {key} should be a HuggingFace model ID"

    @patch("provenance.detectors.transformer.hf_classifier.AutoModelForSequenceClassification", True)
    @patch("provenance.detectors.transformer.hf_classifier.AutoTokenizer", True)
    @patch("provenance.detectors.transformer.hf_classifier.pipeline")
    def test_custom_model_id_accepted(self, mock_pipeline_fn):
        mock_pipe = _make_mock_pipeline()
        mock_pipeline_fn.return_value = mock_pipe

        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        det = HuggingFaceClassifierDetector(model_id="customorg/custom-model")
        assert det.model_id == "customorg/custom-model"


class TestInitErrors:
    def test_missing_transformers_raises_init_error(self):
        with patch.dict("sys.modules", {"transformers": None}):
            import importlib
            import provenance.detectors.transformer.hf_classifier as mod

            importlib.reload(mod)
            with pytest.raises(DetectorInitError, match="transformers is required"):
                mod.HuggingFaceClassifierDetector()

    @patch("provenance.detectors.transformer.hf_classifier.AutoModelForSequenceClassification", True)
    @patch("provenance.detectors.transformer.hf_classifier.AutoTokenizer", True)
    @patch("provenance.detectors.transformer.hf_classifier.pipeline")
    def test_pipeline_failure_raises_init_error(self, mock_pipeline_fn):
        mock_pipeline_fn.side_effect = OSError("Model not found")

        from provenance.detectors.transformer.hf_classifier import (
            HuggingFaceClassifierDetector,
        )

        with pytest.raises(DetectorInitError, match="Failed to initialize"):
            HuggingFaceClassifierDetector()
