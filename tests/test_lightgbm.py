"""Tests for LightGBM detector (default model, feature mismatch, SHAP handling)."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from provenance.core.errors import DetectorInitError, ModelNotFoundError


class TestDefaultModelWarning:
    @patch("provenance.detectors.stylometric.lightgbm_detector.lgb")
    @patch("provenance.detectors.stylometric.lightgbm_detector.shap")
    @patch(
        "provenance.detectors.stylometric.feature_extractor.FeatureExtractor",
        create=True,
    )
    def test_default_model_uses_dummy_data(self, mock_fe_cls, mock_shap, mock_lgb):
        from provenance.detectors.stylometric.feature_extractor import FeatureExtractor

        with patch(
            "provenance.detectors.stylometric.feature_extractor.FeatureExtractor",
            mock_fe_cls,
        ):
            mock_fe = MagicMock()
            mock_fe.get_feature_names.return_value = ["feat_a", "feat_b"]
            mock_fe_cls.return_value = mock_fe

            mock_model = MagicMock()
            mock_lgb.train.return_value = mock_model
            mock_lgb.Dataset.return_value = MagicMock()
            mock_shap.Explainer.return_value = MagicMock()

            from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector

            det = LightGBMDetector()
            assert det.model is mock_model
            mock_lgb.train.assert_called_once()

    @patch("provenance.detectors.stylometric.lightgbm_detector.lgb")
    @patch("provenance.detectors.stylometric.lightgbm_detector.shap")
    def test_default_model_predicts_with_extracted_features(self, mock_shap, mock_lgb):
        mock_fe = MagicMock()
        mock_fe.get_feature_names.return_value = ["feat_a", "feat_b"]
        mock_fe.extract.return_value = {"feat_a": 1.0, "feat_b": 2.0}
        mock_fe.to_vector.return_value = [1.0, 2.0]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.65]
        mock_lgb.train.return_value = mock_model
        mock_lgb.Dataset.return_value = MagicMock()

        mock_explainer = MagicMock()
        mock_sv = MagicMock()
        mock_sv.values = [MagicMock()]
        mock_sv.values[0].tolist.return_value = [0.1, -0.2]
        mock_explainer.return_value = mock_sv
        mock_shap.Explainer.return_value = mock_explainer

        with patch(
            "provenance.detectors.stylometric.feature_extractor.FeatureExtractor",
            return_value=mock_fe,
        ):
            from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector

            det = LightGBMDetector()
            result = det.detect("Some text to analyze")
            assert result.score == 0.65
            assert result.confidence > 0.0


class TestFeatureDimensionMismatch:
    @patch("provenance.detectors.stylometric.lightgbm_detector.lgb")
    @patch("provenance.detectors.stylometric.lightgbm_detector.shap")
    def test_short_vector_padded(self, mock_shap, mock_lgb):
        mock_fe = MagicMock()
        mock_fe.get_feature_names.return_value = ["a", "b", "c"]
        mock_fe.extract.return_value = {"a": 1.0, "b": 2.0}
        mock_fe.to_vector.return_value = [1.0, 2.0]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_lgb.train.return_value = mock_model
        mock_lgb.Dataset.return_value = MagicMock()

        mock_explainer = MagicMock()
        mock_sv = MagicMock()
        mock_sv.values = [MagicMock()]
        mock_sv.values[0].tolist.return_value = [0.1, 0.2, 0.3]
        mock_explainer.return_value = mock_sv
        mock_shap.Explainer.return_value = mock_explainer

        with patch(
            "provenance.detectors.stylometric.feature_extractor.FeatureExtractor",
            return_value=mock_fe,
        ):
            from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector

            det = LightGBMDetector()
            result = det.detect("Some text")

            predict_args = mock_model.predict.call_args[0][0]
            assert len(predict_args[0]) == 3
            assert predict_args[0][2] == 0.0

    @patch("provenance.detectors.stylometric.lightgbm_detector.lgb")
    @patch("provenance.detectors.stylometric.lightgbm_detector.shap")
    def test_long_vector_truncated(self, mock_shap, mock_lgb):
        mock_fe = MagicMock()
        mock_fe.get_feature_names.return_value = ["a", "b"]
        mock_fe.extract.return_value = {"a": 1.0, "b": 2.0, "c": 3.0}
        mock_fe.to_vector.return_value = [1.0, 2.0, 3.0]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_lgb.train.return_value = mock_model
        mock_lgb.Dataset.return_value = MagicMock()

        mock_explainer = MagicMock()
        mock_sv = MagicMock()
        mock_sv.values = [MagicMock()]
        mock_sv.values[0].tolist.return_value = [0.1, 0.2]
        mock_explainer.return_value = mock_sv
        mock_shap.Explainer.return_value = mock_explainer

        with patch(
            "provenance.detectors.stylometric.feature_extractor.FeatureExtractor",
            return_value=mock_fe,
        ):
            from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector

            det = LightGBMDetector()
            result = det.detect("Some text")

            predict_args = mock_model.predict.call_args[0][0]
            assert len(predict_args[0]) == 2


class TestSHAPDependency:
    def test_missing_shap_raises_init_error(self):
        with patch.dict("sys.modules", {"lightgbm": MagicMock(), "shap": None}):
            import importlib
            import provenance.detectors.stylometric.lightgbm_detector as mod

            importlib.reload(mod)
            with pytest.raises(DetectorInitError, match="lightgbm and shap"):
                mod.LightGBMDetector()

    def test_missing_lightgbm_raises_init_error(self):
        with patch.dict("sys.modules", {"lightgbm": None, "shap": MagicMock()}):
            import importlib
            import provenance.detectors.stylometric.lightgbm_detector as mod

            importlib.reload(mod)
            with pytest.raises(DetectorInitError, match="lightgbm and shap"):
                mod.LightGBMDetector()


class TestMissingModelFile:
    @patch("provenance.detectors.stylometric.lightgbm_detector.lgb")
    @patch("provenance.detectors.stylometric.lightgbm_detector.shap")
    def test_nonexistent_model_path_raises(self, mock_shap, mock_lgb):
        from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector

        with pytest.raises(ModelNotFoundError):
            LightGBMDetector(model_path="/nonexistent/path/model.pkl")


class TestDetectWithoutModel:
    @patch("provenance.detectors.stylometric.lightgbm_detector.lgb")
    @patch("provenance.detectors.stylometric.lightgbm_detector.shap")
    def test_detect_with_none_model_returns_default(self, mock_shap, mock_lgb):
        mock_fe = MagicMock()
        mock_fe.get_feature_names.return_value = ["a"]

        mock_model = MagicMock()
        mock_lgb.train.return_value = mock_model
        mock_lgb.Dataset.return_value = MagicMock()
        mock_shap.Explainer.return_value = MagicMock()

        with patch(
            "provenance.detectors.stylometric.feature_extractor.FeatureExtractor",
            return_value=mock_fe,
        ):
            from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector

            det = LightGBMDetector()
            det.model = None
            result = det.detect("Some text")
            assert result.score == 0.5
            assert result.confidence == 0.0
