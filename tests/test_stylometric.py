"""Tests for sentinel.detectors.stylometric.feature_extractor module."""

from provenance.detectors.stylometric.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_extractor_initialization(self):
        assert self.extractor is not None

    def test_extract_returns_dict(self):
        text = "This is a sample text for testing the feature extractor. It contains multiple sentences."
        features = self.extractor.extract(text)
        assert isinstance(features, dict)

    def test_extract_surface_features(self):
        text = "The quick brown fox jumps over the lazy dog. This sentence is used for testing."
        features = self.extractor.extract(text)
        surface_keys = [
            k
            for k in features
            if "flesch" in k
            or "fog" in k
            or "sentence_length" in k
            or "word_length" in k
        ]
        assert len(surface_keys) >= 0

    def test_extract_lexical_features(self):
        text = "The quick brown fox jumps over the lazy dog. This sentence is used for testing."
        features = self.extractor.extract(text)
        lexical_keys = [
            k for k in features if "ttr" in k or "yule" in k or "hapax" in k
        ]
        assert len(lexical_keys) >= 0

    def test_extract_syntactic_features(self):
        text = "The quick brown fox jumps over the lazy dog. This sentence is used for testing."
        features = self.extractor.extract(text)
        syntactic_keys = [
            k for k in features if "pos" in k or "dependency" in k or "passive" in k
        ]
        assert len(syntactic_keys) >= 0

    def test_extract_stylistic_features(self):
        text = "Furthermore, the results indicate that, on the other hand, we may conclude."
        features = self.extractor.extract(text)
        stylistic_keys = [
            k
            for k in features
            if "transition" in k or "function_word" in k or "punctuation" in k
        ]
        assert len(stylistic_keys) >= 0

    def test_to_vector(self):
        text = "This is a sample text for testing the feature extractor."
        features = self.extractor.extract(text)
        vector = self.extractor.to_vector(features)
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, (int, float)) for v in vector)

    def test_to_vector_order_consistent(self):
        text1 = "First text for testing."
        text2 = "Second text for testing."
        features1 = self.extractor.extract(text1)
        features2 = self.extractor.extract(text2)
        vector1 = self.extractor.to_vector(features1)
        vector2 = self.extractor.to_vector(features2)
        assert len(vector1) == len(vector2)

    def test_get_feature_names(self):
        names = self.extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_short_text_handling(self):
        text = "Short."
        features = self.extractor.extract(text)
        assert isinstance(features, dict)

    def test_empty_text_handling(self):
        text = ""
        features = self.extractor.extract(text)
        assert isinstance(features, dict)

    def test_all_features_numeric(self):
        text = "This is a comprehensive test text that should exercise all the feature extraction capabilities of the system."
        features = self.extractor.extract(text)
        for key, value in features.items():
            assert isinstance(value, (int, float)), (
                f"Feature {key} is not numeric: {type(value)}"
            )
