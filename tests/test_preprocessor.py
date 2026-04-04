"""Tests for sentinel.core.preprocessor module."""

from provenance.core.preprocessor import PreprocessedText, Preprocessor, TextChunk


class TestPreprocessor:
    def test_preprocessor_init(self):
        p = Preprocessor()
        assert p.window_size == 512
        assert p.window_overlap == 128
        assert p.min_chunk_length == 50

    def test_normalize_nfc(self):
        p = Preprocessor()
        text = "café"
        normalized = p.normalize(text)
        import unicodedata

        assert unicodedata.is_normalized("NFC", normalized)

    def test_normalize_whitespace(self):
        p = Preprocessor()
        text = "hello    world\n\n\ttab"
        normalized = p.normalize(text)
        assert "\n" not in normalized
        assert "\t" not in normalized
        assert "  " not in normalized

    def test_detect_language_english(self):
        p = Preprocessor()
        lang = p.detect_language("This is a simple English sentence for testing.")
        assert lang in ["en", "unknown"]

    def test_detect_language_german(self):
        p = Preprocessor()
        lang = p.detect_language("Dies ist ein einfacher deutscher Satz zum Testen.")
        assert lang in ["de", "unknown"]

    def test_detect_language_french(self):
        p = Preprocessor()
        lang = p.detect_language("Ceci est une phrase française pour tester.")
        assert lang in ["fr", "unknown"]

    def test_split_sentences(self):
        p = Preprocessor()
        text = "Hello world. This is a test. How are you?"
        sentences = p.split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "Hello world."
        assert sentences[1] == "This is a test."
        assert sentences[2] == "How are you?"

    def test_split_sentences_empty(self):
        p = Preprocessor()
        sentences = p.split_sentences("")
        assert sentences == []

    def test_chunk_text_short_text(self):
        p = Preprocessor()
        text = "Short text."
        chunks = p.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_chunk_text_long_text(self):
        p = Preprocessor(window_size=50, window_overlap=10, min_chunk_length=20)
        text = (
            "This is a longer piece of text that should be split into multiple chunks. "
            * 10
        )
        chunks = p.chunk_text(text)
        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.start_char < c.end_char for c in chunks)

    def test_chunk_text_respects_overlap(self):
        p = Preprocessor(window_size=50, window_overlap=10, min_chunk_length=20)
        text = (
            "This is a longer piece of text that should be split into multiple chunks. "
            * 10
        )
        chunks = p.chunk_text(text)
        if len(chunks) > 1:
            assert chunks[1].start_char < chunks[0].end_char

    def test_preprocess_full(self):
        p = Preprocessor()
        text = "Hello world. This is a test. How are you?"
        result = p.preprocess(text)
        assert isinstance(result, PreprocessedText)
        assert result.original_text == text
        assert result.language in ["en", "unknown"]
        assert len(result.sentences) == 3
        assert len(result.chunks) >= 1

    def test_preprocess_iter(self):
        p = Preprocessor(window_size=50, window_overlap=10, min_chunk_length=20)
        text = (
            "This is a longer piece of text that should be split into multiple chunks. "
            * 10
        )
        chunks = list(p.preprocess_iter(text))
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)
