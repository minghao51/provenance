from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import langdetect
import spacy


@lru_cache(maxsize=1)
def _get_default_nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


@dataclass
class TextChunk:
    text: str
    start_char: int
    end_char: int
    chunk_index: int


@dataclass
class PreprocessedText:
    original_text: str
    normalized_text: str
    language: str
    sentences: list[str]
    chunks: list[TextChunk]


class Preprocessor:
    def __init__(
        self,
        nlp=None,
        window_size: int = 512,
        window_overlap: int = 128,
        min_chunk_length: int = 50,
    ):
        self.nlp = nlp or _get_default_nlp()
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.min_chunk_length = min_chunk_length

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def detect_language(self, text: str) -> str:
        try:
            return str(langdetect.detect(text))
        except Exception:
            return "unknown"

    def split_sentences(self, text: str) -> list[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences

    def chunk_text(self, text: str) -> list[TextChunk]:
        if len(text) <= self.min_chunk_length:
            return [
                TextChunk(
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    chunk_index=0,
                )
            ]

        chunks: list[TextChunk] = []
        start = 0
        chunk_index = 0
        stride = self.window_size - self.window_overlap

        while start < len(text):
            end = start + self.window_size
            if end > len(text):
                end = len(text)
                start = max(0, end - self.window_size)

            chunk_text = text[start:end]
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break
            start += stride

        return chunks

    def preprocess(self, text: str) -> PreprocessedText:
        normalized = self.normalize(text)
        language = self.detect_language(normalized)
        sentences = self.split_sentences(normalized)
        chunks = self.chunk_text(normalized)

        return PreprocessedText(
            original_text=text,
            normalized_text=normalized,
            language=language,
            sentences=sentences,
            chunks=chunks,
        )

    def preprocess_iter(self, text: str) -> Iterator[TextChunk]:
        yield from self.chunk_text(text)

    def tokenize_words(self, text: str) -> list[str]:
        """Extract words from text, lowercase them.

        Args:
            text: Input text.

        Returns:
            List of lowercase words.
        """
        return re.findall(r"\b[a-zA-Z]+\b", text.lower())
