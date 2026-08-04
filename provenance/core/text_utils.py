from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text)


def compute_word_statistics(text: str) -> dict[str, float]:
    words = tokenize_words(text)
    if not words:
        return {"word_count": 0.0, "avg_word_length": 0.0, "word_length_variance": 0.0}
    word_lengths = [len(w) for w in words]
    avg = sum(word_lengths) / len(word_lengths)
    variance = sum((l - avg) ** 2 for l in word_lengths) / len(word_lengths)
    return {
        "word_count": float(len(words)),
        "avg_word_length": avg,
        "word_length_variance": variance,
    }
