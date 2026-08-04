"""Pytest fixtures and configuration for Sentinel tests."""

import os
import sys
from pathlib import Path

import pytest

os.environ["PROVENANCE_SKIP_ENTRY_POINTS"] = "1"

provenance_path = Path(__file__).parent.parent
sys.path.insert(0, str(provenance_path))


@pytest.fixture
def registry_isolation():
    from provenance.core.registry import DetectorRegistry

    reg = DetectorRegistry()
    saved_detectors = dict(reg._detectors)
    saved_ep_loaded = reg._entry_points_loaded
    yield reg
    with reg._detectors.__class__.__mro__[1].__dict__.get(
        "_registry_lock", None
    ) or reg.__class__.__dict__.get("_instance", None):
        pass
    reg._detectors = saved_detectors
    reg._entry_points_loaded = saved_ep_loaded


@pytest.fixture
def sample_human_text():
    return """
    The quick brown fox jumps over the lazy dog. This sentence has been used
    for centuries to test typewriters and printing presses because it contains
    every letter of the alphabet at least once. It's remarkable how such a
    simple phrase can be so useful for testing purposes. Writers often use it
    to check their instruments, and teachers use it to help students learn the
    alphabet. The fox and dog make for an interesting pair of characters,
    one wild and one domesticated, representing the spectrum of animal life
    that humans have relationships with.
    """.strip()


@pytest.fixture
def sample_ai_text():
    return """
    The utilization of canids in agricultural contexts represents a historical
    practice predating modern industrialization. Specifically, the Vulpes genus
    demonstrates remarkable adaptability across diverse ecological niches.
    Conversely, domestic Canis lupus familiaris exhibits evolved social cognition
    facilitating cooperative interactions with Homo sapiens. The phenotypic
    variance between these taxa underscores evolutionary divergence while
    maintaining ancestral genomic commonality. Furthermore, the predator-prey
    dynamic manifests contextually rather than categorically in anthropocentric
    environments. Additionally, domesticated specimens exhibit heightened
    prolactin expression correlating with affiliative behavioral modulation.
    """.strip()


@pytest.fixture
def sample_short_text():
    return "This is a short text."


@pytest.fixture
def sample_code_text():
    return """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)

    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """.strip()


@pytest.fixture
def sample_long_text():
    paragraphs = []
    base = (
        "The development of natural language processing has undergone significant "
        "transformation over the past several decades. Researchers have explored "
        "various approaches to understanding and generating human language, from "
        "rule-based systems to statistical methods and now deep learning architectures. "
        "Each paradigm shift has brought both advantages and challenges to the field. "
        "Modern transformer models have demonstrated remarkable capabilities in text "
        "generation, comprehension, and analysis tasks."
    )
    for i in range(12):
        paragraphs.append(f"{base} Passage number {i + 1} adds additional variation.")
    return "\n\n".join(paragraphs)


@pytest.fixture
def sample_multilingual_text():
    return {
        "en": (
            "The quick brown fox jumps over the lazy dog. This sentence contains "
            "every letter of the English alphabet and has been used for testing purposes "
            "for many decades. Writers and typists alike have found it useful."
        ),
        "es": (
            "El veloz murciélago hindú comía feliz cardillo y kiwi. La cigüeña tocaba "
            "el saxofón detrás del palenque de paja. Esta es una oración de prueba."
        ),
        "fr": (
            "Portez ce vieux whisky au juge blond qui fume sur son île intérieure. "
            "À côté de l'île, une femme prépare un délicieux repas pour sa famille."
        ),
        "de": (
            "Victor jagt zwölf Boxkämpfer quer über den großen Sylter Deich. "
            "Dies ist ein Testsatz, der alle Buchstaben des Alphabets enthält."
        ),
    }


@pytest.fixture
def sample_mixed_text():
    human_part = (
        "I went to the store yesterday and bought some groceries. "
        "The weather was nice and I enjoyed the walk. "
        "My neighbor said hello and we chatted for a bit about the local sports team."
    )
    ai_part = (
        "Furthermore, the implementation of advanced algorithms facilitates "
        "the optimization of resource allocation. Consequently, organizations "
        "can achieve substantial improvements in operational efficiency. "
        "Additionally, the utilization of machine learning models enables "
        "predictive analytics capabilities."
    )
    return f"{human_part}\n\n{ai_part}"
