"""Pytest fixtures and configuration for Sentinel tests."""

import os
import sys
from pathlib import Path

import pytest

os.environ["PROVENANCE_SKIP_ENTRY_POINTS"] = "1"

provenance_path = Path(__file__).parent.parent
sys.path.insert(0, str(provenance_path))


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
