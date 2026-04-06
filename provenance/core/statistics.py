"""Statistical utility functions for provenance detectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def compute_mean_variance_std(values: Sequence[float]) -> tuple[float, float, float]:
    """Compute mean, variance, and standard deviation of values.

    Args:
        values: Sequence of numeric values.

    Returns:
        Tuple of (mean, variance, standard_deviation).
    """
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5
    return mean, variance, std


def compute_cv(
    values: Sequence[float], mean: float | None = None, std: float | None = None
) -> float:
    """Compute coefficient of variation.

    Args:
        values: Sequence of numeric values.
        mean: Pre-computed mean (optional).
        std: Pre-computed standard deviation (optional).

    Returns:
        Coefficient of variation (std/mean), or 0.0 if mean <= 0.
    """
    if mean is None or std is None:
        mean, _, std = compute_mean_variance_std(values)
    return std / mean if mean > 0 else 0.0
