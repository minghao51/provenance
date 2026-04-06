"""PyTorch utility functions for provenance detectors."""

from __future__ import annotations

import torch


def get_torch_device(device: str = "auto") -> str:
    """Get PyTorch device string, auto-detecting if requested.

    Args:
        device: Device string ("auto", "cpu", "cuda", etc.)

    Returns:
        Device string to use with PyTorch.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
