"""Standardized exception hierarchy for detector errors."""

from __future__ import annotations


class DetectorError(Exception):
    """Base exception for all detector-related errors.

    This exception and its subclasses are used for initialization errors
    that should prevent detector creation.
    """

    pass


class ModelNotFoundError(DetectorError):
    """Raised when a detector model file is not found.

    This is an initialization error - the detector cannot function
    without its model file.
    """

    def __init__(self, message: str, model_path: str | None = None):
        super().__init__(message)
        self.model_path = model_path


class DetectorInitError(DetectorError):
    """Raised when a detector fails to initialize.

    This is used for initialization errors that are not related to
    missing model files (e.g., dependency issues, invalid configuration).
    """

    pass


class DetectionError(RuntimeError):
    """Runtime error for detection failures.

    Note: This is typically not raised directly. Detection errors should
    return a default DetectorResult with error metadata instead of raising
    exceptions. This class is provided for consistency if needed.
    """

    pass
