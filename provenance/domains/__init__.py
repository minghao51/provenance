"""Domain-specific detectors for code, academic, and multilingual text."""

from provenance.domains.academic import AcademicDetector
from provenance.domains.code import CodeDetector
from provenance.domains.multilingual import MultilingualDetector

__all__ = ["CodeDetector", "AcademicDetector", "MultilingualDetector"]


def register(registry) -> None:
    registry.register(CodeDetector)
    registry.register(AcademicDetector)
    registry.register(MultilingualDetector)
