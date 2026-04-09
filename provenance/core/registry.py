from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import threading
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .base import BaseDetector

logger = logging.getLogger(__name__)

_registry_lock = threading.Lock()


class DetectorRegistry:
    _instance: DetectorRegistry | None = None
    _detectors: dict[str, type[BaseDetector]]
    _entry_points_loaded: bool = False

    def __new__(cls) -> DetectorRegistry:
        if cls._instance is None:
            with _registry_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._detectors = {}
                    cls._instance._entry_points_loaded = False
        return cls._instance

    def register(self, detector_class: type[BaseDetector]) -> None:
        with _registry_lock:
            self._detectors[detector_class.name] = detector_class

    def get(self, name: str) -> BaseDetector | None:
        with _registry_lock:
            detector_class = self._detectors.get(name)
        if detector_class is None:
            return None
        try:
            return detector_class()
        except Exception as e:
            logger.warning(f"Failed to initialize detector {name}: {e}")
            return None

    def list_detectors(
        self,
        latency_tier: Literal["fast", "medium", "slow"] | None = None,
        domain: str | None = None,
    ) -> list[BaseDetector]:
        with _registry_lock:
            detector_classes = list(self._detectors.values())
        results: list[BaseDetector] = []
        for detector_class in detector_classes:
            try:
                detector = detector_class()
            except Exception as e:
                logger.warning(
                    f"Failed to initialize detector {detector_class.__name__}: {e}"
                )
                continue
            if latency_tier is not None and detector.latency_tier != latency_tier:
                continue
            if domain is not None and domain not in detector.domains:
                continue
            results.append(detector)
        return results

    def load_entry_points(self, force: bool = False) -> None:
        with _registry_lock:
            already_loaded = self._entry_points_loaded
        if already_loaded and not force:
            return
        if os.environ.get("PROVENANCE_SKIP_ENTRY_POINTS"):
            with _registry_lock:
                self._entry_points_loaded = True
            return
        try:
            eps = importlib.metadata.entry_points(group="provenance.detectors")
            for ep in eps:
                try:
                    module = importlib.import_module(ep.module)
                    if hasattr(module, "register"):
                        module.register(self)
                except ImportError as e:
                    logger.debug(f"Failed to import entry point {ep.name}: {e}")
                except AttributeError as e:
                    logger.debug(
                        f"Entry point {ep.name} missing register function: {e}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Unexpected error loading entry point {ep.name}: {e}"
                    )
        except (ImportError, OSError) as e:
            logger.debug(f"Failed to load entry points: {e}")
        with _registry_lock:
            self._entry_points_loaded = True

    def clear(self) -> None:
        with _registry_lock:
            self._detectors.clear()
            self._entry_points_loaded = False


def get_registry() -> DetectorRegistry:
    return DetectorRegistry()
