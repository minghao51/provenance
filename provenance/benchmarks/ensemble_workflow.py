"""Benchmark helpers for comparing ensemble strategies."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

from provenance.benchmarks.evaluator import BenchmarkEvaluator
from provenance.benchmarks.models import BenchmarkSuite
from provenance.benchmarks.registry import DatasetRegistry
from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.config import ProvenanceConfig, resolve_provenance_config
from provenance.core.ensemble import Ensemble, EnsembleConfig
from provenance.core.registry import get_registry


@dataclass
class BenchmarkSplit:
    train_texts: list[str]
    train_labels: list[int]
    test_texts: list[str]
    test_labels: list[int]


class EnsembleBenchmarkDetector(BaseDetector):
    latency_tier = "slow"
    domains = ["prose"]

    def __init__(
        self,
        *,
        name: str,
        detectors: list[BaseDetector],
        strategy: Literal["weighted_average", "stacking", "uncertainty_aware"],
        weights: dict[str, float] | None = None,
    ):
        self.name = name
        self.strategy = strategy
        self.ensemble = Ensemble(
            config=EnsembleConfig(strategy=strategy, weights=weights or {})
        )
        for detector in detectors:
            self.ensemble.add_detector(detector)

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        *,
        stacker_method: Literal["isotonic", "platt"] = "platt",
    ) -> None:
        if self.strategy == "stacking":
            self.ensemble.calibrate(texts, labels, method=stacker_method)

    def detect(self, text: str) -> DetectorResult:
        result = self.ensemble.ensemble_detect(text)
        return DetectorResult(
            score=result.score,
            confidence=result.confidence,
            metadata={
                "label": result.label,
                "top_features": result.top_features,
                "feature_vector": result.feature_vector,
                "strategy": self.strategy,
            },
        )


def build_detector_instances(
    detector_names: list[str],
    config: ProvenanceConfig | str | dict | None = None,
) -> list[BaseDetector]:
    provenance_config = resolve_provenance_config(config)
    registry = get_registry()
    registry.load_entry_points()

    detectors: list[BaseDetector] = []
    for name in detector_names:
        detector = registry.get(name)
        if detector is None:
            continue
        if hasattr(detector, "load_calibration"):
            calibration_path = provenance_config.detector_calibration_paths.get(
                detector.name
            )
            if calibration_path:
                detector.load_calibration(calibration_path)
            elif (
                provenance_config.calibration_model_dir
                and hasattr(detector, "load_default_calibration")
            ):
                detector.load_default_calibration(provenance_config.calibration_model_dir)
        detectors.append(detector)

    return detectors


def stratified_train_test_split(
    texts: list[str],
    labels: list[int],
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> BenchmarkSplit:
    if not texts or len(texts) != len(labels):
        raise ValueError("texts and labels must be non-empty and aligned")

    try:
        from sklearn.model_selection import train_test_split

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )
    except Exception:
        split_index = max(1, min(len(texts) - 1, int(len(texts) * (1 - test_size))))
        train_texts = texts[:split_index]
        test_texts = texts[split_index:]
        train_labels = labels[:split_index]
        test_labels = labels[split_index:]

    return BenchmarkSplit(
        train_texts=train_texts,
        train_labels=[int(label) for label in train_labels],
        test_texts=test_texts,
        test_labels=[int(label) for label in test_labels],
    )


def validate_binary_split(split: BenchmarkSplit) -> None:
    train_classes = set(split.train_labels)
    test_classes = set(split.test_labels)
    if len(train_classes) < 2:
        raise ValueError(
            "Training split must contain both classes for stacker fitting. "
            "Increase --limit or adjust --test-size."
        )
    if len(test_classes) < 2:
        raise ValueError(
            "Held-out split must contain both classes for benchmark metrics. "
            "Increase --limit or adjust --test-size."
        )


def benchmark_ensemble_strategies(
    *,
    detector_names: list[str],
    dataset_name: str,
    config: ProvenanceConfig | str | dict | None = None,
    sample_limit: int | None = None,
    threshold: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
    stacker_method: Literal["isotonic", "platt"] = "platt",
) -> BenchmarkSuite:
    dataset_config = DatasetRegistry.get(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    evaluator = BenchmarkEvaluator()
    texts, labels, _ = evaluator.dataset_loader.load(
        dataset_config, sample_limit=sample_limit, seed=seed
    )
    paired = list(zip(texts, labels, strict=False))
    random.Random(seed).shuffle(paired)
    texts = [text for text, _ in paired]
    labels = [int(label) for _, label in paired]
    split = stratified_train_test_split(texts, labels, test_size=test_size, seed=seed)
    validate_binary_split(split)

    base_detectors = build_detector_instances(detector_names, config=config)
    if not base_detectors:
        raise ValueError("No valid detectors available for ensemble comparison")

    strategies: list[tuple[str, Literal["weighted_average", "stacking", "uncertainty_aware"]]] = [
        ("calibrated_weighted_average", "weighted_average"),
        ("uncertainty_aware_ensemble", "uncertainty_aware"),
        ("learned_stacker", "stacking"),
    ]

    results = []
    for model_name, strategy in strategies:
        detectors = build_detector_instances(detector_names, config=config)
        model = EnsembleBenchmarkDetector(
            name=model_name,
            detectors=detectors,
            strategy=strategy,
        )
        model.fit(
            split.train_texts,
            split.train_labels,
            stacker_method=stacker_method,
        )
        result = evaluator.evaluate_detector(
            model,
            split.test_texts,
            split.test_labels,
            threshold=threshold,
            dataset_name=dataset_name,
            show_progress=False,
        )
        result.metadata["train_samples"] = len(split.train_texts)
        result.metadata["test_samples"] = len(split.test_texts)
        result.metadata["base_detectors"] = detector_names
        results.append(result)

    return BenchmarkSuite(
        name=f"ensemble_comparison_{dataset_name}",
        description="Comparison of calibrated ensemble strategies on a held-out split.",
        results=results,
        config={
            "detectors": detector_names,
            "datasets": [dataset_name],
            "threshold": threshold,
            "sample_limit": sample_limit,
            "test_size": test_size,
            "seed": seed,
            "stacker_method": stacker_method,
        },
    )
