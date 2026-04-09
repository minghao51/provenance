"""Benchmark suite for evaluating detectors on standard datasets."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from provenance.core.base import BaseDetector

from provenance.benchmarks.evaluator import BenchmarkEvaluator
from provenance.benchmarks.loaders import HuggingFaceDatasetLoader
from provenance.benchmarks.models import BenchmarkResult
from provenance.benchmarks.registry import DatasetRegistry, register_default_datasets

register_default_datasets()


class BenchmarkHarness:
    def __init__(
        self, dataset_path: str | None = None, sample_limit: int | None = None
    ):
        self.dataset_path = dataset_path
        self.results: list[BenchmarkResult] = []
        self.sample_limit = sample_limit
        self.detector: BaseDetector | None = None
        self.dataset_loader = HuggingFaceDatasetLoader()
        self.evaluator = BenchmarkEvaluator(dataset_loader=self.dataset_loader)

    def load_dataset(self, name: str) -> tuple[list[str], list[int]]:
        config = DatasetRegistry.get(name)
        if config is None:
            raise ValueError(f"Unknown dataset: {name}")
        try:
            texts, labels, _ = self.dataset_loader.load(
                config, sample_limit=self.sample_limit
            )
        except Exception:
            return [], []
        return texts, labels

    def evaluate(
        self,
        detector,
        texts: list[str],
        labels: list[int],
        threshold: float = 0.5,
        dataset_name: str = "unknown",
    ) -> BenchmarkResult:
        result = self.evaluator.evaluate_detector(
            detector,
            texts,
            labels,
            threshold=threshold,
            dataset_name=dataset_name,
        )
        self.results.append(result)
        return result

    def audit_fpr(
        self,
        texts: list[str],
        labels: list[int],
        demographic: list[str] | None = None,
    ) -> dict:
        non_native_mask = [d == "non-native" for d in (demographic or [])]
        native_mask = [d == "native" for d in (demographic or [])]

        results = {}

        if any(non_native_mask):
            nt_texts = [
                text
                for text, is_non_native in zip(texts, non_native_mask, strict=False)
                if is_non_native
            ]
            nt_labels = [
                label
                for label, is_non_native in zip(labels, non_native_mask, strict=False)
                if is_non_native
            ]
            results["non_native_fpr"] = self._compute_fpr_simple(nt_texts, nt_labels)

        if any(native_mask):
            n_texts = [
                text
                for text, is_native in zip(texts, native_mask, strict=False)
                if is_native
            ]
            n_labels = [
                label
                for label, is_native in zip(labels, native_mask, strict=False)
                if is_native
            ]
            results["native_fpr"] = self._compute_fpr_simple(n_texts, n_labels)

        return results

    def _compute_fpr_simple(self, texts: list[str], labels: list[int]) -> float:
        if not texts:
            return 0.0

        detector = getattr(self, "detector", None)
        if detector is None:
            try:
                from provenance.detectors.stylometric import StylometricDetector

                detector = StylometricDetector()
            except Exception:
                return 0.0

        fp = 0
        n = 0
        scores = self.evaluator.score_texts(detector, texts)
        for label, score in zip(labels, scores, strict=False):
            if label == 0:
                n += 1
                if score >= 0.5:
                    fp += 1

        return fp / n if n > 0 else 0.0

    def generate_report(self, results: list[BenchmarkResult]) -> str:
        lines = ["# Benchmark Results\n"]

        for result in results:
            lines.append(f"\n## {result.detector_name}\n")
            lines.append(f"- **Dataset**: {result.dataset}")
            lines.append(f"- **AUROC**: {result.auroc:.4f}")
            lines.append(f"- **F1**: {result.f1:.4f}")
            lines.append(f"- **TPR@1%FPR**: {result.tpr_at_1fpr:.4f}")
            lines.append(f"- **TPR@5%FPR**: {result.tpr_at_5fpr:.4f}")
            lines.append(f"- **FPR@10%TPR**: {result.fpr_at_10tpr:.4f}")
            lines.append(f"- **Precision**: {result.precision:.4f}")
            lines.append(f"- **Recall**: {result.recall:.4f}")
            lines.append(f"- **Accuracy**: {result.accuracy:.4f}")
            lines.append(f"- **Samples**: {result.num_samples}")

        return "\n".join(lines)

    def save_results(self, results: list[BenchmarkResult], path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                [
                    {
                        "detector_name": r.detector_name,
                        "dataset": r.dataset,
                        "auroc": r.auroc,
                        "f1": r.f1,
                        "tpr_at_1fpr": r.tpr_at_1fpr,
                        "tpr_at_5fpr": r.tpr_at_5fpr,
                        "fpr_at_10tpr": r.fpr_at_10tpr,
                        "precision": r.precision,
                        "recall": r.recall,
                        "accuracy": r.accuracy,
                        "num_samples": r.num_samples,
                        "num_positives": r.num_positives,
                        "num_negatives": r.num_negatives,
                        "eval_time_seconds": r.eval_time_seconds,
                        "metadata": r.metadata,
                        "stratified_results": r.stratified_results,
                    }
                    for r in results
                ],
                f,
                indent=2,
            )


def run_audit(
    detector,
    texts: list[str],
    labels: list[int],
    languages: list[str] | None = None,
) -> dict:
    harness = BenchmarkHarness()
    if detector is not None:
        harness.detector = detector
    return harness.audit_fpr(texts, labels, languages)
