"""Benchmark suite for evaluating detectors on standard datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from provenance.core.base import BaseDetector

from provenance.benchmarks.metrics import (
    compute_accuracy,
    compute_auroc_fallback,
    compute_f1,
    compute_fpr_at_tpr_fallback,
    compute_precision,
    compute_recall,
)


@dataclass
class BenchmarkResult:
    detector_name: str
    dataset: str
    auroc: float
    f1: float
    fpr_at_10tpr: float
    precision: float
    recall: float
    accuracy: float
    num_samples: int
    metadata: dict


class BenchmarkHarness:
    def __init__(
        self, dataset_path: str | None = None, sample_limit: int | None = None
    ):
        self.dataset_path = dataset_path
        self.results: list[BenchmarkResult] = []
        self.sample_limit = sample_limit
        self.detector: BaseDetector | None = None

    def load_dataset(self, name: str) -> tuple[list[str], list[int]]:
        if name == "hc3":
            return self._load_hc3()
        elif name == "raid":
            return self._load_raid()
        elif name == "m4":
            return self._load_m4()
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def _load_hc3(self) -> tuple[list[str], list[int]]:
        try:
            from datasets import load_dataset

            ds = load_dataset("Hello-SimpleAI/HC3", name="all", split="validation")
            texts = []
            labels = []
            for item in ds:
                text = item.get("text", "")
                label = item.get("label", "")
                if text and label:
                    texts.append(text)
                    labels.append(0 if label == "human" else 1)
            if self.sample_limit:
                texts = texts[: self.sample_limit]
                labels = labels[: self.sample_limit]
            return texts, labels
        except Exception:
            return [], []

    def _load_raid(self) -> tuple[list[str], list[int]]:
        try:
            from datasets import load_dataset

            ds = load_dataset("liamdugan/raid", split="train")

            texts = []
            labels = []
            for item in ds:
                if "text" in item and "label" in item:
                    texts.append(item["text"])
                    labels.append(int(item["label"]))
            if self.sample_limit:
                texts = texts[: self.sample_limit]
                labels = labels[: self.sample_limit]
            return texts, labels
        except Exception:
            return [], []

    def _load_m4(self) -> tuple[list[str], list[int]]:
        try:
            from datasets import load_dataset

            ds = load_dataset("NickyNicky/M4", split="train")
            texts = []
            labels = []
            for item in ds:
                if "text" in item and "label" in item:
                    texts.append(item["text"])
                    labels.append(int(item["label"]))
            if self.sample_limit:
                texts = texts[: self.sample_limit]
                labels = labels[: self.sample_limit]
            return texts, labels
        except Exception:
            return [], []

    def _compute_auroc(self, y_true: list[int], y_score: list[float]) -> float:
        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return compute_auroc_fallback(y_true, y_score)

    def _compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_score: list[float],
    ) -> dict:
        accuracy = compute_accuracy(y_true, y_pred)
        precision = compute_precision(y_true, y_pred)
        recall = compute_recall(y_true, y_pred)
        f1 = compute_f1(precision, recall)

        try:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_true, y_score)
            fpr_at_10tpr = float(np.interp(0.9, tpr, fpr))
        except Exception:
            fpr_at_10tpr = compute_fpr_at_tpr_fallback(y_true, y_score, 0.9)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr_at_10tpr": fpr_at_10tpr,
        }

    def evaluate(
        self,
        detector,
        texts: list[str],
        labels: list[int],
        threshold: float = 0.5,
        dataset_name: str = "unknown",
    ) -> BenchmarkResult:
        y_score = []
        for text in texts:
            try:
                result = detector.detect(text)
                y_score.append(result.score)
            except Exception:
                y_score.append(0.5)

        y_pred = [1 if s >= threshold else 0 for s in y_score]

        metrics = self._compute_metrics(labels, y_pred, y_score)
        auroc = self._compute_auroc(labels, y_score)

        return BenchmarkResult(
            detector_name=detector.name,
            dataset=dataset_name,
            auroc=auroc,
            f1=metrics["f1"],
            fpr_at_10tpr=metrics["fpr_at_10tpr"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            accuracy=metrics["accuracy"],
            num_samples=len(texts),
            metadata={},
        )

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
        for text, label in zip(texts, labels, strict=False):
            if label == 0:
                n += 1
                result = detector.detect(text)
                if result.score >= 0.5:
                    fp += 1

        return fp / n if n > 0 else 0.0

    def generate_report(self, results: list[BenchmarkResult]) -> str:
        lines = ["# Benchmark Results\n"]

        for result in results:
            lines.append(f"\n## {result.detector_name}\n")
            lines.append(f"- **Dataset**: {result.dataset}")
            lines.append(f"- **AUROC**: {result.auroc:.4f}")
            lines.append(f"- **F1**: {result.f1:.4f}")
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
                        "fpr_at_10tpr": r.fpr_at_10tpr,
                        "precision": r.precision,
                        "recall": r.recall,
                        "accuracy": r.accuracy,
                        "num_samples": r.num_samples,
                        "metadata": r.metadata,
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
