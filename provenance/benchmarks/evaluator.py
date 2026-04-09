"""Shared benchmark evaluation primitives."""

from __future__ import annotations

import time

import numpy as np

from provenance.benchmarks.loaders import HuggingFaceDatasetLoader
from provenance.benchmarks.metrics import (
    compute_accuracy,
    compute_auprc_fallback,
    compute_auroc_fallback,
    compute_confusion_matrix_fallback,
    compute_f1,
    compute_fpr_at_tpr_fallback,
    compute_precision,
    compute_recall,
    compute_tpr_at_fpr_fallback,
)
from provenance.benchmarks.models import BenchmarkResult


class BenchmarkEvaluator:
    def __init__(
        self,
        dataset_loader: HuggingFaceDatasetLoader | None = None,
        cache_dir: str | None = None,
    ):
        self.dataset_loader = dataset_loader or HuggingFaceDatasetLoader(cache_dir)

    def compute_auroc(self, y_true: list[int], y_score: list[float]) -> float:
        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return compute_auroc_fallback(y_true, y_score)

    def compute_auprc(self, y_true: list[int], y_score: list[float]) -> float:
        try:
            from sklearn.metrics import average_precision_score

            return float(average_precision_score(y_true, y_score))
        except Exception:
            return compute_auprc_fallback(y_true, y_score)

    def compute_fpr_at_tpr(
        self, y_true: list[int], y_score: list[float], target_tpr: float = 0.9
    ) -> float:
        try:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_true, y_score)
            return float(np.interp(target_tpr, tpr, fpr))
        except Exception:
            return compute_fpr_at_tpr_fallback(y_true, y_score, target_tpr)

    def compute_tpr_at_fpr(
        self, y_true: list[int], y_score: list[float], target_fpr: float
    ) -> float:
        try:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_true, y_score)
            eligible = tpr[fpr <= target_fpr]
            if len(eligible) == 0:
                return 0.0
            return float(np.max(eligible))
        except Exception:
            return compute_tpr_at_fpr_fallback(y_true, y_score, target_fpr)

    def compute_confusion_matrix(self, y_true: list[int], y_pred: list[int]) -> dict:
        try:
            from sklearn.metrics import confusion_matrix

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        except Exception:
            return compute_confusion_matrix_fallback(y_true, y_pred)

    def compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_score: list[float],
    ) -> dict:
        precision = compute_precision(y_true, y_pred)
        recall = compute_recall(y_true, y_pred)
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)

        return {
            "accuracy": compute_accuracy(y_true, y_pred),
            "f1": compute_f1(precision, recall),
            "precision": precision,
            "recall": recall,
            "auroc": self.compute_auroc(y_true, y_score),
            "auprc": self.compute_auprc(y_true, y_score),
            "fpr_at_10tpr": self.compute_fpr_at_tpr(y_true, y_score, 0.9),
            "fpr_at_20tpr": self.compute_fpr_at_tpr(y_true, y_score, 0.8),
            "tpr_at_1fpr": self.compute_tpr_at_fpr(y_true, y_score, 0.01),
            "tpr_at_5fpr": self.compute_tpr_at_fpr(y_true, y_score, 0.05),
            "confusion_matrix": confusion_matrix,
        }

    def score_texts(
        self, detector, texts: list[str], show_progress: bool = False
    ) -> list[float]:
        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(texts, desc=f"Evaluating {detector.name}")
            except ImportError:
                pass

        scores = []
        for text in iterator:
            try:
                result = detector.detect(text)
                scores.append(float(result.score))
            except Exception:
                scores.append(0.5)
        return scores

    def evaluate_scores(
        self,
        detector_name: str,
        y_true: list[int],
        y_score: list[float],
        threshold: float = 0.5,
        dataset_name: str = "unknown",
        eval_time_seconds: float = 0.0,
        metadata: dict | None = None,
        stratified_results: dict | None = None,
    ) -> BenchmarkResult:
        threshold = float(threshold)
        y_pred = [1 if float(score) >= threshold else 0 for score in y_score]
        metrics = self.compute_metrics(y_true, y_pred, y_score)

        return BenchmarkResult(
            detector_name=detector_name,
            dataset=dataset_name,
            auroc=metrics["auroc"],
            f1=metrics["f1"],
            fpr_at_10tpr=metrics["fpr_at_10tpr"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            accuracy=metrics["accuracy"],
            num_samples=len(y_true),
            num_positives=sum(y_true),
            num_negatives=len(y_true) - sum(y_true),
            eval_time_seconds=eval_time_seconds,
            tpr_at_1fpr=metrics["tpr_at_1fpr"],
            tpr_at_5fpr=metrics["tpr_at_5fpr"],
            metadata={
                "auprc": metrics["auprc"],
                "threshold": threshold,
                "confusion_matrix": metrics["confusion_matrix"],
                **(metadata or {}),
            },
            stratified_results=stratified_results,
        )

    def evaluate_detector(
        self,
        detector,
        texts: list[str],
        labels: list[int],
        threshold: float = 0.5,
        dataset_name: str = "unknown",
        show_progress: bool = False,
    ) -> BenchmarkResult:
        start_time = time.time()
        y_score = self.score_texts(detector, texts, show_progress=show_progress)

        return self.evaluate_scores(
            detector_name=detector.name,
            y_true=labels,
            y_score=y_score,
            threshold=threshold,
            dataset_name=dataset_name,
            eval_time_seconds=time.time() - start_time,
            stratified_results={
                "overall": self.compute_metrics(
                    labels,
                    [1 if s >= float(threshold) else 0 for s in y_score],
                    y_score,
                )
            },
        )

    def evaluate_stratified(
        self,
        detector,
        texts: list[str],
        labels: list[int],
        metadata: list[dict],
        threshold: float = 0.5,
        dataset_name: str = "unknown",
    ) -> BenchmarkResult:
        base_result = self.evaluate_detector(
            detector, texts, labels, threshold, dataset_name
        )

        stratified = dict(base_result.stratified_results or {})
        stratify_keys: set[str] = set()
        for item_metadata in metadata:
            stratify_keys.update(item_metadata.keys())

        for key in stratify_keys:
            key_texts, key_labels = [], []
            for text, label, item_metadata in zip(
                texts, labels, metadata, strict=False
            ):
                if key in item_metadata:
                    key_texts.append(text)
                    key_labels.append(label)

            if len(key_texts) >= 10:
                key_scores = self.score_texts(detector, key_texts)
                key_pred = [
                    1 if score >= float(threshold) else 0 for score in key_scores
                ]
                stratified[key] = {
                    "n_samples": len(key_texts),
                    "n_positives": sum(key_labels),
                    **self.compute_metrics(key_labels, key_pred, key_scores),
                }

        short_mask = [len(text.split()) < 150 for text in texts]
        if sum(short_mask) >= 10:
            short_texts = [
                text
                for text, is_short in zip(texts, short_mask, strict=False)
                if is_short
            ]
            short_labels = [
                label
                for label, is_short in zip(labels, short_mask, strict=False)
                if is_short
            ]
            short_scores = self.score_texts(detector, short_texts)
            short_pred = [
                1 if score >= float(threshold) else 0 for score in short_scores
            ]
            stratified["short_texts"] = {
                "n_samples": len(short_texts),
                "n_positives": sum(short_labels),
                **self.compute_metrics(short_labels, short_pred, short_scores),
            }

        base_result.stratified_results = stratified
        return base_result
