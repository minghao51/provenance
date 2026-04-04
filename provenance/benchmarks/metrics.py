"""Shared metric computation functions for benchmark evaluation."""

from __future__ import annotations

import numpy as np


def compute_auroc_fallback(y_true: list[int], y_score: list[float]) -> float:
    positives = sum(1 for label in y_true if label == 1)
    negatives = sum(1 for label in y_true if label == 0)
    if positives == 0 or negatives == 0 or len(y_true) != len(y_score):
        return 0.5

    order = np.argsort(np.asarray(y_score, dtype=float))
    sorted_scores = np.asarray(y_score, dtype=float)[order]
    ranks = np.zeros(len(y_score), dtype=float)

    start = 0
    while start < len(sorted_scores):
        end = start
        while (
            end + 1 < len(sorted_scores)
            and sorted_scores[end + 1] == sorted_scores[start]
        ):
            end += 1
        avg_rank = (start + end + 2) / 2
        ranks[order[start : end + 1]] = avg_rank
        start = end + 1

    positive_ranks = sum(
        rank for rank, label in zip(ranks.tolist(), y_true, strict=False) if label == 1
    )
    auc = (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def compute_auprc_fallback(y_true: list[int], y_score: list[float]) -> float:
    positives = sum(1 for label in y_true if label == 1)
    if positives == 0 or len(y_true) != len(y_score):
        return 0.0

    thresholds = sorted({float(score) for score in y_score}, reverse=True)
    precisions = [1.0]
    recalls = [0.0]

    for threshold in thresholds:
        predictions = [1 if float(score) >= threshold else 0 for score in y_score]
        precisions.append(compute_precision(y_true, predictions))
        recalls.append(compute_recall(y_true, predictions))

    precisions.append(0.0)
    recalls.append(1.0)
    return float(np.trapz(precisions, recalls))


def compute_fpr_at_tpr_fallback(
    y_true: list[int], y_score: list[float], target_tpr: float
) -> float:
    positives = sum(1 for label in y_true if label == 1)
    negatives = sum(1 for label in y_true if label == 0)
    if positives == 0 or negatives == 0 or len(y_true) != len(y_score):
        return 1.0

    thresholds = sorted({float(score) for score in y_score}, reverse=True)
    thresholds.append(float("-inf"))
    tpr_values: list[float] = []
    fpr_values: list[float] = []

    for threshold in thresholds:
        predictions = [1 if float(score) >= threshold else 0 for score in y_score]
        tp = sum(
            1
            for pred, label in zip(predictions, y_true, strict=False)
            if pred == 1 and label == 1
        )
        fp = sum(
            1
            for pred, label in zip(predictions, y_true, strict=False)
            if pred == 1 and label == 0
        )
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)

    return float(np.interp(target_tpr, tpr_values, fpr_values))


def compute_confusion_matrix_fallback(
    y_true: list[int], y_pred: list[int]
) -> dict[str, int]:
    tn = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 0 and pred == 0
    )
    fp = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 0 and pred == 1
    )
    fn = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 1 and pred == 0
    )
    tp = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 1 and pred == 1
    )
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def compute_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(
        1 for truth, pred in zip(y_true, y_pred, strict=False) if truth == pred
    )
    return correct / len(y_true)


def compute_precision(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 1 and pred == 1
    )
    fp = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 0 and pred == 1
    )
    return tp / (tp + fp) if (tp + fp) else 0.0


def compute_recall(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 1 and pred == 1
    )
    fn = sum(
        1
        for truth, pred in zip(y_true, y_pred, strict=False)
        if truth == 1 and pred == 0
    )
    return tp / (tp + fn) if (tp + fn) else 0.0


def compute_f1(precision: float, recall: float) -> float:
    return (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
