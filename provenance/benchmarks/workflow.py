"""Benchmark workflow for evaluating AI text detection techniques."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    load_dataset = None

try:
    import pandas as pd

    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

import numpy as np

from provenance.benchmarks.metrics import (
    compute_accuracy,
    compute_auprc_fallback,
    compute_auroc_fallback,
    compute_confusion_matrix_fallback,
    compute_f1,
    compute_fpr_at_tpr_fallback,
    compute_precision,
    compute_recall,
)


@dataclass
class DatasetConfig:
    name: str
    repo_id: str
    config_name: str | None = None
    split: str = "train"
    text_field: str = "text"
    label_field: str = "label"
    label_map: dict = field(default_factory=lambda: {"human": 0, "ai": 1})
    meta_fields: dict = field(default_factory=dict)
    cache_dir: str | None = None


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
    num_positives: int
    num_negatives: int
    eval_time_seconds: float
    metadata: dict
    stratified_results: dict | None = None


@dataclass
class BenchmarkSuite:
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    results: list[BenchmarkResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)


class DatasetRegistry:
    REGISTRY: dict[str, DatasetConfig] = {}

    @classmethod
    def register(cls, config: DatasetConfig):
        cls.REGISTRY[config.name] = config

    @classmethod
    def get(cls, name: str) -> DatasetConfig | None:
        return cls.REGISTRY.get(name)

    @classmethod
    def list_datasets(cls) -> list[str]:
        return list(cls.REGISTRY.keys())

    @classmethod
    def available_datasets(cls) -> dict[str, DatasetConfig]:
        return cls.REGISTRY.copy()


class HuggingFaceDatasetLoader:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or os.environ.get(
            "HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"
        )
        self._cache: dict[str, Any] = {}

    def load(
        self,
        config: DatasetConfig,
        sample_limit: int | None = None,
        seed: int = 42,
        force_refresh: bool = False,
    ) -> tuple[list[str], list[int], list[dict]]:
        cache_key = (
            f"{config.repo_id}_{config.config_name or 'default'}_{sample_limit}_{seed}"
        )
        cache_path = Path(self.cache_dir).expanduser() / f"{cache_key}.json"

        if not force_refresh and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return cached["texts"], cached["labels"], cached["metadata"]

        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library is required for HuggingFaceDatasetLoader"
            )

        try:
            ds = load_dataset(
                config.repo_id,
                name=config.config_name,
                split=config.split,
                streaming=True,
            )
        except Exception as e:
            try:
                ds = load_dataset(config.repo_id, split=config.split, streaming=True)
            except Exception as e2:
                try:
                    ds = load_dataset(
                        config.repo_id,
                        config.config_name,
                        split=config.split,
                        streaming=True,
                    )
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to load dataset {config.repo_id}: {e} -> {e2}"
                    ) from e3

        texts, labels, metadata = [], [], []

        take_count = (sample_limit * 20) if sample_limit else 10000
        items_iter = ds.take(take_count)

        for item in items_iter:
            text = item.get(config.text_field, "")
            label_raw = item.get(config.label_field, "")

            if not text or label_raw == "":
                continue

            if isinstance(label_raw, str):
                if label_raw in config.label_map:
                    label = config.label_map[label_raw]
                elif config.label_map:
                    label = 1
                else:
                    label = int(label_raw)
            else:
                label = int(label_raw)

            texts.append(text)
            labels.append(label)

            meta = {}
            for meta_key, meta_field in config.meta_fields.items():
                if meta_field in item:
                    meta[meta_key] = item[meta_field]
            metadata.append(meta)

        if not texts:
            return [], [], []

        if sample_limit and sample_limit < len(texts):
            np.random.seed(seed)
            indices = np.random.choice(len(texts), sample_limit, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            metadata = [metadata[i] for i in indices]
        elif sample_limit is None and len(texts) > 10000:
            np.random.seed(seed)
            indices = np.random.choice(len(texts), 10000, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            metadata = [metadata[i] for i in indices]
        else:
            np.random.seed(seed)
            indices = np.random.permutation(len(texts))
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            metadata = [metadata[i] for i in indices]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "texts": texts,
                    "labels": labels,
                    "metadata": metadata,
                }
            )
        )

        self._cache[cache_key] = (texts, labels, metadata)
        return texts, labels, metadata


def _register_datasets():
    DatasetRegistry.register(
        DatasetConfig(
            name="raid",
            repo_id="liamdugan/raid",
            split="train",
            text_field="generation",
            label_field="model",
            label_map={"human": 0},
            meta_fields={"domain": "domain", "source": "source_id"},
        )
    )

    DatasetRegistry.register(
        DatasetConfig(
            name="mage",
            repo_id="yaful/MAGE",
            config_name="default",
            split="test",
            text_field="text",
            label_field="label",
            label_map={"human": 0, "ai": 1, "mixed": 2},
            meta_fields={"edit_type": "edit_type"},
        )
    )

    DatasetRegistry.register(
        DatasetConfig(
            name="hc3",
            repo_id="Hello-SimpleAI/HC3",
            config_name="all",
            split="validation",
            text_field="text",
            label_field="label",
            label_map={"human": 0, "ChatGPT": 1},
        )
    )

    DatasetRegistry.register(
        DatasetConfig(
            name="m4",
            repo_id="NickyNicky/M4",
            split="train",
            text_field="text",
            label_field="label",
            label_map={"human": 0, "ai": 1},
        )
    )

    DatasetRegistry.register(
        DatasetConfig(
            name="hc3",
            repo_id="Hello-SimpleAI/HC3",
            config_name="all",
            split="validation",
            text_field="text",
            label_field="label",
            label_map={"human": 0, "ChatGPT": 1},
        )
    )

    DatasetRegistry.register(
        DatasetConfig(
            name="m4",
            repo_id="NickyNicky/M4",
            split="train",
            text_field="text",
            label_field="label",
            label_map={"human": 0, "ai": 1},
        )
    )


_register_datasets()


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
            "confusion_matrix": confusion_matrix,
        }

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
        y_score = []

        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(texts, desc=f"Evaluating {detector.name}")
            except ImportError:
                pass

        for text in iterator:
            try:
                result = detector.detect(text)
                score = float(result.score)
                y_score.append(score)
            except Exception:
                y_score.append(0.5)

        threshold = float(threshold)
        y_pred = [1 if float(s) >= threshold else 0 for s in y_score]
        metrics = self.compute_metrics(labels, y_pred, y_score)
        eval_time = time.time() - start_time

        return BenchmarkResult(
            detector_name=detector.name,
            dataset=dataset_name,
            auroc=metrics["auroc"],
            f1=metrics["f1"],
            fpr_at_10tpr=metrics["fpr_at_10tpr"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            accuracy=metrics["accuracy"],
            num_samples=len(texts),
            num_positives=sum(labels),
            num_negatives=len(labels) - sum(labels),
            eval_time_seconds=eval_time,
            metadata={
                "auprc": metrics["auprc"],
                "threshold": threshold,
                "confusion_matrix": metrics["confusion_matrix"],
            },
            stratified_results={"overall": metrics},
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

        stratified = {}
        stratify_keys: set[str] = set()
        for m in metadata:
            stratify_keys.update(m.keys())

        for key in stratify_keys:
            key_texts, key_labels = [], []
            for text, label, item_metadata in zip(
                texts, labels, metadata, strict=False
            ):
                if key in item_metadata:
                    key_texts.append(text)
                    key_labels.append(label)

            if len(key_texts) >= 10:
                key_scores = []
                for text in key_texts:
                    try:
                        result = detector.detect(text)
                        key_scores.append(result.score)
                    except Exception:
                        key_scores.append(0.5)

                key_pred = [1 if s >= threshold else 0 for s in key_scores]
                metrics = self.compute_metrics(key_labels, key_pred, key_scores)
                stratified[key] = {
                    "n_samples": len(key_texts),
                    "n_positives": sum(key_labels),
                    **metrics,
                }

        short_mask = [len(t.split()) < 150 for t in texts]
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
            short_scores = []
            for text in short_texts:
                try:
                    result = detector.detect(text)
                    short_scores.append(result.score)
                except Exception:
                    short_scores.append(0.5)
            short_pred = [1 if s >= threshold else 0 for s in short_scores]
            metrics = self.compute_metrics(short_labels, short_pred, short_scores)
            stratified["short_texts"] = {
                "n_samples": len(short_texts),
                "n_positives": sum(short_labels),
                **metrics,
            }

        base_result.stratified_results = stratified
        return base_result


class BenchmarkRunner:
    def __init__(
        self,
        evaluator: BenchmarkEvaluator | None = None,
        output_dir: str = "benchmark_results",
    ):
        self.evaluator = evaluator or BenchmarkEvaluator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.suite: BenchmarkSuite | None = None

    def run_benchmark(
        self,
        detector,
        datasets: list[str] | None = None,
        sample_limit: int | None = None,
        threshold: float = 0.5,
        stratified: bool = True,
        show_progress: bool = False,
    ) -> BenchmarkSuite:
        if datasets is None:
            datasets = DatasetRegistry.list_datasets()

        results = []
        for dataset_name in datasets:
            config = DatasetRegistry.get(dataset_name)
            if config is None:
                print(f"Dataset {dataset_name} not found in registry")
                continue

            try:
                texts, labels, metadata = self.evaluator.dataset_loader.load(
                    config, sample_limit=sample_limit
                )
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                continue

            print(
                f"Evaluating {detector.name} on {dataset_name} ({len(texts)} samples)"
            )

            if stratified and metadata:
                result = self.evaluator.evaluate_stratified(
                    detector, texts, labels, metadata, threshold, dataset_name
                )
            else:
                result = self.evaluator.evaluate_detector(
                    detector, texts, labels, threshold, dataset_name, show_progress
                )

            results.append(result)

        self.suite = BenchmarkSuite(
            name=f"{detector.name}_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            results=results,
            config={
                "detector": detector.name,
                "datasets": datasets,
                "sample_limit": sample_limit,
                "threshold": threshold,
                "stratified": stratified,
            },
        )
        return self.suite

    def compare_detectors(
        self,
        detectors: list,
        datasets: list[str] | None = None,
        sample_limit: int | None = None,
        threshold: float = 0.5,
    ) -> BenchmarkSuite:
        all_results = []

        for detector in detectors:
            suite = self.run_benchmark(
                detector, datasets, sample_limit, threshold, stratified=False
            )
            all_results.extend(suite.results)

        return BenchmarkSuite(
            name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            results=all_results,
            config={
                "detectors": [detector.name for detector in detectors],
                "datasets": datasets or DatasetRegistry.list_datasets(),
                "sample_limit": sample_limit,
                "threshold": threshold,
            },
        )

    def generate_report(
        self,
        suite: BenchmarkSuite,
        output_format: Literal["markdown", "json", "csv", "all"] = "markdown",
    ) -> str:
        if output_format == "markdown" or output_format == "all":
            md_path = self.output_dir / f"{suite.name}.md"
            md_path.write_text(self._generate_markdown(suite))
            print(f"Markdown report: {md_path}")

        if output_format == "json" or output_format == "all":
            json_path = self.output_dir / f"{suite.name}.json"
            json_path.write_text(self._generate_json(suite))
            print(f"JSON report: {json_path}")

        if (output_format == "csv" or output_format == "all") and PD_AVAILABLE:
            csv_path = self.output_dir / f"{suite.name}.csv"
            csv_path.write_text(self._generate_csv(suite))
            print(f"CSV report: {csv_path}")

        return self._generate_markdown(suite)

    def _generate_markdown(self, suite: BenchmarkSuite) -> str:
        lines = [
            "# Benchmark Report",
            f"\n**Suite**: {suite.name}",
            f"**Generated**: {suite.created_at}",
            f"**Detector**: {suite.config.get('detector', ', '.join(suite.config.get('detectors', ['N/A'])))}",
            f"**Datasets**: {', '.join(suite.config.get('datasets', []))}",
            f"**Threshold**: {suite.config.get('threshold', 0.5)}",
            "\n---\n",
            "\n## Summary",
            "\n| Detector | Dataset | AUROC | F1 | FPR@10%TPR | Precision | Recall | Accuracy | Samples |",
            "|----------|---------|-------|----|------------|-----------|--------|----------|---------|",
        ]

        for r in suite.results:
            lines.append(
                f"| {r.detector_name} | {r.dataset} | {r.auroc:.4f} | {r.f1:.4f} | {r.fpr_at_10tpr:.4f} | "
                f"{r.precision:.4f} | {r.recall:.4f} | {r.accuracy:.4f} | {r.num_samples} |"
            )

        lines.append("\n## Detailed Results\n")

        for r in suite.results:
            lines.append(f"\n### {r.detector_name} on {r.dataset}\n")
            lines.append(f"- **AUROC**: {r.auroc:.4f}")
            lines.append(f"- **F1**: {r.f1:.4f}")
            lines.append(f"- **FPR@10%TPR**: {r.fpr_at_10tpr:.4f}")
            lines.append(f"- **Precision**: {r.precision:.4f}")
            lines.append(f"- **Recall**: {r.recall:.4f}")
            lines.append(f"- **Accuracy**: {r.accuracy:.4f}")
            lines.append(
                f"- **Samples**: {r.num_samples} ({r.num_positives} AI, {r.num_negatives} human)"
            )
            lines.append(f"- **Eval Time**: {r.eval_time_seconds:.2f}s")

            if r.stratified_results and "overall" in r.stratified_results:
                lines.append("\n#### Stratified Results\n")
                for key, metrics in r.stratified_results.items():
                    if key == "overall":
                        continue
                    lines.append(f"\n**{key}** (n={metrics.get('n_samples', 'N/A')})")
                    lines.append(f"- AUROC: {metrics.get('auroc', 0):.4f}")
                    lines.append(f"- F1: {metrics.get('f1', 0):.4f}")
                    lines.append(f"- FPR@10%TPR: {metrics.get('fpr_at_10tpr', 0):.4f}")

            cm = r.metadata.get("confusion_matrix", {})
            if cm:
                lines.append(
                    f"\n**Confusion Matrix**: TP={cm.get('tp', 0)}, TN={cm.get('tn', 0)}, FP={cm.get('fp', 0)}, FN={cm.get('fn', 0)}"
                )

        return "\n".join(lines)

    def _generate_json(self, suite: BenchmarkSuite) -> str:
        data = {
            "name": suite.name,
            "created_at": suite.created_at,
            "config": suite.config,
            "results": [
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
                    "num_positives": r.num_positives,
                    "num_negatives": r.num_negatives,
                    "eval_time_seconds": r.eval_time_seconds,
                    "metadata": r.metadata,
                    "stratified_results": r.stratified_results,
                }
                for r in suite.results
            ],
        }
        return json.dumps(data, indent=2)

    def _generate_csv(self, suite: BenchmarkSuite) -> str:
        rows = []
        for r in suite.results:
            row = {
                "detector": r.detector_name,
                "dataset": r.dataset,
                "auroc": r.auroc,
                "f1": r.f1,
                "fpr_at_10tpr": r.fpr_at_10tpr,
                "precision": r.precision,
                "recall": r.recall,
                "accuracy": r.accuracy,
                "num_samples": r.num_samples,
                "eval_time_s": r.eval_time_seconds,
            }
            if r.stratified_results:
                for key, metrics in r.stratified_results.items():
                    if key != "overall":
                        row[f"strat_{key}_auroc"] = metrics.get("auroc", "")
                        row[f"strat_{key}_f1"] = metrics.get("f1", "")
            rows.append(row)

        if PD_AVAILABLE:
            df = pd.DataFrame(rows)
            return str(df.to_csv(index=False))
        return ""

    def load_previous_results(self, path: str) -> BenchmarkSuite | None:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        results = []
        for r in data.get("results", []):
            results.append(BenchmarkResult(**r))
        return BenchmarkSuite(
            name=data["name"],
            created_at=data["created_at"],
            config=data.get("config", {}),
            results=results,
        )
