"""Benchmark report generation and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from provenance.benchmarks.models import BenchmarkResult, BenchmarkSuite

try:
    import pandas as pd

    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False


class BenchmarkReportWriter:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        suite: BenchmarkSuite,
        output_format: Literal["markdown", "json", "csv", "all"] = "markdown",
    ) -> str:
        if output_format == "markdown" or output_format == "all":
            md_path = self.output_dir / f"{suite.name}.md"
            md_path.write_text(self.generate_markdown(suite))
            print(f"Markdown report: {md_path}")

        if output_format == "json" or output_format == "all":
            json_path = self.output_dir / f"{suite.name}.json"
            json_path.write_text(self.generate_json(suite))
            print(f"JSON report: {json_path}")

        if (output_format == "csv" or output_format == "all") and PD_AVAILABLE:
            csv_path = self.output_dir / f"{suite.name}.csv"
            csv_path.write_text(self.generate_csv(suite))
            print(f"CSV report: {csv_path}")

        return self.generate_markdown(suite)

    def generate_markdown(self, suite: BenchmarkSuite) -> str:
        lines = [
            "# Benchmark Report",
            f"\n**Suite**: {suite.name}",
            f"**Generated**: {suite.created_at}",
            f"**Detector**: {suite.config.get('detector', ', '.join(suite.config.get('detectors', ['N/A'])))}",
            f"**Datasets**: {', '.join(suite.config.get('datasets', []))}",
            f"**Threshold**: {suite.config.get('threshold', 0.5)}",
            "\n---\n",
            "\n## Summary",
            "\n| Detector | Dataset | AUROC | F1 | TPR@1%FPR | TPR@5%FPR | FPR@10%TPR | Precision | Recall | Accuracy | Samples |",
            "|----------|---------|-------|----|-----------|-----------|------------|-----------|--------|----------|---------|",
        ]

        for result in suite.results:
            lines.append(
                f"| {result.detector_name} | {result.dataset} | {result.auroc:.4f} | {result.f1:.4f} | {result.tpr_at_1fpr:.4f} | {result.tpr_at_5fpr:.4f} | {result.fpr_at_10tpr:.4f} | "
                f"{result.precision:.4f} | {result.recall:.4f} | {result.accuracy:.4f} | {result.num_samples} |"
            )

        lines.append("\n## Detailed Results\n")

        for result in suite.results:
            lines.append(f"\n### {result.detector_name} on {result.dataset}\n")
            lines.append(f"- **AUROC**: {result.auroc:.4f}")
            lines.append(f"- **F1**: {result.f1:.4f}")
            lines.append(f"- **TPR@1%FPR**: {result.tpr_at_1fpr:.4f}")
            lines.append(f"- **TPR@5%FPR**: {result.tpr_at_5fpr:.4f}")
            lines.append(f"- **FPR@10%TPR**: {result.fpr_at_10tpr:.4f}")
            lines.append(f"- **Precision**: {result.precision:.4f}")
            lines.append(f"- **Recall**: {result.recall:.4f}")
            lines.append(f"- **Accuracy**: {result.accuracy:.4f}")
            lines.append(
                f"- **Samples**: {result.num_samples} ({result.num_positives} AI, {result.num_negatives} human)"
            )
            lines.append(f"- **Eval Time**: {result.eval_time_seconds:.2f}s")

            if result.stratified_results and "overall" in result.stratified_results:
                lines.append("\n#### Stratified Results\n")
                for key, metrics in result.stratified_results.items():
                    if key == "overall":
                        continue
                    lines.append(f"\n**{key}** (n={metrics.get('n_samples', 'N/A')})")
                    lines.append(f"- AUROC: {metrics.get('auroc', 0):.4f}")
                    lines.append(f"- F1: {metrics.get('f1', 0):.4f}")
                    lines.append(f"- TPR@1%FPR: {metrics.get('tpr_at_1fpr', 0):.4f}")
                    lines.append(f"- TPR@5%FPR: {metrics.get('tpr_at_5fpr', 0):.4f}")
                    lines.append(f"- FPR@10%TPR: {metrics.get('fpr_at_10tpr', 0):.4f}")

            confusion_matrix = result.metadata.get("confusion_matrix", {})
            if confusion_matrix:
                lines.append(
                    f"\n**Confusion Matrix**: TP={confusion_matrix.get('tp', 0)}, TN={confusion_matrix.get('tn', 0)}, FP={confusion_matrix.get('fp', 0)}, FN={confusion_matrix.get('fn', 0)}"
                )

        return "\n".join(lines)

    def generate_json(self, suite: BenchmarkSuite) -> str:
        data = {
            "name": suite.name,
            "created_at": suite.created_at,
            "config": suite.config,
            "results": [
                {
                    "detector_name": result.detector_name,
                    "dataset": result.dataset,
                    "auroc": result.auroc,
                    "f1": result.f1,
                    "tpr_at_1fpr": result.tpr_at_1fpr,
                    "tpr_at_5fpr": result.tpr_at_5fpr,
                    "fpr_at_10tpr": result.fpr_at_10tpr,
                    "precision": result.precision,
                    "recall": result.recall,
                    "accuracy": result.accuracy,
                    "num_samples": result.num_samples,
                    "num_positives": result.num_positives,
                    "num_negatives": result.num_negatives,
                    "eval_time_seconds": result.eval_time_seconds,
                    "metadata": result.metadata,
                    "stratified_results": result.stratified_results,
                }
                for result in suite.results
            ],
        }
        return json.dumps(data, indent=2)

    def generate_csv(self, suite: BenchmarkSuite) -> str:
        rows = []
        for result in suite.results:
            row = {
                "detector": result.detector_name,
                "dataset": result.dataset,
                "auroc": result.auroc,
                "f1": result.f1,
                "tpr_at_1fpr": result.tpr_at_1fpr,
                "tpr_at_5fpr": result.tpr_at_5fpr,
                "fpr_at_10tpr": result.fpr_at_10tpr,
                "precision": result.precision,
                "recall": result.recall,
                "accuracy": result.accuracy,
                "num_samples": result.num_samples,
                "eval_time_s": result.eval_time_seconds,
            }
            if result.stratified_results:
                for key, metrics in result.stratified_results.items():
                    if key != "overall":
                        row[f"strat_{key}_auroc"] = metrics.get("auroc", "")
                        row[f"strat_{key}_f1"] = metrics.get("f1", "")
            rows.append(row)

        if PD_AVAILABLE:
            df = pd.DataFrame(rows)
            return str(df.to_csv(index=False))
        return ""

    def load_previous_results(self, path: str) -> BenchmarkSuite | None:
        saved_path = Path(path)
        if not saved_path.exists():
            return None

        data = json.loads(saved_path.read_text())
        results = [BenchmarkResult(**result) for result in data.get("results", [])]
        return BenchmarkSuite(
            name=data["name"],
            created_at=data["created_at"],
            config=data.get("config", {}),
            results=results,
        )
