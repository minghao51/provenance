"""Benchmark run orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from provenance.benchmarks.evaluator import BenchmarkEvaluator
from provenance.benchmarks.models import BenchmarkSuite
from provenance.benchmarks.registry import DatasetRegistry
from provenance.benchmarks.reporting import BenchmarkReportWriter


class BenchmarkRunner:
    def __init__(
        self,
        evaluator: BenchmarkEvaluator | None = None,
        output_dir: str = "benchmark_results",
    ):
        self.evaluator = evaluator or BenchmarkEvaluator()
        self.report_writer = BenchmarkReportWriter(output_dir=output_dir)
        self.output_dir = self.report_writer.output_dir
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
            except Exception as exc:
                print(f"Failed to load {dataset_name}: {exc}")
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
        return self.report_writer.generate_report(suite, output_format=output_format)

    def _generate_markdown(self, suite: BenchmarkSuite) -> str:
        return self.report_writer.generate_markdown(suite)

    def _generate_json(self, suite: BenchmarkSuite) -> str:
        return self.report_writer.generate_json(suite)

    def _generate_csv(self, suite: BenchmarkSuite) -> str:
        return self.report_writer.generate_csv(suite)

    def load_previous_results(self, path: str) -> BenchmarkSuite | None:
        return self.report_writer.load_previous_results(path)
