"""Provenance CLI - Command line interface for AI text detection."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import click


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Provenance AI - Detect AI-generated text with modular, explainable analysis."""
    pass


@main.command()
@click.argument("text", required=False)
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, path_type=Path),
    help="Input file to analyze",
)
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.option("--detectors", "-d", multiple=True, help="Specific detectors to use")
@click.option(
    "--ensemble",
    "-e",
    type=click.Choice(["weighted_average", "stacking", "uncertainty_aware"]),
    default="weighted_average",
    help="Ensemble strategy",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    help="Optional JSON/YAML config file",
)
def detect(
    text: str | None,
    file: Path | None,
    output: str,
    detectors: tuple[str, ...],
    ensemble: str,
    config_path: Path | None,
) -> None:
    """Detect AI-generated text in the provided text or file."""
    if file:
        content = file.read_text(encoding="utf-8")
        target_text = content
    elif text:
        target_text = text
    else:
        click.echo("Error: Provide text or use --file option", err=True)
        sys.exit(1)

    from provenance import Provenance
    from provenance.core.registry import get_registry

    registry = get_registry()
    registry.load_entry_points()

    if not detectors:
        available = registry.list_detectors(latency_tier="fast")
        detector_names = [d.name for d in available]
    else:
        detector_names = list(detectors)

    strategy = cast(
        Literal["weighted_average", "stacking", "uncertainty_aware"], ensemble
    )
    provenance = Provenance(
        detectors=detector_names,
        ensemble_strategy=strategy,
        config=config_path,
    )
    result = provenance.detect(target_text)

    if output == "json":
        click.echo(
            json.dumps(
                {
                    "score": result.score,
                    "label": result.label,
                    "confidence": result.confidence,
                    "detector_scores": {
                        name: {
                            "score": dr.score,
                            "confidence": dr.confidence,
                            "metadata": dr.metadata,
                        }
                        for name, dr in result.detector_scores.items()
                    },
                    "heatmap": [(ts.token, ts.score) for ts in result.heatmap[:20]],
                    "feature_vector": result.feature_vector,
                },
                indent=2,
            )
        )
    else:
        click.echo(f"Score: {result.score:.4f}")
        click.echo(f"Label: {result.label}")
        click.echo(f"Confidence: {result.confidence:.4f}")
        if result.detector_scores:
            click.echo("\nDetector Breakdown:")
            for name, dr in result.detector_scores.items():
                click.echo(f"  {name}: {dr.score:.4f} (conf: {dr.confidence:.2f})")


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dataset JSONL file",
)
@click.option(
    "--model", "-m", type=click.Choice(["lgbm"]), default="lgbm", help="Model type"
)
@click.option(
    "--eval-split", "-s", type=float, default=0.2, help="Evaluation split ratio"
)
def train(dataset: Path, model: str, eval_split: float) -> None:
    """Train a detector on custom data."""
    click.echo(f"Training {model} on {dataset}...")

    import json

    texts = []
    labels = []

    with open(dataset) as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(int(item["label"]))

    click.echo(f"Loaded {len(texts)} samples")

    if model == "lgbm":
        from provenance.detectors.stylometric.lightgbm_detector import LightGBMDetector

        detector = LightGBMDetector()
        detector.train(texts, labels)

        output_path = Path(f"model_{model}.pkl")
        detector.save_model(str(output_path))
        click.echo(f"Model saved to {output_path}")


@main.command()
@click.option("--detector", "-d", help="Detector name to benchmark")
@click.option("--dataset", "-ds", default="raid", help="Dataset to use")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Output file for results"
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit number of samples (for quick testing)",
)
@click.option(
    "--threshold", "-t", type=float, default=0.5, help="Classification threshold"
)
def benchmark(
    detector: str | None,
    dataset: str,
    output: Path | None,
    limit: int | None,
    threshold: float,
) -> None:
    """Run benchmark evaluation on a detector."""
    from provenance.benchmarks.workflow import (
        BenchmarkRunner,
        BenchmarkSuite,
        DatasetRegistry,
    )
    from provenance.core.registry import get_registry

    registry = get_registry()
    registry.load_entry_points()

    available_datasets = DatasetRegistry.list_datasets()
    if dataset not in available_datasets:
        click.echo(
            f"Dataset '{dataset}' not found. Available datasets: {', '.join(available_datasets)}",
            err=True,
        )
        sys.exit(1)

    if detector:
        det = registry.get(detector)
        if not det:
            click.echo(f"Detector '{detector}' not found", err=True)
            sys.exit(1)
        detectors = [det]
    else:
        detectors = registry.list_detectors()

    click.echo(f"Benchmarking {len(detectors)} detectors on {dataset}...")
    runner = BenchmarkRunner()
    all_results = []
    for det in detectors:
        click.echo(f"  Evaluating {det.name}...")
        try:
            suite = runner.run_benchmark(
                detector=det,
                datasets=[dataset],
                sample_limit=limit,
                threshold=threshold,
                stratified=True,
            )
            all_results.extend(suite.results)
        except Exception as e:
            click.echo(f"    Error: {e}", err=True)

    if not all_results:
        click.echo("No benchmark results were generated", err=True)
        sys.exit(1)

    report_suite = BenchmarkSuite(
        name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        results=all_results,
        config={
            "detectors": [det.name for det in detectors],
            "datasets": [dataset],
            "sample_limit": limit,
            "threshold": threshold,
            "stratified": True,
        },
    )
    report = runner._generate_markdown(report_suite)

    if output:
        output.write_text(report)
        output.with_name(f"{output.stem}_results.json").write_text(
            runner._generate_json(report_suite)
        )
        click.echo(f"Results saved to {output}")
    else:
        click.echo("\n" + report)


@main.command("benchmark-compare")
@click.option(
    "--detectors", "-d", multiple=True, required=True, help="Detector names to compare"
)
@click.option(
    "--datasets", "-ds", multiple=True, default=["raid"], help="Datasets to evaluate on"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("benchmark_results"),
    help="Output directory",
)
@click.option("--limit", "-l", type=int, default=None, help="Limit samples per dataset")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "json", "csv", "all"]),
    default="all",
    help="Output format",
)
@click.option(
    "--stratify/--no-stratify",
    default=True,
    help="Include stratified benchmark breakdowns",
)
def benchmark_compare(
    detectors: tuple[str, ...],
    datasets: tuple[str, ...],
    output_dir: Path,
    limit: int | None,
    format: str,
    stratify: bool,
) -> None:
    """Compare multiple detectors across datasets with comprehensive reporting."""
    from provenance.benchmarks.workflow import (
        BenchmarkRunner,
        BenchmarkSuite,
        DatasetRegistry,
    )
    from provenance.core.registry import get_registry

    registry = get_registry()
    registry.load_entry_points()

    detector_objs = []
    for name in detectors:
        det = registry.get(name)
        if not det:
            click.echo(f"Detector '{name}' not found", err=True)
            continue
        detector_objs.append(det)

    if not detector_objs:
        click.echo("No valid detectors specified", err=True)
        sys.exit(1)

    click.echo(f"Comparing {len(detector_objs)} detectors on {len(datasets)} datasets")
    click.echo(f"Available datasets: {', '.join(DatasetRegistry.list_datasets())}")

    runner = BenchmarkRunner(output_dir=str(output_dir))

    all_results = []
    for det in detector_objs:
        click.echo(f"\n=== Evaluating {det.name} ===")
        suite = runner.run_benchmark(
            detector=det,
            datasets=list(datasets),
            sample_limit=limit,
            stratified=stratify,
            show_progress=True,
        )
        all_results.extend(suite.results)

    comparison_suite = BenchmarkSuite(
        name=f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        results=all_results,
        config={
            "detectors": [det.name for det in detector_objs],
            "datasets": list(datasets),
            "sample_limit": limit,
            "threshold": 0.5,
            "stratified": stratify,
        },
    )

    click.echo("\n=== Generating report ===")
    runner.generate_report(
        comparison_suite,
        output_format=cast(Literal["markdown", "json", "csv", "all"], format),
    )

    click.echo("\nSummary:")
    for r in all_results:
        click.echo(
            f"  {r.detector_name} on {r.dataset}: AUROC={r.auroc:.4f}, F1={r.f1:.4f}"
        )


@main.command("benchmark-datasets")
def benchmark_datasets() -> None:
    """List available benchmark datasets."""
    from provenance.benchmarks.workflow import DatasetRegistry

    datasets = DatasetRegistry.available_datasets()
    click.echo(f"Available datasets ({len(datasets)}):\n")
    for name, config in datasets.items():
        click.echo(f"  {name}")
        click.echo(f"    Repo: {config.repo_id}")
        click.echo(f"    Config: {config.config_name or 'default'}")
        click.echo(f"    Split: {config.split}")
        click.echo(f"    Text field: {config.text_field}")
        click.echo(f"    Label field: {config.label_field}")
        click.echo(f"    Label map: {config.label_map}")
        if config.meta_fields:
            click.echo(f"    Meta fields: {config.meta_fields}")
        click.echo()


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dataset JSONL file",
)
@click.option("--languages", "-l", multiple=True, help="Language tags for FPR audit")
def audit(dataset: Path, languages: tuple[str, ...]) -> None:
    """Run FPR bias audit on a dataset."""
    import json

    from provenance.benchmarks.evaluation import BenchmarkHarness
    from provenance.core.registry import get_registry

    registry = get_registry()
    registry.load_entry_points()

    texts = []
    labels = []
    lang_tags = []

    with open(dataset) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(int(item["label"]))
            if languages:
                lang_tags.append(
                    languages[i % len(languages)]
                    if i < len(languages) * 100
                    else "unknown"
                )

    click.echo(f"Loaded {len(texts)} samples")

    harness = BenchmarkHarness()

    all_detectors = registry.list_detectors()
    fast_detectors = [d for d in all_detectors if d.latency_tier == "fast"]

    if not fast_detectors:
        click.echo("No fast detectors available for audit", err=True)
        sys.exit(1)

    click.echo(f"Auditing with {len(fast_detectors)} fast detectors...")

    for det in fast_detectors:
        click.echo(f"  {det.name}:")
        fpr = harness._compute_fpr_simple(texts, labels)
        click.echo(f"    Overall FPR: {fpr:.4f}")

    audit_results = harness.audit_fpr(texts, labels, lang_tags if lang_tags else None)
    click.echo("\nFPR Audit Results:")
    for key, val in audit_results.items():
        click.echo(f"  {key}: {val:.4f}")


@main.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8080, type=int, help="Port to bind to")
def serve(host: str, port: int) -> None:
    """Start the Provenance REST API server."""
    from provenance.api import run_server

    click.echo(f"Starting server on {host}:{port}...")
    run_server(host=host, port=port)


@main.command()
def list_detectors() -> None:
    """List all available detectors."""
    from provenance.core.registry import get_registry

    registry = get_registry()
    registry.load_entry_points()
    detectors = registry.list_detectors()

    if not detectors:
        click.echo("No detectors registered")
        return

    click.echo(f"Available detectors ({len(detectors)}):")
    for detector in detectors:
        click.echo(
            f"  {detector.name} [{detector.latency_tier}] - domains: {', '.join(detector.domains)}"
        )


if __name__ == "__main__":
    main()
