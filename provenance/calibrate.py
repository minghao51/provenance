"""Calibration training script for all detectors using benchmark datasets.

Usage:
    uv run python -m provenance.calibrate train --detector entropy --dataset hc3 --limit 500
    uv run python -m provenance.calibrate train --detector all --dataset hc3 --limit 1000
    uv run python -m provenance.calibrate evaluate --detector entropy --dataset hc3 --limit 500
    uv run python -m provenance.calibrate compare --dataset hc3 --limit 500
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from provenance.benchmarks.workflow import (
    BenchmarkEvaluator,
    DatasetRegistry,
    HuggingFaceDatasetLoader,
)
from provenance.core.calibration import SKLEARN_AVAILABLE

DETECTOR_MAP = {
    "entropy": ("provenance.detectors.statistical.entropy", "EntropyDetector"),
    "burstiness": ("provenance.detectors.statistical.burstiness", "BurstinessDetector"),
    "repetition": ("provenance.detectors.statistical.repetition", "RepetitionDetector"),
    "curvature": ("provenance.detectors.statistical.curvature", "CurvatureDetector"),
    "surprisal": ("provenance.detectors.statistical.surprisal", "SurprisalDetector"),
    "stylometric": (
        "provenance.detectors.stylometric.feature_extractor",
        "StylometricDetector",
    ),
    "cognitive": ("provenance.detectors.stylometric.cognitive", "CognitiveDetector"),
    "academic": ("provenance.domains.academic", "AcademicDetector"),
    "code": ("provenance.domains.code", "CodeDetector"),
    "multilingual": ("provenance.domains.multilingual", "MultilingualDetector"),
    "all": None,
}

OUTPUT_DIR = Path("calibration_models")


def _import_detector(name: str):
    entry = DETECTOR_MAP.get(name)
    if entry is None:
        raise ValueError(f"Unknown detector: {name}")
    module_path, class_name = entry
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)()


def _load_data(dataset: str, limit: int | None, seed: int):
    config = DatasetRegistry.get(dataset)
    if config is None:
        click.echo(f"Dataset '{dataset}' not found in registry.", err=True)
        sys.exit(1)

    loader = HuggingFaceDatasetLoader()
    texts, labels, metadata = loader.load(config, sample_limit=limit, seed=seed)

    if not texts:
        click.echo(f"No data loaded from '{dataset}'.", err=True)
        sys.exit(1)

    labels = [int(label) for label in labels]
    unique = set(labels)
    if len(unique) < 2:
        click.echo(f"Dataset '{dataset}' has only one class: {unique}", err=True)
        sys.exit(1)

    click.echo(
        f"Loaded {len(texts)} samples from '{dataset}' ({sum(labels)} AI, {len(labels) - sum(labels)} human)"
    )
    return texts, labels


def _evaluate_detector(detector, texts, labels, evaluator, name):
    threshold = 0.5
    y_score = []
    for text in texts:
        try:
            result = detector.detect(text)
            y_score.append(result.score)
        except Exception:
            y_score.append(0.5)

    y_pred = [1 if s >= threshold else 0 for s in y_score]
    metrics = evaluator.compute_metrics(labels, y_pred, y_score)
    metrics["name"] = name
    return metrics


@click.group()
def cli():
    """Calibration training and evaluation for provenance detectors."""
    pass


@cli.command()
@click.option("--detector", "-d", default="all", help="Detector name or 'all'")
@click.option("--dataset", "-s", default="hc3", help="Dataset name from registry")
@click.option("--limit", "-l", default=500, type=int, help="Number of samples to use")
@click.option(
    "--method",
    default="isotonic",
    type=click.Choice(["isotonic", "sigmoid"]),
    help="Calibration method",
)
@click.option("--cv", default=5, type=int, help="Cross-validation folds")
@click.option(
    "--output-dir", default="calibration_models", help="Directory to save models"
)
@click.option("--seed", default=42, type=int, help="Random seed")
def train(detector, dataset, limit, method, cv, output_dir, seed):
    """Train calibration models for detectors."""
    if not SKLEARN_AVAILABLE:
        click.echo(
            "Error: scikit-learn is required. Install with: pip install scikit-learn",
            err=True,
        )
        sys.exit(1)

    texts, labels = _load_data(dataset, limit, seed)

    split = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split], labels[:split]
    test_texts, test_labels = texts[split:], labels[split:]

    evaluator = BenchmarkEvaluator()
    results = {}

    if detector == "all":
        detector_names = [k for k in DETECTOR_MAP if k != "all"]
    else:
        detector_names = [detector]

    for det_name in detector_names:
        if det_name not in DETECTOR_MAP:
            click.echo(f"Unknown detector: {det_name}", err=True)
            continue

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Training calibration for: {det_name}")
        click.echo(f"{'=' * 60}")

        try:
            det = _import_detector(det_name)
        except Exception as e:
            click.echo(f"  Failed to load detector: {e}", err=True)
            continue

        before = _evaluate_detector(
            det, test_texts, test_labels, evaluator, "heuristic"
        )
        click.echo(
            f"  Before calibration: AUROC={before['auroc']:.4f}, F1={before['f1']:.4f}"
        )

        try:
            start = time.time()
            det.calibrate(train_texts, train_labels, method=method, cv=cv)
            train_time = time.time() - start
            click.echo(f"  Calibration trained in {train_time:.2f}s")
        except Exception as e:
            click.echo(f"  Failed to calibrate: {e}", err=True)
            continue

        after = _evaluate_detector(
            det, test_texts, test_labels, evaluator, "calibrated"
        )
        click.echo(
            f"  After calibration:  AUROC={after['auroc']:.4f}, F1={after['f1']:.4f}"
        )

        delta_auroc = after["auroc"] - before["auroc"]
        delta_f1 = after["f1"] - before["f1"]
        click.echo(f"  Delta: AUROC={delta_auroc:+.4f}, F1={delta_f1:+.4f}")

        results[det_name] = {
            "before": before,
            "after": after,
            "delta_auroc": delta_auroc,
            "delta_f1": delta_f1,
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / f"{det_name}_{dataset}.pkl"
        det.save_calibration(model_path)
        click.echo(f"  Model saved to: {model_path}")

    if results:
        click.echo(f"\n{'=' * 60}")
        click.echo("Summary")
        click.echo(f"{'=' * 60}")
        click.echo(
            f"{'Detector':<20} {'AUROC Before':>12} {'AUROC After':>12} {'Delta':>8} {'F1 Before':>10} {'F1 After':>10} {'Delta':>8}"
        )
        click.echo("-" * 80)
        for name, r in results.items():
            click.echo(
                f"{name:<20} {r['before']['auroc']:>12.4f} {r['after']['auroc']:>12.4f} {r['delta_auroc']:>+8.4f} "
                f"{r['before']['f1']:>10.4f} {r['after']['f1']:>10.4f} {r['delta_f1']:>+8.4f}"
            )

        summary_path = Path(output_dir) / f"calibration_results_{dataset}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2, default=str))
        click.echo(f"\nResults saved to: {summary_path}")


@cli.command()
@click.option("--detector", "-d", default="all", help="Detector name or 'all'")
@click.option("--dataset", "-s", default="hc3", help="Dataset name from registry")
@click.option("--limit", "-l", default=500, type=int, help="Number of samples")
@click.option("--seed", default=42, type=int, help="Random seed")
def compare(detector, dataset, limit, seed):
    """Compare heuristic vs calibrated performance across detectors."""
    if not SKLEARN_AVAILABLE:
        click.echo("Error: scikit-learn is required.", err=True)
        sys.exit(1)

    texts, labels = _load_data(dataset, limit, seed)
    split = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split], labels[:split]
    test_texts, test_labels = texts[split:], labels[split:]

    evaluator = BenchmarkEvaluator()
    detector_names = (
        [k for k in DETECTOR_MAP if k != "all"] if detector == "all" else [detector]
    )

    results = []
    for det_name in detector_names:
        if det_name not in DETECTOR_MAP or DETECTOR_MAP[det_name] is None:
            continue

        try:
            det = _import_detector(det_name)
        except Exception as e:
            click.echo(f"Skipping {det_name}: {e}", err=True)
            continue

        before = _evaluate_detector(
            det, test_texts, test_labels, evaluator, "heuristic"
        )

        try:
            det.calibrate(train_texts, train_labels, cv=5)
        except Exception as e:
            click.echo(f"  Calibration failed for {det_name}: {e}", err=True)
            continue

        after = _evaluate_detector(
            det, test_texts, test_labels, evaluator, "calibrated"
        )
        results.append(
            {
                "detector": det_name,
                "auroc_heuristic": before["auroc"],
                "auroc_calibrated": after["auroc"],
                "f1_heuristic": before["f1"],
                "f1_calibrated": after["f1"],
            }
        )

    click.echo(
        f"\n{'Detector':<20} {'AUROC (heuristic)':>18} {'AUROC (calibrated)':>18} {'F1 (heuristic)':>15} {'F1 (calibrated)':>15}"
    )
    click.echo("-" * 86)
    for r in results:
        click.echo(
            f"{r['detector']:<20} {r['auroc_heuristic']:>18.4f} {r['auroc_calibrated']:>18.4f} "
            f"{r['f1_heuristic']:>15.4f} {r['f1_calibrated']:>15.4f}"
        )


@cli.command()
@click.option("--detector", "-d", required=True, help="Detector name")
@click.option("--model", "-m", required=True, help="Path to calibration model (.pkl)")
@click.option("--dataset", "-s", default="hc3", help="Dataset name")
@click.option("--limit", "-l", default=500, type=int, help="Number of samples")
@click.option("--seed", default=42, type=int, help="Random seed")
def evaluate(detector, model, dataset, limit, seed):
    """Evaluate a pre-trained calibration model."""
    texts, labels = _load_data(dataset, limit, seed)
    evaluator = BenchmarkEvaluator()

    det = _import_detector(detector)
    det.load_calibration(model)

    metrics = _evaluate_detector(det, texts, labels, evaluator, "calibrated")
    click.echo(f"\nCalibrated {detector} on {dataset}:")
    for k, v in metrics.items():
        if k != "name" and isinstance(v, (int, float)):
            click.echo(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    cli()
