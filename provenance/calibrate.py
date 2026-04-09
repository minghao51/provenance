"""Calibration training script for all detectors using benchmark datasets.

Usage:
    uv run python -m provenance.calibrate train --detector entropy --dataset hc3 --limit 500
    uv run python -m provenance.calibrate train --detector all --dataset hc3 --limit 1000
    uv run python -m provenance.calibrate evaluate --detector entropy --dataset hc3 --limit 500
    uv run python -m provenance.calibrate compare --dataset hc3 --limit 500
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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


def _import_detector(name: str, *, autoload_calibration: bool = True):
    entry = DETECTOR_MAP.get(name)
    if entry is None:
        raise ValueError(f"Unknown detector: {name}")
    module_path, class_name = entry
    previous_disable = os.environ.get("PROVENANCE_DISABLE_AUTO_CALIBRATION")
    if not autoload_calibration:
        os.environ["PROVENANCE_DISABLE_AUTO_CALIBRATION"] = "1"
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)()
    finally:
        if not autoload_calibration:
            if previous_disable is None:
                os.environ.pop("PROVENANCE_DISABLE_AUTO_CALIBRATION", None)
            else:
                os.environ["PROVENANCE_DISABLE_AUTO_CALIBRATION"] = previous_disable


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


def _preferred_calibration_key(detector_name: str) -> str:
    try:
        detector = _import_detector(detector_name, autoload_calibration=False)
    except Exception:
        return detector_name

    aliases = getattr(detector, "calibration_aliases", ())
    if aliases:
        return str(aliases[-1])
    return detector_name


def _resolve_model_path(
    detector_name: str, dataset: str, summary: dict[str, Any], output_dir: str
) -> str | None:
    candidate_keys = [detector_name, _preferred_calibration_key(detector_name)]
    selected_models = summary.get("selected_models", {})

    for key in candidate_keys:
        model_path = selected_models.get(key)
        if model_path:
            return str(model_path)

    fallback_path = Path(output_dir) / f"{detector_name}_{dataset}.pkl"
    if fallback_path.exists():
        return str(fallback_path)

    return None


def _build_rejection_reasons(
    metrics: dict[str, Any],
    *,
    min_auroc_improvement: float,
    max_f1_regression: float,
    min_tpr_at_1fpr_improvement: float | None,
) -> list[str]:
    before = metrics.get("before", {})
    after = metrics.get("after", {})
    delta = metrics.get("delta", {})
    reasons: list[str] = []

    delta_auroc = float(
        delta.get("auroc", float(after.get("auroc", 0.0)) - float(before.get("auroc", 0.0)))
    )
    if delta_auroc < min_auroc_improvement:
        reasons.append(
            f"delta_auroc {delta_auroc:.4f} below threshold {min_auroc_improvement:.4f}"
        )

    f1_regression = float(before.get("f1", 0.0)) - float(after.get("f1", 0.0))
    if f1_regression > max_f1_regression:
        reasons.append(
            f"f1 regression {f1_regression:.4f} exceeds threshold {max_f1_regression:.4f}"
        )

    if min_tpr_at_1fpr_improvement is not None:
        before_tpr = before.get("tpr_at_1fpr")
        after_tpr = after.get("tpr_at_1fpr")
        if before_tpr is None or after_tpr is None:
            reasons.append("tpr_at_1fpr unavailable for requested policy gate")
        else:
            delta_tpr = float(after_tpr) - float(before_tpr)
            if delta_tpr < min_tpr_at_1fpr_improvement:
                reasons.append(
                    f"delta_tpr_at_1fpr {delta_tpr:.4f} below threshold {min_tpr_at_1fpr_improvement:.4f}"
                )

    return reasons


def _render_curated_config(model_paths: dict[str, str], calibration_dir: str) -> str:
    lines = [
        "provenance:",
        f"  calibration_model_dir: {calibration_dir}",
        "  detector_calibration_paths:",
    ]
    for detector_name, model_path in model_paths.items():
        lines.append(f"    {detector_name}: {model_path}")
    return "\n".join(lines) + "\n"


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
            det = _import_detector(det_name, autoload_calibration=False)
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
            det = _import_detector(det_name, autoload_calibration=False)
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

    det = _import_detector(detector, autoload_calibration=False)
    det.load_calibration(model)

    metrics = _evaluate_detector(det, texts, labels, evaluator, "calibrated")
    click.echo(f"\nCalibrated {detector} on {dataset}:")
    for k, v in metrics.items():
        if k != "name" and isinstance(v, (int, float)):
            click.echo(f"  {k}: {v:.4f}")


@cli.command("promote")
@click.option(
    "--summary-path",
    default=str(OUTPUT_DIR / "raid_calibration_summary.json"),
    type=click.Path(exists=True, path_type=Path),
    help="Calibration summary JSON to evaluate for promotion",
)
@click.option(
    "--output-config",
    default="provenance.calibrated.yaml",
    type=click.Path(path_type=Path),
    help="Path for the generated curated config",
)
@click.option(
    "--output-summary",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path for the promotion decision summary",
)
@click.option(
    "--calibration-dir",
    default=str(OUTPUT_DIR),
    help="Calibration model directory recorded in the generated config",
)
@click.option(
    "--min-auroc-improvement",
    default=0.01,
    type=float,
    show_default=True,
    help="Minimum AUROC improvement required to promote a calibrated model",
)
@click.option(
    "--max-f1-regression",
    default=0.0,
    type=float,
    show_default=True,
    help="Maximum allowable F1 regression for promoted models",
)
@click.option(
    "--min-tpr-at-1fpr-improvement",
    default=None,
    type=float,
    help="Optional minimum TPR@1%%FPR improvement gate",
)
def promote(
    summary_path: Path,
    output_config: Path,
    output_summary: Path | None,
    calibration_dir: str,
    min_auroc_improvement: float,
    max_f1_regression: float,
    min_tpr_at_1fpr_improvement: float | None,
):
    """Promote calibration artifacts into a curated config using explicit policy gates."""
    summary = json.loads(summary_path.read_text())
    dataset = summary.get("dataset", "unknown")
    results = summary.get("results", {})

    selected_models: dict[str, str] = {}
    rejected_models: dict[str, dict[str, Any]] = {}

    for detector_name, metrics in results.items():
        reasons = _build_rejection_reasons(
            metrics,
            min_auroc_improvement=min_auroc_improvement,
            max_f1_regression=max_f1_regression,
            min_tpr_at_1fpr_improvement=min_tpr_at_1fpr_improvement,
        )
        model_path = _resolve_model_path(detector_name, dataset, summary, calibration_dir)

        if model_path is None:
            reasons.append("no calibration model artifact found")

        if reasons:
            rejected_models[detector_name] = {
                "reasons": reasons,
                "model_path": model_path,
                "before": metrics.get("before", {}),
                "after": metrics.get("after", {}),
                "delta": metrics.get("delta", {}),
            }
            continue

        selected_models[_preferred_calibration_key(detector_name)] = model_path

    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        _render_curated_config(selected_models, calibration_dir), encoding="utf-8"
    )

    decision_summary = {
        "dataset": dataset,
        "source_summary": str(summary_path),
        "policy": {
            "min_auroc_improvement": min_auroc_improvement,
            "max_f1_regression": max_f1_regression,
            "min_tpr_at_1fpr_improvement": min_tpr_at_1fpr_improvement,
        },
        "selected_models": selected_models,
        "rejected_models": rejected_models,
    }

    summary_output_path = output_summary or output_config.with_suffix(".summary.json")
    summary_output_path.write_text(
        json.dumps(decision_summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    click.echo(f"Generated curated config: {output_config}")
    click.echo(f"Promotion summary: {summary_output_path}")
    click.echo(
        f"Promoted {len(selected_models)} model(s); rejected {len(rejected_models)} model(s)."
    )


if __name__ == "__main__":
    cli()
