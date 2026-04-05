"""Comprehensive benchmark comparing all detectors on RAID and MAGE datasets."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from provenance.benchmarks.workflow import (
    BenchmarkEvaluator,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    DatasetRegistry,
)


def get_fast_detectors():
    """Get detectors that don't require model downloads."""
    detectors = []

    try:
        from provenance.detectors.stylometric.lightgbm_detector import (
            LightGBMDetector,
        )

        detectors.append(LightGBMDetector())
        print("✓ LightGBMDetector loaded")
    except Exception as e:
        print(f"✗ LightGBMDetector skipped: {e}")

    try:
        from provenance.detectors.stylometric.cognitive import CognitiveDetector

        detectors.append(CognitiveDetector())
        print("✓ CognitiveDetector loaded")
    except Exception as e:
        print(f"✗ CognitiveDetector skipped: {e}")

    return detectors


def get_transformer_detectors():
    """Get transformer-based detectors."""
    detectors = []

    try:
        from provenance.detectors.transformer.hf_classifier import (
            ChatGPTDetector,
            OpenAIDetector,
            RADARDetector,
        )

        detectors.append(OpenAIDetector())
        print("✓ OpenAIDetector loaded")
        detectors.append(ChatGPTDetector())
        print("✓ ChatGPTDetector loaded")
        detectors.append(RADARDetector())
        print("✓ RADARDetector loaded")
    except Exception as e:
        print(f"✗ Transformer detectors skipped: {e}")

    return detectors


def get_statistical_detectors():
    """Get statistical detectors."""
    detectors = []

    try:
        from provenance.detectors.statistical.perplexity import (
            PerplexityDetector,
            PerplexityDetectorNeo,
        )

        detectors.append(PerplexityDetector())
        print("✓ PerplexityDetector (GPT-2) loaded")
        detectors.append(PerplexityDetectorNeo())
        print("✓ PerplexityDetectorNeo (GPT-Neo) loaded")
    except Exception as e:
        print(f"✗ Perplexity detectors skipped: {e}")

    try:
        from provenance.detectors.statistical.curvature import CurvatureDetector

        detectors.append(CurvatureDetector(n_perturbations=5))
        print("✓ CurvatureDetector loaded")
    except Exception as e:
        print(f"✗ CurvatureDetector skipped: {e}")

    try:
        from provenance.detectors.statistical.surprisal import SurprisalDetector

        detectors.append(SurprisalDetector())
        print("✓ SurprisalDetector loaded")
    except Exception as e:
        print(f"✗ SurprisalDetector skipped: {e}")

    try:
        from provenance.detectors.statistical.burstiness import BurstinessDetector

        detectors.append(BurstinessDetector())
        print("✓ BurstinessDetector loaded")
    except Exception as e:
        print(f"✗ BurstinessDetector skipped: {e}")

    return detectors


def run_comprehensive_benchmark(
    sample_limit: int = 50,
    output_dir: str = "benchmark_results",
    run_transformers: bool = True,
):
    """Run comprehensive benchmark across all detector categories."""

    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sample limit: {sample_limit}")
    print("=" * 80)

    evaluator = BenchmarkEvaluator()
    runner = BenchmarkRunner(output_dir=output_dir)

    all_results: list[BenchmarkResult] = []

    datasets_to_run = ["raid"]

    for dataset_name in datasets_to_run:
        config = DatasetRegistry.get(dataset_name)
        if config is None:
            print(f"Dataset {dataset_name} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Loading {dataset_name} dataset...")
        print(f"{'=' * 60}")

        try:
            texts, labels, metadata = evaluator.dataset_loader.load(
                config, sample_limit=sample_limit
            )
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue

        print(
            f"Loaded {len(texts)} samples ({sum(labels)} AI, {len(labels) - sum(labels)} human)"
        )

        if not texts:
            continue

        detector_categories = [
            ("Statistical", get_statistical_detectors),
            ("Stylometric", get_fast_detectors),
        ]

        if run_transformers:
            detector_categories.append(("Transformer", get_transformer_detectors))

        for category_name, detector_fn in detector_categories:
            print(f"\n{'=' * 60}")
            print(f"Running {category_name} detectors...")
            print(f"{'=' * 60}")

            detectors = detector_fn()

            for detector in detectors:
                print(f"\n  Evaluating {detector.name}...")
                start = time.time()

                try:
                    result = evaluator.evaluate_detector(
                        detector,
                        texts,
                        labels,
                        threshold=0.5,
                        dataset_name=dataset_name,
                        show_progress=True,
                    )
                    all_results.append(result)
                    elapsed = time.time() - start
                    print(
                        f"  ✓ {detector.name}: "
                        f"AUROC={result.auroc:.4f} "
                        f"F1={result.f1:.4f} "
                        f"FPR@10TPR={result.fpr_at_10tpr:.4f} "
                        f"({elapsed:.1f}s)"
                    )
                except Exception as e:
                    print(f"  ✗ {detector.name} failed: {e}")

    suite = BenchmarkSuite(
        name=f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Comprehensive benchmark of all detectors",
        results=all_results,
        config={
            "sample_limit": sample_limit,
            "datasets": datasets_to_run,
            "run_transformers": run_transformers,
        },
    )

    runner.generate_report(suite, output_format="all")

    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(
        f"{'Detector':<30} {'AUROC':>8} {'F1':>8} {'FPR@10TPR':>10} {'Precision':>10} {'Recall':>8} {'Time(s)':>8}"
    )
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x.auroc, reverse=True):
        print(
            f"{r.detector_name:<30} {r.auroc:>8.4f} {r.f1:>8.4f} {r.fpr_at_10tpr:>10.4f} "
            f"{r.precision:>10.4f} {r.recall:>8.4f} {r.eval_time_seconds:>8.1f}"
        )

    results_path = Path(output_dir) / f"{suite.name}.json"
    print(f"\nFull results: {results_path}")

    return suite


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive detector benchmark")
    parser.add_argument(
        "-l",
        "--sample-limit",
        type=int,
        default=50,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--output", default="benchmark_results", help="Output directory"
    )
    parser.add_argument(
        "--no-transformers",
        action="store_true",
        help="Skip transformer-based detectors (slow)",
    )

    args = parser.parse_args()

    run_comprehensive_benchmark(
        sample_limit=args.sample_limit,
        output_dir=args.output,
        run_transformers=not args.no_transformers,
    )
