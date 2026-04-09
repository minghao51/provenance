"""Provenance - Main entry point for AI text detection."""

from __future__ import annotations

from typing import Literal

from provenance.core.base import DetectorResult, SentinelResult, TokenScore
from provenance.core.config import ProvenanceConfig, resolve_provenance_config
from provenance.core.ensemble import Ensemble, EnsembleConfig
from provenance.core.preprocessor import Preprocessor
from provenance.core.registry import get_registry
from provenance.explainability.heatmaps import (
    compute_sentence_scores,
    generate_token_heatmap,
)


class Provenance:
    MIN_TEXT_LENGTH = 150

    def __init__(
        self,
        detectors: list[str] | None = None,
        ensemble_strategy: Literal[
            "weighted_average", "stacking", "uncertainty_aware"
        ] = "weighted_average",
        weights: dict[str, float] | None = None,
        latency_budget: Literal["fast", "medium", "slow"] | None = None,
        preprocessor: Preprocessor | None = None,
        config: ProvenanceConfig | str | None = None,
    ):
        self.config = resolve_provenance_config(config)
        self.preprocessor = preprocessor or Preprocessor()
        self.registry = get_registry()
        self.registry.load_entry_points()

        if detectors is None:
            available = self.registry.list_detectors(
                latency_tier=latency_budget,
            )
            detector_names = [d.name for d in available]
        else:
            detector_names = detectors

        self.ensemble = Ensemble(
            config=EnsembleConfig(
                strategy=ensemble_strategy,
                weights=weights or {},
            )
        )

        for name in detector_names:
            detector = self.registry.get(name)
            if detector:
                if hasattr(detector, "load_calibration"):
                    calibration_path = self.config.detector_calibration_paths.get(
                        detector.name
                    )
                    if calibration_path:
                        detector.load_calibration(calibration_path)
                    elif (
                        self.config.calibration_model_dir
                        and hasattr(detector, "load_default_calibration")
                    ):
                        detector.load_default_calibration(
                            self.config.calibration_model_dir
                        )
                self.ensemble.add_detector(detector)

    def _aggregate_chunk_results(
        self,
        chunk_results: list[SentinelResult],
        text: str,
    ) -> SentinelResult:
        if not chunk_results:
            return SentinelResult(
                score=0.5,
                label="uncertain",
                confidence=0.0,
                detector_scores={},
                heatmap=[],
                sentence_scores=[],
                feature_vector={},
                top_features=[],
            )

        avg_score = sum(r.score for r in chunk_results) / len(chunk_results)
        avg_confidence = sum(r.confidence for r in chunk_results) / len(chunk_results)

        all_detector_scores: dict[str, list[DetectorResult]] = {}
        for result in chunk_results:
            for name, dr in result.detector_scores.items():
                if name not in all_detector_scores:
                    all_detector_scores[name] = []
                all_detector_scores[name].append(dr)

        aggregated_detector_scores: dict[str, DetectorResult] = {}
        for name, results in all_detector_scores.items():
            avg_dr_score = sum(r.score for r in results) / len(results)
            avg_dr_confidence = sum(r.confidence for r in results) / len(results)
            merged_metadata: dict = {}
            for r in results:
                merged_metadata.update(r.metadata)
            aggregated_detector_scores[name] = DetectorResult(
                score=avg_dr_score,
                confidence=avg_dr_confidence,
                metadata=merged_metadata,
            )

        all_heatmaps: list[TokenScore] = []
        for result in chunk_results:
            all_heatmaps.extend(result.heatmap)
        if not all_heatmaps:
            all_heatmaps = generate_token_heatmap(text)

        sentences = self.preprocessor.split_sentences(text)
        num_tokens_per_sent: list[int] = []
        for sent in sentences:
            num_tokens_per_sent.append(len(sent.split()))

        sentence_boundaries: list[tuple[int, int]] = []
        token_idx = 0
        for count in num_tokens_per_sent:
            if count > 0:
                sentence_boundaries.append((token_idx, token_idx + count))
                token_idx += count
            else:
                sentence_boundaries.append((token_idx, token_idx))

        sentence_scores = compute_sentence_scores(all_heatmaps, sentence_boundaries)

        all_feature_vectors: list[dict] = []
        for result in chunk_results:
            if result.feature_vector:
                all_feature_vectors.append(result.feature_vector)
        merged_feature_vector: dict = {}
        for fv in all_feature_vectors:
            merged_feature_vector.update(fv)

        all_top_features: list[tuple[str, float]] = []
        for result in chunk_results:
            all_top_features.extend(result.top_features)

        label = self.ensemble._determine_label(avg_score, avg_confidence)

        return SentinelResult(
            score=avg_score,
            label=label,
            confidence=avg_confidence,
            detector_scores=aggregated_detector_scores,
            heatmap=all_heatmaps[: self.config.max_heatmap_tokens],
            sentence_scores=sentence_scores,
            feature_vector=merged_feature_vector,
            top_features=all_top_features[: self.config.max_top_features],
        )

    def detect(self, text: str) -> SentinelResult:
        preprocessed = self.preprocessor.preprocess(text)

        if (
            len(preprocessed.sentences) == 0
            or len(text.split()) < self.config.min_text_length
        ):
            short_text_warning = self.config.short_text_warning_template.format(
                min_length=self.config.min_text_length
            )
            dummy_result = DetectorResult(
                score=0.5,
                confidence=0.3,
                metadata={"warning": short_text_warning},
            )
            return SentinelResult(
                score=0.5,
                label="uncertain",
                confidence=self.config.short_text_confidence,
                detector_scores={"short_text_warning": dummy_result},
                heatmap=[],
                sentence_scores=[],
                feature_vector={},
                top_features=[],
            )

        chunks = preprocessed.chunks
        if not chunks:
            chunks = [self.preprocessor.chunk_text(text)[0]]

        chunk_results: list[SentinelResult] = []
        for chunk in chunks:
            result = self.ensemble.ensemble_detect(chunk.text)
            chunk_results.append(result)

        return self._aggregate_chunk_results(
            chunk_results, preprocessed.normalized_text
        )

    def audit(
        self,
        texts: list[str],
        labels: list[int],
        languages: list[str] | None = None,
    ) -> dict:
        from provenance.benchmarks.evaluation import run_audit

        return run_audit(
            detector=None,
            texts=texts,
            labels=labels,
            languages=languages,
        )

    def apply_detector_thresholds(
        self,
        detector_name: str,
        thresholds: "EntropyThresholds | BurstinessThresholds | RepetitionThresholds | CurvatureThresholds | SurprisalThresholds",
    ) -> bool:
        """Apply threshold configuration to a specific detector in the ensemble.

        Args:
            detector_name: Name of the detector to configure
            thresholds: Threshold configuration object

        Returns:
            True if thresholds were applied, False if detector not found
        """
        from provenance.core.config import (
            BurstinessThresholds,
            CurvatureThresholds,
            EntropyThresholds,
            RepetitionThresholds,
            SurprisalThresholds,
        )

        for detector in self.ensemble.detectors:
            if detector.name == detector_name and hasattr(detector, "thresholds"):
                detector.thresholds = thresholds
                return True
        return False
