"""Ensemble layer with weighted voting, calibration, and confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .base import BaseDetector, DetectorResult, SentinelResult


@dataclass
class EnsembleConfig:
    strategy: Literal["weighted_average", "stacking", "uncertainty_aware"] = (
        "weighted_average"
    )
    weights: dict[str, float] = field(default_factory=dict)
    confidence_threshold: float = 0.6
    calibration_method: Literal["isotonic", "platt"] | None = None


class Ensemble:
    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        self.detectors: list[BaseDetector] = []
        self.calibration_models: dict[str, object] = {}
        self._stacker = None
        self._feature_names: list[str] = []

    def add_detector(self, detector: BaseDetector) -> None:
        self.detectors.append(detector)
        self._feature_names = [d.name for d in self.detectors]

    def _compute_weighted_average(
        self, detector_scores: dict[str, DetectorResult]
    ) -> float:
        weights = self.config.weights

        if not detector_scores:
            return 0.5

        if not weights:
            return self._compute_average_score(detector_scores)

        total_weight = sum(weights.get(name, 1.0) for name in detector_scores)
        weighted_sum = sum(
            detector_scores[name].score * weights.get(name, 1.0)
            for name in detector_scores
        )
        return weighted_sum / total_weight

    def _compute_average_score(
        self, detector_scores: dict[str, DetectorResult]
    ) -> float:
        """Compute average score across all detector results."""
        if not detector_scores:
            return 0.5
        return sum(r.score for r in detector_scores.values()) / len(detector_scores)

    def _compute_stacking(self, detector_scores: dict[str, DetectorResult]) -> float:
        if self._stacker is None:
            return self._compute_average_score(detector_scores)

        features = [
            detector_scores[name].score
            for name in self._feature_names
            if name in detector_scores
        ]
        if len(features) != len(self._feature_names):
            return self._compute_average_score(detector_scores)

        features_arr = [[f] for f in features]
        try:
            calibrated = self._stacker.predict_proba(features_arr)[0][1]
            return float(calibrated)
        except Exception:
            return self._compute_average_score(detector_scores)

    def _compute_uncertainty_aware_vote(
        self, detector_scores: dict[str, DetectorResult]
    ) -> tuple[float, float]:
        confidence_threshold = self.config.confidence_threshold

        total_weight = 0.0
        weighted_score = 0.0

        for result in detector_scores.values():
            weight = (
                result.confidence
                if result.confidence >= confidence_threshold
                else result.confidence * 0.5
            )

            weighted_score += result.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5, 0.0

        final_score = weighted_score / total_weight
        avg_confidence = sum(r.confidence for r in detector_scores.values()) / len(
            detector_scores
        )

        return final_score, avg_confidence

    def _determine_label(
        self, score: float, confidence: float
    ) -> Literal["human", "ai", "mixed", "uncertain"]:
        if confidence < 0.3:
            return "uncertain"

        if score < 0.35:
            return "human"
        if score > 0.65:
            return "ai"
        if confidence < 0.5:
            return "uncertain"
        return "mixed"

    def _compute_average_confidence(
        self, detector_scores: dict[str, DetectorResult]
    ) -> float:
        """Compute average confidence across all detector results."""
        if not detector_scores:
            return 0.0
        return sum(r.confidence for r in detector_scores.values()) / len(
            detector_scores
        )

    def _collect_heatmap(self, detector_scores: dict[str, DetectorResult]) -> list:
        from provenance.core.base import TokenScore

        all_heatmaps: list[TokenScore] = []

        for result in detector_scores.values():
            if "heatmap" in result.metadata and result.metadata["heatmap"]:
                hm = result.metadata["heatmap"]
                if isinstance(hm, list) and len(hm) > 0:
                    if isinstance(hm[0], TokenScore):
                        all_heatmaps.extend(hm)
                    elif isinstance(hm[0], dict):
                        for item in hm:
                            if "token" in item and "score" in item:
                                all_heatmaps.append(
                                    TokenScore(token=item["token"], score=item["score"])
                                )

        if not all_heatmaps:
            return []

        return all_heatmaps

    def _collect_feature_vector(
        self, detector_scores: dict[str, DetectorResult]
    ) -> dict:
        merged: dict = {}
        for result in detector_scores.values():
            if "features" in result.metadata and result.metadata["features"]:
                merged.update(result.metadata["features"])
        return merged

    def _collect_top_features(self, detector_scores: dict[str, DetectorResult]) -> list:
        all_features: list[tuple[str, float]] = []
        for result in detector_scores.values():
            if "top_features" in result.metadata and result.metadata["top_features"]:
                tf = result.metadata["top_features"]
                if isinstance(tf, list) and len(tf) > 0:
                    if isinstance(tf[0], tuple):
                        all_features.extend(tf)
                    elif isinstance(tf[0], dict):
                        for item in tf:
                            if "feature" in item and "importance" in item:
                                all_features.append(
                                    (item["feature"], item["importance"])
                                )
        all_features.sort(key=lambda x: abs(x[1]), reverse=True)
        return all_features[:20]

    def ensemble_detect(self, text: str) -> SentinelResult:
        from .base import SentinelResult
        from .base import DetectorResult

        detector_scores: dict[str, DetectorResult] = {}
        for detector in self.detectors:
            try:
                result = detector.detect(text)
            except Exception as e:
                result = detector.build_error_result(
                    "Detector execution failed",
                    exception=e,
                )
            detector_scores[detector.name] = result

        if not detector_scores:
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

        if self.config.strategy == "weighted_average":
            score = self._compute_weighted_average(detector_scores)
            confidence = self._compute_average_confidence(detector_scores)
        elif self.config.strategy == "stacking":
            score = self._compute_stacking(detector_scores)
            confidence = self._compute_average_confidence(detector_scores)
        elif self.config.strategy == "uncertainty_aware":
            score, confidence = self._compute_uncertainty_aware_vote(detector_scores)
        else:
            score = self._compute_average_score(detector_scores)
            confidence = self._compute_average_confidence(detector_scores)

        label = self._determine_label(score, confidence)

        heatmap = self._collect_heatmap(detector_scores)
        feature_vector = self._collect_feature_vector(detector_scores)
        top_features = self._collect_top_features(detector_scores)

        return SentinelResult(
            score=score,
            label=label,
            confidence=confidence,
            detector_scores=detector_scores,
            heatmap=heatmap,
            sentence_scores=[],
            feature_vector=feature_vector,
            top_features=top_features,
        )

    def calibrate(
        self,
        texts: list[str],
        labels: list[int],
        method: Literal["isotonic", "platt"] = "isotonic",
    ) -> None:
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return

        if not texts or len(texts) < 10:
            return

        features_list: list[list[float]] = []
        for text in texts:
            text_features: list[float] = []
            for detector in self.detectors:
                result = detector.detect(text)
                text_features.append(result.score)
            features_list.append(text_features)

        if len(features_list) < 10 or len(set(labels)) < 2:
            return

        if method == "platt":
            self._stacker = LogisticRegression(C=1.0, max_iter=1000)
        else:
            from sklearn.calibration import CalibratedClassifierCV

            base = LogisticRegression(C=1.0, max_iter=1000)
            self._stacker = CalibratedClassifierCV(base, cv=3, method="isotonic")

        try:
            if self._stacker is None:
                return
            self._stacker.fit(features_list, labels)
            self.calibration_models["stacker"] = self._stacker
        except Exception as e:
            self._stacker = None
            raise RuntimeError(f"Calibration failed: {e}") from e

    def optimize_weights(self, texts: list[str], labels: list[int]) -> dict[str, float]:
        from scipy.optimize import minimize

        if not texts or len(texts) < 10:
            return self.config.weights

        detector_scores: dict[str, list[float]] = {d.name: [] for d in self.detectors}

        for text in texts:
            for detector in self.detectors:
                result = detector.detect(text)
                detector_scores[detector.name].append(result.score)

        def objective(weights):
            weights = weights / weights.sum()
            weighted_scores = sum(
                weights[i] * detector_scores[detector.name]
                for i, detector in enumerate(self.detectors)
            )
            from sklearn.metrics import brier_score_loss

            return brier_score_loss(labels, weighted_scores)

        n_detectors = len(self.detectors)
        initial_weights = [1.0 / n_detectors] * n_detectors
        bounds = [(0.01, 1.0)] * n_detectors

        result = minimize(
            objective,
            initial_weights,
            bounds=bounds,
            method="L-BFGS-B",
        )

        optimal_weights = result.x / result.x.sum()
        self.config.weights = {
            detector.name: float(optimal_weights[i])
            for i, detector in enumerate(self.detectors)
        }

        return self.config.weights


def compute_confidence_interval(
    scores: list[float],
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    import math
    import statistics

    if not scores:
        return 0.0, 0.0

    mean = statistics.mean(scores)
    n = len(scores)

    if n < 2:
        return mean, 0.0

    stdev = statistics.stdev(scores)
    z = 1.96 if confidence_level == 0.95 else 2.576
    margin = stdev * z / math.sqrt(n)

    return mean - margin, mean + margin
