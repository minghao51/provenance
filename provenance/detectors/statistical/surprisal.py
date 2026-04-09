"""Surprisal-based features inspired by DivEye for paraphrase-robust detection."""

from __future__ import annotations

import math
from typing import cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin
from provenance.core.config import SurprisalThresholds


class SurprisalDetector(CalibratedDetectorMixin, BaseDetector):
    """Detect AI text using surprisal (negative log probability) patterns.

    Inspired by DivEye framework. AI text shows characteristic patterns:
    - Low surprisal variance (uniform token probabilities)
    - Low surprisal autocorrelation (no natural "bursts")
    - Predictable surprisal sequences
    """

    name = "surprisal_diveye"
    latency_tier = "medium"
    domains = ["prose", "academic"]
    calibration_aliases = ("surprisal", "surprisal_diveye")

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-125M",
        window_size: int = 512,
        device: str = "auto",
        thresholds: SurprisalThresholds | None = None,
    ):
        self.model_name = model_name
        self.window_size = window_size
        self.thresholds = thresholds or SurprisalThresholds()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def _compute_token_surprisals(self, text: str) -> list[float]:
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.window_size,
            add_special_tokens=True,
        )
        input_ids = encodings["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return cast(list[float], token_losses.cpu().tolist())

    def _compute_surprisal_variance(self, surprisals: list[float]) -> float:
        if len(surprisals) < 2:
            return 0.0
        mean_s = sum(surprisals) / len(surprisals)
        return sum((s - mean_s) ** 2 for s in surprisals) / len(surprisals)

    def _compute_surprisal_autocorrelation(
        self, surprisals: list[float], lag: int = 1
    ) -> float:
        n = len(surprisals)
        if n < lag + 2:
            return 0.0

        mean_s = sum(surprisals) / n
        variance = sum((s - mean_s) ** 2 for s in surprisals) / n

        if variance < 1e-10:
            return 0.0

        autocov = sum(
            (surprisals[i] - mean_s) * (surprisals[i + lag] - mean_s)
            for i in range(n - lag)
        ) / (n - lag)

        return autocov / variance

    def _compute_surprisal_burstiness(self, surprisals: list[float]) -> float:
        if len(surprisals) < 2:
            return 0.0

        mean_s = sum(surprisals) / len(surprisals)
        if mean_s < 1e-10:
            return 0.0

        variance_sum = sum((s - mean_s) ** 2 for s in surprisals)
        std_s = (variance_sum / len(surprisals)) ** 0.5
        return cast(float, std_s / mean_s)

    def _compute_surprisal_entropy(self, surprisals: list[float]) -> float:
        if not surprisals:
            return 0.0

        total = sum(surprisals)
        if total < 1e-10:
            return 0.0

        probs = [s / total for s in surprisals]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _compute_surprisal_trend(self, surprisals: list[float]) -> float:
        n = len(surprisals)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2
        y_mean = sum(surprisals) / n

        numerator = sum((i - x_mean) * (surprisals[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def _extract_features(self, text: str) -> list[float]:
        surprisals = self._compute_token_surprisals(text)
        if len(surprisals) < 5:
            return [0.0] * 6

        variance = self._compute_surprisal_variance(surprisals)
        autocorr = self._compute_surprisal_autocorrelation(surprisals)
        burstiness = self._compute_surprisal_burstiness(surprisals)
        entropy = self._compute_surprisal_entropy(surprisals)
        trend = self._compute_surprisal_trend(surprisals)
        mean_surprisal = sum(surprisals) / len(surprisals)

        return [variance, autocorr, burstiness, entropy, trend, mean_surprisal]

    def _extract_feature_names(self) -> list[str]:
        return [
            "surprisal_variance",
            "surprisal_autocorr",
            "surprisal_burstiness",
            "surprisal_entropy",
            "surprisal_trend",
            "mean_surprisal",
        ]

    def detect(self, text: str) -> DetectorResult:
        try:
            surprisals = self._compute_token_surprisals(text)

            if len(surprisals) < self.thresholds.min_surprisal_tokens:
                return self.build_error_result("Text too short for surprisal analysis")

            variance = self._compute_surprisal_variance(surprisals)
            autocorr = self._compute_surprisal_autocorrelation(surprisals)
            burstiness = self._compute_surprisal_burstiness(surprisals)
            entropy = self._compute_surprisal_entropy(surprisals)
            trend = self._compute_surprisal_trend(surprisals)
            mean_surprisal = sum(surprisals) / len(surprisals)

            calibrated = self._get_calibrated_score(text)
            if calibrated is not None:
                score, confidence = calibrated
            else:
                ai_score = 0.0

                for threshold, increment in self.thresholds.variance_bands:
                    if variance < threshold:
                        ai_score += increment
                        break

                for threshold, increment in self.thresholds.autocorr_bands:
                    if autocorr < threshold:
                        ai_score += increment
                        break

                for threshold, increment in self.thresholds.burstiness_bands:
                    if burstiness < threshold:
                        ai_score += increment
                        break

                for threshold, increment in self.thresholds.mean_surprisal_bands:
                    if mean_surprisal < threshold:
                        ai_score += increment
                        break

                abs_trend = abs(trend)
                for threshold, increment in self.thresholds.trend_bands:
                    if abs_trend < threshold:
                        ai_score += increment
                        break

                score = min(1.0, max(0.0, ai_score))

                if variance < 3.0 and burstiness < 0.4:
                    confidence = 0.85
                elif variance < 7.0 and burstiness < 0.6:
                    confidence = 0.7
                else:
                    confidence = 0.55

            return DetectorResult(
                score=score,
                confidence=confidence,
                metadata={
                    "surprisal_variance": variance,
                    "surprisal_autocorr": autocorr,
                    "surprisal_burstiness": burstiness,
                    "surprisal_entropy": entropy,
                    "surprisal_trend": trend,
                    "mean_surprisal": mean_surprisal,
                    "calibrated": calibrated is not None,
                },
            )
        except Exception as e:
            return self.build_error_result(
                "Surprisal analysis failed",
                exception=e,
            )


def register(registry=None) -> None:
    registry.register(SurprisalDetector)
