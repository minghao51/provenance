"""DetectGPT-style curvature detector using probability curvature."""

from __future__ import annotations

import random
from typing import cast

from provenance.core.base import BaseDetector, DetectorResult
from provenance.core.calibration import CalibratedDetectorMixin

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None  # type: ignore[assignment,misc]
    AutoModelForCausalLM = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]


class CurvatureDetector(CalibratedDetectorMixin, BaseDetector):
    """DetectGPT-style detector using probability curvature.

    Measures how the log-probability of text changes under perturbations.
    AI-generated text shows "negative curvature" - small changes cause
    large drops in probability, while human text is more robust.
    """

    name = "curvature_detectgpt"
    latency_tier = "medium"
    domains = ["prose", "academic"]

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-125M",
        n_perturbations: int = 10,
        mask_ratio: float = 0.15,
        device: str = "auto",
        seed: int = 42,
    ):
        if torch is None:
            raise ImportError(
                "torch and transformers are required for CurvatureDetector"
            )

        self.model_name = model_name
        self.n_perturbations = n_perturbations
        self.mask_ratio = mask_ratio
        self.rng = random.Random(seed)

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

    def _compute_log_prob(self, input_ids: torch.Tensor) -> float:
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            return cast(float, -outputs.loss.item())

    def _perturb_text(self, text: str) -> str:
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        input_ids = encodings["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        probs = torch.softmax(logits[:, :-1, :], dim=-1)
        perturbed_ids = input_ids.clone()

        n_tokens = input_ids.size(1) - 1
        n_mask = max(1, int(n_tokens * self.mask_ratio))
        mask_indices = self.rng.sample(range(1, n_tokens), n_mask)

        for idx in mask_indices:
            token_probs = probs[0, idx - 1]
            top_k = min(50, token_probs.size(0))
            top_probs, top_indices = torch.topk(token_probs, top_k)
            top_probs = top_probs / top_probs.sum()
            sampled_idx = cast(int, torch.multinomial(top_probs, 1).item())
            perturbed_ids[0, idx] = top_indices[sampled_idx]

        return cast(
            str, self.tokenizer.decode(perturbed_ids[0], skip_special_tokens=True)
        )

    def _extract_features(self, text: str) -> list[float]:
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        input_ids = encodings["input_ids"].to(self.device)

        if input_ids.size(1) <= 2:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        original_log_prob = self._compute_log_prob(input_ids)

        perturbed_log_probs = []
        for _ in range(self.n_perturbations):
            perturbed_text = self._perturb_text(text)
            if not perturbed_text.strip():
                continue

            perturbed_enc = self.tokenizer(
                perturbed_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )
            perturbed_ids = perturbed_enc["input_ids"].to(self.device)
            log_prob = self._compute_log_prob(perturbed_ids)
            perturbed_log_probs.append(log_prob)

        if not perturbed_log_probs:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        mean_perturbed = sum(perturbed_log_probs) / len(perturbed_log_probs)
        curvature = original_log_prob - mean_perturbed
        variance = sum((p - mean_perturbed) ** 2 for p in perturbed_log_probs) / len(
            perturbed_log_probs
        )
        std_perturbed = variance**0.5

        return [
            curvature,
            original_log_prob,
            mean_perturbed,
            std_perturbed,
            abs(curvature),
        ]

    def _extract_feature_names(self) -> list[str]:
        return [
            "curvature",
            "original_log_prob",
            "mean_perturbed_log_prob",
            "std_perturbed_log_prob",
            "abs_curvature",
        ]

    def detect(self, text: str) -> DetectorResult:
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )
        input_ids = encodings["input_ids"].to(self.device)

        if input_ids.size(1) <= 2:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Text too short for curvature analysis"},
            )

        original_log_prob = self._compute_log_prob(input_ids)

        perturbed_log_probs = []
        for _ in range(self.n_perturbations):
            perturbed_text = self._perturb_text(text)
            if not perturbed_text.strip():
                continue

            perturbed_enc = self.tokenizer(
                perturbed_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )
            perturbed_ids = perturbed_enc["input_ids"].to(self.device)
            log_prob = self._compute_log_prob(perturbed_ids)
            perturbed_log_probs.append(log_prob)

        if not perturbed_log_probs:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Failed to generate perturbations"},
            )

        mean_perturbed = sum(perturbed_log_probs) / len(perturbed_log_probs)
        curvature = original_log_prob - mean_perturbed

        variance = sum((p - mean_perturbed) ** 2 for p in perturbed_log_probs) / len(
            perturbed_log_probs
        )
        std_perturbed = variance**0.5

        calibrated = self._get_calibrated_score(text)
        if calibrated is not None:
            score, confidence = calibrated
        else:
            abs_curvature = abs(curvature)

            if abs_curvature < 0.05:
                score = 0.5
                confidence = 0.3
            elif abs_curvature < 0.15:
                score = 0.55 if curvature > 0 else 0.45
                confidence = 0.5
            elif abs_curvature < 0.3:
                score = 0.65 if curvature > 0 else 0.35
                confidence = 0.6
            elif abs_curvature < 0.5:
                score = 0.75 if curvature > 0 else 0.25
                confidence = 0.7
            else:
                score = 0.85 if curvature > 0 else 0.15
                confidence = 0.8

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "curvature": curvature,
                "original_log_prob": original_log_prob,
                "mean_perturbed_log_prob": mean_perturbed,
                "std_perturbed_log_prob": std_perturbed,
                "n_perturbations": len(perturbed_log_probs),
                "calibrated": calibrated is not None,
            },
        )


def register(registry=None) -> None:
    registry.register(CurvatureDetector)
