"""Perplexity-based detector using GPT-2."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from provenance.core.base import BaseDetector, DetectorResult


class PerplexityDetector(BaseDetector):
    name = "perplexity_gpt2"
    latency_tier = "medium"
    domains = ["prose", "academic"]

    def __init__(
        self,
        model_name: str = "gpt2",
        window_size: int = 512,
        stride: int = 256,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.window_size = window_size
        self.stride = stride

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def _compute_windowed_ppl(self, text: str) -> list[float]:
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )

        input_ids = encodings["input_ids"].to(self.device)
        seq_len = input_ids.size(1)

        if seq_len <= 1:
            return []

        max_length = self.window_size
        stride = self.stride

        nlls = []

        for start_loc in range(0, seq_len, stride):
            end_loc = min(start_loc + max_length, seq_len)
            if end_loc - start_loc <= 1:
                break
            slice_ids = input_ids[:, start_loc:end_loc]

            with torch.no_grad():
                outputs = self.model(slice_ids, labels=slice_ids)
                neg_log_likelihood = outputs.loss.item()

            nlls.append(neg_log_likelihood)

        perplexities = [torch.exp(torch.tensor(nll)).item() for nll in nlls]
        return perplexities

    def detect(self, text: str) -> DetectorResult:
        window_perplexities = self._compute_windowed_ppl(text)

        if not window_perplexities:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": "Text too short to compute perplexity"},
            )

        mean_ppl = sum(window_perplexities) / len(window_perplexities)
        variance_ppl = sum((p - mean_ppl) ** 2 for p in window_perplexities) / len(
            window_perplexities
        )
        std_ppl = variance_ppl**0.5

        score = min(1.0, max(0.0, (mean_ppl - 10) / 90))
        score = 1.0 - score

        if std_ppl < 5:
            confidence = 0.9
        elif std_ppl < 15:
            confidence = 0.7
        else:
            confidence = 0.5

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "mean_perplexity": mean_ppl,
                "std_perplexity": std_ppl,
                "window_perplexities": window_perplexities,
            },
        )


class PerplexityDetectorNeo(PerplexityDetector):
    """Perplexity-based detector using GPT-Neo for better modern AI detection."""

    name = "perplexity_gptneo"
    latency_tier = "slow"  # type: ignore[assignment]
    domains = ["prose", "academic"]

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-125M",
        window_size: int = 512,
        stride: int = 256,
        device: str = "auto",
    ):
        super().__init__(
            model_name=model_name,
            window_size=window_size,
            stride=stride,
            device=device,
        )


def register(registry=None) -> None:
    registry.register(PerplexityDetector)
    registry.register(PerplexityDetectorNeo)
