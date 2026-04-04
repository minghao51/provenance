"""LLM-based detectors using local and cloud LLM backends."""

from __future__ import annotations

import json
import os

from provenance.core.base import BaseDetector, DetectorResult

try:
    import ollama
except ImportError:
    ollama = None

try:
    import litellm
except ImportError:
    litellm = None


class OllamaLogProbDetector(BaseDetector):
    name = "logprob_local_llm"
    latency_tier = "slow"
    domains = ["prose", "academic", "code"]

    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        if ollama is None:
            raise ImportError("ollama is required for OllamaLogProbDetector")

        self.model = model
        self.client = ollama.Client(host=host)

    def detect(self, text: str) -> DetectorResult:
        try:
            response = self.client.generate(
                model=self.model,
                prompt=text,
                options={
                    "num_predict": 1,
                    "temperature": 0.0,
                },
            )

            logprobs = response.get("logprobs", [])

            if not logprobs:
                return DetectorResult(
                    score=0.5,
                    confidence=0.3,
                    metadata={"error": "Could not extract log probabilities"},
                )

            mean_logprob = sum(logprobs) / len(logprobs)

            score = 1.0 - (mean_logprob / -10.0)
            score = max(0.0, min(1.0, score))

            return DetectorResult(
                score=score,
                confidence=0.7,
                metadata={
                    "mean_logprob": mean_logprob,
                    "model": self.model,
                    "num_tokens": len(logprobs),
                },
            )
        except Exception as e:
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                metadata={"error": str(e)},
            )


class DetectGPTDetector(BaseDetector):
    name = "detectgpt"
    latency_tier = "slow"
    domains = ["prose", "academic"]

    def __init__(
        self,
        model: str | None = None,
        n_perturbations: int = 5,
        backend: str | None = None,
    ):
        if litellm is None:
            raise ImportError("litellm is required for DetectGPTDetector")

        self.model = model or os.environ.get("LITELLM_MODEL", "openai/gpt-3.5-turbo")
        self.n_perturbations = n_perturbations
        self.backend = backend or self.model

    def _generate_perturbations(self, text: str) -> list[str]:
        perturbation_prompt = f"""Rewrite the following text in different words, maintaining the same meaning but changing the phrasing.
Keep the length similar. Return ONLY the rewritten text, nothing else.

Text: {text}"""

        perturbations = []
        for _ in range(self.n_perturbations):
            try:
                response = litellm.completion(
                    model=self.backend,
                    messages=[{"role": "user", "content": perturbation_prompt}],
                    temperature=0.9,
                )
                perturbed = response["choices"][0]["message"]["content"]
                perturbations.append(perturbed)
            except Exception:
                pass

        return perturbations

    def _compute_ai_scores(self, texts: list[str]) -> list[float]:
        log_probs = []

        for text in texts:
            try:
                response = litellm.completion(
                    model=self.backend,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Score the probability this text is AI-generated on a scale of 0-10: {text}",
                        }
                    ],
                    temperature=0.0,
                )
                score_text = response["choices"][0]["message"]["content"]
                try:
                    score = float(score_text) / 10.0
                except ValueError:
                    score = 0.5
                log_probs.append(score)
            except Exception:
                log_probs.append(0.5)

        return log_probs

    def detect(self, text: str) -> DetectorResult:
        original_score = self._compute_ai_scores([text])[0]

        perturbations = self._generate_perturbations(text)
        if not perturbations:
            return DetectorResult(
                score=original_score,
                confidence=0.3,
                metadata={"error": "Could not generate perturbations"},
            )

        perturbed_scores = self._compute_ai_scores(perturbations)
        mean_perturbed = sum(perturbed_scores) / len(perturbed_scores)

        difference = original_score - mean_perturbed

        if difference > 0.2:
            score = 0.8
            confidence = 0.7
        elif difference > 0.1:
            score = 0.6
            confidence = 0.6
        elif difference < -0.1:
            score = 0.3
            confidence = 0.5
        else:
            score = 0.5
            confidence = 0.4

        return DetectorResult(
            score=score,
            confidence=confidence,
            metadata={
                "original_score": original_score,
                "mean_perturbed_score": mean_perturbed,
                "difference": difference,
                "n_perturbations": len(perturbations),
            },
        )


class LLMMetaReasoningDetector(BaseDetector):
    name = "llm_meta_reasoning"
    latency_tier = "slow"
    domains = ["prose", "academic"]

    ESCALATION_PROMPT = """You are an expert forensic linguist analyzing text for AI generation.

Text:
{text}

Automated Analysis:
- Perplexity Score: {perplexity_score}
- Burstiness CV: {burstiness_cv}
- Top flagged features: {top_features}
- Ensemble Score: {ensemble_score}

Analyze the text step by step for signs of AI generation, considering:
1. Repetitiveness and lack of natural variation
2. Overuse of transition phrases
3. Unusual sentence structure patterns
4. Generic and non-specific language

Return a JSON object with your analysis:
{{"score": float (0.0-1.0, higher = more likely AI), "reasoning": str, "confidence": "low"|"medium"|"high"}}
"""

    def __init__(
        self,
        model: str | None = None,
        backend: str | None = None,
        escalation_threshold: float = 0.6,
    ):
        if litellm is None:
            raise ImportError("litellm is required for LLMMetaReasoningDetector")

        self.model = model or os.environ.get("LITELLM_MODEL", "openai/gpt-4")
        self.backend = backend or self.model
        self.escalation_threshold = escalation_threshold

    def detect(
        self,
        text: str,
        ensemble_score: float | None = None,
        perplexity_score: float | None = None,
        burstiness_cv: float | None = None,
        top_features: list[str] | None = None,
    ) -> DetectorResult:
        prompt = self.ESCALATION_PROMPT.format(
            text=text[:2000],
            perplexity_score=perplexity_score or "N/A",
            burstiness_cv=burstiness_cv or "N/A",
            top_features=", ".join(top_features[:5]) if top_features else "N/A",
            ensemble_score=ensemble_score or "N/A",
        )

        try:
            response = litellm.completion(
                model=self.backend,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response["choices"][0]["message"]["content"]

            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result_json = json.loads(content[json_start:json_end])
                score = float(result_json.get("score", 0.5))
                confidence_str = result_json.get("confidence", "medium")
                reasoning = result_json.get("reasoning", "")

                confidence_map = {"low": 0.4, "medium": 0.7, "high": 0.9}
                confidence = confidence_map.get(confidence_str, 0.5)

                return DetectorResult(
                    score=score,
                    confidence=confidence,
                    metadata={
                        "reasoning": reasoning,
                        "model": self.model,
                    },
                )
        except Exception:
            pass

        return DetectorResult(
            score=0.5,
            confidence=0.0,
            metadata={"error": "LLM reasoning failed"},
        )


def register(registry) -> None:
    if ollama is not None:
        registry.register(OllamaLogProbDetector)
    if litellm is not None:
        registry.register(DetectGPTDetector)
        registry.register(LLMMetaReasoningDetector)
