"""HuggingFace transformer-based classifiers for AI text detection."""

from __future__ import annotations

from provenance.core.base import BaseDetector, DetectorResult, TokenScore

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None
    torch = None

try:
    from captum.attr import IntegratedGradients

    CAPTUM_AVAILABLE = True
except ImportError:
    IntegratedGradients = None
    CAPTUM_AVAILABLE = False


class HuggingFaceClassifierDetector(BaseDetector):
    name = "hf_classifier"
    latency_tier = "slow"
    domains = ["prose", "academic"]

    MODEL_REGISTRY: dict[str, str] = {
        "openai_detector": "openai-community/roberta-base-openai-detector",
        "chatgpt_detector": "Hello-SimpleAI/chatgpt-detector-roberta",
        "radar": "coai/roberta-ai-detector-v2",
    }

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "auto",
        truncation: bool = True,
        max_length: int = 512,
    ):
        if AutoModelForSequenceClassification is None:
            raise ImportError(
                "transformers is required for HuggingFaceClassifierDetector"
            )

        self.model_id = model_id or self.MODEL_REGISTRY["chatgpt_detector"]

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.truncation = truncation
        self.max_length = max_length

        self.classifier = pipeline(
            "text-classification",
            model=self.model_id,
            device=0 if self.device == "cuda" else -1,
            truncation=truncation,
            max_length=max_length,
        )
        self.model = self.classifier.model
        self.tokenizer = self.classifier.tokenizer

        self.model_id = model_id or self.MODEL_REGISTRY["chatgpt_detector"]

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.truncation = truncation
        self.max_length = max_length

        self.classifier = pipeline(
            "text-classification",
            model=self.model_id,
            device=0 if self.device == "cuda" else -1,
            truncation=truncation,
            max_length=max_length,
        )
        self.model = self.classifier.model
        self.tokenizer = self.classifier.tokenizer

    def detect(self, text: str) -> DetectorResult:
        result = self.classifier(text)[0]

        label = result["label"].lower()
        if "human" in label or "real" in label:
            score = 1.0 - result["score"]
        elif "ai" in label or "fake" in label or "gpt" in label:
            score = result["score"]
        else:
            score = result["score"]

        score = float(score)
        confidence = float(result["score"])

        return DetectorResult(
            score=score,
            confidence=min(1.0, confidence + 0.1),
            metadata={
                "model_id": self.model_id,
                "raw_label": result["label"],
                "raw_score": result["score"],
            },
        )


class OpenAIDetector(HuggingFaceClassifierDetector):
    name = "hf_openai_detector"
    latency_tier = "slow"
    domains = ["prose", "academic"]

    def __init__(self, device: str = "auto"):
        super().__init__(
            model_id=self.MODEL_REGISTRY["openai_detector"],
            device=device,
        )


class ChatGPTDetector(HuggingFaceClassifierDetector):
    name = "hf_chatgpt_detector"
    latency_tier = "slow"
    domains = ["prose", "academic"]

    def __init__(self, device: str = "auto"):
        super().__init__(
            model_id=self.MODEL_REGISTRY["chatgpt_detector"],
            device=device,
        )


class RADARDetector(HuggingFaceClassifierDetector):
    name = "hf_radar_detector"
    latency_tier = "slow"
    domains = ["prose", "academic", "code"]

    def __init__(self, device: str = "auto"):
        super().__init__(
            model_id=self.MODEL_REGISTRY["radar"],
            device=device,
        )


class RAIDDetection(HuggingFaceClassifierDetector):
    """Fine-tuned RoBERTa detector trained on RAID dataset."""

    name = "hf_raid_detector"
    latency_tier = "slow"
    domains = ["prose", "academic", "code"]

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
    ):
        if model_path is None:
            model_path = "models/raid_roberta"

        super().__init__(
            model_id=model_path,
            device=device,
            truncation=True,
            max_length=512,
        )


class AttentionHeatmapDetector(HuggingFaceClassifierDetector):
    name = "hf_attention_heatmap"
    latency_tier = "slow"
    domains = ["prose", "academic"]

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "auto",
        max_length: int = 512,
        use_captum: bool = True,
    ):
        super().__init__(
            model_id=model_id or "Hello-SimpleAI/chatgpt-detector-roberta",
            device=device,
            max_length=max_length,
        )
        self.use_captum = use_captum and CAPTUM_AVAILABLE
        self.integrated_gradients = None

        if self.use_captum:
            self.integrated_gradients = IntegratedGradients(self.model)

    def _extract_integrated_gradients_heatmap(self, text: str) -> list[TokenScore]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        embeddings = self.model.roberta.embeddings(input_ids)
        embeddings.requires_grad_(True)

        def forward_func(embeds):
            outputs = self.model.roberta(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
            )
            pooled = outputs.last_hidden_state[:, 0, :]
            return self.model.classifier(pooled)[:, 1]

        ig = IntegratedGradients(forward_func)
        attributions, _ = ig.attribute(
            embeddings,
            target=1,
            return_convergence_delta=False,
        )

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / attributions.sum()

        heatmap = []
        for token, score in zip(tokens, attributions, strict=False):
            if token not in [
                self.tokenizer.pad_token,
                self.tokenizer.bos_token,
                self.tokenizer.eos_token,
            ]:
                heatmap.append(TokenScore(token=token, score=float(score.item())))

        return heatmap

    def _extract_attention_heatmap(self, text: str) -> list[TokenScore]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_attentions=True)
        attentions = outputs.attentions

        if not attentions:
            return []

        last_layer_attention = attentions[-1]
        avg_attention = last_layer_attention.mean(dim=0)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        attention_scores = avg_attention.sum(dim=1)
        attention_scores = attention_scores / attention_scores.max()

        heatmap = []
        for token, score in zip(tokens, attention_scores, strict=False):
            if token not in [
                self.tokenizer.pad_token,
                self.tokenizer.bos_token,
                self.tokenizer.eos_token,
            ]:
                heatmap.append(TokenScore(token=token, score=float(score.item())))

        return heatmap

    def detect(self, text: str) -> DetectorResult:
        result = super().detect(text)

        if self.use_captum:
            heatmap = self._extract_integrated_gradients_heatmap(text)
        else:
            heatmap = self._extract_attention_heatmap(text)

        result.metadata["heatmap"] = heatmap
        result.metadata["heatmap_method"] = (
            "integrated_gradients" if self.use_captum else "attention"
        )
        return result


def register(registry) -> None:
    if AutoModelForSequenceClassification is not None:
        registry.register(HuggingFaceClassifierDetector)
        registry.register(OpenAIDetector)
        registry.register(ChatGPTDetector)
        registry.register(RADARDetector)
        registry.register(AttentionHeatmapDetector)
