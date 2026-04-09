"""Dataset loading utilities for benchmark workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from provenance.benchmarks.models import DatasetConfig

try:
    from datasets import load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    load_dataset = None


class HuggingFaceDatasetLoader:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or os.environ.get(
            "HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"
        )
        self._cache: dict[str, Any] = {}

    def load(
        self,
        config: DatasetConfig,
        sample_limit: int | None = None,
        seed: int = 42,
        force_refresh: bool = False,
    ) -> tuple[list[str], list[int], list[dict]]:
        cache_key = (
            f"{config.repo_id}_{config.config_name or 'default'}_{sample_limit}_{seed}"
        )
        cache_path = Path(self.cache_dir).expanduser() / f"{cache_key}.json"

        if not force_refresh and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return cached["texts"], cached["labels"], cached["metadata"]

        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library is required for HuggingFaceDatasetLoader"
            )

        ds = self._load_streaming_dataset(config)

        texts, labels, metadata = [], [], []
        take_count = (sample_limit * 20) if sample_limit else 10000

        for item in ds.take(take_count):
            text = item.get(config.text_field, "")
            label_raw = item.get(config.label_field, "")

            if not text or label_raw == "":
                continue

            labels.append(self._coerce_label(label_raw, config))
            texts.append(text)
            metadata.append(self._extract_metadata(item, config))

        if not texts:
            return [], [], []

        texts, labels, metadata = self._sample_records(
            texts, labels, metadata, sample_limit, seed
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "texts": texts,
                    "labels": labels,
                    "metadata": metadata,
                }
            )
        )

        self._cache[cache_key] = (texts, labels, metadata)
        return texts, labels, metadata

    def _load_streaming_dataset(self, config: DatasetConfig):
        try:
            return load_dataset(
                config.repo_id,
                name=config.config_name,
                split=config.split,
                streaming=True,
            )
        except Exception as first_error:
            try:
                return load_dataset(config.repo_id, split=config.split, streaming=True)
            except Exception as second_error:
                try:
                    return load_dataset(
                        config.repo_id,
                        config.config_name,
                        split=config.split,
                        streaming=True,
                    )
                except Exception as third_error:
                    raise RuntimeError(
                        f"Failed to load dataset {config.repo_id}: "
                        f"{first_error} -> {second_error}"
                    ) from third_error

    def _coerce_label(self, label_raw: Any, config: DatasetConfig) -> int:
        if isinstance(label_raw, str):
            if label_raw in config.label_map:
                return int(config.label_map[label_raw])
            if config.label_map:
                return 1
            return int(label_raw)
        return int(label_raw)

    def _extract_metadata(self, item: dict, config: DatasetConfig) -> dict:
        meta = {}
        for meta_key, meta_field in config.meta_fields.items():
            if meta_field in item:
                meta[meta_key] = item[meta_field]
        return meta

    def _sample_records(
        self,
        texts: list[str],
        labels: list[int],
        metadata: list[dict],
        sample_limit: int | None,
        seed: int,
    ) -> tuple[list[str], list[int], list[dict]]:
        np.random.seed(seed)

        if sample_limit and sample_limit < len(texts):
            indices = np.random.choice(len(texts), sample_limit, replace=False)
        elif sample_limit is None and len(texts) > 10000:
            indices = np.random.choice(len(texts), 10000, replace=False)
        else:
            indices = np.random.permutation(len(texts))

        return (
            [texts[i] for i in indices],
            [labels[i] for i in indices],
            [metadata[i] for i in indices],
        )
