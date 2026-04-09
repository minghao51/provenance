"""Shared benchmark data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DatasetConfig:
    name: str
    repo_id: str
    config_name: str | None = None
    split: str = "train"
    text_field: str = "text"
    label_field: str = "label"
    label_map: dict = field(default_factory=lambda: {"human": 0, "ai": 1})
    meta_fields: dict = field(default_factory=dict)
    cache_dir: str | None = None


@dataclass
class BenchmarkResult:
    detector_name: str
    dataset: str
    auroc: float
    f1: float
    fpr_at_10tpr: float
    precision: float
    recall: float
    accuracy: float
    num_samples: int
    metadata: dict
    num_positives: int = 0
    num_negatives: int = 0
    eval_time_seconds: float = 0.0
    stratified_results: dict | None = None
    tpr_at_1fpr: float = 0.0
    tpr_at_5fpr: float = 0.0


@dataclass
class BenchmarkSuite:
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    results: list[BenchmarkResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)
