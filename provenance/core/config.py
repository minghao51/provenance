"""Configuration dataclasses for provenance detection thresholds and parameters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ProvenanceConfig:
    """Top-level configuration for the Provenance detector."""

    min_text_length: int = 150
    max_heatmap_tokens: int = 200
    max_top_features: int = 20
    short_text_confidence: float = 0.3
    short_text_warning_template: str = (
        "Text is shorter than recommended minimum ({min_length} words). "
        "Results may be unreliable."
    )
    calibration_model_dir: str | None = None
    detector_calibration_paths: dict[str, str] = field(default_factory=dict)


T = TypeVar("T")


def _unwrap_dataclass_type(type_hint: Any) -> type[Any] | None:
    if is_dataclass(type_hint):
        return type_hint

    origin = get_origin(type_hint)
    if origin is None:
        return None

    for arg in get_args(type_hint):
        if arg is type(None):
            continue
        if is_dataclass(arg):
            return arg
    return None


def _coerce_value(type_hint: Any, value: Any) -> Any:
    nested_type = _unwrap_dataclass_type(type_hint)
    if nested_type is not None and isinstance(value, dict):
        return _build_dataclass(nested_type, value)
    return value


def _build_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    field_map = {item.name: item for item in fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_map:
            continue
        kwargs[key] = _coerce_value(field_map[key].type, value)
    return cls(**kwargs)


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_data(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config files")
        data = yaml.safe_load(raw_text)
    else:
        raise ValueError(
            f"Unsupported config file format: {config_path.suffix or '<none>'}"
        )

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON/YAML object")

    return data


def load_provenance_config(
    source: str | Path | dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> ProvenanceConfig:
    if isinstance(source, (str, Path)):
        data = load_config_data(source)
    else:
        data = dict(source)

    if "provenance" in data and isinstance(data["provenance"], dict):
        data = data["provenance"]

    if overrides:
        data = _merge_dicts(data, overrides)

    return _build_dataclass(ProvenanceConfig, data)


def resolve_provenance_config(
    config: ProvenanceConfig | str | Path | dict[str, Any] | None,
    overrides: dict[str, Any] | None = None,
) -> ProvenanceConfig:
    if config is None:
        base = ProvenanceConfig()
        if overrides:
            return load_provenance_config(asdict(base), overrides=overrides)
        return base

    if isinstance(config, ProvenanceConfig):
        if not overrides:
            return config
        return load_provenance_config(asdict(config), overrides=overrides)

    return load_provenance_config(config, overrides=overrides)


@dataclass
class EntropyThresholds:
    """Thresholds for the entropy-based detector."""

    kl_div_high: float = 2.0
    kl_div_high_score: float = 0.8
    kl_div_high_confidence: float = 0.7
    kl_div_medium: float = 1.0
    kl_div_medium_score: float = 0.6
    kl_div_medium_confidence: float = 0.6
    kl_div_low: float = 0.5
    kl_div_low_score: float = 0.4
    kl_div_low_confidence: float = 0.5
    kl_div_default_score: float = 0.2
    kl_div_default_confidence: float = 0.6


@dataclass
class BurstinessThresholds:
    """Thresholds for the burstiness detector."""

    min_sentence_length: int = 15
    cv_high: float = 0.2
    cv_high_confidence: float = 0.7
    cv_medium: float = 0.4
    cv_medium_confidence: float = 0.6
    cv_low: float = 0.6
    cv_low_confidence: float = 0.5
    cv_default_confidence: float = 0.4


@dataclass
class RepetitionThresholds:
    """Thresholds for the repetition detector."""

    min_word_count: int = 10
    ngram_threshold_count: int = 2
    max_reported_ngrams: int = 10
    self_bleu_boost_threshold: float = 0.5
    self_bleu_boost_score: float = 0.7
    high_confidence_word_count: int = 50
    high_confidence: float = 0.8
    low_confidence: float = 0.4


@dataclass
class CurvatureThresholds:
    """Thresholds for the curvature detector."""

    default_n_perturbations: int = 10
    default_mask_ratio: float = 0.15
    default_model: str = "EleutherAI/gpt-neo-125M"
    max_token_length: int = 512
    top_k_sampling: int = 50
    curvature_bands: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.05, 0.5, 0.3),
            (0.15, 0.55, 0.5),
            (0.3, 0.65, 0.6),
            (0.5, 0.75, 0.7),
            (float("inf"), 0.85, 0.8),
        ]
    )


@dataclass
class SurprisalThresholds:
    """Thresholds for the surprisal detector."""

    default_model: str = "EleutherAI/gpt-neo-125M"
    default_window_size: int = 512
    min_surprisal_tokens: int = 5
    variance_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(2.0, 0.25), (5.0, 0.15), (10.0, 0.05)]
    )
    autocorr_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.1, 0.2), (0.3, 0.1), (0.5, 0.05)]
    )
    burstiness_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.3, 0.2), (0.5, 0.1), (0.7, 0.05)]
    )
    mean_surprisal_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(3.0, 0.15), (5.0, 0.1), (7.0, 0.05)]
    )
    trend_bands: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.01, 0.1), (0.05, 0.05)]
    )
