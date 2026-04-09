"""Dataset registry and default benchmark dataset definitions."""

from __future__ import annotations

from provenance.benchmarks.models import DatasetConfig


class DatasetRegistry:
    REGISTRY: dict[str, DatasetConfig] = {}

    @classmethod
    def register(cls, config: DatasetConfig):
        cls.REGISTRY[config.name] = config

    @classmethod
    def get(cls, name: str) -> DatasetConfig | None:
        return cls.REGISTRY.get(name)

    @classmethod
    def list_datasets(cls) -> list[str]:
        return list(cls.REGISTRY.keys())

    @classmethod
    def available_datasets(cls) -> dict[str, DatasetConfig]:
        return cls.REGISTRY.copy()


DEFAULT_DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig(
        name="raid",
        repo_id="liamdugan/raid",
        split="train",
        text_field="generation",
        label_field="model",
        label_map={"human": 0},
        meta_fields={"domain": "domain", "source": "source_id"},
    ),
    DatasetConfig(
        name="mage",
        repo_id="yaful/MAGE",
        config_name="default",
        split="test",
        text_field="text",
        label_field="label",
        label_map={"human": 0, "ai": 1, "mixed": 2},
        meta_fields={"edit_type": "edit_type"},
    ),
    DatasetConfig(
        name="hc3",
        repo_id="Hello-SimpleAI/HC3",
        config_name="all",
        split="validation",
        text_field="text",
        label_field="label",
        label_map={"human": 0, "ChatGPT": 1},
    ),
    DatasetConfig(
        name="m4",
        repo_id="NickyNicky/M4",
        split="train",
        text_field="text",
        label_field="label",
        label_map={"human": 0, "ai": 1},
    ),
)

_DEFAULTS_REGISTERED = False


def register_default_datasets(force: bool = False) -> None:
    """Populate the registry with built-in datasets exactly once by default."""
    global _DEFAULTS_REGISTERED

    if _DEFAULTS_REGISTERED and not force:
        return

    for config in DEFAULT_DATASETS:
        DatasetRegistry.register(config)

    _DEFAULTS_REGISTERED = True
