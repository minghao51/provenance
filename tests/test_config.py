"""Tests for JSON/YAML provenance config loading."""

from pathlib import Path

import pytest

from provenance.core.config import ProvenanceConfig, load_provenance_config, resolve_provenance_config


def test_load_json_config(tmp_path: Path):
    config_path = tmp_path / "provenance.json"
    config_path.write_text(
        """
        {
          "provenance": {
            "min_text_length": 42,
            "short_text_confidence": 0.15,
            "calibration_model_dir": "models"
          }
        }
        """.strip()
    )

    config = load_provenance_config(config_path)

    assert config.min_text_length == 42
    assert config.short_text_confidence == 0.15
    assert config.calibration_model_dir == "models"


def test_load_yaml_config(tmp_path: Path):
    pytest.importorskip("yaml")

    config_path = tmp_path / "provenance.yaml"
    config_path.write_text(
        """
        provenance:
          min_text_length: 64
          max_top_features: 8
        """.strip()
    )

    config = load_provenance_config(config_path)

    assert config.min_text_length == 64
    assert config.max_top_features == 8


def test_load_yaml_config_with_detector_calibration_paths(tmp_path: Path):
    pytest.importorskip("yaml")

    config_path = tmp_path / "provenance.yaml"
    config_path.write_text(
        """
        provenance:
          detector_calibration_paths:
            repetition: calibration_models/repetition_raid.pkl
        """.strip()
    )

    config = load_provenance_config(config_path)

    assert config.detector_calibration_paths == {
        "repetition": "calibration_models/repetition_raid.pkl"
    }


def test_resolve_existing_config_with_overrides():
    config = resolve_provenance_config(
        ProvenanceConfig(min_text_length=150),
        overrides={"min_text_length": 12},
    )

    assert config.min_text_length == 12
