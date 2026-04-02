"""Tests for model download spec consistency and path derivation pure functions."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from state import RuntimeConfig
from runtime_config.model_download_specs import (
    DEFAULT_MODEL_DOWNLOAD_SPECS,
    DEFAULT_REQUIRED_MODEL_TYPES,
    MODEL_FILE_ORDER,
    ModelFileDownloadSpec,
    resolve_downloading_dir,
    resolve_downloading_path,
    resolve_downloading_target_path,
    resolve_model_path,
    resolve_required_model_types,
)
from state.app_state_types import ModelFileType


def _build_config(tmp_path):
    models_dir = tmp_path / "models"
    return RuntimeConfig(
        device="cpu",
        default_models_dir=models_dir,
        model_download_specs=DEFAULT_MODEL_DOWNLOAD_SPECS,
        required_model_types=DEFAULT_REQUIRED_MODEL_TYPES,
        outputs_dir=tmp_path / "outputs",
        settings_file=tmp_path / "settings.json",
        ltx_api_base_url="https://api.ltx.video",
        force_api_generations=False,
        use_sage_attention=False,
        camera_motion_prompts={},
        default_negative_prompt="",
        dev_mode=False,
    )


def test_specs_cover_all_model_types():
    expected_types = set(get_args(ModelFileType))
    assert set(DEFAULT_MODEL_DOWNLOAD_SPECS.keys()) == expected_types
    assert set(MODEL_FILE_ORDER) == expected_types


def test_model_path_resolves_from_relative_path(tmp_path):
    config = _build_config(tmp_path)
    specs = config.model_download_specs
    models_dir = config.default_models_dir
    spec = config.spec_for("text_encoder")
    assert resolve_model_path(models_dir, specs, "text_encoder") == models_dir / spec.relative_path


def test_downloading_path_is_derived_from_specs(tmp_path):
    config = _build_config(tmp_path)
    specs = config.model_download_specs
    models_dir = config.default_models_dir
    downloading_dir = resolve_downloading_dir(models_dir)

    assert resolve_downloading_path(models_dir, specs, "checkpoint") == downloading_dir
    assert resolve_downloading_path(models_dir, specs, "ic_lora") == downloading_dir
    assert resolve_downloading_path(models_dir, specs, "depth_processor") == downloading_dir / "dpt-hybrid-midas"
    assert resolve_downloading_path(models_dir, specs, "person_detector") == downloading_dir
    assert resolve_downloading_path(models_dir, specs, "pose_processor") == downloading_dir
    assert resolve_downloading_path(models_dir, specs, "zit") == downloading_dir / "Z-Image-Turbo"
    assert resolve_downloading_path(models_dir, specs, "text_encoder") == downloading_dir / "gemma-3-12b-it-qat-q4_0-unquantized"


def test_downloading_path_supports_nested_relative_parents(tmp_path):
    config = _build_config(tmp_path)
    custom_specs = dict(config.model_download_specs)
    custom_specs["ic_lora"] = ModelFileDownloadSpec(
        relative_path=Path("nested/ic-loras/union/model.safetensors"),
        expected_size_bytes=custom_specs["ic_lora"].expected_size_bytes,
        is_folder=False,
        repo_id=custom_specs["ic_lora"].repo_id,
        description=custom_specs["ic_lora"].description,
    )
    models_dir = config.default_models_dir
    downloading_dir = resolve_downloading_dir(models_dir)

    assert resolve_downloading_path(models_dir, custom_specs, "ic_lora") == downloading_dir / "nested" / "ic-loras" / "union"
    assert resolve_downloading_target_path(models_dir, custom_specs, "ic_lora") == downloading_dir / "nested" / "ic-loras" / "union" / "model.safetensors"


def test_model_paths_reject_parent_traversal(tmp_path):
    config = _build_config(tmp_path)
    custom_specs = dict(config.model_download_specs)
    custom_specs["ic_lora"] = ModelFileDownloadSpec(
        relative_path=Path("../escape.safetensors"),
        expected_size_bytes=custom_specs["ic_lora"].expected_size_bytes,
        is_folder=False,
        repo_id=custom_specs["ic_lora"].repo_id,
        description=custom_specs["ic_lora"].description,
    )

    with pytest.raises(ValueError):
        resolve_model_path(config.default_models_dir, custom_specs, "ic_lora")


def test_required_model_types_remain_dynamic_for_text_encoder():
    required_with_api = resolve_required_model_types(DEFAULT_REQUIRED_MODEL_TYPES, has_api_key=True)
    required_without_api = resolve_required_model_types(DEFAULT_REQUIRED_MODEL_TYPES, has_api_key=False)

    assert "text_encoder" not in required_with_api
    assert "text_encoder" in required_without_api


def test_required_model_types_empty_base_stays_empty():
    required = resolve_required_model_types(
        frozenset(),
        has_api_key=False,
    )
    assert required == frozenset()
