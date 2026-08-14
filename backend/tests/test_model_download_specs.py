"""Tests for checkpoint specs and pure path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from api_types import ModelCheckpointID
from runtime_config.model_download_specs import (
    ALL_MODEL_CP_IDS,
    ALL_LTX_LOCAL_MODEL_IDS,
    LTX_2_5_FAMILY_DIR,
    ModelCheckpointSpec,
    delete_cp_path,
    get_existing_cp_path,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_cps,
    get_ltx_model_cp_ids,
    get_ltx_model_spec,
    get_local_prompt_enhancer_cp,
    get_model_cp_spec,
    is_cp_downloaded,
    selected_video_vae_cp,
    unused_video_vae_cp,
    is_duration_head_ready,
    local_prompt_enhancer_candidates,
    resolve_downloaded_prompt_enhancer_cp,
    resolve_downloading_dir,
    resolve_downloading_path,
    resolve_downloading_target_path,
    resolve_model_path,
)

_LTX_2_5_NATIVE_CPS: tuple[ModelCheckpointID, ...] = (
    "ltx-2.5-22b-distilled",
    "ltx-2.5-spatial-upscaler-x2-1.0",
    "ltx-2.5-video-vae",
    "ltx-2.5-video-vae-conv",
    "ltx-2.5-audio-vae",
    "ltx-2.5-duration-head",
    "gemma4-12b-with-proj-ltx-2.5",
)


def test_specs_cover_all_checkpoint_ids():
    assert set(ALL_MODEL_CP_IDS) == {cp_id for cp_id in ALL_MODEL_CP_IDS}


def test_primary_ltx_checkpoints_map_1_to_1_with_ltx_models():
    assert len(get_ltx_cps()) == len(ALL_LTX_LOCAL_MODEL_IDS)


def test_latest_ltx_model_is_relevant():
    latest = get_latest_ltx_model_id()
    assert latest == "ltx-2.5-22b-distilled"
    spec = get_ltx_model_spec(latest)
    assert spec.model_cp in get_ltx_cps()
    assert spec.video_vae_cp is not None
    assert spec.audio_vae_cp is not None
    assert spec.ic_loras_spec is None


def test_ic_lora_cp_ids_are_deduped_for_2_3():
    spec = get_ltx_model_spec("ltx-2.3-22b-distilled-1.1")
    assert get_ic_loras_cp_ids(spec.ic_loras_spec) == ("ltx-2.3-22b-ic-lora-union-control-ref0.5",)


def test_ltx_2_5_model_cp_ids_include_split_vaes():
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    assert get_ltx_model_cp_ids("ltx-2.5-22b-distilled") == (
        spec.model_cp,
        spec.upscale_cp,
        spec.text_encoder_cp,
        spec.video_vae_cp,
        spec.video_vae_conv_cp,
        spec.audio_vae_cp,
        spec.duration_head_cp,
    )
    assert spec.video_vae_conv_cp == "ltx-2.5-video-vae-conv"
    assert spec.duration_head_cp == "ltx-2.5-duration-head"


def test_2_3_has_no_split_video_vaes():
    spec = get_ltx_model_spec("ltx-2.3-22b-distilled-1.1")
    assert spec.video_vae_cp is None
    assert spec.video_vae_conv_cp is None
    assert selected_video_vae_cp(spec, use_conv_vae=True) is None
    assert selected_video_vae_cp(spec, use_conv_vae=False) is None


def test_selected_video_vae_cp_follows_toggle():
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    assert selected_video_vae_cp(spec, use_conv_vae=True) == spec.video_vae_conv_cp
    assert selected_video_vae_cp(spec, use_conv_vae=False) == spec.video_vae_cp
    assert unused_video_vae_cp(spec, use_conv_vae=True) == spec.video_vae_cp
    assert unused_video_vae_cp(spec, use_conv_vae=False) == spec.video_vae_conv_cp


def test_2_3_has_no_duration_head_cp():
    assert get_ltx_model_spec("ltx-2.3-22b-distilled-1.1").duration_head_cp is None


def test_duration_head_ready_requires_the_split_file(tmp_path: Path):
    model_id = "ltx-2.5-22b-distilled"
    assert is_duration_head_ready(tmp_path, model_id) is False
    spec = get_ltx_model_spec(model_id)
    assert spec.duration_head_cp is not None
    path = resolve_model_path(tmp_path, spec.duration_head_cp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 1024)
    assert is_duration_head_ready(tmp_path, model_id) is True
    assert is_duration_head_ready(tmp_path, "ltx-2.3-22b-distilled-1.1") is False


def test_duration_head_download_filename_is_nested():
    spec = get_model_cp_spec("ltx-2.5-duration-head")
    assert spec.download_filename == "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    assert spec.repo_id == "Lightricks/LTX-2.5"


def test_ltx_2_5_download_filenames_are_nested():
    transformer = get_model_cp_spec("ltx-2.5-22b-distilled")
    assert transformer.download_filename.startswith("diffusion_models/")
    assert transformer.repo_id == "Lightricks/LTX-2.5"


def test_ltx_2_5_native_weights_live_under_family_dir():
    for cp_id in _LTX_2_5_NATIVE_CPS:
        relative = get_model_cp_spec(cp_id).relative_path
        assert relative.parts[0] == LTX_2_5_FAMILY_DIR.name, cp_id
        assert relative.parent == LTX_2_5_FAMILY_DIR


def test_shared_and_2_3_checkpoints_stay_at_models_root():
    for cp_id in ALL_MODEL_CP_IDS:
        if cp_id in _LTX_2_5_NATIVE_CPS:
            continue
        assert get_model_cp_spec(cp_id).relative_path.parts[0] != LTX_2_5_FAMILY_DIR.name, cp_id


def test_model_path_resolves_from_relative_path(tmp_path):
    cp_id: ModelCheckpointID = "gemma-3-12b-it-qat-q4_0-unquantized"
    spec = get_model_cp_spec(cp_id)
    assert resolve_model_path(tmp_path, cp_id) == tmp_path / spec.relative_path


def test_downloading_path_is_derived_from_spec():
    models_dir = Path("/tmp/models")
    downloading_dir = resolve_downloading_dir(models_dir)

    assert resolve_downloading_path(models_dir, "ltx-2.3-22b-distilled") == downloading_dir
    assert (
        resolve_downloading_path(models_dir, "gemma-3-12b-it-qat-q4_0-unquantized")
        == downloading_dir / "gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert resolve_downloading_target_path(models_dir, "ltx-2.3-22b-distilled") == downloading_dir / "ltx-2.3-22b-distilled.safetensors"
    assert resolve_downloading_target_path(models_dir, "ltx-2.5-22b-distilled") == (
        downloading_dir / LTX_2_5_FAMILY_DIR / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    )


def test_2_5_write_path_is_family_dir(tmp_path):
    path = resolve_model_path(tmp_path, "ltx-2.5-22b-distilled")
    assert path.parent == tmp_path / LTX_2_5_FAMILY_DIR
    assert path.name == "ltx-2.5-22b-distilled-transformer-bf16.safetensors"


def test_2_5_reads_legacy_flat_file_at_models_root(tmp_path):
    spec = get_model_cp_spec("ltx-2.5-video-vae")
    leftover = tmp_path / spec.relative_path.name
    leftover.write_bytes(b"legacy")
    assert is_cp_downloaded(tmp_path, "ltx-2.5-video-vae") is True
    assert get_existing_cp_path(tmp_path, "ltx-2.5-video-vae") == leftover


def test_2_5_prefers_family_dir_over_legacy_root(tmp_path):
    spec = get_model_cp_spec("ltx-2.5-video-vae")
    leftover = tmp_path / spec.relative_path.name
    leftover.write_bytes(b"legacy")
    canonical = resolve_model_path(tmp_path, "ltx-2.5-video-vae")
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    assert get_existing_cp_path(tmp_path, "ltx-2.5-video-vae") == canonical


def test_2_5_delete_removes_family_dir_and_legacy_root(tmp_path):
    spec = get_model_cp_spec("ltx-2.5-audio-vae")
    leftover = tmp_path / spec.relative_path.name
    leftover.write_bytes(b"legacy")
    canonical = resolve_model_path(tmp_path, "ltx-2.5-audio-vae")
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    delete_cp_path(tmp_path, "ltx-2.5-audio-vae")
    assert not leftover.exists()
    assert not canonical.exists()
    assert is_cp_downloaded(tmp_path, "ltx-2.5-audio-vae") is False


def test_2_3_has_no_root_fallback(tmp_path):
    assert is_cp_downloaded(tmp_path, "ltx-2.3-22b-distilled") is False
    with pytest.raises(FileNotFoundError):
        get_existing_cp_path(tmp_path, "ltx-2.3-22b-distilled")


def test_relative_paths_are_unique():
    relative_paths = {get_model_cp_spec(cp_id).relative_path for cp_id in ALL_MODEL_CP_IDS}
    assert len(relative_paths) == len(ALL_MODEL_CP_IDS)


def test_model_path_rejects_parent_traversal(monkeypatch, tmp_path):
    bad_spec = ModelCheckpointSpec(
        relative_path=Path("../escape.safetensors"),
        expected_size_bytes=1,
        is_folder=False,
        repo_id="x/y",
        description="bad",
    )
    monkeypatch.setattr(
        "runtime_config.model_download_specs.get_model_cp_spec",
        lambda _cp_id: bad_spec,
    )
    with pytest.raises(ValueError, match="cannot traverse parents"):
        resolve_model_path(tmp_path, "ltx-2.3-22b-distilled")


def _write_folder_cp(models_dir: Path, cp_id: ModelCheckpointID) -> None:
    path = resolve_model_path(models_dir, cp_id)
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(b"x")


def test_2_5_enhancer_candidates_prefer_e2b_then_gemma3():
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    assert get_local_prompt_enhancer_cp(spec) == "gemma-4-e2b-it"
    assert local_prompt_enhancer_candidates(spec) == (
        "gemma-4-e2b-it",
        "gemma-3-12b-it-qat-q4_0-unquantized",
    )


def test_2_3_enhancer_is_the_encoder_only():
    spec = get_ltx_model_spec("ltx-2.3-22b-distilled-1.1")
    assert local_prompt_enhancer_candidates(spec) == (spec.text_encoder_cp,)
    assert "gemma-4-e2b-it" not in local_prompt_enhancer_candidates(spec)


def test_2_5_resolves_gemma3_when_e2b_is_missing(tmp_path: Path):
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    assert resolve_downloaded_prompt_enhancer_cp(tmp_path, spec) is None
    _write_folder_cp(tmp_path, "gemma-3-12b-it-qat-q4_0-unquantized")
    assert resolve_downloaded_prompt_enhancer_cp(tmp_path, spec) == "gemma-3-12b-it-qat-q4_0-unquantized"


def test_2_5_prefers_e2b_over_gemma3(tmp_path: Path):
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    _write_folder_cp(tmp_path, "gemma-3-12b-it-qat-q4_0-unquantized")
    _write_folder_cp(tmp_path, "gemma-4-e2b-it")
    assert resolve_downloaded_prompt_enhancer_cp(tmp_path, spec) == "gemma-4-e2b-it"
