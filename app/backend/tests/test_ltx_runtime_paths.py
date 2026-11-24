"""Split 2.5 DurationHead path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime_config.ltx_runtime_paths import resolve_ltx_runtime_paths
from runtime_config.model_download_specs import get_ltx_model_spec, resolve_model_path
from services.ltx_pipeline_common import build_model_paths


def _write_cp(models_dir: Path, cp_id: str) -> Path:
    path = resolve_model_path(models_dir, cp_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _write_2_5_bundle(
    models_dir: Path,
    *,
    include_diff_vae: bool = True,
    include_conv_vae: bool = True,
    include_duration_head: bool = True,
) -> None:
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    cps = [spec.model_cp, spec.upscale_cp, spec.audio_vae_cp]
    if include_diff_vae:
        cps.append(spec.video_vae_cp)
    if include_conv_vae:
        cps.append(spec.video_vae_conv_cp)
    if include_duration_head:
        cps.append(spec.duration_head_cp)
    for cp_id in cps:
        if cp_id is not None:
            _write_cp(models_dir, cp_id)


def test_split_model_paths_pass_duration_head() -> None:
    paths = build_model_paths(
        "transformer.safetensors",
        "gemma",
        video_vae_path="video.safetensors",
        audio_vae_path="audio.safetensors",
        duration_head_path="duration.safetensors",
    )
    assert paths.mode == "split"
    assert paths.duration_head_path == "duration.safetensors"


def test_split_model_paths_omit_duration_head_when_missing() -> None:
    paths = build_model_paths(
        "transformer.safetensors",
        "gemma",
        video_vae_path="video.safetensors",
        audio_vae_path="audio.safetensors",
    )
    assert paths.duration_head_path is None


def test_monolith_duration_head_is_the_fat_checkpoint() -> None:
    paths = build_model_paths("monolith.safetensors", "gemma")
    assert paths.mode == "monolith"
    assert paths.duration_head_path == "monolith.safetensors"


def test_resolve_runtime_paths_includes_downloaded_duration_head(tmp_path: Path) -> None:
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    _write_2_5_bundle(tmp_path)
    paths = resolve_ltx_runtime_paths(
        tmp_path, "ltx-2.5-22b-distilled", gemma_root=None, use_conv_vae=False
    )
    assert paths.duration_head_path == str(resolve_model_path(tmp_path, spec.duration_head_cp))


def test_resolve_runtime_paths_omits_missing_duration_head(tmp_path: Path) -> None:
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    _write_2_5_bundle(tmp_path, include_duration_head=False)
    paths = resolve_ltx_runtime_paths(
        tmp_path, "ltx-2.5-22b-distilled", gemma_root=None, use_conv_vae=False
    )
    assert paths.duration_head_path is None
    assert paths.video_vae_path == str(resolve_model_path(tmp_path, spec.video_vae_cp))


def test_resolve_runtime_paths_picks_conv_vae(tmp_path: Path) -> None:
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    _write_2_5_bundle(tmp_path)
    paths = resolve_ltx_runtime_paths(
        tmp_path, "ltx-2.5-22b-distilled", gemma_root=None, use_conv_vae=True
    )
    assert paths.video_vae_path == str(resolve_model_path(tmp_path, spec.video_vae_conv_cp))


def test_resolve_runtime_paths_picks_diffvae(tmp_path: Path) -> None:
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled")
    _write_2_5_bundle(tmp_path)
    paths = resolve_ltx_runtime_paths(
        tmp_path, "ltx-2.5-22b-distilled", gemma_root=None, use_conv_vae=False
    )
    assert paths.video_vae_path == str(resolve_model_path(tmp_path, spec.video_vae_cp))


def test_resolve_runtime_paths_2_3_ignores_conv_toggle(tmp_path: Path) -> None:
    spec = get_ltx_model_spec("ltx-2.3-22b-distilled-1.1")
    for cp_id in (spec.model_cp, spec.upscale_cp):
        _write_cp(tmp_path, cp_id)
    for use_conv_vae in (True, False):
        paths = resolve_ltx_runtime_paths(
            tmp_path, "ltx-2.3-22b-distilled-1.1", gemma_root=None, use_conv_vae=use_conv_vae
        )
        assert paths.video_vae_path is None


def test_resolve_runtime_paths_does_not_fall_back_to_other_vae(tmp_path: Path) -> None:
    _write_2_5_bundle(tmp_path, include_conv_vae=False)
    with pytest.raises(FileNotFoundError, match="ltx-2.5-video-vae-conv"):
        resolve_ltx_runtime_paths(
            tmp_path, "ltx-2.5-22b-distilled", gemma_root=None, use_conv_vae=True
        )
