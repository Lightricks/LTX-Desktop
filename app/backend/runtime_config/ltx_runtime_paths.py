"""Resolve LTX model component paths for pipeline construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from api_types import LTXLocalModelId
from runtime_config.model_download_specs import (
    get_existing_cp_path,
    get_ltx_model_spec,
    is_cp_downloaded,
    selected_video_vae_cp,
)


@dataclass(frozen=True, slots=True)
class ResolvedLtxModelPaths:
    """Filesystem paths for the active local LTX model bundle."""

    checkpoint_path: str
    upsampler_path: str
    gemma_root: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    duration_head_path: str | None


def resolve_ltx_runtime_paths(
    models_dir: Path,
    model_id: LTXLocalModelId,
    *,
    gemma_root: str | None,
    use_conv_vae: bool,
) -> ResolvedLtxModelPaths:
    """Resolve transformer/upscaler/(optional)VAE paths for ``model_id``.

    ``gemma_root`` is passed in from the text handler (may be None in API-encode mode).
    Split VAEs are required for 2.5 and omitted for monolith 2.3. DurationHead is
    a small split-pack file on 2.5; missing it still lets explicit-duration
    generation run (AutoDuration then fails in the pipeline).

    ``use_conv_vae`` selects the 2.5 conv VAE (fast decode) vs DiffVAE. The decoder
    class is chosen by vendored ltx-core from checkpoint metadata — Desktop only
    swaps the path. Missing the selected file is an error; do not fall back.
    """
    spec = get_ltx_model_spec(model_id)
    video_vae_path: str | None = None
    audio_vae_path: str | None = None
    duration_head_path: str | None = None
    video_vae_cp = selected_video_vae_cp(spec, use_conv_vae=use_conv_vae)
    if video_vae_cp is not None:
        video_vae_path = str(get_existing_cp_path(models_dir, video_vae_cp))
    if spec.audio_vae_cp is not None:
        audio_vae_path = str(get_existing_cp_path(models_dir, spec.audio_vae_cp))
    if spec.duration_head_cp is not None and is_cp_downloaded(models_dir, spec.duration_head_cp):
        duration_head_path = str(get_existing_cp_path(models_dir, spec.duration_head_cp))
    return ResolvedLtxModelPaths(
        checkpoint_path=str(get_existing_cp_path(models_dir, spec.model_cp)),
        upsampler_path=str(get_existing_cp_path(models_dir, spec.upscale_cp)),
        gemma_root=gemma_root,
        video_vae_path=video_vae_path,
        audio_vae_path=audio_vae_path,
        duration_head_path=duration_head_path,
    )
