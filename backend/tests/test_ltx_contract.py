"""Upstream LTX-2 contracts that desktop's forked pipelines and patches rely on.

Each assertion names something desktop calls, overrides or monkey-patches. When
an LTX bump breaks one, this file fails first with a message that points at the
exact contract, instead of a shape error deep inside a fork.
Re-check this file (and `.cursor/rules/ltx2-patch-reverify.mdc`) on every bump.
"""

from __future__ import annotations

import dataclasses
import inspect

import ltx_pipelines.chunks.conditionings as chunk_conditionings
import ltx_pipelines.utils.helpers as helpers
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline, ICLoraStageConfig
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from ltx_pipelines.utils.types import ModalitySpec, PipelineOutput, VideoAudio


def _params(fn: object) -> list[str]:
    return list(inspect.signature(fn).parameters)  # type: ignore[arg-type]


def test_diffusion_stage_call_takes_modalities_and_an_optional_loop() -> None:
    assert _params(DiffusionStage.__call__) == [
        "self",
        "denoiser",
        "sigmas",
        "noiser",
        "modalities",
        "loop",
        "max_batch_size",
    ]


def test_diffusion_stage_exposes_transformer_scale_factors() -> None:
    """Latents must be sized with these, not with the decoder's tiling scale."""
    assert "self.video_scale_factors" in inspect.getsource(DiffusionStage.__init__)


def test_modality_spec_fields() -> None:
    fields = {field.name for field in dataclasses.fields(ModalitySpec)}
    assert {"latent", "conditioning_fps", "context", "conditionings", "noise_scale", "frozen"} <= fields
    assert "initial_latent" not in fields


def test_pipeline_output_is_a_four_field_tuple() -> None:
    assert PipelineOutput._fields == ("video", "audio", "num_frames", "tiling_config")


def test_video_audio_unpacks_as_video_then_audio() -> None:
    assert tuple(VideoAudio(video=1, audio=2)) == (1, 2)


def test_ancestral_loop_parameters_used_by_the_sampler_partials() -> None:
    assert {"noise_seed", "model_dtype", "eta", "s_noise"} <= set(_params(euler_ancestral_denoising_loop))


def test_distilled_pipeline_sampler_kwargs_hook() -> None:
    assert _params(DistilledPipeline._sampler_kwargs)[:3] == ["self", "seed", "noise_seed_offset"]  # type: ignore[attr-defined]


def test_initial_latent_helpers_take_explicit_scale_factors() -> None:
    assert "scale_factors" in _params(helpers.create_initial_video_latent)
    assert "video_scale_factors" in _params(helpers.create_initial_av_latents)
    assert "video_scale_factors" in _params(helpers.create_initial_audio_latent)


def test_keyframe_swap_targets_exist_in_both_modules() -> None:
    """`distilled_keyframe_guiding` rebinds this name in both modules."""
    for module in (helpers, chunk_conditionings):
        assert callable(module.combined_image_conditionings)
    assert callable(helpers.image_conditionings_by_adding_guiding_latent)


def test_chunk_keyframes_resolve_the_swappable_name_from_the_chunks_module() -> None:
    """The per-chunk factory looks the name up in its module globals at call time.

    If upstream stops doing that (e.g. binds it as a default argument), swapping the
    module attribute silently stops changing multi-keyframe behaviour.
    """
    from services.fast_video_pipeline.distilled_keyframe_guiding import distilled_keyframe_guiding

    factory_globals = chunk_conditionings.image_conditionings_for_chunk.__globals__
    assert factory_globals is vars(chunk_conditionings)

    with distilled_keyframe_guiding():
        assert factory_globals["combined_image_conditionings"] is helpers.image_conditionings_by_adding_guiding_latent

    assert factory_globals["combined_image_conditionings"] is not helpers.image_conditionings_by_adding_guiding_latent


def test_ic_lora_override_point_and_stage_config() -> None:
    assert _params(ICLoraPipeline._video_conditionings) == ["self", "ctx", "stage_config", "frame_sources"]  # type: ignore[attr-defined]
    assert "apply_ic_lora" in {field.name for field in dataclasses.fields(ICLoraStageConfig)}
    assert {"encode_tiling", "downscale_factor", "reference_temporal_scale_factor"} <= set(
        _params(chunk_conditionings.reference_video_conditionings_for_chunk)
    )
