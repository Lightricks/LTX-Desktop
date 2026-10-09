"""GPU-free checks for Retake/Extend distilled sampler selection.

2.5 distilled T2V samples stage 1 ancestrally and cleans the noise in a short
Euler refine. The masked region starts at sigma=1, so Euler-only never lets
the prompt take (retake reconstructs the source; extend morphs into sludge)
and ancestral-only leaves texture boiling. Both features need the same pair.
``_run()`` cannot be exercised here without loading real checkpoints, so this
pins the sampler plan and the ``DiffusionStage`` call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from ltx_pipelines.distilled import (
    ANCESTRAL_ETA,
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_S_NOISE,
    should_use_ancestral_sampler,
)
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from safetensors.torch import save_file

from services.denoising_progress import distilled_total_steps
from services.retake_pipeline.ltx_retake_pipeline import LTXRetakePipeline
from services.retake_pipeline.retake_sampler import resolve_sampler_plan


def _checkpoint_with_version(tmp_path: Path, version: str) -> str:
    path = tmp_path / f"transformer-{version}.safetensors"
    save_file({"dummy": torch.zeros(1)}, str(path), metadata={"model_version": version})
    return str(path)


def _assert_ancestral_kwargs(kwargs: dict[str, Any], *, seed: int, dtype: torch.dtype) -> None:
    assert "stepper" not in kwargs
    loop = kwargs["loop"]
    assert isinstance(loop, partial)
    assert loop.func is euler_ancestral_denoising_loop
    assert loop.keywords["noise_seed"] == seed + ANCESTRAL_NOISE_SEED_OFFSET
    assert loop.keywords["model_dtype"] is dtype
    assert loop.keywords["eta"] == ANCESTRAL_ETA
    assert loop.keywords["s_noise"] == ANCESTRAL_S_NOISE


def test_distilled_2_5_selects_ancestral_sampler(tmp_path: Path) -> None:
    seed = 7
    dtype = torch.bfloat16
    path = _checkpoint_with_version(tmp_path, "2.5")
    assert should_use_ancestral_sampler(path)
    plan = resolve_sampler_plan(
        distilled=True,
        use_ancestral=should_use_ancestral_sampler(path),
        seed=seed,
        dtype=dtype,
    )
    _assert_ancestral_kwargs(plan.stage1, seed=seed, dtype=dtype)


def test_distilled_2_3_keeps_deterministic_defaults(tmp_path: Path) -> None:
    path = _checkpoint_with_version(tmp_path, "2.3")
    assert not should_use_ancestral_sampler(path)
    plan = resolve_sampler_plan(
        distilled=True,
        use_ancestral=should_use_ancestral_sampler(path),
        seed=1,
        dtype=torch.bfloat16,
    )
    assert plan.stage1 == {}


def test_guided_path_never_selects_ancestral_sampler() -> None:
    plan = resolve_sampler_plan(
        distilled=False,
        use_ancestral=True,
        seed=1,
        dtype=torch.bfloat16,
    )
    assert plan.stage1 == {}


def test_2_5_uses_ancestral_then_euler_refine() -> None:
    plan = resolve_sampler_plan(
        distilled=True,
        use_ancestral=True,
        seed=1,
        dtype=torch.bfloat16,
    )
    assert plan.two_stage is True
    assert plan.total_steps == distilled_total_steps()


def test_2_3_stays_single_stage_euler() -> None:
    plan = resolve_sampler_plan(
        distilled=True,
        use_ancestral=False,
        seed=1,
        dtype=torch.bfloat16,
    )
    assert plan.two_stage is False
    assert plan.total_steps == distilled_total_steps(stage_2=False)


def _pipeline_with_recording_stage(
    *, use_ancestral: bool
) -> tuple[LTXRetakePipeline, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    @dataclass
    class _State:
        latent: str

    def fake_stage(**kwargs: Any) -> tuple[object, object]:
        calls.append(kwargs)
        return _State(f"v{len(calls)}"), _State(f"a{len(calls)}")

    pipeline = cast(LTXRetakePipeline, object.__new__(LTXRetakePipeline))
    pipeline.dtype = torch.bfloat16
    pipeline.device = torch.device("cpu")
    pipeline.use_ancestral_sampler = use_ancestral
    pipeline.stage = fake_stage  # type: ignore[method-assign]
    return pipeline, calls


def test_resolve_sampler_plan_forwards_ancestral_kwargs() -> None:
    seed = 7
    dtype = torch.bfloat16
    plan = resolve_sampler_plan(
        distilled=True,
        use_ancestral=True,
        seed=seed,
        dtype=dtype,
    )
    assert plan.two_stage is True
    assert plan.total_steps == distilled_total_steps()
    _assert_ancestral_kwargs(plan.stage1, seed=seed, dtype=dtype)
    assert plan.refine == {}


def test_retake_denoise_runs_ancestral_then_euler_refine() -> None:
    from ltx_pipelines.utils.types import ModalitySpec

    seed = 7
    pipeline, calls = _pipeline_with_recording_stage(use_ancestral=True)
    video = ModalitySpec(latent="v0", conditioning_fps=24.0)
    audio = ModalitySpec(latent="a0", conditioning_fps=24.0)
    video_state, audio_state = pipeline._denoise_masked_latents(
        distilled=True,
        seed=seed,
        denoiser=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        noiser=object(),
        video=video,
        audio=audio,
    )

    assert len(calls) == 2
    _assert_ancestral_kwargs(calls[0], seed=seed, dtype=pipeline.dtype)
    assert "stepper" not in calls[1]
    assert "loop" not in calls[1]
    assert calls[1]["modalities"].video.latent == "v1"
    assert calls[1]["modalities"].audio.latent == "a1"
    assert calls[1]["modalities"].video.noise_scale == calls[1]["modalities"].audio.noise_scale
    assert calls[1]["modalities"].video.noise_scale > 0
    assert video_state.latent == "v2"
    assert audio_state.latent == "a2"


def _frozen_audio_run(*, use_ancestral: bool) -> tuple[list[dict[str, Any]], Any, Any]:
    from ltx_pipelines.utils.types import ModalitySpec

    pipeline, calls = _pipeline_with_recording_stage(use_ancestral=use_ancestral)
    video_state, audio_state = pipeline._denoise_masked_latents(
        distilled=True,
        seed=7,
        denoiser=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        noiser=object(),
        video=ModalitySpec(latent="v0", conditioning_fps=24.0),
        audio=ModalitySpec(latent="a0", conditioning_fps=24.0, frozen=True),
    )
    return calls, video_state, audio_state


def test_frozen_modality_is_never_noised_in_any_stage() -> None:
    calls, _, _ = _frozen_audio_run(use_ancestral=True)

    assert len(calls) == 2
    for call in calls:
        modalities = call["modalities"]
        assert modalities.audio.frozen is True
        assert modalities.audio.noise_scale == 0.0
    # The live side keeps the schedule's noise: pinning must not leak onto it.
    assert calls[0]["modalities"].video.noise_scale == 1.0
    assert calls[1]["modalities"].video.noise_scale > 0


def test_frozen_modality_returns_the_input_latent_and_feeds_it_to_stage_2() -> None:
    calls, video_state, audio_state = _frozen_audio_run(use_ancestral=True)

    assert calls[1]["modalities"].audio.latent == "a0"
    assert audio_state.latent == "a0"
    assert video_state.latent == "v2"


def test_frozen_video_is_pinned_and_restored_on_the_single_stage_path() -> None:
    from ltx_pipelines.utils.types import ModalitySpec

    pipeline, calls = _pipeline_with_recording_stage(use_ancestral=False)
    video_state, audio_state = pipeline._denoise_masked_latents(
        distilled=True,
        seed=7,
        denoiser=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        noiser=object(),
        video=ModalitySpec(latent="v0", conditioning_fps=24.0, frozen=True),
        audio=ModalitySpec(latent="a0", conditioning_fps=24.0),
    )

    assert len(calls) == 1
    assert calls[0]["modalities"].video.noise_scale == 0.0
    assert calls[0]["modalities"].audio.noise_scale == 1.0
    assert video_state.latent == "v0"
    assert audio_state.latent == "a1"
