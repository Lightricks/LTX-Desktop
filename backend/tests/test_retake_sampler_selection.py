"""GPU-free checks for Retake/Extend distilled sampler selection.

``_run()`` cannot be exercised here without loading real checkpoints. The helper
it calls, plus ``_invoke_diffusion_stage`` (the ``self.stage`` call site), are
the whole decision.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_pipelines.distilled import (
    ANCESTRAL_ETA,
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_S_NOISE,
    should_use_ancestral_sampler,
)
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
from safetensors.torch import save_file

from services.retake_pipeline.ltx_retake_pipeline import (
    LTXRetakePipeline,
    distilled_stage_sampler_kwargs,
)


def _checkpoint_with_version(tmp_path: Path, version: str) -> str:
    path = tmp_path / f"transformer-{version}.safetensors"
    save_file({"dummy": torch.zeros(1)}, str(path), metadata={"model_version": version})
    return str(path)


def _assert_ancestral_kwargs(kwargs: dict[str, Any], *, seed: int, dtype: torch.dtype) -> None:
    stepper = kwargs["stepper"]
    assert isinstance(stepper, EulerAncestralDiffusionStep)
    assert stepper.eta == ANCESTRAL_ETA
    assert stepper.s_noise == ANCESTRAL_S_NOISE
    loop = kwargs["loop"]
    assert isinstance(loop, partial)
    assert loop.func is euler_ancestral_denoising_loop
    assert loop.keywords["noise_seed"] == seed + ANCESTRAL_NOISE_SEED_OFFSET
    assert loop.keywords["model_dtype"] is dtype


def test_distilled_2_5_selects_ancestral_sampler(tmp_path: Path) -> None:
    seed = 7
    dtype = torch.bfloat16
    path = _checkpoint_with_version(tmp_path, "2.5")
    assert should_use_ancestral_sampler(path)
    kwargs = distilled_stage_sampler_kwargs(
        distilled=True,
        use_ancestral=should_use_ancestral_sampler(path),
        seed=seed,
        dtype=dtype,
    )
    _assert_ancestral_kwargs(kwargs, seed=seed, dtype=dtype)


def test_distilled_2_3_keeps_deterministic_defaults(tmp_path: Path) -> None:
    path = _checkpoint_with_version(tmp_path, "2.3")
    assert not should_use_ancestral_sampler(path)
    kwargs = distilled_stage_sampler_kwargs(
        distilled=True,
        use_ancestral=should_use_ancestral_sampler(path),
        seed=1,
        dtype=torch.bfloat16,
    )
    assert kwargs == {}


def test_guided_path_never_selects_ancestral_sampler() -> None:
    kwargs = distilled_stage_sampler_kwargs(
        distilled=False,
        use_ancestral=True,
        seed=1,
        dtype=torch.bfloat16,
    )
    assert kwargs == {}


def _pipeline_with_recording_stage(*, use_ancestral: bool) -> tuple[LTXRetakePipeline, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def fake_stage(**kwargs: Any) -> tuple[object, object]:
        calls.append(kwargs)
        return object(), object()

    pipeline = cast(LTXRetakePipeline, object.__new__(LTXRetakePipeline))
    pipeline.dtype = torch.bfloat16
    pipeline.use_ancestral_sampler = use_ancestral
    pipeline.stage = fake_stage  # type: ignore[method-assign]
    return pipeline, calls


def test_invoke_diffusion_stage_forwards_ancestral_kwargs() -> None:
    seed = 7
    pipeline, calls = _pipeline_with_recording_stage(use_ancestral=True)
    denoiser = object()
    pipeline._invoke_diffusion_stage(distilled=True, seed=seed, denoiser=denoiser)

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["denoiser"] is denoiser
    _assert_ancestral_kwargs(kwargs, seed=seed, dtype=pipeline.dtype)


def test_invoke_diffusion_stage_omits_sampler_overrides_when_not_ancestral() -> None:
    pipeline, calls = _pipeline_with_recording_stage(use_ancestral=False)
    pipeline._invoke_diffusion_stage(distilled=True, seed=7, denoiser=object())

    assert len(calls) == 1
    assert "stepper" not in calls[0]
    assert "loop" not in calls[0]
