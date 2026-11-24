"""Distilled 2.5 sampler plan for Home/Remote retake and extend.

T2V 2.5 injects ancestral noise in stage 1 and relies on a short Euler refine
to clean it. The masked region starts at sigma=1 (pure noise), so Euler-only
on that 8-step schedule never lets the prompt take — retake reconstructs the
source, extend morphs the last frame into sludge. Ancestral-only leaves
texture boiling. Both features need the same pair as T2V: ancestral stage 1
plus a same-resolution Euler refine (no 2× upsample).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from ltx_pipelines.distilled import (
    ANCESTRAL_ETA,
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_S_NOISE,
)
from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop

from services.denoising_progress import distilled_total_steps


def _distilled_stage_sampler_kwargs(
    *,
    distilled: bool,
    use_ancestral: bool,
    seed: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Optional ``loop`` override for :class:`DiffusionStage`.

    Distilled 2.5+ checkpoints need the ancestral sampler; 2.3 and the guided
    (non-distilled) path keep DiffusionStage's deterministic Euler default.
    ``eta`` and ``s_noise`` belong on the loop, not on the step constructor.
    """
    if not distilled or not use_ancestral:
        return {}
    return {
        "loop": partial(
            euler_ancestral_denoising_loop,
            noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            model_dtype=dtype,
            eta=ANCESTRAL_ETA,
            s_noise=ANCESTRAL_S_NOISE,
        ),
    }


def _retake_uses_two_stage(*, distilled: bool, use_ancestral: bool) -> bool:
    return bool(distilled and use_ancestral)


def _retake_denoise_steps(
    *, distilled: bool, use_ancestral: bool, guided_steps: int = 40
) -> int:
    if not distilled:
        return guided_steps
    return distilled_total_steps(stage_2=use_ancestral)


@dataclass(frozen=True)
class SamplerPlan:
    two_stage: bool
    total_steps: int
    stage1: dict[str, Any]
    refine: dict[str, Any]


def resolve_sampler_plan(
    *,
    distilled: bool,
    use_ancestral: bool,
    seed: int,
    dtype: torch.dtype,
) -> SamplerPlan:
    two_stage = _retake_uses_two_stage(distilled=distilled, use_ancestral=use_ancestral)
    return SamplerPlan(
        two_stage=two_stage,
        total_steps=_retake_denoise_steps(distilled=distilled, use_ancestral=two_stage),
        stage1=_distilled_stage_sampler_kwargs(
            distilled=distilled, use_ancestral=two_stage, seed=seed, dtype=dtype
        ),
        refine=_distilled_stage_sampler_kwargs(
            distilled=distilled, use_ancestral=False, seed=seed, dtype=dtype
        ),
    )
