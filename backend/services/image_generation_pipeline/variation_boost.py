"""Variation boost for distilled, guidance-free image models (Z-Image Turbo, Krea 2 Turbo).

Distilled turbo models let the prompt pin the composition, so different seeds barely move
it. Adding seeded noise to a random subset of the text-embedding values for the first
step(s) — where layout is decided — knocks each seed into a different composition; the
clean embeddings are restored afterwards so the remaining steps refine detail normally.
Costs nothing measurable (the prompt is encoded once, same as without the boost), and the
same seed + variation reproduces the same image.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import torch

from services.generation_interrupt import diffusers_step_callback

logger = logging.getLogger(__name__)

StepCallback = Callable[[object, int, object, dict[str, Any]], dict[str, Any]]
_E = TypeVar("_E", torch.Tensor, list[torch.Tensor])


def perturb_embeds(
    clean: _E,
    *,
    seed: int,
    variation: float,
    max_noise: float,
    mask_fraction: float = 0.5,
) -> _E:
    """Noise with std = variation x max_noise x the embeddings' own std, on a random mask."""
    # Seeded on CPU so results don't depend on device RNG.
    gen = torch.Generator(device="cpu").manual_seed((seed * 7919 + 1) % (2**63))

    def one(emb: torch.Tensor) -> torch.Tensor:
        shape = tuple(emb.shape)
        noise = torch.randn(shape, generator=gen, dtype=torch.float32)
        mask = torch.rand(shape, generator=gen) < mask_fraction
        scale = float(emb.float().std()) * max_noise * variation
        return emb + (noise * mask * scale).to(device=emb.device, dtype=emb.dtype)

    if isinstance(clean, torch.Tensor):
        return one(clean)
    return [one(e) for e in clean]


def boost_step_count(num_inference_steps: int, boost_fraction: float) -> int:
    return max(1, round(num_inference_steps * boost_fraction))


def restoring_callback(clean: object, boost_steps: int) -> StepCallback:
    """Interrupt-aware step callback that swaps the clean embeds back in after boost_steps."""

    def callback(pipe: object, step_index: int, timestep: object, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = diffusers_step_callback(pipe, step_index, timestep, kwargs)
        if step_index == boost_steps - 1:
            kwargs["prompt_embeds"] = clean
        return kwargs

    return callback


def clamp_variation(variation: float) -> float:
    return max(0.0, min(1.0, variation))


def log_boost(model: str, variation: float, boost_steps: int, num_inference_steps: int) -> None:
    logger.info(
        "%s variation boost %.2f: noisy prompt embeds for %d/%d steps",
        model,
        variation,
        boost_steps,
        num_inference_steps,
    )
