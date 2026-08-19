"""Monkey-patch: abort denoising between transformer forwards.

``ltx_pipelines`` sampler loops have no interrupt callback. Every local video
path (distilled t2v/i2v, A2V, retake, extend, IC-LoRA) goes through
``DiffusionStage.__call__`` → ``loop(..., denoiser=...)``. Wrapping that
denoiser checks cancel at every denoise call (res2s calls it twice per step).

Raising from the denoiser unwinds ``DiffusionStage.__call__``'s transformer
context, so stage 2 / VAE / ffmpeg never run and ``diffusion_stage_cache``
``_in_use`` drops. GPU weights stay loaded.

Remove once ltx-pipelines denoiser/loop accepts an interrupt callback.

Usage:
    import services.patches.diffusion_interrupt  # noqa: F401
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast

from ltx_pipelines.utils.blocks import DiffusionStage

from services.generation_interrupt import wrap_denoiser

assert callable(getattr(DiffusionStage, "__call__", None)), (
    "ltx_pipelines.utils.blocks.DiffusionStage.__call__ missing — re-verify this patch on rev bump"
)
_original_call = DiffusionStage.__call__
assert "denoiser" in inspect.signature(_original_call).parameters, (
    "DiffusionStage.__call__ has no 'denoiser' parameter — re-verify this patch on rev bump"
)
_signature_fn: object | None = None
_cached_signature: inspect.Signature | None = None


def _original_signature() -> inspect.Signature:
    global _signature_fn, _cached_signature
    fn = _original_call
    if _signature_fn is not fn or _cached_signature is None:
        _cached_signature = inspect.signature(fn)
        _signature_fn = fn
    return _cached_signature


def _call_with_interrupt(self: DiffusionStage, *args: Any, **kwargs: Any) -> Any:
    bound = _original_signature().bind(self, *args, **kwargs)
    bound.apply_defaults()
    original_denoiser = cast(Callable[..., Any], bound.arguments["denoiser"])
    bound.arguments["denoiser"] = wrap_denoiser(original_denoiser)
    return _original_call(*bound.args, **bound.kwargs)


DiffusionStage.__call__ = _call_with_interrupt  # type: ignore[method-assign]
