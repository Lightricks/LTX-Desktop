"""Monkey-patch: abort denoising between transformer forwards, and count steps.

``ltx_pipelines`` sampler loops have no interrupt or progress callback. Every
local video path (distilled t2v/i2v, A2V, retake, extend, IC-LoRA) goes through
``DiffusionStage.__call__`` → ``loop(..., denoiser=...)``. Wrapping that
denoiser checks cancel at every denoise call (res2s calls it twice per step).

Wrapping ``loop`` and intercepting ``tqdm`` iterations counts real denoise
steps once per loop iteration (not once per denoiser call). Totals come from
``services.denoising_progress.distilled_total_steps``, which uses the same
sigma schedules the pipelines pass in. Step-based progress reporting was
proposed by Sergio Gil (@jimeneztion).

Raising from the denoiser unwinds ``DiffusionStage.__call__``'s transformer
context, so stage 2 / VAE / ffmpeg never run and ``diffusion_stage_cache``
``_in_use`` drops. GPU weights stay loaded.

Wrap the argument named ``denoiser`` via ``signature.bind``, not ``args[0]``.
A positional reorder still hits the denoiser; a missing name raises at the
``__call__`` (once per stage), not per denoise step.

Remove once ltx-pipelines denoiser/loop accepts an interrupt callback.

Usage:
    import services.patches.diffusion_interrupt  # noqa: F401
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from ltx_pipelines.utils.blocks import DiffusionStage

from services.generation_interrupt import wrap_denoiser

assert callable(getattr(DiffusionStage, "__call__", None)), (
    "ltx_pipelines.utils.blocks.DiffusionStage.__call__ missing — re-verify this patch on rev bump"
)
_original_call = DiffusionStage.__call__
_call_params = list(inspect.signature(_original_call).parameters)
assert len(_call_params) >= 2 and _call_params[0] == "self" and _call_params[1] == "denoiser", (
    "DiffusionStage.__call__ first positional after self is not 'denoiser' — "
    "re-verify this patch on rev bump"
)


@dataclass
class _TqdmStepCount:
    yielded: int = 0
    counted: int = 0

    def flush_completed(self) -> None:
        """Count iterations the consumer completed but did not resume past.

        ``note_step`` runs after each ``yield``. ``euler_ancestral_denoising_loop``
        ``break``s on the terminal zero-sigma step, so that last resume never
        happens even though the denoise ran. Flush only after a normal return —
        cancel/exception must not count the aborted iteration.
        """
        from services.denoising_progress import note_step

        while self.counted < self.yielded:
            self.counted += 1
            note_step()


@contextmanager
def _count_tqdm_iterations() -> Iterator[_TqdmStepCount]:
    """Count each sampler-loop tqdm iteration as one denoise step."""
    import ltx_pipelines.utils.samplers as samplers

    from services.denoising_progress import note_step

    original_tqdm = samplers.tqdm
    state = _TqdmStepCount()

    def counting_tqdm(iterable: Any = None, **kwargs: Any) -> Any:
        inner = original_tqdm(iterable, **kwargs)

        class _CountingTqdm:
            def __iter__(self) -> Iterator[Any]:
                for item in inner:
                    state.yielded += 1
                    yield item
                    state.counted += 1
                    note_step()

            def __len__(self) -> int:
                return len(inner)

        return _CountingTqdm()

    samplers.tqdm = counting_tqdm  # type: ignore[misc]
    try:
        yield state
    finally:
        samplers.tqdm = original_tqdm  # type: ignore[misc]


def _loop_with_step_count(loop: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _count_tqdm_iterations() as state:
            result = loop(*args, **kwargs)
            state.flush_completed()
            return result

    return wrapped


@functools.wraps(_original_call)
def _call_with_interrupt(self: DiffusionStage, *args: Any, **kwargs: Any) -> Any:
    bound = inspect.signature(_original_call).bind(self, *args, **kwargs)
    bound.apply_defaults()
    if "denoiser" not in bound.arguments:
        raise TypeError("DiffusionStage.__call__ missing required argument: 'denoiser'")
    bound.arguments["denoiser"] = wrap_denoiser(cast(Callable[..., Any], bound.arguments["denoiser"]))
    if "loop" in bound.arguments:
        loop = bound.arguments["loop"]
        if loop is None:
            from ltx_pipelines.utils.samplers import euler_denoising_loop

            loop = euler_denoising_loop
        bound.arguments["loop"] = _loop_with_step_count(cast(Callable[..., Any], loop))
    return _original_call(*bound.args, **bound.kwargs)


DiffusionStage.__call__ = _call_with_interrupt  # type: ignore[method-assign]
