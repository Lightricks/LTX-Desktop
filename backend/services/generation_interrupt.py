"""Process-wide cooperative cancel for in-flight generation.

The denoise thread must not take AppState's RLock. Cancel HTTP sets this Event;
the inference loop (and Diffusers step callback) polls it between steps.

On MPS, ``Event.wait`` is used as a GIL yield so ``POST /api/generate/cancel``
and ``GET /health`` can run between steps. CUDA kernels already release the GIL,
so the cancel check is a non-blocking ``is_set``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

_cancel = threading.Event()
_MPS_YIELD_S = 0.001
_yield_timeout_s: float | None = None

_D = TypeVar("_D")


class GenerationCancelledError(RuntimeError):
    def __init__(self, message: str = "Generation was cancelled") -> None:
        super().__init__(message)


def request() -> None:
    _cancel.set()


def clear() -> None:
    _cancel.clear()


def is_requested() -> bool:
    return _cancel.is_set()


def raise_if_requested() -> None:
    if _cancel.is_set():
        raise GenerationCancelledError()


def _gil_yield_timeout_s() -> float:
    global _yield_timeout_s
    if _yield_timeout_s is None:
        timeout = 0.0
        try:
            import torch

            if torch.backends.mps.is_available():
                timeout = _MPS_YIELD_S
        except Exception:
            timeout = 0.0
        _yield_timeout_s = timeout
    return _yield_timeout_s


def yield_and_check() -> None:
    """Abort if cancel was requested. On MPS, yield the GIL first so /health can run."""
    timeout = _gil_yield_timeout_s()
    if timeout <= 0:
        raise_if_requested()
        return
    if _cancel.wait(timeout):
        raise GenerationCancelledError()


def wrap_denoiser(denoiser: Callable[..., _D]) -> Callable[..., _D]:
    def wrapped(*args: Any, **kwargs: Any) -> _D:
        yield_and_check()
        return denoiser(*args, **kwargs)

    return wrapped


def is_cancel_exception(exc: BaseException) -> bool:
    """True only for GenerationCancelledError, including when wrapped as __cause__/__context__."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, GenerationCancelledError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def diffusers_step_callback(
    pipeline: object,
    step_index: int,
    timestep: object,
    callback_kwargs: dict[str, Any],
) -> dict[str, Any]:
    del pipeline, step_index, timestep
    yield_and_check()
    return callback_kwargs
