"""Fail a Mac generation when a denoise step returns NaN or Inf.

A non-finite latent decodes to a solid green video, and the job still ends as succeeded
(LTX-Desktop#199). The check reads the host once per denoiser call, so it runs on MPS only.
CUDA keeps its asynchronous queue.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class NonFiniteLatentError(RuntimeError):
    def __init__(self, modality: str) -> None:
        super().__init__(
            f"A {modality} denoise step returned NaN or Inf values. "
            "The output would be corrupt, so the job was stopped."
        )


def raise_if_nonfinite(result: Any) -> None:
    """Raise when a denoiser result (a ``VideoAudio``) holds a non-finite ``denoised`` tensor on MPS."""
    for modality in ("video", "audio"):
        denoised = getattr(getattr(result, modality, None), "denoised", None)
        if not isinstance(denoised, torch.Tensor) or denoised.device.type != "mps":
            continue
        if not bool(torch.isfinite(denoised).all()):
            raise NonFiniteLatentError(modality)


def wrap_denoiser(denoiser: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = denoiser(*args, **kwargs)
        raise_if_nonfinite(result)
        return result

    return wrapped
