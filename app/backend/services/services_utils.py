"""Shared types/protocols for backend service modules."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, BinaryIO, Protocol, TypeAlias

import torch
from PIL.Image import Image as PILImage

from inference.device import device_supports_fp8 as device_supports_fp8
from inference.device import get_device_type

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
RequestFieldValue: TypeAlias = str | bytes | int | float | bool | None
RequestData: TypeAlias = bytes | str | Mapping[str, RequestFieldValue] | BinaryIO | None
PromptInput: TypeAlias = str | Sequence[str]

TensorType: TypeAlias = torch.Tensor
PILImageType: TypeAlias = PILImage

if TYPE_CHECKING:
    FrameArray: TypeAlias = NDArray[np.uint8]
else:
    FrameArray: TypeAlias = object

TensorOrNone: TypeAlias = TensorType | None

logger = logging.getLogger(__name__)


def compute_edit_dimensions(width: int, height: int) -> tuple[int, int]:
    """Floor each dimension to a multiple of 16 (Z-Image VAE constraint), min 16.

    Image-to-image runs at the source image's resolution; the diffusers pipeline raises
    if a dimension is not divisible by vae_scale (=16 for Z-Image).
    """
    return (max(16, (width // 16) * 16), max(16, (height // 16) * 16))


def clamp_strength(strength: float) -> float:
    """Clamp img2img strength into the range the diffusers pipeline accepts (0, 1]."""
    return min(1.0, max(1e-3, strength))


def effective_edit_steps(num_inference_steps: int, strength: float) -> int:
    """Mirror ZImageImg2ImgPipeline.get_timesteps' effective-step count exactly.

    init_timestep = min(steps * strength, steps); t_start = int(max(steps - init_timestep, 0));
    effective = steps - t_start. Only strength == 0.0 can drive this to 0 (any positive
    strength, however small, yields at least 1 step) — used unclamped so the guard doesn't
    mask that one real case.
    """
    init_timestep = min(num_inference_steps * strength, num_inference_steps)
    t_start = int(max(num_inference_steps - init_timestep, 0))
    return num_inference_steps - t_start


def sync_device(device: str | torch.device | object | None) -> None:
    device_type = get_device_type(device)
    if device_type == "cuda":
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.warning("torch.cuda.synchronize() failed", exc_info=True)
        return

    if device_type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:
            logger.warning("torch.mps.synchronize() failed", exc_info=True)


def empty_device_cache(device: str | torch.device | object | None) -> None:
    device_type = get_device_type(device)
    if device_type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.warning("torch.cuda.empty_cache() failed", exc_info=True)
        return

    if device_type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            logger.warning("torch.mps.empty_cache() failed", exc_info=True)


class LatentStateLike(Protocol):
    latent: torch.Tensor


class VideoCaptureLike(Protocol):
    def get(self, prop_id: int) -> float:
        ...

    def set(self, prop_id: int, value: float) -> bool:
        ...

    def read(self) -> tuple[bool, FrameArray | None]:
        ...

    def release(self) -> None:
        ...

    def isOpened(self) -> bool:
        ...


class VideoWriterLike(Protocol):
    def write(self, frame: FrameArray) -> None:
        ...

    def release(self) -> None:
        ...


class ImagePipelineOutputLike(Protocol):
    images: Sequence[PILImageType]
