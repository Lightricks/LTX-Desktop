"""Shared types/protocols for backend service modules."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, BinaryIO, Protocol, TypeAlias

import torch
from PIL.Image import Image as PILImage

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ltx_core.model.video_vae import TilingConfig


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
RequestFieldValue: TypeAlias = str | bytes | int | float | bool | None
RequestData: TypeAlias = bytes | str | Mapping[str, RequestFieldValue] | BinaryIO | None
PromptInput: TypeAlias = str | Sequence[str]

TensorType: TypeAlias = torch.Tensor
PILImageType: TypeAlias = PILImage

if TYPE_CHECKING:
    from ltx_core.types import Audio as AudioType

    FrameArray: TypeAlias = NDArray[np.uint8]
    TilingConfigType: TypeAlias = TilingConfig
else:
    FrameArray: TypeAlias = object
    TilingConfigType: TypeAlias = object
    AudioType: TypeAlias = object

TensorOrNone: TypeAlias = TensorType | None
AudioOrNone: TypeAlias = AudioType | None

logger = logging.getLogger(__name__)


def get_device_type(device: str | torch.device | object | None) -> str:
    if device is None:
        return "cpu"

    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type

    if isinstance(device, str):
        try:
            return str(torch.device(device).type)
        except Exception:
            logger.warning("Could not parse device string '%s', using it as-is", device, exc_info=True)
            return device

    return "cpu"


def is_rocm_device() -> bool:
    """Return True if PyTorch is running on an AMD ROCm/HIP backend."""
    return getattr(torch.version, "hip", None) is not None


def device_supports_fp8(device: str | torch.device | object | None) -> bool:
    if get_device_type(device) != "cuda":
        return False
    # ROCm backend: FP8 is not reliably hardware-accelerated on RDNA 3.x (gfx1151)
    if is_rocm_device():
        return False
    # NVIDIA: FP8 requires Ada Lovelace (sm_89) or newer
    try:
        major, minor = torch.cuda.get_device_capability()
        return major > 8 or (major == 8 and minor >= 9)
    except Exception:
        return False


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
