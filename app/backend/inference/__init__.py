"""Headless inference library. Public exports only; no desktop infra."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

# Hosts should configure the allocator before importing inference. This
# setdefault protects direct library imports without overriding host policy.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from inference.types import (
    ExtendMode,
    ImageConditioningInput,
    LTXLocalModelId,
    RetakeMode,
)

if TYPE_CHECKING:
    from inference.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline as LTXFastVideoPipeline

__all__ = [
    "ExtendMode",
    "ImageConditioningInput",
    "LTXFastVideoPipeline",
    "LTXLocalModelId",
    "RetakeMode",
]


def __getattr__(name: str) -> object:
    # TODO: the Pod image in the next PR imports LTXFastVideoPipeline from this package.
    if name == "LTXFastVideoPipeline":
        from inference.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline

        return LTXFastVideoPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
