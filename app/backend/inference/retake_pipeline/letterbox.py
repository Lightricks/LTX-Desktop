"""Fit a picture onto the ÷32 canvas, then crop the black bars back off.

Retake and extend both encode at ``SpatialLetterbox.canvas_*`` and stitch at
``content_*``. Padding keeps source rows on the same rows as the generated
span. A center-crop or a stretch moves them and the seam jumps. The encode and
the stitch both resize through ``resize_rgb`` so they share one kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

_SPATIAL_FACTOR = 32


@dataclass(frozen=True)
class SpatialLetterbox:
    """Picture rect centered on the ÷32 canvas the model actually encodes."""

    content_width: int
    content_height: int
    canvas_width: int
    canvas_height: int
    left: int
    top: int

    @property
    def right(self) -> int:
        return self.canvas_width - self.content_width - self.left

    @property
    def bottom(self) -> int:
        return self.canvas_height - self.content_height - self.top


def generation_letterbox(width: int, height: int) -> SpatialLetterbox:
    """Pad ``width``×``height`` up to the next multiple of 32, split evenly.

    An odd remainder gives the extra pixel to the right or the bottom.
    """
    canvas_width = ((width + _SPATIAL_FACTOR - 1) // _SPATIAL_FACTOR) * _SPATIAL_FACTOR
    canvas_height = ((height + _SPATIAL_FACTOR - 1) // _SPATIAL_FACTOR) * _SPATIAL_FACTOR
    pad_x = canvas_width - width
    pad_y = canvas_height - height
    return SpatialLetterbox(
        content_width=width,
        content_height=height,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        left=pad_x // 2,
        top=pad_y // 2,
    )


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))


def fit_and_pad(frame: np.ndarray, box: SpatialLetterbox) -> np.ndarray:
    """Resize ``frame`` to the picture, then center it on a black canvas."""
    fitted = resize_rgb(frame, box.content_width, box.content_height)
    if box.canvas_width == box.content_width and box.canvas_height == box.content_height:
        return fitted
    canvas = np.zeros((box.canvas_height, box.canvas_width, frame.shape[2]), dtype=frame.dtype)
    canvas[
        box.top : box.top + box.content_height,
        box.left : box.left + box.content_width,
    ] = fitted
    return canvas


def crop_letterbox(frame: np.ndarray, box: SpatialLetterbox) -> np.ndarray:
    """Drop the bars added by ``fit_and_pad``."""
    if box.top == 0 and box.left == 0 and frame.shape[0] == box.content_height and frame.shape[1] == box.content_width:
        return frame
    return frame[
        box.top : box.top + box.content_height,
        box.left : box.left + box.content_width,
    ].copy()
