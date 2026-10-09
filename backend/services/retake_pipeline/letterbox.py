"""Fit a picture onto the ÷32 canvas, then crop the black bars back off.

Retake and extend both encode at ``SpatialLetterbox.canvas_*`` and stitch at
``content_*``. Padding keeps source rows on the same rows as the generated
span. A center-crop or a stretch moves them and the seam jumps. The encode and
the stitch both resize through ``resize_rgb`` so they share one kernel.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from handlers.video_resolution import SpatialLetterbox


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
