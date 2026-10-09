"""Letterbox pad and crop shared by retake and extend."""

from __future__ import annotations

import numpy as np

from handlers.video_resolution import generation_letterbox
from services.retake_pipeline.letterbox import crop_letterbox, fit_and_pad


def test_pad_then_crop_restores_the_1080p_picture() -> None:
    box = generation_letterbox(8, 6)
    picture = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(6, 8, 3)
    canvas = fit_and_pad(picture, box)
    assert canvas.shape == (box.canvas_height, box.canvas_width, 3)
    assert canvas[: box.top].sum() == 0
    assert canvas[box.top + box.content_height :].sum() == 0
    assert np.array_equal(crop_letterbox(canvas, box), picture)


def test_fit_and_pad_scales_a_larger_source_into_the_picture_before_padding() -> None:
    box = generation_letterbox(4, 2)
    source = np.full((4, 8, 3), 90, dtype=np.uint8)
    canvas = fit_and_pad(source, box)
    cropped = crop_letterbox(canvas, box)
    assert cropped.shape == (2, 4, 3)
    assert cropped.min() == 90
