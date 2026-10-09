"""A cutout matte must line up with the clip in time and in space."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest

from services.ffmpeg import run_ffmpeg
from services.features.video.cutout_alignment import (
    GRID_PAD_FRAMES,
    CanvasFit,
    conform_matte_to_clip,
    fit_clip_to_canvas,
    prepare_clip_for_model,
)
from services.media_probe import video_frame_count

_FPS = 24


def _write_clip(path: Path, frames: list[np.ndarray]) -> Path:
    writer = imageio.get_writer(str(path), fps=_FPS, codec="libx264", macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return path


def _solid_clip(path: Path, *, width: int, height: int, frames: int) -> Path:
    return _write_clip(path, [np.full((height, width, 3), 90, dtype=np.uint8)] * frames)


def _first_frame(path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(path))
    try:
        return np.asarray(reader.get_data(0))
    finally:
        reader.close()


def _decoded_frame_count(path: Path) -> int:
    """Frames a decoder returns. Packets alone can hide a duplicated or a late frame."""
    run = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(re.findall(r"frame= *(\d+)", run.stderr)[-1])


def _fit(width: int, height: int, canvas_width: int, canvas_height: int) -> CanvasFit:
    return fit_clip_to_canvas(width, height, canvas_width, canvas_height)


def test_a_wide_clip_is_fitted_inside_a_taller_canvas_with_no_offset_on_the_full_side() -> None:
    fit = _fit(640, 360, 512, 256)

    assert (fit.width, fit.height) == (454, 256)
    # The clip fills the canvas height, so the vertical offset is 0 and not rounded up.
    assert (fit.x, fit.y) == (28, 0)
    assert not fit.fills_canvas


def test_the_fit_offsets_and_sizes_stay_even_and_inside_the_canvas() -> None:
    for clip in [(640, 360), (360, 640), (801, 599), (100, 100)]:
        for canvas in [(512, 256), (256, 512), (640, 384), (704, 704)]:
            fit = _fit(*clip, *canvas)
            assert all(value % 2 == 0 for value in (fit.width, fit.height, fit.x, fit.y))
            assert 0 <= fit.x and fit.x + fit.width <= canvas[0]
            assert 0 <= fit.y and fit.y + fit.height <= canvas[1]


def test_a_clip_with_the_canvas_shape_fills_the_canvas() -> None:
    assert _fit(1024, 512, 512, 256).fills_canvas


def test_the_model_clip_is_the_canvas_size_and_repeats_the_last_frame(tmp_path: Path) -> None:
    source = _solid_clip(tmp_path / "source.mp4", width=160, height=90, frames=20)
    fit = _fit(160, 90, 128, 64)

    model_clip = prepare_clip_for_model(source, tmp_path / "fit.mp4", fit=fit, keep_audio=False)

    assert _first_frame(model_clip).shape[:2] == (64, 128)
    assert video_frame_count(model_clip) == 20 + GRID_PAD_FRAMES


def test_the_border_of_the_model_clip_repeats_the_edge_pixels(tmp_path: Path) -> None:
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[:, :] = (200, 40, 40)
    source = _write_clip(tmp_path / "source.mp4", [frame] * 9)
    fit = _fit(160, 90, 128, 64)

    pixels = _first_frame(prepare_clip_for_model(source, tmp_path / "fit.mp4", fit=fit, keep_audio=False))

    # Black padding would read as background to the model. The smear keeps the clip color.
    assert abs(int(pixels[32, 1, 0]) - 200) < 20
    assert abs(int(pixels[32, 126, 0]) - 200) < 20


@pytest.mark.parametrize("frames", [9, 20, 25, 145, 150])
def test_the_matte_is_cut_to_the_clip_frame_count(tmp_path: Path, frames: int) -> None:
    """A time cut keeps extra frames on a B-frame stream. The frame count does not."""
    source = _solid_clip(tmp_path / "source.mp4", width=160, height=90, frames=frames)
    fit = _fit(160, 90, 128, 64)
    model_clip = prepare_clip_for_model(source, tmp_path / "fit.mp4", fit=fit, keep_audio=False)
    # The model returns the 8k+1 frames the pipeline snaps down to.
    grid = (video_frame_count(model_clip) - 1) // 8 * 8 + 1
    matte = tmp_path / "matte.mp4"
    run_ffmpeg(["-y", "-i", str(model_clip), "-frames:v", str(grid), "-c:v", "libx264", str(matte)])
    assert grid >= frames

    conform_matte_to_clip(matte, frames=frames, fps=_FPS, fit=fit)

    assert video_frame_count(matte) == frames


def test_a_matte_that_fills_the_canvas_is_cut_to_the_exact_frame_count(tmp_path: Path) -> None:
    """With B-frames, a stream copy cut can leave a frame past the cut on some builds."""
    fit = _fit(128, 64, 128, 64)
    assert fit.fills_canvas
    for total, keep in [(153, 150), (33, 26), (17, 10)]:
        matte = _write_clip(
            tmp_path / "matte.mp4",
            [np.full((64, 128, 3), 10 + index, dtype=np.uint8) for index in range(total)],
        )

        conform_matte_to_clip(matte, frames=keep, fps=_FPS, fit=fit)

        assert video_frame_count(matte) == keep
        assert _decoded_frame_count(matte) == keep


def test_a_matte_that_needs_no_cut_is_left_as_the_model_wrote_it(tmp_path: Path) -> None:
    fit = _fit(128, 64, 128, 64)
    matte = _solid_clip(tmp_path / "matte.mp4", width=128, height=64, frames=145)
    before = matte.read_bytes()

    conform_matte_to_clip(matte, frames=145, fps=_FPS, fit=fit)

    assert matte.read_bytes() == before


def test_the_matte_covers_the_same_view_as_the_clip_after_the_border_is_cut(tmp_path: Path) -> None:
    """A 16:9 clip in a 2:1 canvas. A crop would lose the top and the bottom of the clip."""
    width, height = 640, 360
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[300:350, 40:120] = 255  # a bright square near the bottom edge
    frame[10:60, 520:600] = 255  # a bright square near the top edge
    source = _write_clip(tmp_path / "source.mp4", [frame] * 9)
    fit = _fit(width, height, 512, 256)
    # The model is stood in by the model clip itself: its bright pixels are the matte.
    matte = prepare_clip_for_model(source, tmp_path / "matte.mp4", fit=fit, keep_audio=False)

    conform_matte_to_clip(matte, frames=9, fps=_FPS, fit=fit)

    conformed = _first_frame(matte)[:, :, 0] > 128
    assert conformed.shape == (fit.height, fit.width)
    scale_x, scale_y = fit.width / width, fit.height / height
    for top, bottom, left, right in [(300, 350, 40, 120), (10, 60, 520, 600)]:
        rows = slice(int(top * scale_y), int(bottom * scale_y))
        columns = slice(int(left * scale_x), int(right * scale_x))
        expected = np.zeros_like(conformed)
        expected[rows, columns] = True
        # Compare inside a window around this square, so the other square is not counted.
        window = np.zeros_like(conformed)
        window[max(0, rows.start - 8) : rows.stop + 8, max(0, columns.start - 8) : columns.stop + 8] = True
        overlap = (conformed & expected & window).sum() / (expected | (conformed & window)).sum()
        assert overlap > 0.8
