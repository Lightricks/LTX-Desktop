"""Silence stands in for a silent source's audio in retake/extend stitches. It must end."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from services.ffmpeg import run_ffmpeg
from services.records import MediaError
from services.retake_pipeline.stitch_silence import append_silence_input


def _silent_video(path: Path, *, frames: int, fps: int) -> str:
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return str(path)


def test_silence_input_is_added_after_every_input_and_ends_the_output_with_the_video() -> None:
    args = ["-i", "video.mp4"]
    append_silence_input(args)
    assert args[-1] == "-shortest"
    assert args.index("-shortest") > max(i for i, arg in enumerate(args) if arg == "-i")


def test_a_stitch_that_keeps_the_silence_tail_finishes_with_the_video(tmp_path: Path) -> None:
    """Retake with a suffix and extend at the start keep ``atrim=start=...`` of the silence;
    without ``-shortest`` ffmpeg encodes silence until the timeout."""
    video = _silent_video(tmp_path / "video.mp4", frames=50, fps=25)
    dest = tmp_path / "out.mp4"
    args = ["-y", "-i", video]
    append_silence_input(args)
    args += [
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS[v];[1:a]atrim=start=1,asetpts=PTS-STARTPTS[a]",
        "-map", "[v]", "-map", "[a]", str(dest),
    ]

    try:
        run_ffmpeg(args, timeout=30)
    except MediaError as exc:
        pytest.fail(f"ffmpeg did not stop on its own: {exc}")

    with av.open(str(dest)) as container:
        assert float(container.duration) / av.time_base == pytest.approx(2.0, abs=0.2)
