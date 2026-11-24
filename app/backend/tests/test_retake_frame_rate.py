"""The retake/extend window must play at the source rate, not ``int(fps)``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import av
import imageio_ffmpeg
import pytest

from services.retake_pipeline.frame_rate import restamp_to_source_rate

NTSC_FILM = 24000 / 1001
FRAMES = 46


def _window_written_at_23fps(path: Path) -> Path:
    """What the upstream encoder produces for a 23.976 fps source."""
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=s=64x64:r=23:n=1,trim=end_frame={FRAMES}",
            "-f",
            "lavfi",
            "-i",
            f"sine=d={FRAMES / NTSC_FILM}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _frame_times(path: Path) -> list[float]:
    with av.open(str(path)) as container:
        return sorted(float(frame.time) for frame in container.decode(video=0))


def _audio_seconds(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        assert stream.duration is not None
        return float(stream.duration * stream.time_base)


def test_restamp_puts_every_frame_at_the_source_rate(tmp_path: Path) -> None:
    window = _window_written_at_23fps(tmp_path / "w.window.mp4")
    audio_before = _audio_seconds(window)

    restamp_to_source_rate(str(window), source_fps=NTSC_FILM)

    times = _frame_times(window)
    assert len(times) == FRAMES
    for index, time in enumerate(times):
        assert time == pytest.approx(index / NTSC_FILM, abs=1e-3)
    assert _audio_seconds(window) == pytest.approx(audio_before, abs=1e-3)
    assert not (tmp_path / "w.window.restamp.mp4").exists()


def test_restamp_leaves_an_integer_rate_untouched(tmp_path: Path) -> None:
    window = _window_written_at_23fps(tmp_path / "w.window.mp4")
    before = window.stat().st_mtime_ns

    restamp_to_source_rate(str(window), source_fps=23.0)

    assert window.stat().st_mtime_ns == before
