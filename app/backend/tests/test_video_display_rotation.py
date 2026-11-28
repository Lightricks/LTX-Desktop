"""Display rotation comes from the clip, not an ffmpeg banner spawn."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
from services.media_probe import probe_file


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _landscape(path: Path) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x16:r=8:d=0.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _with_display_rotation(src: Path, dest: Path, angle: str) -> None:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-display_rotation",
            angle,
            "-i",
            str(src),
            "-c",
            "copy",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )


def test_probe_swaps_dimensions_for_a_90_degree_display_matrix(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    path = tmp_path / "rotated.mp4"
    _landscape(src)
    _with_display_rotation(src, path, "90")
    _kind, _mime, metadata = probe_file(path)
    assert metadata.mediaType == "video"
    assert (metadata.metadata.width, metadata.metadata.height) == (16, 32)


def test_probe_keeps_coded_size_when_the_clip_has_no_rotation(tmp_path: Path) -> None:
    path = tmp_path / "plain.mp4"
    _landscape(path)
    _kind, _mime, metadata = probe_file(path)
    assert metadata.mediaType == "video"
    assert (metadata.metadata.width, metadata.metadata.height) == (32, 16)


def test_probe_keeps_coded_size_when_rotation_cannot_be_read(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _landscape(src)
    broken = tmp_path / "broken.mp4"
    data = src.read_bytes()
    broken.write_bytes(data[: int(len(data) * 0.7)])
    _kind, _mime, metadata = probe_file(broken)
    assert metadata.mediaType == "video"
    assert (metadata.metadata.width, metadata.metadata.height) == (32, 16)
