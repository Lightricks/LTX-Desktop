"""Small real MP4 files for tests that decode video (the sampler is real PyAV, never faked)."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

from services.ffmpeg import run_ffmpeg


def write_mp4(
    path: Path, *, seconds: float = 6, fps: int = 12, width: int = 64, height: int = 32
) -> Path:
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(max(1, round(fps * seconds))):
        writer.append_data(frame)
    writer.close()
    return path


def write_pattern_clip(
    path: Path,
    *,
    frames: int,
    audio: bool = False,
    fps: int = 24,
    size: str = "64x64",
    pix_fmt: str = "yuv420p",
) -> Path:
    """A real clip of exactly ``frames`` frames: a moving test pattern, so frames differ.

    ``audio`` adds a sine track as long as the video. The container sets the codec:
    mp4 gets AAC, webm gets Opus.
    """
    args = ["-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    args += ["-frames:v", str(frames), "-c:v", "libx264" if path.suffix == ".mp4" else "libvpx"]
    args += ["-pix_fmt", pix_fmt]
    if audio:
        audio_codec = "aac" if path.suffix == ".mp4" else "libopus"
        args += ["-c:a", audio_codec, "-t", f"{frames / fps:.4f}"]
    run_ffmpeg([*args, str(path)])
    return path


def write_mp4_with_display_rotation(path: Path, *, angle: str) -> Path:
    """A landscape clip whose container says "show it rotated", as a phone records it."""
    source = write_mp4(path.with_name(f"{path.stem}-source.mp4"), seconds=1)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-display_rotation",
            angle,
            "-i",
            str(source),
            "-c",
            "copy",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def write_mp4_with_timestamp_offset(path: Path, *, offset_seconds: int) -> Path:
    """A clip whose timeline starts at ``offset_seconds``, as a cut or re-muxed clip does."""
    source = write_mp4(path.with_name(f"{path.stem}-source.mp4"), seconds=6, fps=12)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(source),
            "-c",
            "copy",
            "-output_ts_offset",
            str(offset_seconds),
            "-muxdelay",
            "0",
            "-muxpreload",
            "0",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def write_noisy_mp4(path: Path, *, frames: int = 72, fps: int = 12) -> Path:
    """Random pixels, so the file is large enough that damaging its middle hits real video data."""
    rng = np.random.default_rng(0)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=None)
    for _ in range(frames):
        writer.append_data(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    writer.close()
    return path


def damage_middle(source: Path, dest: Path, *, seed: int) -> Path:
    """Overwrite random bytes in the middle half: the header stays readable, the frames do not."""
    rng = random.Random(seed)
    data = bytearray(source.read_bytes())
    for _ in range(300):
        data[rng.randrange(len(data) // 4, len(data) * 3 // 4)] = rng.randrange(256)
    dest.write_bytes(bytes(data))
    return dest
