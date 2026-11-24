"""Real-ffmpeg check that the retake stitch neither skips nor repeats a frame."""

from __future__ import annotations

import subprocess
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np

from services.retake_pipeline.frame_rate import restamp_to_source_rate
from services.retake_pipeline.retake_stitch import stitch_retake_output
from services.retake_pipeline.window import window_for_retake

NTSC_FILM = 24000 / 1001
SOURCE_FRAMES = 300
SIZE = 64


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", *args], check=True, capture_output=True
    )


def _source(path: Path) -> Path:
    # Brightness steps 20 levels per frame, so neighbours are easy to tell apart.
    _ffmpeg(
        "-f", "lavfi", "-i",
        f"nullsrc=s={SIZE}x{SIZE}:r=24000/1001,"
        f"format=gray,geq=lum='mod(N*20,240)+8',trim=end_frame={SOURCE_FRAMES}",
        "-f", "lavfi", "-i", f"sine=d={SOURCE_FRAMES / NTSC_FILM}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(path),
    )
    return path


def _window_like_the_encoder(source: Path, path: Path, start: int, frames: int) -> Path:
    """The source's own frames written as ``ltx_pipelines`` does, then restamped.

    PyAV at ``rate=23`` plus the restamp leaves every timestamp a hair early,
    which is what made seconds-based trims drop the boundary frame.
    """
    with av.open(str(source)) as reader, av.open(str(path), "w") as writer:
        stream = writer.add_stream("libx264", rate=int(NTSC_FILM), options={"crf": "12"})
        stream.width = stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        for index, frame in enumerate(reader.decode(video=0)):
            if start <= index < start + frames:
                image = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"))
                for packet in stream.encode(image):
                    writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
    restamp_to_source_rate(str(path), source_fps=NTSC_FILM)
    return path


def _gray_frames(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack(
            [
                frame.reformat(width=32, height=32, format="gray")
                .to_ndarray()
                .astype(np.float32)
                for frame in container.decode(video=0)
            ]
        )


def test_stitched_retake_keeps_every_source_frame_in_place(tmp_path: Path) -> None:
    source = _source(tmp_path / "source.mp4")
    window = window_for_retake(
        source_frames=SOURCE_FRAMES,
        mask_start_frame=140,
        mask_end_frame=170,
        max_input_frames=161,
        fps=NTSC_FILM,
    )
    windowed = _window_like_the_encoder(
        source,
        tmp_path / "out.window.mp4",
        window.encode_start_frame,
        window.encode_frames,
    )
    dest = tmp_path / "out.mp4"

    stitch_retake_output(
        source_path=str(source),
        windowed_path=str(windowed),
        dest_path=str(dest),
        window=window,
        fps=NTSC_FILM,
        target_width=SIZE,
        target_height=SIZE,
    )

    expected = _gray_frames(source)
    stitched = _gray_frames(dest)
    assert len(stitched) == SOURCE_FRAMES
    for index, frame in enumerate(stitched):
        lo, hi = max(0, index - 2), min(SOURCE_FRAMES, index + 3)
        errors = [float(np.mean((frame - expected[j]) ** 2)) for j in range(lo, hi)]
        assert lo + int(np.argmin(errors)) == index, f"frame {index} is out of place"
