"""Guide clip of a recipe that refines an upscaled clip (Refine Details).

The model card says to scale the clip to the output canvas with Lanczos and feed that as the
guide. The upstream pipeline would scale the clip itself, but with a bilinear filter, and
that gives the model a softer picture than the one it trained on. So the recipe writes the
guide at the exact canvas. The pipeline then has nothing to scale.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from services.ffmpeg import (
    FFMPEG_PROTOCOL_WHITELIST,
    FfmpegCancelledError,
    run_ffmpeg,
    timeout_for_pixels,
)
from services.generation_interrupt import GenerationCancelledError, is_requested
from services.media_probe import video_frame_count
from services.records import MediaError


def lanczos_guide_filter(width: int, height: int, output_fps: float | None = None) -> str:
    """Resample to ``output_fps``, scale to fill the canvas, then cut the overflow.

    The clip is already near the canvas shape, so the cut removes only the aspect slack.
    ``fps`` picks the nearest source frame, as ``ic_lora_video_filters`` does.
    """
    scale = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}"
    )
    return scale if output_fps is None else f"fps={output_fps:g},{scale}"


def prepare_lanczos_guide(
    source: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    keep_audio: bool,
    output_fps: float | None = None,
) -> Path:
    """Write ``dest``: the clip at ``width`` x ``height`` through a Lanczos filter.

    The guide stays lossless (``-qp 0``), so the model reads the pixels the filter made.
    The colour tags match ``center_crop_video``, the other recipes' model input.
    """
    args = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        lanczos_guide_filter(width, height, output_fps),
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
    ]
    args += ["-map", "0:a?", "-c:a", "aac"] if keep_audio else ["-an"]
    args += ["-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, str(dest)]
    try:
        run_ffmpeg(
            args,
            timeout=timeout_for_pixels(width, height, video_frame_count(source)),
            should_cancel=is_requested,
        )
    except FfmpegCancelledError:
        raise GenerationCancelledError() from None
    except subprocess.CalledProcessError as exc:
        raise MediaError("UNREADABLE_MEDIA", "clip preparation failed") from exc
    return dest
