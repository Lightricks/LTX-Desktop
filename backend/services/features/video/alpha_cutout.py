"""AlphaGen cutout. Merge the source video and the generated matte into a WebM and a GIF with alpha.

The model writes a grayscale matte (white is the subject). The cutout is the source
video with that matte as its alpha channel. VP9 alpha plays in Chromium, so the player
shows the cutout over any CSS background and the file is the download. The GIF is the
same cutout for apps that do not read a WebM. It has one bit of alpha, so its edges are
harder than the WebM edges.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import NamedTuple

from services.ffmpeg import (
    FFMPEG_PROTOCOL_WHITELIST,
    FfmpegCancelledError,
    run_ffmpeg,
    timeout_for_source_duration,
)
from services.generation_interrupt import GenerationCancelledError, is_requested
from services.generation_queue.types import OutputAllocation, OutputPlan
from services.media_probe import probe_file, video_fps, video_frame_count
from services.records import MediaError

logger = logging.getLogger(__name__)

CUTOUT_OUTPUT_PLAN = OutputPlan(
    slot="output",
    media_kind="video",
    mime_type="video/webm",
    name="output.webm",
)
MATTE_OUTPUT_PLAN = OutputPlan(
    slot="matte",
    media_kind="video",
    mime_type="video/mp4",
    name="matte.mp4",
)
GIF_OUTPUT_PLAN = OutputPlan(
    slot="gif",
    media_kind="image",
    mime_type="image/gif",
    name="output.gif",
    # The WebM and the matte are the result. A run without the GIF still succeeds.
    optional=True,
)
# The order is stored. The UI reads the WebM first, the matte second and the GIF third.
CUTOUT_OUTPUT_PLANS = (CUTOUT_OUTPUT_PLAN, MATTE_OUTPUT_PLAN, GIF_OUTPUT_PLAN)

# The GIF fits in a box of this size. A smaller clip keeps its size.
GIF_MAX_EDGE = 720


def require_cutout_outputs(
    outputs: tuple[OutputAllocation, ...],
) -> tuple[OutputAllocation, OutputAllocation, OutputAllocation]:
    """The cutout the user sees, the raw matte the model writes, then the GIF."""
    if tuple(output.plan.slot for output in outputs) != tuple(
        plan.slot for plan in CUTOUT_OUTPUT_PLANS
    ):
        raise ValueError("a cutout recipe requires 'output', 'matte' and 'gif' allocations")
    return outputs[0], outputs[1], outputs[2]


class _MatteGeometry(NamedTuple):
    width: int
    height: int
    fps: float
    frames: int
    timeout: float


def _matte_geometry(matte: Path) -> _MatteGeometry:
    _kind, _mime, probed = probe_file(matte)
    if probed.mediaType != "video":
        raise MediaError("UNREADABLE_MEDIA", "the matte is not a video")
    return _MatteGeometry(
        width=probed.metadata.width,
        height=probed.metadata.height,
        fps=video_fps(matte),
        frames=video_frame_count(matte),
        timeout=timeout_for_source_duration(probed.metadata.durationMs / 1000),
    )


def _alpha_merge_filters(geometry: _MatteGeometry) -> list[str]:
    """The source with the matte as its alpha, as the straight RGBA stream ``[rgba]``."""
    return [
        f"[0:v]scale={geometry.width}:{geometry.height}:flags=lanczos,fps={geometry.fps:g},"
        "setpts=PTS-STARTPTS,format=rgba[rgb]",
        # format=gray already maps the matte's limited range (Y 16..235) to full range. A
        # second expansion would crush soft alpha such as hair and glass edges.
        "[1:v]format=gray,setpts=PTS-STARTPTS[alpha]",
        # shortest=1 ends one frame early on two streams of the same length. The matte is
        # already cut to the clip, so the output is capped at the matte length instead
        # (`-frames:v` in `_run_bake`).
        "[rgb][alpha]alphamerge[rgba]",
    ]


def _run_bake(
    source: Path,
    matte: Path,
    filters: list[str],
    output_args: list[str],
    geometry: _MatteGeometry,
    failure: str,
) -> None:
    args = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(source),
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(matte),
        "-filter_complex",
        ";".join(filters),
        "-frames:v",
        str(geometry.frames),
        *output_args,
    ]
    try:
        run_ffmpeg(
            args,
            timeout=geometry.timeout,
            # A cancel during the encode stops ffmpeg instead of waiting for it.
            should_cancel=is_requested,
        )
    except FfmpegCancelledError:
        raise GenerationCancelledError() from None
    except subprocess.CalledProcessError as exc:
        # A bake miss must not read as a crop failure at the caller.
        raise MediaError("UNREADABLE_MEDIA", failure) from exc


def bake_alpha_cutout(source: Path, matte: Path, dest: Path) -> None:
    """Write ``dest`` (WebM, VP9 ``yuva420p``). The size and rate follow the matte.

    The source is scaled to the matte, which the model sized inside the 540p envelope.
    Audio comes from the matte file, where the model already placed the chosen audio.
    """
    geometry = _matte_geometry(matte)
    _run_bake(
        source,
        matte,
        _alpha_merge_filters(geometry),
        [
            "-map",
            "[rgba]",
            "-map",
            "1:a?",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-auto-alt-ref",
            "0",
            "-b:v",
            "0",
            "-crf",
            "18",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-row-mt",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
            str(dest),
        ],
        geometry,
        "alpha cutout failed",
    )


def bake_alpha_gif(source: Path, matte: Path, dest: Path) -> None:
    """Write ``dest`` (GIF, looping, no audio) of the same cutout as the WebM.

    One palette is built from the whole clip, with a slot kept for transparency. A GIF
    pixel is clear or solid, so alpha under 128 becomes clear. The clip is not scaled up.
    """
    geometry = _matte_geometry(matte)
    _run_bake(
        source,
        matte,
        [
            *_alpha_merge_filters(geometry),
            f"[rgba]scale='min({GIF_MAX_EDGE},iw)':'min({GIF_MAX_EDGE},ih)'"
            ":force_original_aspect_ratio=decrease:flags=lanczos,split[frames][palette_in]",
            "[palette_in]palettegen=reserve_transparent=1[palette]",
            "[frames][palette]paletteuse=alpha_threshold=128[gif]",
        ],
        ["-map", "[gif]", "-an", "-c:v", "gif", "-loop", "0", str(dest)],
        geometry,
        "alpha gif failed",
    )
    # A GIF that ffmpeg wrote but nothing can read must not reach the store.
    probe_file(dest)


def bake_alpha_gif_if_possible(source: Path, matte: Path, dest: Path) -> None:
    """Write the GIF, or leave no file. The run keeps the WebM and the matte.

    A cancel still stops the run. Only a media failure of the GIF itself is skipped.
    """
    try:
        bake_alpha_gif(source, matte, dest)
    except MediaError:
        logger.warning("alpha gif skipped for %s", dest.name, exc_info=True)
        dest.unlink(missing_ok=True)
