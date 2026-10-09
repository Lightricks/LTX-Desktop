"""Alignment for a cutout recipe. The matte must line up with the clip in time and in space.

Time: the pipeline snaps a frame count DOWN to 8k+1, so a 150 frame clip would lose 5 frames.
The model reads the clip with its last frame repeated, which lifts the snap UP to the grid.

Space: the pipeline fills the stage canvas with a center crop of the guide clip. A clip whose
shape differs from the canvas (the canvas sits on a 64 px grid) would show the model a smaller
view than the bake sees, and the matte would drift from the subject. So the clip is fitted
INTO the canvas with its edge pixels smeared into the border. The pipeline then only scales it.

After the run, the matte is cut back to the clip: the border off, the extra frames off.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from services.ffmpeg import (
    FFMPEG_PROTOCOL_WHITELIST,
    FfmpegCancelledError,
    ffmpeg_seconds,
    run_ffmpeg,
)
from services.generation_interrupt import GenerationCancelledError, is_requested
from services.media_probe import video_frame_count
from services.records import MediaError

# One frame grid step is 8 frames. Seven repeats lift any clip length to the next 8k+1,
# and the snap down drops the rest. A clip already on the grid keeps its length.
GRID_PAD_FRAMES = 7
# The matte is stored again when the border is cut off. Near lossless keeps soft alpha edges.
_MATTE_CRF = "10"


def _even(value: int) -> int:
    """yuv420p needs even sides and even offsets."""
    return value - value % 2


@dataclass(frozen=True, slots=True)
class CanvasFit:
    """Where the clip sits inside the stage canvas the model runs at."""

    canvas_width: int
    canvas_height: int
    width: int
    height: int
    x: int
    y: int

    @property
    def fills_canvas(self) -> bool:
        return self.width == self.canvas_width and self.height == self.canvas_height


def fit_clip_to_canvas(
    clip_width: int, clip_height: int, canvas_width: int, canvas_height: int
) -> CanvasFit:
    """The largest copy of the clip that fits the canvas, centered. The clip shape is kept."""
    scale = min(canvas_width / clip_width, canvas_height / clip_height)
    width = min(canvas_width, max(2, _even(round(clip_width * scale))))
    height = min(canvas_height, max(2, _even(round(clip_height * scale))))
    return CanvasFit(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        width=width,
        height=height,
        x=_even((canvas_width - width) // 2),
        y=_even((canvas_height - height) // 2),
    )


def _fit_filter(fit: CanvasFit) -> str:
    scale = f"scale={fit.width}:{fit.height}:flags=lanczos"
    if fit.fills_canvas:
        return scale
    right = fit.canvas_width - fit.x - fit.width
    bottom = fit.canvas_height - fit.y - fit.height
    return (
        f"{scale},pad={fit.canvas_width}:{fit.canvas_height}:{fit.x}:{fit.y},"
        f"fillborders=left={fit.x}:right={right}:top={fit.y}:bottom={bottom}:mode=smear"
    )


def _run(args: list[str], failure: str) -> None:
    try:
        run_ffmpeg(args, should_cancel=is_requested)
    except FfmpegCancelledError:
        raise GenerationCancelledError() from None
    except subprocess.CalledProcessError as exc:
        raise MediaError("UNREADABLE_MEDIA", failure) from exc


def prepare_clip_for_model(
    source: Path, dest: Path, *, fit: CanvasFit, keep_audio: bool
) -> Path:
    """Write ``dest``: the clip fitted into the canvas, with its last frame repeated.

    The guide stays lossless (``-qp 0``), so the model reads the same pixels the fit made.
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
        f"{_fit_filter(fit)},tpad=stop_mode=clone:stop={GRID_PAD_FRAMES}",
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ]
    if keep_audio:
        args += ["-map", "0:a?", "-c:a", "aac"]
    else:
        args += ["-an"]
    args += ["-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, str(dest)]
    _run(args, "clip preparation failed")
    return dest


def conform_matte_to_clip(matte: Path, *, frames: int, fps: float, fit: CanvasFit) -> None:
    """Rewrite the matte in place: ``frames`` frames, with the canvas border cut off.

    The video is cut by frame count. A time cut keeps extra frames, because a B-frame
    stream starts its timestamps late. The time limit only ends a source audio track.
    A cut encodes the video again. A stream copy cuts in decode order, and with
    B-frames that can leave a frame past the cut.
    """
    if fit.fills_canvas and video_frame_count(matte) == frames:
        # No border and no extra frame: nothing to cut, so keep the model file as it is.
        return
    conformed = matte.with_name(f"{matte.stem}-conform{matte.suffix}")
    video = [
        # A clip that fills the canvas has no border to cut.
        *([] if fit.fills_canvas else ["-vf", f"crop={fit.width}:{fit.height}:{fit.x}:{fit.y}"]),
        "-c:v",
        "libx264",
        "-crf",
        _MATTE_CRF,
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ]
    args = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(matte),
        "-map",
        "0",
        *video,
        "-c:a",
        "copy",
        "-frames:v",
        str(frames),
        "-t",
        ffmpeg_seconds(frames / fps),
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        str(conformed),
    ]
    try:
        _run(args, "matte alignment failed")
        os.replace(conformed, matte)
    finally:
        conformed.unlink(missing_ok=True)
