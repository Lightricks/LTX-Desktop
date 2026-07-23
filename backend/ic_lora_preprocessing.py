"""IC-LoRA preprocessing pipeline: turn user input into an IC-LoRA control video."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from api_types import PreprocessingStep
from frame_math import snap_to_frame_grid
from services.interfaces import VideoProcessor


@dataclass(frozen=True, slots=True)
class MediaArtifact:
    path: str
    kind: Literal["image", "video"]
    fps: float | None = None
    frame_count: int | None = None
    # Outpainting only: path to the keep=white/pad=black conditioning-attention mask video.
    mask_path: str | None = None


@dataclass(frozen=True, slots=True)
class OutpaintParams:
    # Pixels to add on each edge of the source. Extend-only (all ≥ 0, no crop); the UI emits
    # these directly from the canvas editor (aspect presets are computed front-end into pads).
    left: int
    right: int
    top: int
    bottom: int


@dataclass
class PreprocessingContext:
    num_frames: int
    outputs_dir: Path
    video_processor: VideoProcessor
    outpaint: OutpaintParams | None = None


PreprocessingUtility = Callable[[MediaArtifact, dict[str, object], PreprocessingContext], MediaArtifact]


def _coerce_frame_count(n: int) -> int:
    # Pipeline requires (n - 1) % 8 == 0. floor=1: a 1-frame image still yields a valid clip.
    return snap_to_frame_grid(n, floor=1)


def _image_to_frames(
    artifact: MediaArtifact, params: dict[str, object], ctx: PreprocessingContext
) -> MediaArtifact:
    if artifact.kind != "image":
        raise ValueError("image_to_frames requires an image input")
    fps = float(params.get("fps", 24))  # type: ignore[arg-type]
    count = _coerce_frame_count(ctx.num_frames)
    frame = ctx.video_processor.read_image(artifact.path)
    h, w = frame.shape[0], frame.shape[1]
    out_path = str(ctx.outputs_dir / f"_ic_lora_frames_{uuid.uuid4().hex[:8]}.mp4")
    writer = ctx.video_processor.create_writer(out_path, fourcc="mp4v", fps=fps, size=(int(w), int(h)))
    for _ in range(count):
        writer.write(frame)
    ctx.video_processor.release(writer)
    return MediaArtifact(path=out_path, kind="video", fps=fps, frame_count=count)


def _outpaint_canvas(
    artifact: MediaArtifact, params: dict[str, object], ctx: PreprocessingContext
) -> MediaArtifact:
    """Center the source video on a larger green canvas and emit a keep/pad mask video.

    The pad region is filled chroma-green (the training distribution) and the mask marks
    the kept region white (preserve the reference) and the pad black (generate freely) — the
    polarity the upstream `conditioning_attention_mask` expects. The mask is static across
    time and gets the same frame count as the conditioning video.
    """
    del params  # pads come from ctx.outpaint, not static catalog params
    if artifact.kind != "video":
        raise ValueError("outpaint_canvas requires a video input")
    if ctx.outpaint is None:
        raise ValueError("outpaint_canvas requires outpaint params on the context")

    cap = ctx.video_processor.open_video(artifact.path)
    cond_writer = None
    mask_writer = None
    try:
        info = ctx.video_processor.get_video_info(cap)
        src_w, src_h = int(info["width"]), int(info["height"])
        if src_w <= 0 or src_h <= 0:
            raise ValueError("could not read source video dimensions for outpainting")
        fps = float(info["fps"]) or 24.0
        frame_count = int(info["frame_count"])

        left, right, top, bottom = (ctx.outpaint.left, ctx.outpaint.right, ctx.outpaint.top, ctx.outpaint.bottom)
        # Defense in depth (the canvas UI already enforces this): extend-only, no crop, and each
        # edge capped at +100% of its axis so a stray request can't blow up the canvas.
        if min(left, right, top, bottom) < 0:
            raise ValueError("outpaint pads must be non-negative (extend-only, no crop)")
        if left > src_w or right > src_w or top > src_h or bottom > src_h:
            raise ValueError("outpaint pad exceeds +100% of the source dimension")
        new_w, new_h = src_w + left + right, src_h + top + bottom
        # mp4 encoders want even dimensions — absorb any odd pixel into the trailing pad.
        if new_w % 2:
            right += 1
            new_w += 1
        if new_h % 2:
            bottom += 1
            new_h += 1

        cond_path = str(ctx.outputs_dir / f"_outpaint_{uuid.uuid4().hex[:8]}.mp4")
        mask_path = str(ctx.outputs_dir / f"_outpaint_mask_{uuid.uuid4().hex[:8]}.mp4")
        cond_writer = ctx.video_processor.create_writer(cond_path, fourcc="mp4v", fps=fps, size=(new_w, new_h))
        mask_writer = ctx.video_processor.create_writer(mask_path, fourcc="mp4v", fps=fps, size=(new_w, new_h))

        # The keep/pad mask is static across time — write one frame; _load_mask_tensor repeats
        # it to the clip length, so we avoid encoding N identical frames.
        mask_frame = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        mask_frame[top : top + src_h, left : left + src_w] = 255
        mask_writer.write(mask_frame)

        written = 0
        for _ in range(frame_count):
            frame = ctx.video_processor.read_frame(cap)
            if frame is None:
                break
            # Rotated / VFR / corrupt clips can decode frames that don't match the probed size;
            # stop cleanly rather than letting the paste raise a broadcast error mid-write.
            if frame.shape[:2] != (src_h, src_w):
                break
            canvas = np.empty((new_h, new_w, 3), dtype=np.uint8)
            canvas[:] = (0, 255, 0)  # BGR chroma green in the pad region
            canvas[top : top + src_h, left : left + src_w] = frame
            cond_writer.write(canvas)
            written += 1

        return MediaArtifact(path=cond_path, kind="video", fps=fps, frame_count=written, mask_path=mask_path)
    finally:
        ctx.video_processor.release(cap)
        if cond_writer is not None:
            ctx.video_processor.release(cond_writer)
        if mask_writer is not None:
            ctx.video_processor.release(mask_writer)


PREPROCESSING_UTILITIES: dict[str, PreprocessingUtility] = {
    "image_to_frames": _image_to_frames,
    "outpaint_canvas": _outpaint_canvas,
}


def run_preprocessing(
    steps: list[PreprocessingStep],
    input_artifact: MediaArtifact,
    ctx: PreprocessingContext,
) -> MediaArtifact:
    artifact = input_artifact
    for step in steps:
        utility = PREPROCESSING_UTILITIES[step.utility]  # KeyError on unknown utility
        artifact = utility(artifact, dict(step.params), ctx)
    if artifact.kind != "video":
        raise ValueError("Preprocessing pipeline must end in a video artifact")
    return artifact
