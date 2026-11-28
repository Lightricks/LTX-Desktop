"""Temporal windows for a 1440p or 4K Fast or A2V job: the layout, the peak-VRAM log, the streamed encode."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol

import torch

from inference.fast_chunking import CARRY_FRAMES, is_chunked_canvas, last_window_frames
from inference.frame_math import AutoDurationSpec, compute_num_frames
from inference.ic_lora_pipeline.ic_lora_inference import (
    InferenceResult,
    close_streams,
    plan_chunks,
)
from inference.common import encode_segments_output, video_chunks_number
from inference.phase_log import measured_gpu_phase

if TYPE_CHECKING:
    from ltx_pipelines.chunks import ChunkConfig

logger = logging.getLogger(__name__)

LOG_PREFIX = "[hires-video]"


class WindowEstimate(Protocol):
    def __call__(
        self,
        *,
        width: int,
        height: int,
        frames: int,
        windows: int,
        last_frames: int,
    ) -> float: ...


class EditEstimate(Protocol):
    def __call__(self, *, width: int, height: int, frames: int) -> float: ...


def _longest_frames(num_frames: int | AutoDurationSpec, frame_rate: float) -> int:
    """An automatic duration is counted at its longest."""
    if isinstance(num_frames, AutoDurationSpec):
        return compute_num_frames(int(num_frames.max_seconds), int(frame_rate))
    return num_frames


def plan_windows(
    chunk_pixel_frames: int | None,
    *,
    width: int,
    height: int,
    num_frames: int | AutoDurationSpec,
    frame_rate: float,
) -> tuple[ChunkConfig | None, int]:
    """(chunk config, window count)."""
    return plan_chunks(
        chunk_pixel_frames,
        width=width,
        height=height,
        num_frames=_longest_frames(num_frames, frame_rate),
        frame_rate=frame_rate,
        carry_frames=CARRY_FRAMES,
    )


def measured_windows(
    chunk_config: ChunkConfig | None,
    *,
    width: int,
    height: int,
    window_count: int,
    num_frames: int | AutoDurationSpec,
    frame_rate: float,
    device: torch.device,
    estimate_gib: WindowEstimate | None = None,
) -> AbstractContextManager[None]:
    """Phase and peak-VRAM log for a 1440p or 4K run, windowed or one pass. Other runs keep their own logs."""
    if chunk_config is None:
        if not is_chunked_canvas(width, height):
            return nullcontext()
        frames = _longest_frames(num_frames, frame_rate)
        last_frames = frames
        label = f"{width}x{height}, {frames} frames in one pass"
    else:
        frames = chunk_config.chunk_pixel_frames
        last_frames = last_window_frames(
            _longest_frames(num_frames, frame_rate),
            frames,
            chunk_config.next_video_carry_frames,
        )
        label = (
            f"{width}x{height} in {window_count} window(s) of {frames} frames "
            f"(carry {chunk_config.next_video_carry_frames})"
        )
    return measured_gpu_phase(
        logger,
        label,
        device=device,
        prefix=LOG_PREFIX,
        estimate_gib=(
            None
            if estimate_gib is None
            else estimate_gib(
                width=width,
                height=height,
                frames=frames,
                windows=window_count,
                last_frames=last_frames,
            )
        ),
    )


def measured_edit(
    width: int,
    height: int,
    frames: int,
    *,
    estimate_gib: EditEstimate | None = None,
) -> AbstractContextManager[None]:
    """Phase and peak-VRAM log for a 1440p or 4K Retake or Extend run. Other sizes keep their own logs."""
    if not is_chunked_canvas(width, height):
        return nullcontext()
    return measured_gpu_phase(
        logger,
        f"edit {width}x{height}, {frames} frames",
        device=torch.device("cuda"),
        prefix=LOG_PREFIX,
        estimate_gib=(
            None
            if estimate_gib is None
            else estimate_gib(width=width, height=height, frames=frames)
        ),
    )


def encode_result(result: InferenceResult, *, fps: int, output_path: str, streamed: bool) -> None:
    """Encode the video. A windowed job runs its windows while the encoder reads them."""
    try:
        encode_segments_output(
            result.segments,
            fps=fps,
            output_path=output_path,
            video_chunks_number_value=None if streamed else video_chunks_number(result.num_frames, result.tiling_config),
        )
    finally:
        close_streams(result)
