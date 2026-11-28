"""Desktop budget estimates injected into headless window telemetry."""

from __future__ import annotations

import sys

from inference.job_budget import video_tokens
from runtime_config.video_job_budget import (
    decode_tile_tokens,
    edit_job_stream_gib,
    fast_job_stream_gib,
)


def fast_window_estimate_gib(
    *,
    width: int,
    height: int,
    frames: int,
    windows: int,
    last_frames: int,
) -> float:
    return fast_job_stream_gib(
        video_tokens(height, width, frames),
        windows=windows,
        darwin=sys.platform == "darwin",
        last_tokens=video_tokens(height, width, last_frames),
    )


def edit_window_estimate_gib(*, width: int, height: int, frames: int) -> float:
    return edit_job_stream_gib(
        video_tokens(height, width, frames),
        darwin=sys.platform == "darwin",
        decode_tokens=decode_tile_tokens(height, width, frames),
    )
