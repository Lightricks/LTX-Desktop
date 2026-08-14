"""Shared duration<->frame math for video generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoDurationSpec:
    """Ask DurationHead to pick length, clamped to this envelope (seconds)."""

    min_seconds: float
    max_seconds: float


def snap_up_to_multiple(n: int, multiple: int) -> int:
    """Round ``n`` up to the next multiple of ``multiple``.

    For pixel dimensions on a VAE/latent grid, where rounding down loses picture: a value
    already on the grid is returned unchanged.
    """
    return -(-n // multiple) * multiple


def snap_to_frame_grid(n: int, *, floor: int = 9) -> int:
    """Snap a frame count DOWN to the largest valid (n - 1) % 8 == 0 count, clamped to ``floor``.

    Use for snapping an existing/derived frame count to the grid. Distinct from
    compute_num_frames, which sizes a *duration* (it keeps duration*fps intervals and adds the
    trailing frame, so compute_num_frames(5, 24) == 121 where this would give 113).
    """
    return max(floor, ((max(1, n) - 1) // 8) * 8 + 1)


def compute_num_frames(duration_seconds: int, fps: int) -> int:
    """Frame count for a duration, snapped to the pipeline's (n - 1) % 8 == 0 grid."""
    n = ((duration_seconds * fps) // 8) * 8 + 1
    return max(n, 9)
