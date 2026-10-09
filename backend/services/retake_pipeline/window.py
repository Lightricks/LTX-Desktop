"""Encode-window math for Home/Remote retake and extend.

GenSpace callers omit ``max_input_frames`` on extend and never call retake
windowing — they still encode the whole source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from api_types import ExtendMode

MAX_INPUT_FRAMES = 505
VAE_TEMPORAL_DIVISOR = 8
MASK_DELTA_SECONDS = 0.5
RETAKE_BLEND_SECONDS = 0.3
# ltxv-api ``stitch_frames`` lerp band (``MAX_FADE_FRAMES = 16``).
MAX_FADE_FRAMES = 16
_BLEND_GUARD_FRAMES = 1
# The denoise mask covers whole latents, so it spills up to 7 frames past the
# feathered region, and the decoder bleeds about one more latent. The encode
# window keeps that much context so the 16-frame lerp has original frames
# beside the regenerated span.
LATENT_ROUNDING_FRAMES = VAE_TEMPORAL_DIVISOR - 1
STITCH_GUARD_FRAMES = VAE_TEMPORAL_DIVISOR


@dataclass(frozen=True)
class ExtendEncodeWindow:
    context_frames: int
    encode_start_frame: int
    dropped_prefix_frames: int
    dropped_suffix_frames: int

    def start_time(self, fps: float) -> float:
        return self.encode_start_frame / fps

    def max_duration(self, fps: float) -> float:
        return self.context_frames / fps


@dataclass(frozen=True)
class RetakeEncodeWindow:
    encode_start_frame: int
    encode_frames: int
    mask_start_frame: int
    mask_end_frame: int
    context_before_frames: int
    context_after_frames: int
    source_frames: int

    def encode_start_time(self, fps: float) -> float:
        return self.encode_start_frame / fps

    def encode_duration(self, fps: float) -> float:
        return self.encode_frames / fps

    def window_mask_start(self, fps: float) -> float:
        return (self.mask_start_frame - self.encode_start_frame) / fps

    def window_mask_end(self, fps: float) -> float:
        return (self.mask_end_frame - self.encode_start_frame) / fps


def vae_align_down(length: int) -> int:
    """Largest positive ``8 * k + 1`` that is ``<= length``."""
    if length <= 1:
        return 1
    return length - ((length - 1) % VAE_TEMPORAL_DIVISOR)


def duration_to_extend_frames(duration: float, fps: float) -> int:
    """Snap extend duration up to a multiple of 8 so the padded output stays 8k+1."""
    frames = round(duration * fps)
    return (
        (frames + VAE_TEMPORAL_DIVISOR - 1) // VAE_TEMPORAL_DIVISOR
    ) * VAE_TEMPORAL_DIVISOR


def window_for_extend(
    *,
    source_frames: int,
    extend_frames: int,
    mode: ExtendMode,
    max_input_frames: int | None = None,
) -> ExtendEncodeWindow:
    if source_frames <= 0 or extend_frames <= 0:
        raise ValueError("source_frames and extend_frames must be positive")
    if max_input_frames is None:
        return ExtendEncodeWindow(
            context_frames=source_frames,
            encode_start_frame=0,
            dropped_prefix_frames=0,
            dropped_suffix_frames=0,
        )
    if extend_frames >= max_input_frames:
        raise ValueError(
            f"Extend duration ({extend_frames} frames) exceeds limit ({max_input_frames})"
        )
    context_budget = max_input_frames - extend_frames
    context_frames = vae_align_down(min(source_frames, context_budget))
    if context_frames <= 0:
        raise ValueError(
            f"Extend duration ({extend_frames} frames) exceeds limit ({max_input_frames})"
        )
    dropped = max(0, source_frames - context_frames)
    if mode == "end":
        return ExtendEncodeWindow(
            context_frames=context_frames,
            encode_start_frame=dropped,
            dropped_prefix_frames=dropped,
            dropped_suffix_frames=0,
        )
    return ExtendEncodeWindow(
        context_frames=context_frames,
        encode_start_frame=0,
        dropped_prefix_frames=0,
        dropped_suffix_frames=dropped,
    )


def window_for_retake(
    *,
    source_frames: int,
    mask_start_frame: int,
    mask_end_frame: int,
    max_input_frames: int | None = MAX_INPUT_FRAMES,
    fps: float | None = None,
) -> RetakeEncodeWindow:
    if source_frames <= 0:
        raise ValueError("source_frames must be positive")
    mask_start_frame = max(0, min(source_frames, mask_start_frame))
    mask_end_frame = max(mask_start_frame, min(source_frames, mask_end_frame))
    mask_frames = mask_end_frame - mask_start_frame
    if mask_frames <= 0:
        raise ValueError("mask region must contain at least one frame")

    cap = source_frames if max_input_frames is None else min(source_frames, max_input_frames)
    leftover_source = mask_start_frame > 0 or mask_end_frame < source_frames
    if mask_frames > cap or (mask_frames == cap and leftover_source):
        raise ValueError(
            f"Mask region ({mask_frames} frames) too large for max segment size ({cap} frames)"
        )

    available_before = mask_start_frame
    available_after = source_frames - mask_end_frame
    if mask_frames >= cap:
        return RetakeEncodeWindow(
            encode_start_frame=mask_start_frame,
            encode_frames=vae_align_down(mask_frames),
            mask_start_frame=mask_start_frame,
            mask_end_frame=mask_end_frame,
            context_before_frames=0,
            context_after_frames=0,
            source_frames=source_frames,
        )

    per_side = max(0, (cap - mask_frames) // 2)
    context_before = min(per_side, available_before)
    context_after = min(per_side, available_after)

    if fps is not None and fps > 0:
        min_blend = math.ceil((RETAKE_BLEND_SECONDS + MASK_DELTA_SECONDS) * fps)
        min_blend += LATENT_ROUNDING_FRAMES + STITCH_GUARD_FRAMES + _BLEND_GUARD_FRAMES
        grown_before = context_before
        grown_after = context_after
        if context_before > 0:
            grown_before = max(context_before, min(min_blend, available_before))
        if context_after > 0:
            grown_after = max(context_after, min(min_blend, available_after))
        if grown_before + mask_frames + grown_after <= cap:
            context_before = grown_before
            context_after = grown_after

    leftover = cap - (context_before + mask_frames + context_after)
    grow_after = min(leftover, available_after - context_after)
    context_after += max(0, grow_after)
    leftover -= max(0, grow_after)
    grow_before = min(leftover, available_before - context_before)
    context_before += max(0, grow_before)

    context_before, context_after = _align_context(
        context_before,
        mask_frames,
        context_after,
        available_before,
        available_after,
        cap,
    )
    encode_frames = context_before + mask_frames + context_after
    encode_start = mask_start_frame - context_before
    return RetakeEncodeWindow(
        encode_start_frame=encode_start,
        encode_frames=encode_frames,
        mask_start_frame=mask_start_frame,
        mask_end_frame=mask_end_frame,
        context_before_frames=context_before,
        context_after_frames=context_after,
        source_frames=source_frames,
    )


def feathered_mask_times(window: RetakeEncodeWindow, fps: float) -> tuple[float, float]:
    """Widen the denoise mask MASK_DELTA into kept context (Home extend seam)."""
    start = window.window_mask_start(fps)
    end = window.window_mask_end(fps)
    duration = window.encode_duration(fps)
    if window.context_before_frames > 0:
        start = max(0.0, start - MASK_DELTA_SECONDS)
    if window.context_after_frames > 0:
        end = min(duration, end + MASK_DELTA_SECONDS)
    return start, end


def _latent_frame_span(latent: int) -> tuple[int, int]:
    """Window frames ``[start, end)`` one latent decodes to (causal: latent 0 is frame 0)."""
    if latent == 0:
        return 0, 1
    return VAE_TEMPORAL_DIVISOR * latent - (VAE_TEMPORAL_DIVISOR - 1), (
        VAE_TEMPORAL_DIVISOR * latent + 1
    )


def denoised_frame_span(window: RetakeEncodeWindow, fps: float) -> tuple[int, int]:
    """Window frames ``[start, end)`` the retake actually regenerates.

    ``TemporalRegionMask`` denoises every latent whose frames overlap the
    feathered region, so the span snaps outward to latent boundaries.
    """
    region_start, region_end = feathered_mask_times(window, fps)
    latents = (window.encode_frames - 1) // VAE_TEMPORAL_DIVISOR + 1
    spans = [
        span
        for span in map(_latent_frame_span, range(latents))
        if span[1] / fps > region_start and span[0] / fps < region_end
    ]
    if not spans:
        return window.mask_start_frame - window.encode_start_frame, (
            window.mask_end_frame - window.encode_start_frame
        )
    return spans[0][0], min(window.encode_frames, spans[-1][1])


def _align_context(
    context_before: int,
    mask_frames: int,
    context_after: int,
    available_before: int,
    available_after: int,
    cap: int,
) -> tuple[int, int]:
    total = context_before + mask_frames + context_after
    remainder = (total - 1) % VAE_TEMPORAL_DIVISOR
    extra = 0 if remainder == 0 else VAE_TEMPORAL_DIVISOR - remainder
    for _ in range(extra):
        if total >= cap:
            break
        if context_after < available_after:
            context_after += 1
            total += 1
        elif context_before < available_before:
            context_before += 1
            total += 1
        else:
            break
    aligned = vae_align_down(context_before + mask_frames + context_after)
    overflow = context_before + mask_frames + context_after - aligned
    while overflow > 0 and context_after > 0:
        context_after -= 1
        overflow -= 1
    while overflow > 0 and context_before > 0:
        context_before -= 1
        overflow -= 1
    return context_before, context_after
