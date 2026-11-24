"""Encode-window math for Home/Remote retake and extend.

GenSpace callers omit ``max_input_frames`` on extend and never call retake
windowing — they still encode the whole source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

from api_types import ExtendMode
from runtime_config.video_job_budget import VIDEO_JOB_TOO_LARGE
from services.records import CapabilityFailedError

MAX_INPUT_FRAMES = 505
VAE_TEMPORAL_DIVISOR = 8
MASK_DELTA_SECONDS = 0.5
RETAKE_BLEND_SECONDS = 0.3
# Minimum context an extension must leave inside the cap: 121 frames (~5s),
# 8k+1-aligned so the guarantee survives the VAE snap exactly. A product
# coherence floor (approved default), not a measurement: below this the model
# extends from slivers. Longer requests are refused with the max named.
MIN_EXTEND_CONTEXT_FRAMES = 121
# ltxv-api ``stitch_frames`` lerp band (``MAX_FADE_FRAMES = 16``).
MAX_FADE_FRAMES = 16
_BLEND_GUARD_FRAMES = 1
# Out-of-range refusal, shared with the Home retake advisory
# (frontend/ltx-io/screens/Feature/definitions/retake.ts): the advisory must
# read identically to create-time enforcement. Pinned by
# test_retake_out_of_range_matches_home_advisory_copy.
SELECTION_OUT_OF_RANGE_MESSAGE = "Selection is outside the usable video range"
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


def max_extend_frames(frame_cap: int) -> int:
    """Longest extension that preserves the minimum context inside the cap."""
    return frame_cap - MIN_EXTEND_CONTEXT_FRAMES


def extend_exceeds_cap(
    extend_frames: int, frame_cap: int, minimum_context_frames: int = MIN_EXTEND_CONTEXT_FRAMES
) -> bool:
    """True when the extension would leave less than the minimum context."""
    return extend_frames > frame_cap - minimum_context_frames


def extend_too_long_message(
    *,
    duration_s: float,
    target_width: int,
    target_height: int,
    frame_cap: int,
    fps: float,
) -> str:
    """Refusal naming the longest extension this machine fits at this size.

    Shared by the execute-time executor and the create-time preflight: the
    verdict must read identically in both, since either may be the one the
    user sees (create normally, execute on a free-VRAM race). At the hard
    cap the 121-frame context rule binds on every machine, so the copy
    stays machine-neutral; a shortened machine cap keeps "on this machine".
    """
    longest = max(0, max_extend_frames(frame_cap))
    detail = f" (at most {int(longest / fps)}s)" if fps > 0 else ""
    if frame_cap >= MAX_INPUT_FRAMES:
        return (
            f"Extend of {duration_s:g}s is too long for "
            f"{target_width}x{target_height}{detail}."
        )
    return (
        f"Extend of {duration_s:g}s is too long for "
        f"{target_width}x{target_height} on this machine{detail}."
    )


def retake_selection_too_long_message(
    *,
    duration_s: float,
    target_width: int,
    target_height: int,
    frame_cap: int,
    fps: float,
) -> str:
    """Refusal naming the longest selection this machine fits at this size.

    The named max floors below the cap: a mask exactly the cap with leftover
    source still raises in window_for_retake, so frame_cap itself is not an
    achievable promise. Same create/execute sharing rationale as extend.
    """
    longest = max(0, frame_cap - 1)
    detail = f" (at most {int(longest / fps)}s)" if fps > 0 else ""
    return (
        f"Retake selection of {duration_s:g}s is too long for "
        f"{target_width}x{target_height} on this machine{detail}."
    )


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


@dataclass(frozen=True)
class EditRefusal:
    """Window/math refusal shared by execute and preflight verdicts.

    ``cause`` chains the window ValueError; refusals without one (out-of-range
    selections, cap overruns) raise plain on both paths, as before.
    """

    message: str
    code: str
    cause: BaseException | None = None

    def raise_capability_error(self) -> NoReturn:
        """Raise as the executor's failure, preserving ``from exc`` chains."""
        if self.cause is not None:
            raise CapabilityFailedError(self.message, code=self.code) from self.cause
        raise CapabilityFailedError(self.message, code=self.code)


@dataclass(frozen=True)
class PreparedRetakeWindow:
    window: RetakeEncodeWindow
    end_time: float


@dataclass(frozen=True)
class PreparedExtendWindow:
    window: ExtendEncodeWindow
    extend_frames: int


def prepare_retake_window(
    *,
    fps: float,
    target_frames: int,
    target_width: int,
    target_height: int,
    frame_cap: int,
    start_time: float,
    duration: float,
) -> PreparedRetakeWindow | EditRefusal:
    """Mask + window math, one implementation for execute and preflight.

    The executor raises refusals via ``raise_capability_error``; preflights
    return them as failing verdicts. Budget stays per-side (the executor
    loads the pipeline, preflights decide without loading), as do metadata
    reads and size/cap resolution (the matrix shares those per candidate).
    """
    corrected_duration = target_frames / fps
    end_time = min(start_time + duration, corrected_duration)
    if start_time >= end_time:
        return EditRefusal(
            message=SELECTION_OUT_OF_RANGE_MESSAGE,
            code="INVALID_GENERATION_SPEC",
        )
    mask_start_frame = max(0, min(target_frames, round(start_time * fps)))
    mask_end_frame = max(
        mask_start_frame + 1,
        min(target_frames, round(end_time * fps)),
    )
    try:
        window = window_for_retake(
            source_frames=target_frames,
            mask_start_frame=mask_start_frame,
            mask_end_frame=mask_end_frame,
            max_input_frames=frame_cap,
            fps=fps,
        )
    except ValueError as exc:
        # A short selection on a long clip used to pad to 505 and 422. If the
        # selection itself is larger than the window that fits, say so as a
        # resolution limit (a shortened cap means the machine) rather than a
        # mask-size error (a full cap means past the hard 505 window).
        if frame_cap < MAX_INPUT_FRAMES:
            return EditRefusal(
                message=retake_selection_too_long_message(
                    duration_s=duration,
                    target_width=target_width,
                    target_height=target_height,
                    frame_cap=frame_cap,
                    fps=fps,
                ),
                code=VIDEO_JOB_TOO_LARGE,
                cause=exc,
            )
        return EditRefusal(message=str(exc), code="INVALID_GENERATION_SPEC", cause=exc)
    return PreparedRetakeWindow(window=window, end_time=end_time)


def prepare_extend_window(
    *,
    fps: float,
    source_frames: int,
    target_width: int,
    target_height: int,
    frame_cap: int,
    duration: float,
    mode: ExtendMode,
    minimum_context_frames: int = MIN_EXTEND_CONTEXT_FRAMES,
) -> PreparedExtendWindow | EditRefusal:
    """Extend window math, one implementation for execute and preflight.

    Same sharing rationale as ``prepare_retake_window``. ``minimum_context_frames``
    is the coherence floor for admission; the shrink retry passes 1 so an
    admitted job salvages a run with reduced (far-side-trimmed) context
    instead of failing when free VRAM dips below the full-context ask.
    """
    extend_frames = duration_to_extend_frames(duration, fps)
    # The extension must leave the minimum context inside the window
    # (MIN_EXTEND_CONTEXT_FRAMES): without the floor, long extensions
    # silently extend from slivers.
    if extend_exceeds_cap(extend_frames, frame_cap, minimum_context_frames):
        return EditRefusal(
            message=extend_too_long_message(
                duration_s=duration,
                target_width=target_width,
                target_height=target_height,
                frame_cap=frame_cap,
                fps=fps,
            ),
            code=VIDEO_JOB_TOO_LARGE,
        )
    try:
        window = window_for_extend(
            source_frames=source_frames,
            extend_frames=extend_frames,
            mode=mode,
            max_input_frames=frame_cap,
        )
    except ValueError as exc:
        return EditRefusal(message=str(exc), code="INVALID_GENERATION_SPEC", cause=exc)
    return PreparedExtendWindow(window=window, extend_frames=extend_frames)


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
