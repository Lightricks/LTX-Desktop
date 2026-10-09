"""Desktop LTX offering capabilities: feature flags + 16:9 pixel maps.

Duration/fps envelopes stay on LTXVideoGenerationSpec. This matrix is the
feature/pixel SSOT those specs do not have. Local 2.5 is the on-device distilled
offering; its flags are independent of the API Fast rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, assert_never

from api_types import (
    LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    LTXLocalModelId,
    LTXVideoGenAspectRatio,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
)

LtxCapabilityFeature = Literal[
    "t2v",
    "i2v",
    "a2v",
    "ic_lora",
    "retake",
    "extend",
    "multi_keyframe",
    "user_loras",
    "camera_motion",
    "auto_duration",
]
LtxAspectRatio = LTXVideoGenAspectRatio


@dataclass(frozen=True)
class LtxOfferingCapabilities:
    t2v: bool
    i2v: bool
    a2v: bool
    ic_lora: bool
    retake: bool
    extend: bool
    multi_keyframe: bool
    multi_keyframe_max_count: int
    user_loras: bool
    camera_motion: bool
    # t2v/i2v only: send duration=null and the cloud worker picks length from the prompt.
    auto_duration: bool
    # Label → (width, height) for 16:9. Other ratios are derived in pixels_for().
    resolution_pixels_16_9: dict[LTXVideoGenResolution, tuple[int, int]]


@dataclass(frozen=True)
class LocalOfferingCapabilities(LtxOfferingCapabilities):
    """Same shape as the base class — this exists purely so local_caps()'s return
    type can't be confused with api_caps()'s at the type-checker level."""


@dataclass(frozen=True)
class ApiOfferingCapabilities(LtxOfferingCapabilities):
    """Same shape as the base class — this exists purely so api_caps()'s return
    type can't be confused with local_caps()'s at the type-checker level."""


# Real two-pass canvases. Splitting 540p (2.3 960×544 vs 2.5 1024×576) made a
# model switch fail assert_resolution, so 2.3 and 2.5 share this grid.
_TWO_PASS_PIXELS_16_9: dict[LTXVideoGenResolution, tuple[int, int]] = {
    "540p": (1024, 576),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
}

# Text and image labels that denoise the parent canvas and return the first pass.
_STAGE1_PARENT: dict[LTXVideoGenResolution, LTXVideoGenResolution] = {
    "270p": "540p",
    "360p": "720p",
}

# A2V finishes both passes, so these are the picture, not the text-to-video parent.
_A2V_TWO_PASS_PIXELS_16_9: dict[LTXVideoGenResolution, tuple[int, int]] = {
    "270p": (576, 320),
    "360p": (704, 384),
}


def _local_pixel_map() -> dict[LTXVideoGenResolution, tuple[int, int]]:
    pixels = dict(_TWO_PASS_PIXELS_16_9)
    for label, parent in _STAGE1_PARENT.items():
        pixels[label] = _TWO_PASS_PIXELS_16_9[parent]
    return pixels


# 270p and 360p are the parent pair, not a second copy of those numbers.
_LOCAL_PIXELS_16_9: dict[LTXVideoGenResolution, tuple[int, int]] = _local_pixel_map()

CanvasMode = Literal["video", "a2v"]


@dataclass(frozen=True)
class LocalCanvas:
    width: int
    height: int
    skip_stage_2: bool


_API_PIXELS_16_9: dict[LTXVideoGenResolution, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}

_LOCAL_2_3 = LocalOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=True,
    retake=True,
    extend=True,
    multi_keyframe=True,
    multi_keyframe_max_count=LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    user_loras=True,
    camera_motion=True,
    auto_duration=False,
    resolution_pixels_16_9=_LOCAL_PIXELS_16_9,
)

# DistilledA2V is wired for local 2.5. Auto duration is DurationHead on the
# distilled checkpoint (t2v/i2v; A2V length comes from the audio). Advertised
# only when those weights are on disk — see effective_local_caps().
_LOCAL_2_5 = LocalOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=True,
    retake=True,
    extend=True,
    multi_keyframe=True,
    multi_keyframe_max_count=LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    user_loras=True,
    camera_motion=True,
    auto_duration=True,
    resolution_pixels_16_9=_LOCAL_PIXELS_16_9,
)

# API rows follow ltxv-api handlers. camera_motion is a named LoRA on the tia2v
# stack, not a Desktop capability on Fast.
_API_FAST = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=False,
    ic_lora=False,
    retake=False,
    extend=False,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=False,
    auto_duration=False,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)

_API_FAST_2_5 = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=False,
    retake=False,
    extend=False,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=False,
    auto_duration=True,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)

_API_PRO = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=False,
    retake=True,
    extend=True,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=True,
    auto_duration=False,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)

# ltxv-api retake/extend accept ltx-2-pro / ltx-2-3-pro. Auto duration is on both
# 2.5 API variants (t2v/i2v duration=null).
_API_PRO_2_5 = ApiOfferingCapabilities(
    t2v=True,
    i2v=True,
    a2v=True,
    ic_lora=False,
    retake=False,
    extend=False,
    multi_keyframe=False,
    multi_keyframe_max_count=0,
    user_loras=False,
    camera_motion=True,
    auto_duration=True,
    resolution_pixels_16_9=_API_PIXELS_16_9,
)


def local_caps(model_id: LTXLocalModelId) -> LocalOfferingCapabilities:
    match model_id:
        case "ltx-2.5-22b-distilled":
            return _LOCAL_2_5
        case "ltx-2.3-22b-distilled" | "ltx-2.3-22b-distilled-1.1":
            return _LOCAL_2_3
        case _:
            assert_never(model_id)


def effective_local_caps(
    model_id: LTXLocalModelId,
    *,
    duration_head_ready: bool,
) -> LocalOfferingCapabilities:
    """Static offering flags, with Auto duration on only when DurationHead weights are on disk.

    Local-only. API 2.5 Auto duration is independent of this file.
    """
    caps = local_caps(model_id)
    if caps.auto_duration and not duration_head_ready:
        return replace(caps, auto_duration=False)
    return caps


def api_caps(pipeline: LTXVideoGenPipeline) -> ApiOfferingCapabilities:
    match pipeline:
        case "fast":
            return _API_FAST
        case "fast-2.5":
            return _API_FAST_2_5
        case "pro":
            return _API_PRO
        case "pro-2.5":
            return _API_PRO_2_5
        case _:
            assert_never(pipeline)


def supports(caps: LtxOfferingCapabilities, feature: LtxCapabilityFeature) -> bool:
    match feature:
        case "t2v":
            return caps.t2v
        case "i2v":
            return caps.i2v
        case "a2v":
            return caps.a2v
        case "ic_lora":
            return caps.ic_lora
        case "retake":
            return caps.retake
        case "extend":
            return caps.extend
        case "multi_keyframe":
            return caps.multi_keyframe
        case "user_loras":
            return caps.user_loras
        case "camera_motion":
            return caps.camera_motion
        case "auto_duration":
            return caps.auto_duration
        case _:
            assert_never(feature)


_ASPECT_PARTS: dict[LtxAspectRatio, tuple[int, int]] = {
    "21:9": (21, 9),
    "16:9": (16, 9),
    "3:2": (3, 2),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "4:5": (4, 5),
    "9:16": (9, 16),
}


def _snap_nearest_64(value: int) -> int:
    """Nearest multiple of 64. An exact halfway rounds up.

    Always rounding up turned a 4:5 long side of 720 into 768, which is 4:3.
    Halfway still rounds up so a *.5 multiple cannot shrink the way Python's
    half-to-even round did on the two-stage grid.
    """
    remainder = value % 64
    if remainder == 0:
        return value
    if remainder < 32:
        return value - remainder
    return value + (64 - remainder)


def _pixels_from_anchor(
    anchor: tuple[int, int],
    aspect: LtxAspectRatio,
    *,
    snap: bool,
) -> tuple[int, int]:
    """Short side is the 16:9 height. 16:9 itself stays the stored anchor."""
    if aspect == "16:9":
        return anchor
    short = anchor[1]
    width_part, height_part = _ASPECT_PARTS[aspect]
    if width_part >= height_part:
        height = short
        width = round(short * width_part / height_part)
    else:
        width = short
        height = round(short * height_part / width_part)
    if snap:
        return _snap_nearest_64(width), _snap_nearest_64(height)
    return width, height


def local_canvas(resolution: LTXVideoGenResolution, *, mode: CanvasMode) -> LocalCanvas:
    """One record for a local label. Video 270p/360p skip stage 2 on the parent canvas."""
    if resolution not in _LOCAL_PIXELS_16_9:
        raise KeyError(resolution)
    if mode == "a2v":
        override = _A2V_TWO_PASS_PIXELS_16_9.get(resolution)
        if override is not None:
            width, height = override
            return LocalCanvas(width, height, skip_stage_2=False)
    width, height = _LOCAL_PIXELS_16_9[resolution]
    return LocalCanvas(
        width,
        height,
        skip_stage_2=mode == "video" and resolution in _STAGE1_PARENT,
    )


def budget_size(canvas: LocalCanvas) -> tuple[int, int]:
    """Pixels the job denoises. Stage-1-only labels are half the stored canvas."""
    if canvas.skip_stage_2:
        return canvas.width // 2, canvas.height // 2
    return canvas.width, canvas.height


def budget_pixels(
    caps: LtxOfferingCapabilities,
    resolution: LTXVideoGenResolution,
    aspect: LtxAspectRatio,
    *,
    mode: CanvasMode,
) -> tuple[int, int]:
    """Denoised size for one ratio. Video 270p/360p are half the parent frame."""
    if mode == "a2v":
        width, height = a2v_pixels_for(caps, resolution, aspect)
    else:
        width, height = pixels_for(caps, resolution, aspect)
    if mode == "video" and resolution in _STAGE1_PARENT:
        return width // 2, height // 2
    return width, height


def pixels_for(
    caps: LtxOfferingCapabilities,
    resolution: LTXVideoGenResolution,
    aspect: LtxAspectRatio,
) -> tuple[int, int]:
    """Short side is the 16:9 height. 16:9 itself stays the stored anchor.

    API sizes are the rounded short-side formula so they match ltxv-api tiles.
    Local sizes then snap each edge to the nearest multiple of 64.
    """
    size = caps.resolution_pixels_16_9.get(resolution)
    if size is None:
        raise KeyError(resolution)
    return _pixels_from_anchor(
        size,
        aspect,
        snap=isinstance(caps, LocalOfferingCapabilities),
    )


def a2v_pixels_for(
    caps: LtxOfferingCapabilities,
    resolution: LTXVideoGenResolution,
    aspect: LtxAspectRatio,
) -> tuple[int, int]:
    """A2V 270p/360p finish at their own canvas. Other labels use pixels_for."""
    if resolution not in caps.resolution_pixels_16_9:
        raise KeyError(resolution)
    override = _A2V_TWO_PASS_PIXELS_16_9.get(resolution)
    if override is None:
        return pixels_for(caps, resolution, aspect)
    return _pixels_from_anchor(
        override,
        aspect,
        snap=isinstance(caps, LocalOfferingCapabilities),
    )
