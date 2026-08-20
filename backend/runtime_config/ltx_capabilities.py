"""Desktop LTX offering capabilities: feature flags + 16:9 pixel maps.

Duration/fps envelopes stay on LTXVideoGenerationSpec. This matrix is the
feature/pixel SSOT those specs do not have. Local 2.5 is the on-device distilled
offering; its flags are independent of the API Fast rows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, assert_never

from api_types import LTXLocalModelId, LTXVideoGenPipeline, LTXVideoGenResolution

LtxCapabilityFeature = Literal[
    "t2v",
    "i2v",
    "a2v",
    "ic_lora",
    "retake",
    "extend",
    "user_loras",
    "camera_motion",
    "auto_duration",
]
LtxAspectRatio = Literal["16:9", "9:16"]


@dataclass(frozen=True)
class LtxOfferingCapabilities:
    t2v: bool
    i2v: bool
    a2v: bool
    ic_lora: bool
    retake: bool
    extend: bool
    user_loras: bool
    camera_motion: bool
    # t2v/i2v only: send duration=null and the cloud worker picks length from the prompt.
    auto_duration: bool
    # Label → (width, height) for 16:9; 9:16 is swapped at pixels_for().
    resolution_pixels_16_9: dict[LTXVideoGenResolution, tuple[int, int]]


@dataclass(frozen=True)
class LocalOfferingCapabilities(LtxOfferingCapabilities):
    """Same shape as the base class — this exists purely so local_caps()'s return
    type can't be confused with api_caps()'s at the type-checker level."""


@dataclass(frozen=True)
class ApiOfferingCapabilities(LtxOfferingCapabilities):
    """Same shape as the base class — this exists purely so api_caps()'s return
    type can't be confused with local_caps()'s at the type-checker level."""


# Local Fast sizes: one /64 two-stage grid for 2.3 and 2.5. Splitting 540p
# (2.3 960×544 vs 2.5 1024×576) made a model switch fail assert_resolution.
_LOCAL_PIXELS_16_9: dict[LTXVideoGenResolution, tuple[int, int]] = {
    "540p": (1024, 576),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
}

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
        case "user_loras":
            return caps.user_loras
        case "camera_motion":
            return caps.camera_motion
        case "auto_duration":
            return caps.auto_duration
        case _:
            assert_never(feature)


def pixels_for(
    caps: LtxOfferingCapabilities,
    resolution: LTXVideoGenResolution,
    aspect: LtxAspectRatio,
) -> tuple[int, int]:
    size = caps.resolution_pixels_16_9.get(resolution)
    if size is None:
        raise KeyError(resolution)
    width, height = size
    if aspect == "9:16":
        return height, width
    return width, height
