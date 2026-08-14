"""Canonical video-generation capability specs used by backend and frontend."""

from __future__ import annotations

from api_types import (
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    LTXLocalModelId,
    LTXOfferingCapabilitiesSpec,
    LTXVideoGenerationModelSpecItem,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
)
from runtime_config.ltx_capabilities import LtxOfferingCapabilities, api_caps, effective_local_caps
from runtime_config.model_download_specs import get_latest_ltx_model_id, get_ltx_model_spec

# The concrete ltxv-api model id each Desktop-facing pipeline maps to when forced onto
# the API backend. SSOT for this mapping — video_generation_handler and the
# retake/extend handlers all import it from here.
FORCED_API_MODEL_MAP: dict[str, str] = {
    "fast": "ltx-2-3-fast",
    "pro": "ltx-2-3-pro",
    "fast-2.5": "ltx-2-5-fast",
    "pro-2.5": "ltx-2-5-pro",
}


def _resolution_spec(
    *,
    fps_to_durations: dict[LTXVideoGenFps, tuple[LTXVideoGenDuration, ...]],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        },
    )


ltx_api_model_specs: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...] = (
    (
        "fast",
        LTXVideoGenerationSpec(
            display_name="LTX-2.3 Fast (API)",
            supported_resolutions_durations={
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10, 12, 14, 16, 18, 20),
                        25: (6, 8, 10, 12, 14, 16, 18, 20),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "1440p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "2160p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
            },
            # No A2V envelope: ltxv-api audio-to-video does not accept ltx-2-3-fast.
        ),
    ),
    (
        "pro",
        LTXVideoGenerationSpec(
            display_name="LTX-2.3 Pro (API)",
            supported_resolutions_durations={
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "1440p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "2160p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
            },
            a2v_supported_resolutions_durations={
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
            },
        ),
    ),
    # ltx-2-5-fast: t2v/i2v duration envelope matches API Fast. A2V is 1080p
    # (ltxv-api MAX_AUDIO_SECONDS for this model).
    (
        "fast-2.5",
        LTXVideoGenerationSpec(
            display_name="LTX-2.5 Fast (API)",
            supported_resolutions_durations={
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10, 12, 14, 16, 18, 20),
                        25: (6, 8, 10, 12, 14, 16, 18, 20),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "1440p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "2160p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
            },
            a2v_supported_resolutions_durations={
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
            },
        ),
    ),
    (
        "pro-2.5",
        LTXVideoGenerationSpec(
            display_name="LTX-2.5 Pro (API)",
            supported_resolutions_durations={
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "1440p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
                "2160p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
            },
            a2v_supported_resolutions_durations={
                "1080p": _resolution_spec(
                    fps_to_durations={
                        24: (6, 8, 10),
                        25: (6, 8, 10),
                        48: (6, 8, 10),
                        50: (6, 8, 10),
                    },
                ),
            },
        ),
    ),
)


def _capabilities_spec(caps: LtxOfferingCapabilities) -> LTXOfferingCapabilitiesSpec:
    return LTXOfferingCapabilitiesSpec(
        t2v=caps.t2v,
        i2v=caps.i2v,
        a2v=caps.a2v,
        ic_lora=caps.ic_lora,
        retake=caps.retake,
        extend=caps.extend,
        user_loras=caps.user_loras,
        camera_motion=caps.camera_motion,
        auto_duration=caps.auto_duration,
    )


def _item_with_caps(
    pipeline: LTXVideoGenPipeline,
    spec: LTXVideoGenerationSpec,
    caps: LtxOfferingCapabilities,
) -> LTXVideoGenerationModelSpecItem:
    return LTXVideoGenerationModelSpecItem(
        pipeline=pipeline,
        spec=spec.model_copy(
            update={
                "capabilities": _capabilities_spec(caps),
                "a2v_supported_resolutions_durations": (
                    spec.a2v_supported_resolutions_durations if caps.a2v else None
                ),
            }
        ),
    )


def get_local_video_generation_model_specs(
    model_id: LTXLocalModelId | None = None,
    *,
    duration_head_ready: bool = False,
) -> list[LTXVideoGenerationModelSpecItem]:
    resolved_id = model_id or get_latest_ltx_model_id()
    local_model_spec = get_ltx_model_spec(resolved_id)
    caps = effective_local_caps(resolved_id, duration_head_ready=duration_head_ready)
    return [
        _item_with_caps(pipeline, spec, caps)
        for pipeline, spec in local_model_spec.supported_pipelines
    ]


def get_api_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    # API Auto duration is a cloud capability. Do not gate it on the local DurationHead file.
    return [
        _item_with_caps(pipeline, spec, api_caps(pipeline))
        for pipeline, spec in ltx_api_model_specs
    ]


def build_generate_video_model_specs_response(
    local_model_id: LTXLocalModelId | None = None,
    *,
    duration_head_ready: bool = False,
) -> GenerateVideoModelsSpecsResponse:
    return GenerateVideoModelsSpecsResponse(
        local_models=get_local_video_generation_model_specs(
            local_model_id, duration_head_ready=duration_head_ready
        ),
        api_models=get_api_video_generation_model_specs(),
    )


def _get_resolution_spec(
    item: LTXVideoGenerationModelSpecItem,
    *,
    resolution: LTXVideoGenResolution,
    is_a2v: bool,
) -> LTXVideoGenerationResolutionSpec | None:
    if is_a2v:
        # A pipeline with no a2v spec doesn't support audio-conditioned generation at
        # all — must not fall back to the plain (non-a2v) matrix, or an unsupported
        # pipeline looks valid here and only fails downstream at the LTX API.
        if item.spec.a2v_supported_resolutions_durations is None:
            return None
        resolution_map = item.spec.a2v_supported_resolutions_durations
    else:
        resolution_map = item.spec.supported_resolutions_durations
    return resolution_map.get(resolution)


def get_supported_durations(
    resolution_spec: LTXVideoGenerationResolutionSpec,
    *,
    fps: LTXVideoGenFps,
) -> list[LTXVideoGenDuration]:
    return list(resolution_spec.fps_to_durations.get(fps, []))


def supported_duration_range(
    item: LTXVideoGenerationModelSpecItem,
    *,
    resolution: LTXVideoGenResolution,
    fps: LTXVideoGenFps,
    is_a2v: bool = False,
) -> tuple[LTXVideoGenDuration, LTXVideoGenDuration]:
    resolution_spec = _get_resolution_spec(item, resolution=resolution, is_a2v=is_a2v)
    if resolution_spec is None:
        raise KeyError(resolution)
    durations = get_supported_durations(resolution_spec, fps=fps)
    if not durations:
        raise KeyError(fps)
    return min(durations), max(durations)


def validate_generate_video_request(
    req: GenerateVideoRequest,
    *,
    use_api_specs: bool,
    local_model_id: LTXLocalModelId | None = None,
    duration_head_ready: bool = False,
) -> str | None:
    items = (
        get_api_video_generation_model_specs()
        if use_api_specs
        else get_local_video_generation_model_specs(
            local_model_id, duration_head_ready=duration_head_ready
        )
    )
    item = next((candidate for candidate in items if candidate.pipeline == req.model), None)
    generation_backend = "api" if use_api_specs else "local"
    generation_mode = "audio-to-video" if req.audioPath is not None else "image-to-video" if req.imagePath is not None else "text-to-video"

    if item is None:
        return (
            f"Unsupported {generation_backend} video generation pipeline: {req.model}. "
            f"Supported pipelines: {', '.join(candidate.pipeline for candidate in items)}"
        )

    resolution_spec = _get_resolution_spec(item, resolution=req.resolution, is_a2v=req.audioPath is not None)
    if resolution_spec is None:
        return (
            f"Unsupported {generation_backend} {generation_mode} resolution '{req.resolution}' "
            f"for pipeline '{req.model}'"
        )

    if req.fps not in resolution_spec.fps_to_durations:
        return (
            f"Unsupported {generation_backend} {generation_mode} fps '{req.fps}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}'"
        )

    if req.duration is None:
        if req.audioPath is not None:
            return "Automatic duration cannot be combined with audio-to-video"
        if item.spec.capabilities is None or not item.spec.capabilities.auto_duration:
            return (
                f"Automatic duration is not supported for {generation_backend} "
                f"pipeline '{req.model}'"
            )
        return None

    supported_durations = get_supported_durations(resolution_spec, fps=req.fps)
    if req.duration not in supported_durations:
        return (
            f"Unsupported {generation_backend} {generation_mode} duration '{req.duration}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}' and fps '{req.fps}'"
        )

    return None
