"""Canonical video-generation capability specs used by backend and frontend."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from api_types import (
    DownloadedLocalVideoGenerationModelSpecItem,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    LTXLocalModelId,
    LTXOfferingCapabilitiesSpec,
    LTXVideoGenerationModelSpecItem,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    LTXVideoGenAspectRatio,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
)
from keyframe_request import validate_keyframe_inputs
from runtime_config.ltx_capabilities import (
    CanvasMode,
    LtxOfferingCapabilities,
    api_caps,
    budget_size,
    effective_local_caps,
    local_canvas,
    pixels_for,
)
from runtime_config.offerings import (
    OFFERING_IDS,
    offering_id_for_local_model_id,
    resolve_offering_local_model_id,
)
from runtime_config.model_download_specs import (
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    is_duration_head_ready,
)
from runtime_config.runtime_policy import LocalGenerationMode
from runtime_config.video_job_budget import advertised_fast_durations
from server_utils.media_validation import normalize_optional_path

# The concrete ltxv-api model id each Desktop-facing pipeline maps to when forced onto
# the API backend. SSOT for this mapping — video_generation_handler and the
# retake/extend handlers all import it from here.
FORCED_API_MODEL_MAP: dict[str, str] = {
    "fast": "ltx-2-3-fast",
    "pro": "ltx-2-3-pro",
    "fast-2.5": "ltx-2-5-fast",
    "pro-2.5": "ltx-2-5-pro",
}


_ApiDurationEnvelope = tuple[LTXVideoGenDuration, ...]
_ApiFpsDurationMap = dict[LTXVideoGenFps, _ApiDurationEnvelope]
_ApiResolutionMap = dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]

# API duration envelopes (seconds). ltxv-api accepts 2–20s. GenSpace floors the
# picker at 6s; gap fill uses smallest_valid from this full list.
# 20s is Fast 720p/1080p at 24/25 and the A2V standard-tier audio cap.
_API_DURATIONS_TO_10S: _ApiDurationEnvelope = (2, 3, 4, 5, 6, 8, 10)
_API_DURATIONS_TO_20S: _ApiDurationEnvelope = (2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20)

_API_FPS_STANDARD: _ApiFpsDurationMap = {
    24: _API_DURATIONS_TO_10S,
    25: _API_DURATIONS_TO_10S,
    48: _API_DURATIONS_TO_10S,
    50: _API_DURATIONS_TO_10S,
}
_API_FPS_EXTENDED: _ApiFpsDurationMap = {
    24: _API_DURATIONS_TO_20S,
    25: _API_DURATIONS_TO_20S,
    48: _API_DURATIONS_TO_10S,
    50: _API_DURATIONS_TO_10S,
}
# ltx-2-5-pro has no 48 fps in MODEL_CAPABILITY_MATRIX.
_API_FPS_PRO_2_5: _ApiFpsDurationMap = {
    24: _API_DURATIONS_TO_10S,
    25: _API_DURATIONS_TO_10S,
    50: _API_DURATIONS_TO_10S,
}


# 2.3 generate is 16:9/9:16. 2.5 adds 1:1 and 3:2. 21:9, 4:3, and 4:5 are local-only.
_API_ASPECT_RATIOS_2_3: tuple[LTXVideoGenAspectRatio, ...] = ("16:9", "9:16")
_API_ASPECT_RATIOS_2_5: tuple[LTXVideoGenAspectRatio, ...] = ("16:9", "3:2", "1:1", "9:16")


def _resolution_spec(
    fps_to_durations: _ApiFpsDurationMap,
    aspect_ratios: tuple[LTXVideoGenAspectRatio, ...],
) -> LTXVideoGenerationResolutionSpec:
    return LTXVideoGenerationResolutionSpec(
        fps_to_durations={
            fps: list(durations)
            for fps, durations in fps_to_durations.items()
        },
        aspect_ratios=list(aspect_ratios),
    )


def _resolution_map(
    cells: dict[LTXVideoGenResolution, _ApiFpsDurationMap],
    aspect_ratios: tuple[LTXVideoGenAspectRatio, ...],
) -> _ApiResolutionMap:
    return {
        resolution: _resolution_spec(fps_to_durations, aspect_ratios)
        for resolution, fps_to_durations in cells.items()
    }


# Fast t2v/i2v: 720p/1080p get 20s at 24/25; 1440p/4K stay at 10s.
_API_FAST_CELLS: dict[LTXVideoGenResolution, _ApiFpsDurationMap] = {
    "720p": _API_FPS_EXTENDED,
    "1080p": _API_FPS_EXTENDED,
    "1440p": _API_FPS_STANDARD,
    "2160p": _API_FPS_STANDARD,
}
# Pro 2.3 t2v/i2v: 10s at every fps and resolution, including 720p.
_API_PRO_CELLS: dict[LTXVideoGenResolution, _ApiFpsDurationMap] = {
    "720p": _API_FPS_STANDARD,
    "1080p": _API_FPS_STANDARD,
    "1440p": _API_FPS_STANDARD,
    "2160p": _API_FPS_STANDARD,
}
# Pro 2.5 t2v/i2v: 720p+1080p, 24/25/50, 10s. No 48 fps, no 1440p/4K.
_API_PRO_2_5_CELLS: dict[LTXVideoGenResolution, _ApiFpsDurationMap] = {
    "720p": _API_FPS_PRO_2_5,
    "1080p": _API_FPS_PRO_2_5,
}

_API_FAST_RESOLUTIONS = _resolution_map(_API_FAST_CELLS, _API_ASPECT_RATIOS_2_3)
_API_FAST_25_RESOLUTIONS = _resolution_map(_API_FAST_CELLS, _API_ASPECT_RATIOS_2_5)
_API_PRO_RESOLUTIONS = _resolution_map(_API_PRO_CELLS, _API_ASPECT_RATIOS_2_3)
_API_PRO_2_5_RESOLUTIONS = _resolution_map(_API_PRO_2_5_CELLS, _API_ASPECT_RATIOS_2_5)
# A2V stays 16:9/9:16 even on 2.5. 20s audio at 720p/1080p, 10s at 1440p/4K.
_API_A2V_RESOLUTIONS = _resolution_map(_API_FAST_CELLS, _API_ASPECT_RATIOS_2_3)
_API_PRO_2_5_A2V_RESOLUTIONS = _resolution_map(_API_PRO_2_5_CELLS, _API_ASPECT_RATIOS_2_3)


ltx_api_model_specs: tuple[tuple[LTXVideoGenPipeline, LTXVideoGenerationSpec], ...] = (
    (
        "fast",
        LTXVideoGenerationSpec(
            display_name="LTX-2.3 Fast (API)",
            supported_resolutions_durations=_API_FAST_RESOLUTIONS,
            # No A2V envelope: ltxv-api audio-to-video does not accept ltx-2-3-fast.
        ),
    ),
    (
        "pro",
        LTXVideoGenerationSpec(
            display_name="LTX-2.3 Pro (API)",
            supported_resolutions_durations=_API_PRO_RESOLUTIONS,
            a2v_supported_resolutions_durations=_API_A2V_RESOLUTIONS,
        ),
    ),
    (
        "fast-2.5",
        LTXVideoGenerationSpec(
            display_name="LTX-2.5 Fast (API)",
            supported_resolutions_durations=_API_FAST_25_RESOLUTIONS,
            a2v_supported_resolutions_durations=_API_A2V_RESOLUTIONS,
        ),
    ),
    (
        "pro-2.5",
        LTXVideoGenerationSpec(
            display_name="LTX-2.5 Pro (API)",
            supported_resolutions_durations=_API_PRO_2_5_RESOLUTIONS,
            a2v_supported_resolutions_durations=_API_PRO_2_5_A2V_RESOLUTIONS,
        ),
    ),
)


def forced_api_resolution_map() -> dict[str, dict[str, dict[str, str]]]:
    """Pipeline → resolution label → aspect → WxH, taken from each API spec cell."""
    mapped: dict[str, dict[str, dict[str, str]]] = {}
    for pipeline, spec in ltx_api_model_specs:
        caps = api_caps(pipeline)
        by_resolution: dict[str, dict[str, str]] = {}
        for resolution, cell in spec.supported_resolutions_durations.items():
            by_resolution[resolution] = {
                aspect: f"{width}x{height}"
                for aspect in cell.aspect_ratios
                for width, height in (pixels_for(caps, resolution, aspect),)
            }
        mapped[pipeline] = by_resolution
    return mapped


def _capabilities_spec(caps: LtxOfferingCapabilities) -> LTXOfferingCapabilitiesSpec:
    return LTXOfferingCapabilitiesSpec(
        t2v=caps.t2v,
        i2v=caps.i2v,
        a2v=caps.a2v,
        ic_lora=caps.ic_lora,
        retake=caps.retake,
        extend=caps.extend,
        multi_keyframe=caps.multi_keyframe,
        multi_keyframe_max_count=caps.multi_keyframe_max_count,
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


def _filter_local_fast_resolution_map(
    specs: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None,
    caps: LtxOfferingCapabilities,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool,
    mode: CanvasMode = "video",
) -> dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None:
    if specs is None:
        return None
    if process_mode == "unsupported":
        return {}
    filtered: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] = {}
    for resolution, spec in specs.items():
        if resolution not in caps.resolution_pixels_16_9:
            raise KeyError(resolution)
        width, height = budget_size(local_canvas(resolution, mode=mode))
        fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]] = {}
        for fps, durations in spec.fps_to_durations.items():
            kept = advertised_fast_durations(
                width,
                height,
                int(fps),
                durations,
                memory_gb=memory_gb,
                process_mode=process_mode,
                darwin=darwin,
            )
            if kept:
                fps_to_durations[fps] = cast(list[LTXVideoGenDuration], kept)
        if fps_to_durations:
            filtered[resolution] = LTXVideoGenerationResolutionSpec(
                fps_to_durations=fps_to_durations,
                aspect_ratios=list(spec.aspect_ratios),
            )
    return filtered


def _filter_local_fast_item(
    item: LTXVideoGenerationModelSpecItem,
    caps: LtxOfferingCapabilities,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool,
) -> LTXVideoGenerationModelSpecItem:
    spec = item.spec
    return item.model_copy(
        update={
            "spec": spec.model_copy(
                update={
                    "supported_resolutions_durations": _filter_local_fast_resolution_map(
                        spec.supported_resolutions_durations,
                        caps,
                        memory_gb=memory_gb,
                        process_mode=process_mode,
                        darwin=darwin,
                    ),
                    "a2v_supported_resolutions_durations": _filter_local_fast_resolution_map(
                        spec.a2v_supported_resolutions_durations,
                        caps,
                        memory_gb=memory_gb,
                        process_mode=process_mode,
                        darwin=darwin,
                        mode="a2v",
                    ),
                }
            )
        }
    )


def get_local_video_generation_model_specs(
    model_id: LTXLocalModelId | None = None,
    *,
    duration_head_ready: bool = False,
    memory_gb: float | None = None,
    process_mode: LocalGenerationMode | None = None,
    darwin: bool = False,
) -> list[LTXVideoGenerationModelSpecItem]:
    resolved_id = model_id or get_latest_ltx_model_id()
    local_model_spec = get_ltx_model_spec(resolved_id)
    caps = effective_local_caps(resolved_id, duration_head_ready=duration_head_ready)
    items = [
        _item_with_caps(pipeline, spec, caps)
        for pipeline, spec in local_model_spec.supported_pipelines
    ]
    if process_mode is None:
        return items
    return [
        _filter_local_fast_item(
            item,
            caps,
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
        )
        for item in items
    ]


def get_api_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    # API Auto duration is a cloud capability. Do not gate it on the local DurationHead file.
    return [
        _item_with_caps(pipeline, spec, api_caps(pipeline))
        for pipeline, spec in ltx_api_model_specs
    ]


def get_downloaded_local_video_generation_model_specs(
    models_dir: Path,
    *,
    memory_gb: float | None = None,
    process_mode: LocalGenerationMode | None = None,
    darwin: bool = False,
) -> list[DownloadedLocalVideoGenerationModelSpecItem]:
    items: list[DownloadedLocalVideoGenerationModelSpecItem] = []
    for offering in OFFERING_IDS:
        model_id = resolve_offering_local_model_id(models_dir, offering)
        if model_id is None:
            continue
        for local_item in get_local_video_generation_model_specs(
            model_id,
            duration_head_ready=is_duration_head_ready(models_dir, model_id),
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
        ):
            items.append(
                DownloadedLocalVideoGenerationModelSpecItem(
                    model=offering,
                    pipeline=local_item.pipeline,
                    spec=local_item.spec,
                )
            )
    return items


def build_generate_video_model_specs_response(
    local_model_id: LTXLocalModelId | None = None,
    *,
    models_dir: Path | None = None,
    duration_head_ready: bool = False,
    memory_gb: float | None = None,
    process_mode: LocalGenerationMode | None = None,
    darwin: bool = False,
    low_performance_machine: bool = False,
) -> GenerateVideoModelsSpecsResponse:
    downloaded = (
        get_downloaded_local_video_generation_model_specs(
            models_dir,
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
        )
        if models_dir is not None
        else []
    )
    return GenerateVideoModelsSpecsResponse(
        local_models=get_local_video_generation_model_specs(
            local_model_id,
            duration_head_ready=duration_head_ready,
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
        ),
        api_models=get_api_video_generation_model_specs(),
        downloaded_local_models=downloaded,
        active_offering=(
            offering_id_for_local_model_id(local_model_id)
            if local_model_id is not None
            else None
        ),
        low_performance_machine=low_performance_machine,
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


VideoGenerationFeature = Literal[
    "multi-keyframe",
    "audio-to-video",
    "image-to-video",
    "text-to-video",
]


def video_generation_feature(
    *,
    audio_path: str | None,
    image_path: str | None,
    last_image_path: str | None = None,
    keyframes: object = None,
) -> VideoGenerationFeature:
    """User-facing mode. Keyframes win, then audio, then a still.

    Same order as ``videoGenerationModeFromInputs`` in the frontend. Validation
    error text and analytics feature names both use this so a keyframe job is
    not recorded as image-to-video.
    """
    if keyframes:
        return "multi-keyframe"
    if audio_path:
        return "audio-to-video"
    if image_path or last_image_path:
        return "image-to-video"
    return "text-to-video"


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
    image_path = normalize_optional_path(req.imagePath)
    last_image_path = normalize_optional_path(req.lastImagePath)
    audio_path = normalize_optional_path(req.audioPath)
    generation_mode = video_generation_feature(
        audio_path=audio_path,
        image_path=image_path,
        last_image_path=last_image_path,
        keyframes=req.keyframes,
    )

    keyframe_error = validate_keyframe_inputs(
        req,
        use_api_specs=use_api_specs,
        image_path=image_path,
        last_image_path=last_image_path,
        audio_path=audio_path,
    )
    if keyframe_error is not None:
        return keyframe_error

    if last_image_path:
        if not image_path:
            return "Last frame requires a first-frame image"
        if req.duration is None:
            return "Last frame cannot be combined with automatic duration"

    if item is None:
        return (
            f"Unsupported {generation_backend} video generation pipeline: {req.model}. "
            f"Supported pipelines: {', '.join(candidate.pipeline for candidate in items)}"
        )

    resolution_spec = _get_resolution_spec(item, resolution=req.resolution, is_a2v=audio_path is not None)
    if resolution_spec is None:
        return (
            f"Unsupported {generation_backend} {generation_mode} resolution '{req.resolution}' "
            f"for pipeline '{req.model}'"
        )

    if req.aspectRatio not in resolution_spec.aspect_ratios:
        return (
            f"Unsupported {generation_backend} {generation_mode} aspect ratio '{req.aspectRatio}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}'"
        )

    if req.fps not in resolution_spec.fps_to_durations:
        return (
            f"Unsupported {generation_backend} {generation_mode} fps '{req.fps}' "
            f"for pipeline '{req.model}' at resolution '{req.resolution}'"
        )

    if req.duration is None:
        if audio_path is not None:
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

    if req.keyframes:
        caps = item.spec.capabilities
        if caps is None or not caps.multi_keyframe:
            return (
                f"Multi-keyframe is not supported for {generation_backend} "
                f"pipeline '{req.model}'"
            )
        if len(req.keyframes) > caps.multi_keyframe_max_count:
            return f"You can place up to {caps.multi_keyframe_max_count} keyframes"

    return None
