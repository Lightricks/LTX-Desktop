"""Canonical video-generation capability specs used by backend and frontend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from api_types import (
    ChunkedFastCell,
    DownloadedLocalVideoGenerationModelSpecItem,
    DroppedFastAspectCell,
    DroppedFastCell,
    FastCellAsk,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    LTXLocalModelId,
    LTXOfferingCapabilitiesSpec,
    LTXVideoGenerationAspectSpec,
    LTXVideoGenerationModelSpecItem,
    LTXVideoGenerationResolutionSpec,
    LTXVideoGenerationSpec,
    LTXVideoGenAspectRatio,
    LTXVideoGenDuration,
    LTXVideoGenFps,
    LTXVideoGenPipeline,
    LTXVideoGenResolution,
    LocalUnavailableReason,
    WarnedFastAspectCell,
    WarnedFastCell,
)
from frame_math import compute_num_frames
from keyframe_request import validate_keyframe_inputs
from runtime_config.fast_chunking import job_window_count
from runtime_config.ltx_capabilities import (
    CanvasMode,
    LtxOfferingCapabilities,
    api_caps,
    budget_pixels,
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
from runtime_config.measured_cells import BudgetRoute
from runtime_config.memory_meter import (
    job_filter_machine_factor,
    job_memory_ask,
    memory_capacity,
)
from runtime_config.model_download_specs import (
    get_latest_ltx_model_id,
    get_ltx_model_spec,
    is_duration_head_ready,
)
from runtime_config.budget_admission import (
    Admitted,
    DroppedCellFieldsPayload,
    PricedJob,
    Refused,
    RiskAllowance,
    admit,
    dropped_cell_fields,
    warned_cell_parts,
)
from runtime_config.budget_decision import VideoJobReject
from runtime_config.video_job_budget import (
    LocalJobFilter,
    budget_route,
    decide_fast_durations,
    decide_video_job,
    price_alternative_pass,
    reject_message_and_code,
    video_job_reject,
)
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


def _dropped_cell_message(
    decision: VideoJobReject,
    job_filter: LocalJobFilter,
) -> str:
    """List-time refusal copy for one withheld cell.

    Same builder as the request-time 422s over the figures the gate
    compared, so a cell refused in the picker reads identically if it
    ever 422s.
    """
    message, _code = reject_message_and_code(decision, job_filter)
    return message


_UNSUPPORTED_DROPPED_DECISION = VideoJobReject(
    seq=0.0, full_gib=0.0, stream_gib=0.0, reason="unsupported"
)


def _unsupported_dropped_cells(
    specs: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec],
    *,
    audio: bool,
    job_filter: LocalJobFilter,
) -> list[DroppedFastCell]:
    """Every static cell withheld with `unsupported`, no decisions needed."""
    return [
        DroppedFastCell(
            resolution=resolution,
            fps=fps,
            duration=duration,
            reason="unsupported",
            message=_dropped_cell_message(
                _UNSUPPORTED_DROPPED_DECISION, job_filter
            ),
            audio=audio,
        )
        for resolution, spec in specs.items()
        for fps, durations in spec.fps_to_durations.items()
        for duration in durations
    ]


def _dropped_cell_fields(refused: Refused, job_filter: LocalJobFilter) -> DroppedCellFieldsPayload:
    reject = video_job_reject(refused)
    return dropped_cell_fields(
        reject,
        refused.priced.evidence,
        job_filter,
        message=_dropped_cell_message(reject, job_filter),
    )


def _cell_ask(
    priced: PricedJob,
    *,
    route: BudgetRoute,
    system: Literal["Windows", "Linux", "Darwin"] = "Windows",
    resolution: LTXVideoGenResolution,
    fps: LTXVideoGenFps,
    duration: int,
    audio: bool,
    alternative: PricedJob | None,
    machine_factor: float | None = None,
    conv_vae: bool = False,
) -> FastCellAsk:
    alternative_ask = None
    if alternative is not None:
        admitted = admit(alternative, RiskAllowance.for_mode("brave"))
        alternative_ask = job_memory_ask(
            alternative,
            route=route,
            system=system,
            risk=admitted.risk if isinstance(admitted, Admitted) else None,
            machine_factor=machine_factor,
            conv_vae=conv_vae,
        )
    return FastCellAsk(
        **job_memory_ask(
            priced, route=route, system=system, machine_factor=machine_factor, conv_vae=conv_vae
        ).model_dump(),
        resolution=resolution,
        fps=fps,
        duration=cast(LTXVideoGenDuration, duration),
        audio=audio,
        alternative=alternative_ask,
    )


def _filter_local_fast_resolution_map(
    specs: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None,
    caps: LtxOfferingCapabilities,
    *,
    job_filter: LocalJobFilter,
    mode: CanvasMode = "video",
    audio: bool = False,
    aspect: LTXVideoGenAspectRatio = "16:9",
    asks: list[FastCellAsk] | None = None,
) -> tuple[
    dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None,
    list[DroppedFastCell],
    list[DroppedFastAspectCell],
    list[WarnedFastCell],
    list[WarnedFastAspectCell],
    list[ChunkedFastCell],
]:
    if specs is None:
        return None, [], [], [], [], []
    if job_filter.profile.process_mode == "unsupported":
        return {}, _unsupported_dropped_cells(specs, audio=audio, job_filter=job_filter), [], [], [], []
    filtered: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] = {}
    route = budget_route("fast", job_filter.profile)
    dropped: list[DroppedFastCell] = []
    aspect_dropped: list[DroppedFastAspectCell] = []
    warned: list[WarnedFastCell] = []
    aspect_warned: list[WarnedFastAspectCell] = []
    chunked: list[ChunkedFastCell] = []
    for resolution, spec in specs.items():
        if resolution not in caps.resolution_pixels_16_9:
            raise KeyError(resolution)
        width, height = (
            budget_size(local_canvas(resolution, mode=mode))
            if aspect == "16:9"
            else budget_pixels(caps, resolution, aspect, mode=mode)
        )
        fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]] = {}
        for fps, durations in spec.fps_to_durations.items():
            outcome = decide_fast_durations(
                width,
                height,
                int(fps),
                durations,
                job_filter=job_filter,
            )
            if outcome.kept:
                fps_to_durations[fps] = cast(
                    list[LTXVideoGenDuration], outcome.kept
                )
            if asks is not None:
                # Every cell, kept or withheld, so a hover preview can show any option's ask.
                machine_factor = job_filter_machine_factor(job_filter)
                for duration in durations:
                    admission = outcome.admitted.get(duration) or outcome.refused[duration]
                    asks.append(
                        _cell_ask(
                            admission.priced,
                            route=route,
                            system=job_filter.profile.system,
                            resolution=resolution,
                            fps=fps,
                            duration=duration,
                            audio=audio,
                            alternative=price_alternative_pass(
                                width,
                                height,
                                compute_num_frames(duration, int(fps)),
                                job_filter=job_filter,
                                auto=admission.priced,
                            ),
                            machine_factor=machine_factor,
                            conv_vae=job_filter.conv_vae,
                        )
                    )
            for duration, refused in outcome.refused.items():
                reject_window = video_job_reject(refused).chunk_pixel_frames
                dropped.append(
                    DroppedFastCell(
                        resolution=resolution,
                        fps=fps,
                        duration=cast(LTXVideoGenDuration, duration),
                        audio=audio,
                        windows=(
                            None
                            if reject_window is None
                            else job_window_count(
                                compute_num_frames(duration, int(fps)), reject_window
                            )
                        ),
                        window_frames=reject_window,
                        **_dropped_cell_fields(refused, job_filter),
                    )
                )
            # Emit warned cells for kept durations
            for duration, admitted in outcome.admitted.items():
                parts = warned_cell_parts(admitted, job_filter)
                if parts is None:
                    continue
                tier, headroom_gib, message, readout = parts
                warned.append(
                    WarnedFastCell(
                        resolution=resolution,
                        fps=fps,
                        duration=cast(LTXVideoGenDuration, duration),
                        tier=tier,
                        headroom_gib=headroom_gib,
                        message=message,
                        readout=readout,
                        audio=audio,
                    )
                )
            # Kept durations that run in more than one window, for the "takes longer" notice.
            for duration, admitted in outcome.admitted.items():
                window = admitted.load.chunk_pixel_frames
                if window is None:
                    continue
                chunked.append(
                    ChunkedFastCell(
                        resolution=resolution,
                        fps=fps,
                        duration=cast(LTXVideoGenDuration, duration),
                        windows=job_window_count(compute_num_frames(duration, int(fps)), window),
                        window_frames=window,
                        audio=audio,
                    )
                )
            if aspect != "16:9":
                continue
            # Aspect refinement: a kept anchor cell a wider shape cannot run.
            # Narrower-or-equal shapes cost no more than the anchor, so only
            # wider ratios (21:9 today) can emit here; the anchor itself and
            # anchor-dropped cells never do.
            for duration in outcome.kept:
                frames = compute_num_frames(duration, int(fps))
                for ratio in spec.aspect_ratios:
                    selected_width, selected_height = budget_pixels(
                        caps, resolution, ratio, mode=mode
                    )
                    if selected_width * selected_height <= width * height:
                        continue
                    admission = decide_video_job(
                        selected_width, selected_height, frames, job_filter=job_filter
                    )
                    if isinstance(admission, Refused):
                        aspect_dropped.append(
                            DroppedFastAspectCell(
                                resolution=resolution,
                                fps=fps,
                                duration=cast(LTXVideoGenDuration, duration),
                                aspectRatio=ratio,
                                audio=audio,
                                **_dropped_cell_fields(admission, job_filter),
                            )
                        )
                    else:
                        parts = warned_cell_parts(admission, job_filter)
                        if parts is not None:
                            tier, headroom_gib, message, readout = parts
                            aspect_warned.append(
                                WarnedFastAspectCell(
                                    resolution=resolution,
                                    fps=fps,
                                    duration=cast(LTXVideoGenDuration, duration),
                                    aspectRatio=ratio,
                                    tier=tier,
                                    headroom_gib=headroom_gib,
                                    message=message,
                                    readout=readout,
                                    audio=audio,
                                )
                            )
        if fps_to_durations:
            filtered[resolution] = LTXVideoGenerationResolutionSpec(
                fps_to_durations=fps_to_durations,
                aspect_ratios=list(spec.aspect_ratios),
            )
    return filtered, dropped, aspect_dropped, warned, aspect_warned, chunked


def _aspect_ratios_in_spec(spec: LTXVideoGenerationSpec) -> list[LTXVideoGenAspectRatio]:
    seen: list[LTXVideoGenAspectRatio] = []
    for table in (
        spec.supported_resolutions_durations,
        spec.a2v_supported_resolutions_durations or {},
    ):
        for rspec in table.values():
            for ratio in rspec.aspect_ratios:
                if ratio not in seen:
                    seen.append(ratio)
    return seen


def _build_aspect_spec(
    spec: LTXVideoGenerationSpec,
    caps: LtxOfferingCapabilities,
    *,
    job_filter: LocalJobFilter,
    aspect: LTXVideoGenAspectRatio,
) -> tuple[
    LTXVideoGenerationAspectSpec,
    list[DroppedFastAspectCell],
    list[WarnedFastAspectCell],
]:
    asks: list[FastCellAsk] = []
    kept, dropped, aspect_dropped, warned, aspect_warned, chunked = (
        _filter_local_fast_resolution_map(
            spec.supported_resolutions_durations,
            caps,
            job_filter=job_filter,
            aspect=aspect,
            asks=asks,
        )
    )
    a2v_kept, a2v_dropped, a2v_aspect_dropped, a2v_warned, a2v_aspect_warned, a2v_chunked = (
        _filter_local_fast_resolution_map(
            spec.a2v_supported_resolutions_durations,
            caps,
            job_filter=job_filter,
            mode="a2v",
            audio=True,
            aspect=aspect,
            asks=asks,
        )
    )
    return (
        LTXVideoGenerationAspectSpec(
            supported_resolutions_durations=kept or {},
            a2v_supported_resolutions_durations=a2v_kept,
            dropped_cells=dropped + a2v_dropped,
            warned_cells=warned + a2v_warned,
            chunked_cells=chunked + a2v_chunked,
            cell_asks=asks,
        ),
        aspect_dropped + a2v_aspect_dropped,
        aspect_warned + a2v_aspect_warned,
    )


def _filter_local_fast_item(
    item: LTXVideoGenerationModelSpecItem,
    caps: LtxOfferingCapabilities,
    *,
    job_filter: LocalJobFilter,
) -> LTXVideoGenerationModelSpecItem:
    spec = item.spec
    anchor_spec, aspect_dropped, aspect_warned = _build_aspect_spec(
        spec, caps, job_filter=job_filter, aspect="16:9"
    )
    aspect_specs: dict[LTXVideoGenAspectRatio, LTXVideoGenerationAspectSpec] = {
        ratio: (
            anchor_spec
            if ratio in ("16:9", "9:16")
            else _build_aspect_spec(spec, caps, job_filter=job_filter, aspect=ratio)[0]
        )
        for ratio in _aspect_ratios_in_spec(spec)
    }
    return item.model_copy(
        update={
            "spec": spec.model_copy(
                update={
                    "supported_resolutions_durations": anchor_spec.supported_resolutions_durations,
                    "a2v_supported_resolutions_durations": anchor_spec.a2v_supported_resolutions_durations,
                    "dropped_cells": anchor_spec.dropped_cells,
                    "dropped_aspect_cells": aspect_dropped,
                    "warned_cells": anchor_spec.warned_cells,
                    "warned_aspect_cells": aspect_warned,
                    "chunked_cells": anchor_spec.chunked_cells,
                    "cell_asks": anchor_spec.cell_asks,
                    "aspect_specs": aspect_specs,
                }
            )
        }
    )


def get_local_video_generation_model_specs(
    model_id: LTXLocalModelId | None = None,
    *,
    duration_head_ready: bool = False,
    job_filter: LocalJobFilter | None = None,
) -> list[LTXVideoGenerationModelSpecItem]:
    """Specs for one local model, filtered when a job filter is provided.

    ``job_filter=None`` serves the unfiltered static ceiling (request-shape
    validation only). Otherwise each row is evaluated under its own model's
    profile: a downloaded 2.5 row takes the 2.5 Windows profile even while
    2.3 is active, because profile is (system, model, hardware), not offering.
    """
    if job_filter is not None and model_id is None and job_filter.profile.model_id is not None:
        model_id = job_filter.profile.model_id
    resolved_id = model_id or get_latest_ltx_model_id()
    local_model_spec = get_ltx_model_spec(resolved_id)
    caps = effective_local_caps(resolved_id, duration_head_ready=duration_head_ready)
    items = [
        _item_with_caps(pipeline, spec, caps)
        for pipeline, spec in local_model_spec.supported_pipelines
    ]
    if job_filter is None:
        return items
    row_filter = replace(
        job_filter, profile=replace(job_filter.profile, model_id=resolved_id)
    )
    return [_filter_local_fast_item(item, caps, job_filter=row_filter) for item in items]


def get_api_video_generation_model_specs() -> list[LTXVideoGenerationModelSpecItem]:
    # API Auto duration is a cloud capability. Do not gate it on the local DurationHead file.
    return [
        _item_with_caps(pipeline, spec, api_caps(pipeline))
        for pipeline, spec in ltx_api_model_specs
    ]


def get_downloaded_local_video_generation_model_specs(
    models_dir: Path,
    *,
    job_filter: LocalJobFilter,
) -> list[DownloadedLocalVideoGenerationModelSpecItem]:
    items: list[DownloadedLocalVideoGenerationModelSpecItem] = []
    for offering in OFFERING_IDS:
        model_id = resolve_offering_local_model_id(models_dir, offering)
        if model_id is None:
            continue
        for local_item in get_local_video_generation_model_specs(
            model_id,
            duration_head_ready=is_duration_head_ready(models_dir, model_id),
            job_filter=job_filter,
        ):
            items.append(
                DownloadedLocalVideoGenerationModelSpecItem(
                    model=offering,
                    pipeline=local_item.pipeline,
                    spec=local_item.spec,
                )
            )
    return items


@dataclass(frozen=True, slots=True)
class _LocalUnavailableDetails:
    reason: LocalUnavailableReason | None = None
    required_ram_gb: int | None = None
    detected_ram_gb: int | None = None


def _local_unavailable_details(
    local_models: list[LTXVideoGenerationModelSpecItem],
    *,
    job_filter: LocalJobFilter,
) -> _LocalUnavailableDetails:
    """Name RAM when it is the binding constraint on an empty local grid.

    A ``ram_shortfall`` verdict means VRAM and mode already passed, so any
    such verdict under the served filter proves RAM is what hides the grid —
    no sufficient-RAM rerun needed. Unknown RAM yields ``unsupported``, never
    shortfall, so it correctly reports empty details here.
    """
    fast = next((item for item in local_models if item.pipeline == "fast"), None)
    if fast is None or fast.spec.supported_resolutions_durations:
        return _LocalUnavailableDetails()
    video_dropped = [
        cell for cell in (fast.spec.dropped_cells or []) if not cell.audio
    ]
    if not any(cell.reason == "ram_shortfall" for cell in video_dropped):
        return _LocalUnavailableDetails()
    # Shortfall verdicts prove total RAM is known (unknown yields unsupported),
    # so detected is present — the assert narrows what the verdicts guarantee.
    total_ram_bytes = job_filter.budget.total_ram_bytes
    assert total_ram_bytes is not None
    return _LocalUnavailableDetails(
        reason="ram_shortfall",
        required_ram_gb=job_filter.profile.windows_fast_required_ram_bytes // (1024**3),
        detected_ram_gb=total_ram_bytes // (1024**3),
    )


def build_generate_video_model_specs_response(
    local_model_id: LTXLocalModelId | None = None,
    *,
    models_dir: Path | None = None,
    duration_head_ready: bool = False,
    job_filter: LocalJobFilter,
    low_performance_machine: bool = False,
) -> GenerateVideoModelsSpecsResponse:
    if local_model_id is None and job_filter.profile.model_id is not None:
        local_model_id = job_filter.profile.model_id
    downloaded = (
        get_downloaded_local_video_generation_model_specs(
            models_dir,
            job_filter=job_filter,
        )
        if models_dir is not None
        else []
    )
    local_models = get_local_video_generation_model_specs(
        local_model_id,
        duration_head_ready=duration_head_ready,
        job_filter=job_filter,
    )
    unavailable = _local_unavailable_details(
        local_models,
        job_filter=job_filter,
    )
    return GenerateVideoModelsSpecsResponse(
        local_models=local_models,
        api_models=get_api_video_generation_model_specs(),
        downloaded_local_models=downloaded,
        active_offering=(
            offering_id_for_local_model_id(local_model_id)
            if local_model_id is not None
            else None
        ),
        local_unavailable_reason=unavailable.reason,
        local_unavailable_required_ram_gb=unavailable.required_ram_gb,
        local_unavailable_detected_ram_gb=unavailable.detected_ram_gb,
        low_performance_machine=low_performance_machine,
        budgetMode=job_filter.budget_mode,
        neverSplitIntoParts=job_filter.pass_shape == "one_pass",
        memory=memory_capacity(job_filter),
    )


def _get_resolution_spec(
    item: LTXVideoGenerationModelSpecItem,
    *,
    resolution: LTXVideoGenResolution,
    is_a2v: bool,
    aspect_ratio: LTXVideoGenAspectRatio | None = None,
) -> LTXVideoGenerationResolutionSpec | None:
    aspect_spec = (
        item.spec.aspect_specs.get(aspect_ratio)
        if aspect_ratio is not None and item.spec.aspect_specs is not None
        else None
    )
    if is_a2v:
        # A pipeline with no a2v spec doesn't support audio-conditioned generation at
        # all — must not fall back to the plain (non-a2v) matrix, or an unsupported
        # pipeline looks valid here and only fails downstream at the LTX API.
        resolution_map = (
            aspect_spec.a2v_supported_resolutions_durations
            if aspect_spec is not None
            else item.spec.a2v_supported_resolutions_durations
        )
        if resolution_map is None:
            return None
    else:
        resolution_map = (
            aspect_spec.supported_resolutions_durations
            if aspect_spec is not None
            else item.spec.supported_resolutions_durations
        )
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
    aspect_ratio: LTXVideoGenAspectRatio | None = None,
    is_a2v: bool = False,
) -> tuple[LTXVideoGenDuration, LTXVideoGenDuration]:
    resolution_spec = _get_resolution_spec(
        item, resolution=resolution, is_a2v=is_a2v, aspect_ratio=aspect_ratio
    )
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
