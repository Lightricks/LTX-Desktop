"""Remote-safe Asset and Generation projections.

Desktop `Asset`/`Generation` stay the storage and local HTTP contract. The phone
only receives this allowlisted view: no filesystem paths, no LoRA refs, and no
arbitrary stored spec JSON.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_serializer

from api_types import (
    ApiActivityStatus,
    Asset,
    AssetCore,
    ExtendMode,
    IcLoraAudioMode,
    IcLoraControlType,
    ImageProvider,
    Generation,
    GenerationCore,
    NagParams,
    QueueEntry,
    QueueProgress,
    QueueSnapshot,
    RetakeMode,
)
from api_motion_path import MotionPath
from remote.media_urls import MediaUrlSigner
from runtime_config.ic_lora_tiling import (
    MAX_TILE_OVERLAP,
    MIN_CARRY_FRAMES,
    MIN_TILE_OVERLAP,
    MIN_WINDOW_FRAMES,
    IcLoraTileSize,
)
from runtime_config.reject_reasons import PassShape

logger = logging.getLogger(__name__)

_NAG_NUMBER_FIELDS = frozenset(NagParams.model_fields) - {"negativePrompt"}

_SAFE_PARAM_STRINGS = frozenset(
    {
        "prompt",
        "negativePrompt",
        "model",
        "aspectRatio",
        "promptProvenance",
        "provider",
        "mode",
        "audioMode",
        "passShape",
        "controlType",
        "tilingSize",
    }
)
_REMOTE_IMAGE_PROVIDERS = frozenset(get_args(ImageProvider))
_REMOTE_AUDIO_MODES = frozenset(get_args(IcLoraAudioMode))
_REMOTE_PASS_SHAPES = frozenset(get_args(PassShape))
_REMOTE_CONTROL_TYPES = frozenset(get_args(IcLoraControlType))
_REMOTE_TILE_SIZES = frozenset(get_args(IcLoraTileSize))
_SAFE_PARAM_NUMBERS = frozenset({"duration", "fps", "startTime", "tilingOverlap", *_NAG_NUMBER_FIELDS})
_REMOTE_MODES = frozenset((*get_args(RetakeMode), *get_args(ExtendMode)))


class RemoteAsset(AssetCore):
    """Asset identity and metadata without Desktop filesystem paths."""

    model_config = ConfigDict(strict=True, extra="forbid")

    bytes_url: str = Field(min_length=1)
    thumbnail_url: str | None = None


class RemoteAssetListItem(RemoteAsset):
    in_use: bool
    has_thumbnail: bool


class RemoteAssetListResponse(BaseModel):
    items: list[RemoteAssetListItem]
    next_cursor: str | None = None


class RemoteInputAssetRef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    assetId: str = Field(min_length=1)


class RemoteKeyframe(BaseModel):
    """One still of a Keyframes run. Reuse puts it back on the timeline."""

    model_config = ConfigDict(strict=True, extra="forbid")

    assetId: str = Field(min_length=1)
    frameIndex: int = Field(ge=0)
    strength: float = Field(ge=0.0, le=1.0)


class RemoteTargetResolution(BaseModel):
    """Home Retake/Extend target size. Tier ids stay strings."""

    model_config = ConfigDict(strict=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RemoteGenerationParams(BaseModel):
    """Form and result-meta fields only, for video and text-to-image generations."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str | None = None
    negativePrompt: str | None = None
    nagScale: int | float | None = None
    nagAlpha: int | float | None = None
    nagTau: int | float | None = None
    model: str | None = None
    resolution: str | RemoteTargetResolution | None = None
    duration: int | float | None = None
    fps: int | float | None = None
    aspectRatio: str | None = None
    promptProvenance: str | None = None
    scale: int | float | None = None
    # Which saved custom LoRA the run used, so Reuse can select it again. An id, never a path.
    customLoraId: str | None = None
    # The LoRA name at run time, so the queue and results show it on a phone.
    customLoraName: str | None = None
    numFrames: int | None = None
    mode: RetakeMode | ExtendMode | None = None
    startTime: int | float | None = None
    seed: int | None = None
    audioMode: IcLoraAudioMode | None = None
    passShape: PassShape | None = None
    controlType: IcLoraControlType | None = None
    # Advanced tiling controls of a recipe, so Reuse restores the same run on the phone.
    tilingOverlap: float | None = None
    carryFrames: int | None = None
    tilingSize: IcLoraTileSize | None = None
    maxWindowFrames: int | None = None
    width: int | None = None
    height: int | None = None
    provider: ImageProvider | None = None


class RemoteVideoInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    startFrame: RemoteInputAssetRef | None = None
    endFrame: RemoteInputAssetRef | None = None
    audio: RemoteInputAssetRef | None = None
    video: RemoteInputAssetRef | None = None
    image: RemoteInputAssetRef | None = None
    images: list[RemoteInputAssetRef] | None = None
    keyframes: list[RemoteKeyframe] | None = None
    # Motion Track: coordinates and pacing only, so Reuse can draw the paths again.
    paths: list[MotionPath] | None = None


class RemoteGenerationSpec(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: RemoteGenerationParams = Field(default_factory=RemoteGenerationParams)
    inputs: RemoteVideoInputs | None = None


class RemoteGeneration(GenerationCore):
    model_config = ConfigDict(strict=True, extra="forbid")

    spec: RemoteGenerationSpec
    outputs: list[RemoteAsset]

    @field_serializer("spec")
    def _serialize_spec(self, spec: RemoteGenerationSpec) -> dict[str, JsonValue]:
        return spec.model_dump(mode="json", exclude_none=True)


class RemoteQueueEntry(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    generation: RemoteGeneration
    input_assets: list[RemoteAsset]
    progress: QueueProgress | None = None
    cancellable: bool = True
    api_status: ApiActivityStatus | None = None


class RemoteQueueSnapshot(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    active: list[RemoteQueueEntry] = Field(default_factory=list[RemoteQueueEntry])
    queued: list[RemoteQueueEntry]
    done: list[RemoteQueueEntry] = Field(default_factory=list[RemoteQueueEntry])
    failed: list[RemoteQueueEntry] = Field(default_factory=list[RemoteQueueEntry])
    unseen_ids: list[str] = Field(default_factory=list[str])


def to_remote_asset(
    asset: Asset, *, signer: MediaUrlSigner, device_id: str
) -> RemoteAsset:
    thumbnail_url = (
        signer.thumbnail_url(asset.id, device_id)
        if asset.thumbnail_path is not None
        else None
    )
    return RemoteAsset(
        id=asset.id,
        media_kind=asset.media_kind,
        origin=asset.origin,
        mime_type=asset.mime_type,
        name=asset.name,
        metadata=asset.metadata,
        created_at=asset.created_at,
        bytes_url=signer.bytes_url(asset.id, device_id),
        thumbnail_url=thumbnail_url,
    )


def to_remote_list_item(
    asset: Asset, *, in_use: bool, signer: MediaUrlSigner, device_id: str
) -> RemoteAssetListItem:
    projected = to_remote_asset(asset, signer=signer, device_id=device_id)
    return RemoteAssetListItem(
        **projected.model_dump(),
        in_use=in_use,
        has_thumbnail=asset.thumbnail_path is not None,
    )


def to_remote_generation(
    generation: Generation, *, signer: MediaUrlSigner, device_id: str
) -> RemoteGeneration:
    return RemoteGeneration(
        id=generation.id,
        feature=generation.feature,
        contract_version=generation.contract_version,
        status=generation.status,
        spec=project_remote_spec(generation.spec),
        error_code=generation.error_code,
        created_at=generation.created_at,
        queued_at=generation.queued_at,
        attempt_count=generation.attempt_count,
        started_at=generation.started_at,
        finished_at=generation.finished_at,
        risk=generation.risk,
        lane=generation.lane,
        provider_job_id=generation.provider_job_id,
        outputs=[
            to_remote_asset(asset, signer=signer, device_id=device_id)
            for asset in generation.outputs
        ],
    )


def to_remote_queue_entry(
    entry: QueueEntry, *, signer: MediaUrlSigner, device_id: str
) -> RemoteQueueEntry:
    return RemoteQueueEntry(
        generation=to_remote_generation(
            entry.generation, signer=signer, device_id=device_id
        ),
        input_assets=[
            to_remote_asset(asset, signer=signer, device_id=device_id)
            for asset in entry.input_assets
        ],
        progress=entry.progress,
        cancellable=entry.cancellable,
        api_status=entry.api_status,
    )


def to_remote_queue_snapshot(
    snapshot: QueueSnapshot,
    *,
    signer: MediaUrlSigner,
    device_id: str,
    feature_is_allowed: Callable[[str], bool],
) -> RemoteQueueSnapshot:
    done = [
        to_remote_queue_entry(entry, signer=signer, device_id=device_id)
        for entry in snapshot.done
        if feature_is_allowed(entry.generation.feature)
    ]
    visible_done_ids = {entry.generation.id for entry in done}
    return RemoteQueueSnapshot(
        active=[
            to_remote_queue_entry(entry, signer=signer, device_id=device_id)
            for entry in snapshot.active
            if feature_is_allowed(entry.generation.feature)
        ],
        queued=[
            to_remote_queue_entry(entry, signer=signer, device_id=device_id)
            for entry in snapshot.queued
            if feature_is_allowed(entry.generation.feature)
        ],
        done=done,
        failed=[
            to_remote_queue_entry(entry, signer=signer, device_id=device_id)
            for entry in snapshot.failed
            if feature_is_allowed(entry.generation.feature)
        ],
        unseen_ids=[
            generation_id
            for generation_id in snapshot.unseen_ids
            if generation_id in visible_done_ids
        ],
    )


def project_remote_spec(spec: dict[str, JsonValue]) -> RemoteGenerationSpec:
    return RemoteGenerationSpec(
        params=_project_params(spec.get("params")),
        inputs=_project_inputs(spec.get("inputs")),
    )


def _project_params(value: object) -> RemoteGenerationParams:
    mapping = _string_mapping(value)
    if mapping is None:
        return RemoteGenerationParams()
    strings = {
        key: extracted
        for key in _SAFE_PARAM_STRINGS
        if (extracted := _optional_str(mapping.get(key))) is not None
    }
    numbers = {
        key: extracted
        for key in _SAFE_PARAM_NUMBERS
        if (extracted := _optional_number(mapping.get(key))) is not None
    }
    return RemoteGenerationParams(
        prompt=strings.get("prompt"),
        negativePrompt=strings.get("negativePrompt") or None,
        nagScale=_non_negative_number(numbers.get("nagScale")),
        nagAlpha=_non_negative_number(numbers.get("nagAlpha")),
        nagTau=_non_negative_number(numbers.get("nagTau")),
        model=strings.get("model"),
        resolution=_project_resolution(mapping.get("resolution")),
        duration=numbers.get("duration"),
        fps=numbers.get("fps"),
        aspectRatio=strings.get("aspectRatio"),
        promptProvenance=strings.get("promptProvenance"),
        scale=_project_lora_scale(mapping.get("loras")),
        customLoraId=_project_custom_lora_id(mapping.get("loras")),
        customLoraName=_project_custom_lora_name(mapping.get("loras")),
        numFrames=_optional_int(mapping.get("numFrames")),
        mode=_project_mode(strings.get("mode")),
        startTime=_non_negative_number(numbers.get("startTime")),
        seed=_optional_int(mapping.get("seed")),
        audioMode=_project_audio_mode(strings.get("audioMode")),
        passShape=_project_pass_shape(strings.get("passShape")),
        controlType=_project_control_type(strings.get("controlType")),
        tilingOverlap=_project_tiling_overlap(numbers.get("tilingOverlap")),
        carryFrames=_at_least(_optional_int(mapping.get("carryFrames")), MIN_CARRY_FRAMES),
        tilingSize=_project_tile_size(strings.get("tilingSize")),
        maxWindowFrames=_at_least(_optional_int(mapping.get("maxWindowFrames")), MIN_WINDOW_FRAMES),
        width=_positive_int(mapping.get("width")),
        height=_positive_int(mapping.get("height")),
        provider=_project_provider(strings.get("provider")),
    )


def _project_lora_scale(value: object) -> int | float | None:
    """Strength from ``loras[0].scale`` (recipes) — the ref itself is never projected."""
    if not isinstance(value, list) or not value:
        return None
    first = _string_mapping(cast(list[object], value)[0])
    if first is None:
        return None
    return _optional_number(first.get("scale"))


def _project_custom_lora_id(value: object) -> str | None:
    """``loras[0].customLoraId`` when the run used a saved custom LoRA."""
    if not isinstance(value, list) or not value:
        return None
    first = _string_mapping(cast(list[object], value)[0])
    return None if first is None else _optional_str(first.get("customLoraId"))


def _project_custom_lora_name(value: object) -> str | None:
    """``loras[0].displayName`` when the run used a saved custom LoRA."""
    if not isinstance(value, list) or not value:
        return None
    first = _string_mapping(cast(list[object], value)[0])
    if first is None or _optional_str(first.get("customLoraId")) is None:
        return None
    return _optional_str(first.get("displayName"))


def _project_inputs(value: object) -> RemoteVideoInputs | None:
    mapping = _string_mapping(value)
    if mapping is None:
        return None
    start = _input_asset_ref(mapping.get("startFrame"))
    end = _input_asset_ref(mapping.get("endFrame"))
    audio = _input_asset_ref(mapping.get("audio"))
    video = _input_asset_ref(mapping.get("video"))
    image = _input_asset_ref(mapping.get("image"))
    images = _input_asset_refs(mapping.get("images"))
    keyframes = _keyframes(mapping.get("keyframes"))
    paths = _motion_paths(mapping.get("paths"))
    if images is None and keyframes is None and paths is None and all(
        ref is None for ref in (start, end, audio, video, image)
    ):
        return None
    return RemoteVideoInputs(
        startFrame=start,
        endFrame=end,
        audio=audio,
        video=video,
        image=image,
        images=images,
        keyframes=keyframes,
        paths=paths,
    )


def _keyframes(value: object) -> list[RemoteKeyframe] | None:
    """The stills with frame and strength. One unusable entry drops the list, never a part of it."""
    if not isinstance(value, list):
        return None
    keyframes: list[RemoteKeyframe] = []
    for item in cast(list[object], value):
        mapping = _string_mapping(item)
        ref = None if mapping is None else _input_asset_ref(mapping)
        frame_index = None if mapping is None else _optional_int(mapping.get("frameIndex"))
        strength = None if mapping is None else _optional_number(mapping.get("strength"))
        if ref is None or frame_index is None or strength is None:
            return None
        try:
            keyframes.append(
                RemoteKeyframe(assetId=ref.assetId, frameIndex=frame_index, strength=float(strength))
            )
        except ValidationError:
            return None
    return keyframes or None


def _motion_paths(value: object) -> list[MotionPath] | None:
    if not isinstance(value, list):
        return None
    try:
        paths = [MotionPath.model_validate(item) for item in cast(list[object], value)]
    except ValidationError:
        # Create validated every path, so a stored set that fails is a damaged row. Reuse
        # then has no tracks to draw again. It never gets a part of the tracks.
        logger.warning("Dropping stored motion paths that no longer validate")
        return None
    return paths or None


def _input_asset_refs(value: object) -> list[RemoteInputAssetRef] | None:
    if not isinstance(value, list):
        return None
    refs = [_input_asset_ref(item) for item in cast(list[object], value)]
    return [ref for ref in refs if ref is not None] or None


def _input_asset_ref(value: object) -> RemoteInputAssetRef | None:
    mapping = _string_mapping(value)
    if mapping is None:
        return None
    asset_id = _optional_str(mapping.get("assetId"))
    if asset_id is None or asset_id == "":
        return None
    if "/" in asset_id or "\\" in asset_id:
        return None
    return RemoteInputAssetRef(assetId=asset_id)


def _string_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)


def _project_resolution(value: object) -> str | RemoteTargetResolution | None:
    """Tier id (`"720p"`) or a Retake/Extend `{width, height}`. Anything else is dropped."""
    text = _optional_str(value)
    if text is not None and text != "":
        return text
    mapping = _string_mapping(value)
    if mapping is None:
        return None
    width = _positive_int(mapping.get("width"))
    height = _positive_int(mapping.get("height"))
    if width is None or height is None:
        return None
    return RemoteTargetResolution(width=width, height=height)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def _non_negative_number(value: int | float | None) -> int | float | None:
    if value is None or value < 0:
        return None
    return value


def _project_mode(value: str | None) -> RetakeMode | ExtendMode | None:
    if value in _REMOTE_MODES:
        return cast(RetakeMode | ExtendMode, value)
    return None


def _project_provider(value: str | None) -> ImageProvider | None:
    if value in _REMOTE_IMAGE_PROVIDERS:
        return cast(ImageProvider, value)
    return None


def _project_audio_mode(value: str | None) -> IcLoraAudioMode | None:
    if value in _REMOTE_AUDIO_MODES:
        return cast(IcLoraAudioMode, value)
    return None


def _project_pass_shape(value: str | None) -> PassShape | None:
    if value in _REMOTE_PASS_SHAPES:
        return cast(PassShape, value)
    return None


def _project_control_type(value: str | None) -> IcLoraControlType | None:
    if value in _REMOTE_CONTROL_TYPES:
        return cast(IcLoraControlType, value)
    return None


def _project_tile_size(value: str | None) -> IcLoraTileSize | None:
    if value in _REMOTE_TILE_SIZES:
        return cast(IcLoraTileSize, value)
    return None


def _project_tiling_overlap(value: int | float | None) -> float | None:
    if value is None or not MIN_TILE_OVERLAP <= value <= MAX_TILE_OVERLAP:
        return None
    return float(value)


def _at_least(value: int | None, minimum: int) -> int | None:
    if value is None or value < minimum:
        return None
    return value


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _positive_int(value: object) -> int | None:
    number = _optional_int(value)
    if number is None or number <= 0:
        return None
    return number
