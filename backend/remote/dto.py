"""Remote-safe Asset and Generation projections.

Desktop `Asset`/`Generation` stay the storage and local HTTP contract. The phone
only receives this allowlisted view: no filesystem paths, no LoRA refs, and no
arbitrary stored spec JSON.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_serializer

from api_types import (
    Asset,
    AssetCore,
    ExtendMode,
    IcLoraAudioMode,
    Generation,
    GenerationCore,
    QueueEntry,
    QueueProgress,
    QueueSnapshot,
    RetakeMode,
)
from remote.media_urls import MediaUrlSigner

_SAFE_PARAM_STRINGS = frozenset(
    {
        "prompt",
        "model",
        "aspectRatio",
        "promptProvenance",
        "mode",
        "audioMode",
    }
)
_REMOTE_AUDIO_MODES = frozenset(get_args(IcLoraAudioMode))
_SAFE_PARAM_NUMBERS = frozenset({"duration", "fps", "startTime"})
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


class RemoteTargetResolution(BaseModel):
    """Home Retake/Extend target size. Tier ids stay strings."""

    model_config = ConfigDict(strict=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RemoteVideoParams(BaseModel):
    """T2V/I2V/A2V form and result-meta fields only."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str | None = None
    model: str | None = None
    resolution: str | RemoteTargetResolution | None = None
    duration: int | float | None = None
    fps: int | float | None = None
    aspectRatio: str | None = None
    promptProvenance: str | None = None
    scale: int | float | None = None
    numFrames: int | None = None
    mode: RetakeMode | ExtendMode | None = None
    startTime: int | float | None = None
    seed: int | None = None
    audioMode: IcLoraAudioMode | None = None


class RemoteVideoInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    startFrame: RemoteInputAssetRef | None = None
    endFrame: RemoteInputAssetRef | None = None
    audio: RemoteInputAssetRef | None = None
    video: RemoteInputAssetRef | None = None
    image: RemoteInputAssetRef | None = None


class RemoteGenerationSpec(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: RemoteVideoParams = Field(default_factory=RemoteVideoParams)
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


class RemoteQueueSnapshot(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    active: RemoteQueueEntry | None
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
    )


def to_remote_queue_snapshot(
    snapshot: QueueSnapshot,
    *,
    signer: MediaUrlSigner,
    device_id: str,
    feature_is_allowed: Callable[[str], bool],
) -> RemoteQueueSnapshot:
    active = snapshot.active
    done = [
        to_remote_queue_entry(entry, signer=signer, device_id=device_id)
        for entry in snapshot.done
        if feature_is_allowed(entry.generation.feature)
    ]
    visible_done_ids = {entry.generation.id for entry in done}
    return RemoteQueueSnapshot(
        active=(
            to_remote_queue_entry(active, signer=signer, device_id=device_id)
            if active is not None and feature_is_allowed(active.generation.feature)
            else None
        ),
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


def _project_params(value: object) -> RemoteVideoParams:
    mapping = _string_mapping(value)
    if mapping is None:
        return RemoteVideoParams()
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
    return RemoteVideoParams(
        prompt=strings.get("prompt"),
        model=strings.get("model"),
        resolution=_project_resolution(mapping.get("resolution")),
        duration=numbers.get("duration"),
        fps=numbers.get("fps"),
        aspectRatio=strings.get("aspectRatio"),
        promptProvenance=strings.get("promptProvenance"),
        scale=_project_lora_scale(mapping.get("loras")),
        numFrames=_optional_int(mapping.get("numFrames")),
        mode=_project_mode(strings.get("mode")),
        startTime=_non_negative_number(numbers.get("startTime")),
        seed=_optional_int(mapping.get("seed")),
        audioMode=_project_audio_mode(strings.get("audioMode")),
    )


def _project_lora_scale(value: object) -> int | float | None:
    """Strength from ``loras[0].scale`` (recipes) — the ref itself is never projected."""
    if not isinstance(value, list) or not value:
        return None
    first = _string_mapping(cast(list[object], value)[0])
    if first is None:
        return None
    return _optional_number(first.get("scale"))


def _project_inputs(value: object) -> RemoteVideoInputs | None:
    mapping = _string_mapping(value)
    if mapping is None:
        return None
    start = _input_asset_ref(mapping.get("startFrame"))
    end = _input_asset_ref(mapping.get("endFrame"))
    audio = _input_asset_ref(mapping.get("audio"))
    video = _input_asset_ref(mapping.get("video"))
    image = _input_asset_ref(mapping.get("image"))
    if all(ref is None for ref in (start, end, audio, video, image)):
        return None
    return RemoteVideoInputs(
        startFrame=start, endFrame=end, audio=audio, video=video, image=image
    )


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


def _project_audio_mode(value: str | None) -> IcLoraAudioMode | None:
    if value in _REMOTE_AUDIO_MODES:
        return cast(IcLoraAudioMode, value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _positive_int(value: object) -> int | None:
    number = _optional_int(value)
    if number is None or number <= 0:
        return None
    return number
