from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from api_types import (
    AssetMetadata,
    GenerationErrorCode,
    GenerationStatus,
    MediaKind,
    Origin,
    require_failed_has_error_code,
    require_media_kind_matches_metadata,
)

Status = GenerationStatus


class StatusError(Exception):
    def __init__(self, generation_id: str, status: str) -> None:
        self.generation_id = generation_id
        self.status = status
        super().__init__(f"generation {generation_id} is {status}")


class AssetInUseError(Exception):
    def __init__(self, asset_id: str) -> None:
        self.asset_id = asset_id
        super().__init__(f"asset {asset_id} is in use")


class AttemptError(Exception):
    def __init__(self, generation_id: str, expected: int, actual: int) -> None:
        self.generation_id = generation_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"generation {generation_id} attempt mismatch: expected {expected}, actual {actual}"
        )


class MarkSucceededError(Exception):
    generation_id: str
    error_code: GenerationErrorCode

    def __init__(self, generation_id: str, error_code: GenerationErrorCode) -> None:
        self.generation_id = generation_id
        self.error_code = error_code
        super().__init__(error_code)


class CapabilityFailedError(Exception):
    """Executor could not run this generation with the current local capability."""

    def __init__(self, detail: str, *, code: str | None = None) -> None:
        self.detail = detail
        self.code = code
        super().__init__(detail)


class UnavailableError(Exception):
    """Raised when the sqlite store cannot be opened or recovered."""


MediaErrorCode = Literal[
    "UNSUPPORTED_MEDIA",
    "FILE_TOO_LARGE",
    "NO_AUDIO_STREAM",
    "IMAGE_DIMENSIONS_TOO_LARGE",
    "UNREADABLE_MEDIA",
    "INVALID_TRIM_RANGE",
    "TRIM_BUSY",
]


class MediaError(Exception):
    def __init__(self, code: MediaErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class _RecordModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class AssetRecord(_RecordModel):
    id: str = Field(min_length=1)
    media_kind: MediaKind
    origin: Origin
    path: str = Field(min_length=1)
    thumbnail_path: str | None = None
    mime_type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    metadata: AssetMetadata
    created_at: int = Field(ge=0)

    @model_validator(mode="after")
    def _media_kind_matches_metadata(self) -> AssetRecord:
        require_media_kind_matches_metadata(self.media_kind, self.metadata)
        return self


class ListedAsset(_RecordModel):
    asset: AssetRecord
    in_use: bool


class AssetListPage(_RecordModel):
    items: tuple[ListedAsset, ...]
    next_cursor: str | None


class OutputSpec(_RecordModel):
    asset_id: str = Field(min_length=1)
    dest_path: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    mime_type: str = Field(min_length=1)
    name: str = Field(min_length=1)


class PairedDeviceRecord(_RecordModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    token_hash: str = Field(min_length=1)
    created_at: int = Field(ge=0)
    last_seen_at: int = Field(ge=0)
    first_seen_ip: str | None = None
    first_seen_user_agent: str | None = None
    revoked_at: int | None = None


class GenerationRecord(_RecordModel):
    id: str = Field(min_length=1)
    feature: str = Field(min_length=1)
    contract_version: int = Field(ge=1)
    status: Status
    spec: dict[str, JsonValue]
    error_code: GenerationErrorCode | None = None
    created_at: int = Field(ge=0)
    queued_at: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    started_at: int | None = None
    finished_at: int | None = None
    outputs: tuple[AssetRecord, ...]

    @model_validator(mode="after")
    def _failed_has_error_code(self) -> GenerationRecord:
        require_failed_has_error_code(self.status, self.error_code)
        return self


class QueueGenerationRecord(_RecordModel):
    generation: GenerationRecord
    input_assets: tuple[AssetRecord, ...]
