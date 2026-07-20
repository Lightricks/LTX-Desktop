"""Protocol and lightweight value types for media storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Literal, Protocol

MediaType = Literal["image", "audio", "video"]


@dataclass(frozen=True, slots=True)
class StagedMedia:
    token: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MediaRecord:
    id: str
    media_type: MediaType
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    expires_at: datetime
    path: Path


class MediaStore(Protocol):
    def stage_upload(self, source: BinaryIO, *, filename: str, max_bytes: int) -> StagedMedia:
        ...

    def commit_upload(
        self,
        staged: StagedMedia,
        *,
        media_type: MediaType,
        filename: str,
        content_type: str,
    ) -> MediaRecord:
        ...

    def discard_staged(self, staged: StagedMedia) -> None:
        ...

    def resolve_upload(self, media_id: str) -> MediaRecord | None:
        ...

    def delete_upload(self, media_id: str) -> bool:
        ...

    def register_artifact(self, path: Path, *, media_type: MediaType, content_type: str) -> MediaRecord:
        ...

    def resolve_artifact(self, artifact_id: str) -> MediaRecord | None:
        ...

    def delete_artifact(self, artifact_id: str) -> bool:
        ...

    def cleanup_expired(self) -> None:
        ...
