"""Media upload, opaque reference resolution, and artifact registration."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from threading import RLock
from typing import BinaryIO, TYPE_CHECKING

from PIL import Image

from _routes._errors import HTTPError
from api_types import ArtifactRef, MediaRef
from handlers.base import StateHandlerBase
from server_utils.media_validation import (
    MAX_MEDIA_BYTES,
    validate_audio_file,
    validate_image_file,
    validate_video_file,
)
from services.interfaces import (
    MediaRecord,
    MediaStore,
    MediaTooLargeError,
    MediaType,
    VideoProcessor,
)
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

_IMAGE_CONTENT_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


class MediaHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        media_store: MediaStore,
        video_processor: VideoProcessor,
    ) -> None:
        super().__init__(state, lock, config)
        self._store = media_store
        self._video_processor = video_processor
        self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        try:
            self._store.cleanup_expired()
        except Exception:
            logger.warning("Media-store cleanup failed", exc_info=True)

    def upload(self, source: BinaryIO, *, filename: str, media_type: MediaType) -> MediaRef:
        self._cleanup_expired()
        try:
            staged = self._store.stage_upload(
                source,
                filename=filename,
                max_bytes=MAX_MEDIA_BYTES[media_type],
            )
        except MediaTooLargeError:
            raise HTTPError(413, "MEDIA_TOO_LARGE") from None
        except Exception as exc:
            raise HTTPError(500, "MEDIA_UPLOAD_FAILED") from exc
        try:
            content_type = self._validate_and_detect(staged.path, media_type)
            record = self._store.commit_upload(
                staged,
                media_type=media_type,
                filename=filename,
                content_type=content_type,
            )
        except HTTPError as exc:
            self._store.discard_staged(staged)
            raise HTTPError(400, "INVALID_MEDIA") from exc
        except Exception as exc:
            self._store.discard_staged(staged)
            raise HTTPError(500, "MEDIA_UPLOAD_FAILED") from exc
        return self._media_ref(record)

    def _validate_and_detect(self, path: Path, media_type: MediaType) -> str:
        if media_type == "image":
            validate_image_file(str(path))
            with Image.open(path) as image:
                return _IMAGE_CONTENT_TYPES.get(str(image.format or "").upper(), "application/octet-stream")
        if media_type == "audio":
            validate_audio_file(str(path))
        else:
            validate_video_file(str(path), self._video_processor)
        guessed, _encoding = mimetypes.guess_type(path.name)
        return guessed or ("audio/mpeg" if media_type == "audio" else "video/mp4")

    def resolve_input(
        self,
        *,
        media_id: str | None,
        legacy_path: str | None,
        expected_type: MediaType,
        required: bool = False,
    ) -> Path | None:
        normalized_id = media_id.strip() if media_id else ""
        normalized_path = legacy_path.strip() if legacy_path else ""
        if normalized_id and normalized_path:
            raise HTTPError(422, "MEDIA_ID_AND_PATH_ARE_MUTUALLY_EXCLUSIVE")
        if normalized_id:
            record = self._store.resolve_upload(normalized_id)
            if record is None:
                raise HTTPError(404, "MEDIA_NOT_FOUND")
            if record.media_type != expected_type:
                raise HTTPError(409, "MEDIA_TYPE_MISMATCH")
            return record.path
        if normalized_path:
            if not self.config.allow_legacy_path_inputs:
                raise HTTPError(422, "REMOTE_PATH_INPUT_DISABLED")
            return Path(normalized_path)
        if required:
            raise HTTPError(422, "MEDIA_INPUT_REQUIRED")
        return None

    def register_artifact(self, path: str | Path, *, media_type: MediaType) -> ArtifactRef:
        self._cleanup_expired()
        artifact_path = Path(path)
        guessed, _encoding = mimetypes.guess_type(artifact_path.name)
        fallback = "image/png" if media_type == "image" else "video/mp4"
        try:
            record = self._store.register_artifact(
                artifact_path,
                media_type=media_type,
                content_type=guessed or fallback,
            )
        except Exception as exc:
            raise HTTPError(500, "ARTIFACT_REGISTRATION_FAILED") from exc
        return self._artifact_ref(record)

    def resolve_artifact(self, artifact_id: str) -> MediaRecord:
        record = self._store.resolve_artifact(artifact_id)
        if record is None:
            raise HTTPError(404, "ARTIFACT_NOT_FOUND")
        return record

    def delete_upload(self, media_id: str) -> None:
        if not self._store.delete_upload(media_id):
            raise HTTPError(404, "MEDIA_NOT_FOUND")

    def delete_artifact(self, artifact_id: str) -> None:
        if not self._store.delete_artifact(artifact_id):
            raise HTTPError(404, "ARTIFACT_NOT_FOUND")

    @staticmethod
    def _media_ref(record: MediaRecord) -> MediaRef:
        return MediaRef(
            media_id=record.id,
            media_type=record.media_type,
            filename=record.filename,
            content_type=record.content_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            expires_at=record.expires_at,
        )

    @staticmethod
    def _artifact_ref(record: MediaRecord) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=record.id,
            media_type=record.media_type,
            filename=record.filename,
            content_type=record.content_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
            expires_at=record.expires_at,
        )
