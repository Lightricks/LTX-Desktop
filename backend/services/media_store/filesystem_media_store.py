"""Filesystem implementation of the opaque media store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from services.media_store.media_store import MediaRecord, MediaType, StagedMedia

_ID_PATTERN = re.compile(r"^(?:med|art)_[A-Za-z0-9_-]{24,}$")
_CHUNK_BYTES = 1024 * 1024


class MediaTooLargeError(Exception):
    pass


class InvalidMediaStorePathError(Exception):
    pass


class _StoredMetadata(BaseModel):
    model_config = ConfigDict(strict=True)

    schema_version: Literal[1] = 1
    id: str
    media_type: MediaType
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    expires_at: datetime
    relative_path: str


def _now() -> datetime:
    return datetime.now(UTC)


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    return name[:255] or "media"


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        return suffix
    return ".bin"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class FilesystemMediaStore:
    def __init__(
        self,
        *,
        app_data_dir: Path,
        outputs_dir: Path,
        upload_ttl: timedelta = timedelta(hours=24),
        artifact_ttl: timedelta = timedelta(days=7),
    ) -> None:
        self._root = (app_data_dir / "media").resolve()
        self._staging_dir = self._root / "staging"
        self._uploads_dir = self._root / "uploads"
        self._artifacts_dir = self._root / "artifacts"
        self._outputs_dir = outputs_dir.resolve()
        self._upload_ttl = upload_ttl
        self._artifact_ttl = artifact_ttl
        self._lock = threading.RLock()
        for directory in (self._root, self._staging_dir, self._uploads_dir, self._artifacts_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _new_id(prefix: Literal["med", "art"]) -> str:
        return f"{prefix}_{secrets.token_urlsafe(24)}"

    @staticmethod
    def _write_metadata(path: Path, metadata: _StoredMetadata) -> None:
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(metadata.model_dump_json(), encoding="utf-8")
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            pass
        temp_path.replace(path)

    def stage_upload(self, source: BinaryIO, *, filename: str, max_bytes: int) -> StagedMedia:
        token = secrets.token_urlsafe(24)
        path = self._staging_dir / f"{token}.part{_safe_suffix(filename)}"
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("xb") as output:
                while True:
                    chunk = source.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise MediaTooLargeError()
                    digest.update(chunk)
                    output.write(chunk)
            return StagedMedia(token=token, path=path, size_bytes=size, sha256=digest.hexdigest())
        except Exception:
            path.unlink(missing_ok=True)
            raise

    def commit_upload(
        self,
        staged: StagedMedia,
        *,
        media_type: MediaType,
        filename: str,
        content_type: str,
    ) -> MediaRecord:
        with self._lock:
            media_id = self._new_id("med")
            directory = self._uploads_dir / media_id
            directory.mkdir(parents=False, exist_ok=False)
            content_path = directory / f"content{_safe_suffix(filename)}"
            try:
                staged.path.replace(content_path)
                expires_at = _now() + self._upload_ttl
                metadata = _StoredMetadata(
                    id=media_id,
                    media_type=media_type,
                    filename=_safe_filename(filename),
                    content_type=content_type,
                    size_bytes=staged.size_bytes,
                    sha256=staged.sha256,
                    expires_at=expires_at,
                    relative_path=content_path.name,
                )
                self._write_metadata(directory / "metadata.json", metadata)
                return self._to_record(metadata, content_path)
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    def discard_staged(self, staged: StagedMedia) -> None:
        staged.path.unlink(missing_ok=True)

    def _read_record(self, metadata_path: Path, *, root: Path) -> MediaRecord | None:
        try:
            metadata = _StoredMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
            if not _ID_PATTERN.fullmatch(metadata.id) or metadata_path.parent.name != metadata.id:
                return None
            path = (metadata_path.parent / metadata.relative_path).resolve(strict=True)
            resolved_root = root.resolve()
            if not _is_within(path, resolved_root) or not path.is_file():
                return None
            return self._to_record(metadata, path)
        except (OSError, ValidationError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _to_record(metadata: _StoredMetadata, path: Path) -> MediaRecord:
        return MediaRecord(
            id=metadata.id,
            media_type=metadata.media_type,
            filename=metadata.filename,
            content_type=metadata.content_type,
            size_bytes=metadata.size_bytes,
            sha256=metadata.sha256,
            expires_at=metadata.expires_at,
            path=path,
        )

    def resolve_upload(self, media_id: str) -> MediaRecord | None:
        if not media_id.startswith("med_") or not _ID_PATTERN.fullmatch(media_id):
            return None
        with self._lock:
            record = self._read_record(self._uploads_dir / media_id / "metadata.json", root=self._uploads_dir)
            if record is None or record.expires_at <= _now():
                return None
            return record

    def delete_upload(self, media_id: str) -> bool:
        if not media_id.startswith("med_") or not _ID_PATTERN.fullmatch(media_id):
            return False
        with self._lock:
            directory = self._uploads_dir / media_id
            if not directory.is_dir():
                return False
            shutil.rmtree(directory)
            return True

    def register_artifact(self, path: Path, *, media_type: MediaType, content_type: str) -> MediaRecord:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not _is_within(resolved, self._outputs_dir):
            raise InvalidMediaStorePathError("Artifact must be a file inside the outputs directory")
        with self._lock:
            artifact_id = self._new_id("art")
            metadata_path = self._artifacts_dir / artifact_id / "metadata.json"
            metadata_path.parent.mkdir(parents=False, exist_ok=False)
            digest = hashlib.sha256()
            with resolved.open("rb") as artifact_file:
                while chunk := artifact_file.read(_CHUNK_BYTES):
                    digest.update(chunk)
            metadata = _StoredMetadata(
                id=artifact_id,
                media_type=media_type,
                filename=resolved.name,
                content_type=content_type,
                size_bytes=resolved.stat().st_size,
                sha256=digest.hexdigest(),
                expires_at=_now() + self._artifact_ttl,
                relative_path=str(resolved.relative_to(self._outputs_dir)),
            )
            try:
                self._write_metadata(metadata_path, metadata)
            except Exception:
                shutil.rmtree(metadata_path.parent, ignore_errors=True)
                raise
            return self._to_record(metadata, resolved)

    def resolve_artifact(self, artifact_id: str) -> MediaRecord | None:
        if not artifact_id.startswith("art_") or not _ID_PATTERN.fullmatch(artifact_id):
            return None
        with self._lock:
            metadata_path = self._artifacts_dir / artifact_id / "metadata.json"
            try:
                metadata = _StoredMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
                if metadata.id != artifact_id:
                    return None
                path = (self._outputs_dir / metadata.relative_path).resolve(strict=True)
                if not path.is_file() or not _is_within(path, self._outputs_dir):
                    return None
                record = self._to_record(metadata, path)
            except (OSError, ValidationError, ValueError, json.JSONDecodeError):
                return None
            if record.expires_at <= _now():
                return None
            return record

    def delete_artifact(self, artifact_id: str) -> bool:
        record = self.resolve_artifact(artifact_id)
        if record is None:
            return False
        with self._lock:
            record.path.unlink(missing_ok=True)
            shutil.rmtree(self._artifacts_dir / artifact_id, ignore_errors=True)
            return True

    def cleanup_expired(self) -> None:
        now = _now()
        with self._lock:
            for staged in self._staging_dir.iterdir():
                try:
                    if datetime.fromtimestamp(staged.stat().st_mtime, UTC) + timedelta(hours=1) <= now:
                        staged.unlink(missing_ok=True)
                except OSError:
                    continue
            for directory in self._uploads_dir.iterdir():
                record = self._read_record(directory / "metadata.json", root=self._uploads_dir)
                if record is not None and record.expires_at <= now:
                    shutil.rmtree(directory, ignore_errors=True)
            for directory in self._artifacts_dir.iterdir():
                try:
                    metadata = _StoredMetadata.model_validate_json(
                        (directory / "metadata.json").read_text(encoding="utf-8")
                    )
                except (OSError, ValidationError):
                    continue
                if metadata.expires_at > now:
                    continue
                try:
                    artifact_path = (self._outputs_dir / metadata.relative_path).resolve(strict=True)
                    if artifact_path.is_file() and _is_within(artifact_path, self._outputs_dir):
                        artifact_path.unlink(missing_ok=True)
                except OSError:
                    pass
                shutil.rmtree(directory, ignore_errors=True)
