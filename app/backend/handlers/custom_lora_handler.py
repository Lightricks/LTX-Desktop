"""Custom LoRA handler: list, import (file copy or URL download), put weights back, delete.

An import runs in the background and reports progress through one session slot, the same way a
catalog download does. Weights are staged in ``loras/custom/.tmp``, checked, then moved into
``loras/custom/<id>/``. The folder gets its ``meta.json`` last, so a failed or interrupted import
never shows up as a LoRA.

A saved LoRA behaves like a catalog LoRA in Settings > Models. Deleting its weights keeps the
entry, and the weights come back from the saved link or from a chosen file (``restore_weights``).
Removing the LoRA deletes the whole folder.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from _routes._errors import HTTPError
from api_types import (
    ActiveCustomLoraImportResponse,
    CatalogDownloadStatus,
    CustomLoraImportProgressResponse,
    CustomLoraImportRequest,
    CustomLoraImportStartResponse,
    CustomLoraItem,
    CustomLoraLinkInfoResponse,
    CustomLoraListResponse,
    CustomLoraRestoreRequest,
)
from handlers.base import StateHandlerBase
from handlers.download_sessions import DownloadSessionTracker
from handlers.hf_auth_utils import optional_hf_token
from runtime_config.custom_lora_urls import (
    CustomLoraUrlError,
    hugging_face_repo_of,
    page_url_for,
    pick_hugging_face_weights,
    resolve_custom_lora_url,
    saved_source_url,
)
from runtime_config.custom_loras import (
    COVER_MEDIA_TYPES,
    MAX_CUSTOM_LORA_BYTES,
    STAGING_DIRNAME,
    WEIGHTS_SUFFIX,
    CustomLoraFileError,
    CustomLoraRecord,
    custom_lora_cover_path,
    custom_lora_dir,
    custom_loras_root,
    list_custom_loras,
    read_custom_lora,
    slugify_name,
    validate_safetensors_weights,
    write_meta,
)
from server_utils.media_validation import validate_image_file
from services.interfaces import TaskRunner
from services.url_downloader import UrlDownloadError, UrlDownloader, parse_model_card
from state.app_state_types import AppState, DownloadSessionId

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_GENERIC_FAILURE = "The LoRA could not be saved. Check the free disk space and try again."
_COVER_FORMAT_SUFFIX = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
}


def _in_use_error() -> HTTPError:
    return HTTPError(
        409,
        "A queued or running generation uses this LoRA. Wait for it to finish or cancel it, "
        "then try again.",
        code="CUSTOM_LORA_IN_USE",
    )


def _import_progress(
    status: CatalogDownloadStatus,
    *,
    item_id: str | None = None,
    downloaded_bytes: int = 0,
    expected_bytes: int = 0,
    progress: float = 0.0,
    speed_bytes_per_sec: float = 0.0,
    error: str | None = None,
) -> CustomLoraImportProgressResponse:
    return CustomLoraImportProgressResponse(
        status=status,
        custom_lora_id=item_id,
        downloaded_bytes=downloaded_bytes,
        expected_bytes=expected_bytes,
        progress=progress,
        speed_bytes_per_sec=speed_bytes_per_sec,
        error=error,
    )


@dataclass(frozen=True, slots=True)
class _UrlSource:
    """Weights fetched from a checked HTTPS link."""

    fetch_url: str
    page_url: str | None
    hf_token: str | None
    downloader: UrlDownloader


@dataclass(frozen=True, slots=True)
class _FileSource:
    """Weights copied from a file the user chose."""

    path: Path
    size: int


_Source = _UrlSource | _FileSource


class CustomLoraHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        task_runner: TaskRunner,
        config: "RuntimeConfig",
        url_downloader: UrlDownloader | None = None,
        lora_in_use: Callable[[str], bool] = lambda _custom_lora_id: False,
    ) -> None:
        super().__init__(state, lock, config)
        # Whether a queued or running generation holds a saved LoRA. The queue owns that answer.
        self._lora_in_use = lora_in_use
        self._task_runner = task_runner
        # None when the build has no URL downloader: file import still works.
        self._url_downloader = url_downloader
        self._sessions = DownloadSessionTracker(
            state,
            lock,
            session_attr="custom_lora_import_session",
            completed_attr="completed_custom_lora_import_sessions",
            build_progress=_import_progress,
            build_active=lambda session_id, item_id, progress: ActiveCustomLoraImportResponse(
                session_id=session_id, custom_lora_id=item_id, progress=progress
            ),
            busy_message="Another LoRA is being added. Wait for it to finish.",
            busy_code="CUSTOM_LORA_IMPORT_RUNNING",
        )
        # A crash leaves the staging folder of a session. No import runs yet, so all of it is stale.
        shutil.rmtree(custom_loras_root(self.models_dir) / STAGING_DIRNAME, ignore_errors=True)

    # --- listing ---
    def list_loras(self) -> CustomLoraListResponse:
        return CustomLoraListResponse(
            loras=[
                CustomLoraItem(
                    id=r.id,
                    name=r.name,
                    size_bytes=r.size_bytes,
                    source=r.source,
                    downloaded=r.downloaded,
                    can_download=r.source_url is not None,
                    page_url=r.page_url,
                    trigger=r.trigger,
                    description=r.description,
                    examples=list(r.examples),
                    cover_url=(
                        None
                        if (cover := custom_lora_cover_path(self.models_dir, r)) is None
                        else str(cover)
                    ),
                )
                for r in list_custom_loras(self.models_dir)
            ]
        )

    # --- import ---
    def start_import(self, req: CustomLoraImportRequest) -> CustomLoraImportStartResponse:
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_DOWNLOADS_DISABLED_IN_FORCE_API_MODE")
        slug = slugify_name(req.name)
        if not slug:
            raise HTTPError(
                422, "Use at least one letter or digit in the name.", code="CUSTOM_LORA_NAME_INVALID"
            )

        source: _Source
        if req.url is not None:
            source = self._link_source(req.url)
        else:
            assert req.source_path is not None  # the request model guarantees one source
            source = self._file_source(req.source_path)
        cover_source = None if req.cover_path is None else self._cover_source(req.cover_path)

        record = CustomLoraRecord(
            id=slug,
            name=req.name,
            filename=f"{slug}{WEIGHTS_SUFFIX}",
            size_bytes=0,
            source="url" if isinstance(source, _UrlSource) else "file",
            source_url=saved_source_url(source.fetch_url) if isinstance(source, _UrlSource) else None,
            page_url=source.page_url if isinstance(source, _UrlSource) else None,
            trigger=req.trigger,
            description=req.description or "",
            examples=tuple(req.examples),
            cover_filename=None if cover_source is None else cover_source[1],
        )
        session_id = self._open_session(record, source, must_be_new=True)
        return self._run_import(
            session_id,
            record,
            source,
            cover_source=None if cover_source is None else cover_source[0],
            publish=self._publish_new,
        )

    def restore_weights(
        self, custom_lora_id: str, req: CustomLoraRestoreRequest
    ) -> CustomLoraImportStartResponse:
        """Put the weights of a saved LoRA back, from its saved link or from a chosen file."""
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_DOWNLOADS_DISABLED_IN_FORCE_API_MODE")
        record = self._require_record(custom_lora_id)
        if record.downloaded:
            raise HTTPError(
                409, "This LoRA already has its weights.", code="CUSTOM_LORA_ALREADY_DOWNLOADED"
            )
        source: _Source
        if req.source_path is not None:
            source = self._file_source(req.source_path)
        elif record.source_url is not None:
            source = self._link_source(record.source_url)
        else:
            raise HTTPError(
                422, "This LoRA has no saved link. Choose the file again.", code="CUSTOM_LORA_SOURCE_MISSING"
            )
        session_id = self._open_session(record, source, must_be_new=False)
        return self._run_import(session_id, record, source, publish=self._publish_weights)

    def link_info(self, url: str) -> CustomLoraLinkInfoResponse:
        """Validate a link and read form suggestions from the Hugging Face model card behind it.

        A link that fails ``resolve_custom_lora_url`` answers ``accepted=False``. Any accepted
        link answers ``accepted=True``; a non-Hugging Face link, a missing card, or a failed card
        request leaves the suggestions empty so the form is not blocked.
        """
        try:
            resolve_custom_lora_url(url)
        except CustomLoraUrlError:
            return CustomLoraLinkInfoResponse(accepted=False)
        repo = hugging_face_repo_of(url)
        if repo is None or self._url_downloader is None:
            return CustomLoraLinkInfoResponse(accepted=True)
        hf_token = optional_hf_token(self.state, self.lock)
        try:
            card = self._url_downloader.fetch_hugging_face_card(repo, hf_token=hf_token)
        except UrlDownloadError as exc:
            logger.info("No model card for %s: %s", repo.repo_id, exc)
            return CustomLoraLinkInfoResponse(accepted=True)
        suggestions = parse_model_card(card)
        return CustomLoraLinkInfoResponse(
            accepted=True,
            trigger=suggestions.trigger,
            description=suggestions.description,
            examples=list(suggestions.examples),
        )

    def _link_source(self, url: str) -> _UrlSource:
        if self._url_downloader is None:
            raise HTTPError(
                409, "Adding a LoRA from a URL is not available.", code="CUSTOM_LORA_URL_UNAVAILABLE"
            )
        # The token is read before the session opens. Only a Hugging Face host receives it.
        hf_token = optional_hf_token(self.state, self.lock)
        try:
            resolved = resolve_custom_lora_url(url)
            fetch_url = resolved.url
            if resolved.repo is not None:
                # A model page names no file. Take the one weights file the repo holds.
                filename = pick_hugging_face_weights(
                    self._url_downloader.list_hugging_face_files(resolved.repo, hf_token=hf_token)
                )
                fetch_url = resolved.repo.file_url(filename)
        except (CustomLoraUrlError, UrlDownloadError) as exc:
            raise HTTPError(422, str(exc), code="CUSTOM_LORA_URL_INVALID") from exc
        return _UrlSource(
            fetch_url=fetch_url,
            page_url=page_url_for(fetch_url),
            hf_token=hf_token,
            downloader=self._url_downloader,
        )

    @staticmethod
    def _cover_source(cover_path: str) -> tuple[Path, str]:
        """The cover image and its stored filename, or a 422 the form can show."""
        if Path(cover_path).suffix.lower() not in COVER_MEDIA_TYPES:
            raise HTTPError(
                422, "Choose a PNG, JPG, or WebP image.", code="CUSTOM_LORA_COVER_INVALID"
            )
        try:
            file_path = validate_image_file(cover_path)
            with Image.open(file_path) as img:
                suffix = _COVER_FORMAT_SUFFIX.get(str(img.format or "").upper())
                if suffix is None:
                    raise ValueError(img.format)
        except (HTTPError, ValueError, OSError) as exc:
            raise HTTPError(
                422, "Choose a PNG, JPG, or WebP image.", code="CUSTOM_LORA_COVER_INVALID"
            ) from exc
        return file_path, f"cover{suffix}"

    @staticmethod
    def _file_source(source_path: str) -> _FileSource:
        path = Path(source_path)
        if path.suffix.lower() != WEIGHTS_SUFFIX or not path.is_file():
            raise HTTPError(422, "Choose a .safetensors file.", code="CUSTOM_LORA_FILE_INVALID")
        size = path.stat().st_size
        if size > MAX_CUSTOM_LORA_BYTES:
            raise HTTPError(422, "The file is too large for a LoRA.", code="CUSTOM_LORA_FILE_INVALID")
        return _FileSource(path=path, size=size)

    def _open_session(
        self, record: CustomLoraRecord, source: _Source, *, must_be_new: bool
    ) -> DownloadSessionId:
        """Take the one session slot. A new LoRA also needs a free name."""

        def _require_free_name() -> None:
            if must_be_new and custom_lora_dir(self.models_dir, record.id).exists():
                raise HTTPError(
                    409,
                    "A custom LoRA with this name already exists.",
                    code="CUSTOM_LORA_NAME_TAKEN",
                )

        return self._sessions.open(
            record.id,
            source.size if isinstance(source, _FileSource) else 0,
            before_open=_require_free_name,
        )

    def _run_import(
        self,
        session_id: DownloadSessionId,
        record: CustomLoraRecord,
        source: _Source,
        *,
        cover_source: Path | None = None,
        publish: Callable[[Path, Path, CustomLoraRecord], None],
    ) -> CustomLoraImportStartResponse:
        """Fetch or copy the weights in the background, check them, then publish them."""
        models_dir = self.models_dir

        def _work() -> None:
            staging = custom_loras_root(models_dir) / STAGING_DIRNAME / session_id
            staging.mkdir(parents=True, exist_ok=True)
            try:
                if cover_source is not None and record.cover_filename is not None:
                    shutil.copyfile(cover_source, staging / record.cover_filename)
                staged = staging / record.filename
                match source:
                    case _UrlSource():
                        source.downloader.download(
                            source.fetch_url,
                            staged,
                            hf_token=source.hf_token,
                            max_bytes=MAX_CUSTOM_LORA_BYTES,
                            on_progress=lambda done, total: self._sessions.update_progress(
                                session_id, done, total
                            ),
                        )
                    case _FileSource():
                        self._copy(source.path, staged, session_id)
                validate_safetensors_weights(staged)
                publish(models_dir, staged, replace(record, size_bytes=staged.stat().st_size))
            finally:
                shutil.rmtree(staging, ignore_errors=True)
                self._remove_empty_staging_root(models_dir)
            self._sessions.finish(session_id)

        self._task_runner.run_background(
            _work,
            task_name="custom-lora-import",
            on_error=lambda exc: self._fail(session_id, exc),
            daemon=True,
        )
        return CustomLoraImportStartResponse(
            status="started", sessionId=session_id, custom_lora_id=record.id
        )

    def _copy(self, source: Path, dest: Path, session_id: DownloadSessionId) -> None:
        copied = 0
        with open(source, "rb") as src, open(dest, "wb") as out:
            while chunk := src.read(_COPY_CHUNK_BYTES):
                out.write(chunk)
                copied += len(chunk)
                self._sessions.update_progress(session_id, copied, 0)

    @staticmethod
    def _publish_new(models_dir: Path, staged: Path, record: CustomLoraRecord) -> None:
        final_dir = custom_lora_dir(models_dir, record.id)
        final_dir.mkdir(parents=True, exist_ok=False)
        try:
            staged.replace(final_dir / record.filename)
            if record.cover_filename is not None:
                (staged.parent / record.cover_filename).replace(final_dir / record.cover_filename)
            write_meta(final_dir, record)
        except BaseException:
            shutil.rmtree(final_dir, ignore_errors=True)
            raise

    @staticmethod
    def _publish_weights(models_dir: Path, staged: Path, record: CustomLoraRecord) -> None:
        """Move the weights into the folder of an existing LoRA and save the new size."""
        final_dir = custom_lora_dir(models_dir, record.id)
        staged.replace(final_dir / record.filename)
        write_meta(final_dir, record)

    @staticmethod
    def _remove_empty_staging_root(models_dir: Path) -> None:
        root = custom_loras_root(models_dir) / STAGING_DIRNAME
        try:
            root.rmdir()
        except OSError:
            pass  # not empty or already gone

    def _fail(self, session_id: DownloadSessionId, exc: Exception) -> None:
        # These two carry a message written for the user. Anything else (disk, permissions)
        # is logged by the task runner and shown as a generic message.
        known = isinstance(exc, (UrlDownloadError, CustomLoraFileError))
        self._sessions.fail(session_id, str(exc) if known else _GENERIC_FAILURE)

    def get_import_progress(self, session_id: str) -> CustomLoraImportProgressResponse:
        return self._sessions.get_progress(session_id)

    def get_active_import(self) -> ActiveCustomLoraImportResponse:
        return self._sessions.get_active()

    # --- delete ---
    def _require_dir(self, custom_lora_id: str) -> Path:
        root = custom_loras_root(self.models_dir).resolve()
        try:
            target = custom_lora_dir(self.models_dir, custom_lora_id).resolve()
        except ValueError as exc:
            raise HTTPError(404, "CUSTOM_LORA_NOT_FOUND") from exc
        # The route pattern already limits the id. This keeps a bad id from ever leaving the root.
        if target.parent != root or custom_lora_id == STAGING_DIRNAME or not target.is_dir():
            raise HTTPError(404, "CUSTOM_LORA_NOT_FOUND")
        return target

    def _require_record(self, custom_lora_id: str) -> CustomLoraRecord:
        self._require_dir(custom_lora_id)
        record = read_custom_lora(self.models_dir, custom_lora_id)
        if record is None:
            raise HTTPError(404, "CUSTOM_LORA_NOT_FOUND")
        return record

    def _refuse_while_importing(self, custom_lora_id: str) -> None:
        self._sessions.refuse_if_item_active(
            custom_lora_id, message="CUSTOM_LORA_IMPORT_RUNNING"
        )

    def _refuse_while_in_use(self, custom_lora_id: str) -> None:
        if self._lora_in_use(custom_lora_id):
            raise _in_use_error()

    def delete_weights(self, custom_lora_id: str) -> None:
        """Delete the weights and keep the entry, like Delete on a catalog LoRA."""
        record = self._require_record(custom_lora_id)
        self._refuse_while_importing(custom_lora_id)
        self._refuse_while_in_use(custom_lora_id)
        weights = custom_lora_dir(self.models_dir, custom_lora_id) / record.filename
        if not weights.is_file():
            raise HTTPError(404, "CUSTOM_LORA_NOT_DOWNLOADED")
        try:
            weights.unlink()
        except PermissionError as exc:
            # Windows refuses to remove a file the pipeline has open.
            raise _in_use_error() from exc

    def cover_file(self, custom_lora_id: str) -> tuple[Path, str] | None:
        """The cover file and its media type, or ``None`` when this LoRA has no cover."""
        record = read_custom_lora(self.models_dir, custom_lora_id)
        if record is None:
            return None
        path = custom_lora_cover_path(self.models_dir, record)
        if path is None:
            return None
        media_type = COVER_MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            return None
        return path, media_type

    def delete(self, custom_lora_id: str) -> None:
        """Remove the LoRA: weights, entry, and folder."""
        target = self._require_dir(custom_lora_id)
        self._refuse_while_importing(custom_lora_id)
        self._refuse_while_in_use(custom_lora_id)
        try:
            shutil.rmtree(target)
        except PermissionError as exc:
            raise _in_use_error() from exc
