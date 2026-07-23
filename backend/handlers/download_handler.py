"""Checkpoint download session handler."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterable
from threading import RLock
from typing import TYPE_CHECKING
from uuid import uuid4

import requests as http_requests

from _routes._errors import HTTPError
from api_types import (
    ActiveDownloadResponse,
    CheckModelAccessResponse,
    DownloadProgressCompleteResponse,
    DownloadProgressErrorResponse,
    DownloadProgressResponse,
    DownloadProgressRunningResponse,
    ModelAccessStatus,
    ModelCheckpointID,
)
from handlers.base import StateHandlerBase, with_state_lock
from handlers.hf_auth_utils import optional_hf_token
from handlers.models_handler import ModelsHandler
from runtime_config.model_download_specs import (
    ALL_MODEL_CP_IDS,
    get_model_cp_spec,
    is_cp_downloaded,
    resolve_downloading_dir,
    resolve_downloading_path,
    resolve_downloading_target_path,
    resolve_model_path,
)
from services.interfaces import ModelDownloader, TaskRunner
from state.app_state_types import (
    AppState,
    DownloadSessionComplete,
    DownloadSessionError,
    DownloadSessionId,
    DownloadingSession,
    FileDownloadRunning,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class DownloadHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        models_handler: ModelsHandler,
        model_downloader: ModelDownloader,
        task_runner: TaskRunner,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._models_handler = models_handler
        self._model_downloader = model_downloader
        self._task_runner = task_runner

    def _ordered_cp_ids(self, cp_ids: Iterable[ModelCheckpointID]) -> tuple[ModelCheckpointID, ...]:
        cp_id_set = set(cp_ids)
        return tuple(cp_id for cp_id in ALL_MODEL_CP_IDS if cp_id in cp_id_set)

    @with_state_lock
    def is_download_running(self) -> bool:
        return self.state.downloading_session is not None

    def get_active_download(self) -> ActiveDownloadResponse:
        session = self.state.downloading_session
        if session is None:
            return ActiveDownloadResponse(session_id=None, cp_ids=[])
        return ActiveDownloadResponse(
            session_id=str(session.id),
            cp_ids=sorted(session.files_to_download),
        )

    @with_state_lock
    def start_download(self, cp_ids: set[ModelCheckpointID]) -> DownloadSessionId:
        session_id = DownloadSessionId(uuid4().hex)
        self.state.downloading_session = DownloadingSession(
            id=session_id,
            current_running_file=None,
            files_to_download=cp_ids,
            completed_files=set(),
            completed_bytes=0,
        )
        return session_id

    @with_state_lock
    def start_file(self, cp_id: ModelCheckpointID, target: str) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        if session.current_running_file is not None:
            session.completed_bytes += session.current_running_file.downloaded_bytes
            session.completed_files.add(session.current_running_file.file_type)
        session.current_running_file = FileDownloadRunning(
            file_type=cp_id,
            target_path=target,
            downloaded_bytes=0,
            speed_bytes_per_sec=0.0,
        )

    @with_state_lock
    def finish_download(self) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        if session.current_running_file is not None:
            session.completed_bytes += session.current_running_file.downloaded_bytes
            session.completed_files.add(session.current_running_file.file_type)
        self.state.completed_download_sessions[session.id] = DownloadSessionComplete()
        self.state.downloading_session = None

    @with_state_lock
    def update_file_progress(self, cp_id: ModelCheckpointID, downloaded: int, speed_bytes_per_sec: float) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        current = session.current_running_file
        if current is None or current.file_type != cp_id:
            return
        current.downloaded_bytes = downloaded
        current.speed_bytes_per_sec = speed_bytes_per_sec

    @with_state_lock
    def fail_download(self, error: str) -> None:
        logger.error("Checkpoint download failed: %s", error)
        session = self.state.downloading_session
        if session is not None:
            self.state.completed_download_sessions[session.id] = DownloadSessionError(error_message=error)
            self.state.downloading_session = None

    def _make_progress_callback(self, cp_id: ModelCheckpointID) -> Callable[[int], None]:
        last_sample_time = time.monotonic()
        last_sample_bytes = 0
        smoothed_speed = 0.0

        def on_progress(downloaded: int) -> None:
            nonlocal last_sample_time, last_sample_bytes, smoothed_speed
            now = time.monotonic()
            elapsed = now - last_sample_time
            if elapsed >= 1.0:
                instant_speed = (downloaded - last_sample_bytes) / elapsed
                if smoothed_speed == 0.0:
                    smoothed_speed = instant_speed
                else:
                    smoothed_speed = 0.3 * instant_speed + 0.7 * smoothed_speed
                last_sample_time = now
                last_sample_bytes = downloaded
            self.update_file_progress(cp_id, downloaded, smoothed_speed)

        return on_progress

    def _on_background_download_error(self, exc: Exception) -> None:
        self.fail_download(str(exc))

    @with_state_lock
    def get_download_progress(self, session_id: str) -> DownloadProgressResponse:
        typed_session_id = DownloadSessionId(session_id)
        session = self.state.downloading_session
        if session is not None and session.id == typed_session_id:
            current = session.current_running_file
            current_downloaded = current.downloaded_bytes if current else 0
            total_downloaded = session.completed_bytes + current_downloaded
            expected_total_bytes = sum(get_model_cp_spec(cp_id).expected_size_bytes for cp_id in session.files_to_download)

            current_file_progress = 0.0
            if current is not None:
                spec = get_model_cp_spec(current.file_type)
                if spec.expected_size_bytes > 0:
                    current_file_progress = min(99.0, current.downloaded_bytes / spec.expected_size_bytes * 100)

            total_progress = 0.0
            if expected_total_bytes > 0:
                total_progress = min(99.0, total_downloaded / expected_total_bytes * 100)

            return DownloadProgressRunningResponse(
                status="downloading",
                current_downloading_file=current.file_type if current else None,
                current_file_progress=current_file_progress,
                total_progress=total_progress,
                total_downloaded_bytes=total_downloaded,
                expected_total_bytes=expected_total_bytes,
                completed_files=set(session.completed_files),
                all_files=set(session.files_to_download),
                speed_bytes_per_sec=current.speed_bytes_per_sec if current else 0.0,
                error=None,
            )

        result = self.state.completed_download_sessions.get(typed_session_id)
        if result is not None:
            match result:
                case DownloadSessionComplete():
                    return DownloadProgressCompleteResponse(status="complete")
                case DownloadSessionError(error_message=error_message):
                    return DownloadProgressErrorResponse(status="error", error=error_message)

        raise ValueError(f"Unknown download session: {session_id}")

    def cleanup_downloading_dir(self) -> None:
        downloading_dir = resolve_downloading_dir(self.models_dir)
        if downloading_dir.exists():
            shutil.rmtree(downloading_dir)

    def _download_to_staging(self, cp_id: ModelCheckpointID, hf_token: str | None) -> None:
        spec = get_model_cp_spec(cp_id)
        self.start_file(cp_id, spec.name)
        progress_cb = self._make_progress_callback(cp_id)

        resolve_downloading_dir(self.models_dir).mkdir(parents=True, exist_ok=True)

        if spec.is_folder:
            self._model_downloader.download_snapshot(
                repo_id=spec.repo_id,
                local_dir=str(resolve_downloading_path(self.models_dir, cp_id)),
                on_progress=progress_cb,
                token=hf_token,
            )
        else:
            self._model_downloader.download_file(
                repo_id=spec.repo_id,
                filename=spec.name,
                local_dir=str(resolve_downloading_path(self.models_dir, cp_id)),
                on_progress=progress_cb,
                token=hf_token,
            )

    def _commit_staged_checkpoint(self, cp_id: ModelCheckpointID) -> bool:
        src = resolve_downloading_target_path(self.models_dir, cp_id)
        dst = resolve_model_path(self.models_dir, cp_id)
        spec = get_model_cp_spec(cp_id)

        if is_cp_downloaded(self.models_dir, cp_id):
            if src.exists():
                if spec.is_folder:
                    shutil.rmtree(src)
                else:
                    src.unlink(missing_ok=True)
            return False

        dst.parent.mkdir(parents=True, exist_ok=True)
        if spec.is_folder:
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)
        else:
            if dst.exists():
                dst.unlink()
            src.rename(dst)
        return True

    def _rollback_committed_checkpoints(self, cp_ids: Iterable[ModelCheckpointID]) -> None:
        for cp_id in cp_ids:
            spec = get_model_cp_spec(cp_id)
            path = resolve_model_path(self.models_dir, cp_id)
            if spec.is_folder:
                if path.exists():
                    shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _discover_download_cp_ids(self, requested_cp_ids: set[ModelCheckpointID]) -> tuple[ModelCheckpointID, ...]:
        missing: set[ModelCheckpointID] = set()
        for cp_id in requested_cp_ids:
            if not self._models_handler.is_cp_downloaded(cp_id):
                missing.add(cp_id)
        return self._ordered_cp_ids(missing)

    def _download_worker(self, cp_ids: tuple[ModelCheckpointID, ...], *, atomic_commit: bool) -> None:
        if not cp_ids:
            self.finish_download()
            return

        # Base/bundled checkpoints are public — never require sign-in. Attach the in-app token
        # if the user happens to be signed in, otherwise download anonymously. (Gated *catalog*
        # entries are handled separately in lora_catalog_handler and do require auth.)
        hf_token = optional_hf_token(self.state, self._lock)

        try:
            if atomic_commit:
                for cp_id in cp_ids:
                    logger.info("Downloading %s from %s", cp_id, get_model_cp_spec(cp_id).repo_id)
                    self._download_to_staging(cp_id, hf_token)

                committed_cp_ids: list[ModelCheckpointID] = []
                try:
                    for cp_id in cp_ids:
                        if self._commit_staged_checkpoint(cp_id):
                            committed_cp_ids.append(cp_id)
                except Exception:
                    self._rollback_committed_checkpoints(committed_cp_ids)
                    raise
            else:
                for cp_id in cp_ids:
                    logger.info("Downloading %s from %s", cp_id, get_model_cp_spec(cp_id).repo_id)
                    self._download_to_staging(cp_id, hf_token)
                    self._commit_staged_checkpoint(cp_id)
        except Exception:
            self.cleanup_downloading_dir()
            raise

        self.cleanup_downloading_dir()
        self.finish_download()

    def start_model_download(self, *, download_type: str, cp_ids: set[ModelCheckpointID]) -> DownloadSessionId:
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_DOWNLOADS_DISABLED_IN_FORCE_API_MODE")

        # Resolve what to download (may raise) before touching the lock.
        if download_type == "upgrade":
            resolved_upgrade = self._models_handler.resolve_upgrade_download(cp_ids)
            ordered_cp_ids = resolved_upgrade.cp_ids
            atomic_commit = True
        elif download_type == "download":
            ordered_cp_ids = self._discover_download_cp_ids(set(cp_ids))
            atomic_commit = False
        else:
            raise HTTPError(400, "INVALID_DOWNLOAD_REQUEST")

        # Check-and-set in one lock acquisition (RLock is reentrant, so start_download's own
        # lock nests fine) — otherwise two concurrent calls both pass the guard and the second
        # clobbers the first's session, racing on the same staging dir.
        with self._lock:
            if self.state.downloading_session is not None:
                raise HTTPError(409, "DOWNLOAD_ALREADY_RUNNING")
            session_id = self.start_download(set(ordered_cp_ids))
        self._task_runner.run_background(
            lambda: self._download_worker(ordered_cp_ids, atomic_commit=atomic_commit),
            task_name="model-download",
            on_error=self._on_background_download_error,
            daemon=True,
        )
        return session_id

    def check_model_access(self, cp_ids: set[ModelCheckpointID]) -> CheckModelAccessResponse:
        repo_ids = {get_model_cp_spec(cp_id).repo_id for cp_id in cp_ids}

        # Bundled checkpoints are public, so when the user isn't signed in there's nothing to
        # verify — treat them all as authorized. When signed in, do a real per-repo check (covers
        # any future gated checkpoint without forcing sign-in for the common public case).
        hf_token = optional_hf_token(self.state, self._lock)
        if hf_token is None:
            return CheckModelAccessResponse(access={repo_id: "authorized" for repo_id in repo_ids})

        access: dict[str, ModelAccessStatus] = {}
        for repo_id in sorted(repo_ids):
            try:
                response = http_requests.head(
                    f"https://huggingface.co/{repo_id}/resolve/main/.gitattributes",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    allow_redirects=True,
                    timeout=10,
                )
                access[repo_id] = "authorized" if response.status_code == 200 else "not_authorized"
            except Exception:
                access[repo_id] = "not_authorized"

        return CheckModelAccessResponse(access=access)
