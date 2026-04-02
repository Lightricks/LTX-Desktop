"""Model download session handler."""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from threading import RLock
from typing import TYPE_CHECKING
from uuid import uuid4

from api_types import (
    DownloadProgressCompleteResponse,
    DownloadProgressErrorResponse,
    DownloadProgressResponse,
    DownloadProgressRunningResponse,
)
from handlers.base import StateHandlerBase, with_state_lock
from handlers.models_handler import ModelsHandler
from runtime_config.model_download_specs import (
    MODEL_FILE_ORDER,
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
    ModelFileType,
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

    @with_state_lock
    def is_download_running(self) -> bool:
        return self.state.downloading_session is not None

    @with_state_lock
    def start_download(self, files_to_download: set[ModelFileType]) -> DownloadSessionId:
        session_id = DownloadSessionId(uuid4().hex)
        self.state.downloading_session = DownloadingSession(
            id=session_id,
            current_running_file=None,
            files_to_download=files_to_download,
            completed_files=set(),
            completed_bytes=0,
        )
        return session_id

    @with_state_lock
    def start_file(self, file_type: ModelFileType, target: str) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        if session.current_running_file is not None:
            session.completed_bytes += session.current_running_file.downloaded_bytes
            session.completed_files.add(session.current_running_file.file_type)
        session.current_running_file = FileDownloadRunning(
            file_type=file_type,
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
    def update_file_progress(self, file_type: ModelFileType, downloaded: int, speed_bytes_per_sec: float) -> None:
        session = self.state.downloading_session
        if session is None:
            return
        rf = session.current_running_file
        if rf is None or rf.file_type != file_type:
            return
        rf.downloaded_bytes = downloaded
        rf.speed_bytes_per_sec = speed_bytes_per_sec

    @with_state_lock
    def fail_download(self, error: str) -> None:
        logger.error("Model download failed: %s", error)
        session = self.state.downloading_session
        if session is not None:
            self.state.completed_download_sessions[session.id] = DownloadSessionError(error_message=error)
            self.state.downloading_session = None

    def _make_progress_callback(self, file_type: ModelFileType) -> Callable[[int], None]:
        last_sample_time = time.monotonic()
        last_sample_bytes = 0
        smoothed_speed = 0.0

        def on_progress(downloaded: int) -> None:
            nonlocal last_sample_time, last_sample_bytes, smoothed_speed
            now = time.monotonic()
            elapsed = now - last_sample_time
            if elapsed >= 1.0:
                instant_speed = (downloaded - last_sample_bytes) / elapsed
                # EWMA: weight new sample at 30%, keep 70% of previous.
                # On first sample (smoothed_speed == 0) use instant value.
                if smoothed_speed == 0.0:
                    smoothed_speed = instant_speed
                else:
                    smoothed_speed = 0.3 * instant_speed + 0.7 * smoothed_speed
                last_sample_time = now
                last_sample_bytes = downloaded
            self.update_file_progress(file_type, downloaded, smoothed_speed)

        return on_progress

    def _on_background_download_error(self, exc: Exception) -> None:
        self.fail_download(str(exc))

    @with_state_lock
    def get_download_progress(self, session_id: str) -> DownloadProgressResponse:
        typed_session_id = DownloadSessionId(session_id)
        session = self.state.downloading_session
        if session is not None and session.id == typed_session_id:
            rf = session.current_running_file
            current_downloaded = rf.downloaded_bytes if rf else 0
            total_downloaded = session.completed_bytes + current_downloaded

            expected_total_bytes = sum(
                self.config.spec_for(ft).expected_size_bytes for ft in session.files_to_download
            )

            current_file_progress = 0.0
            if rf is not None:
                spec = self.config.spec_for(rf.file_type)
                if spec.expected_size_bytes > 0:
                    current_file_progress = min(99.0, rf.downloaded_bytes / spec.expected_size_bytes * 100)

            total_progress = 0.0
            if expected_total_bytes > 0:
                total_progress = min(99.0, total_downloaded / expected_total_bytes * 100)

            return DownloadProgressRunningResponse(
                status="downloading",
                current_downloading_file=rf.file_type if rf else None,
                current_file_progress=current_file_progress,
                total_progress=total_progress,
                total_downloaded_bytes=total_downloaded,
                expected_total_bytes=expected_total_bytes,
                completed_files=set(session.completed_files),
                all_files=set(session.files_to_download),
                speed_bytes_per_sec=rf.speed_bytes_per_sec if rf else 0.0,
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

    def _move_to_final(self, file_type: ModelFileType) -> None:
        """Move downloaded file/folder from downloading dir to final location."""
        spec = self.config.spec_for(file_type)
        src = resolve_downloading_target_path(self.models_dir, self.config.model_download_specs, file_type)
        dst = resolve_model_path(self.models_dir, self.config.model_download_specs, file_type)

        if spec.is_folder:
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)
        else:
            if dst.exists():
                dst.unlink()
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)

    def cleanup_downloading_dir(self) -> None:
        """Remove stale .downloading/ dir (leftover from crashed downloads)."""
        downloading = resolve_downloading_dir(self.models_dir)
        if downloading.exists():
            shutil.rmtree(downloading)

    def _discover_files_to_download(self, model_types: set[ModelFileType]) -> dict[ModelFileType, str]:
        """Determine which files need downloading (not already available)."""
        self._models_handler.refresh_available_files()
        available = self.state.available_files.copy()

        files_to_download: dict[ModelFileType, str] = {}
        for model_type in MODEL_FILE_ORDER:
            if model_type not in model_types:
                continue
            if available[model_type] is not None:
                continue
            spec = self.config.spec_for(model_type)
            files_to_download[model_type] = spec.name
        return files_to_download

    def _download_models_worker(self, files_to_download: dict[ModelFileType, str]) -> None:
        if not files_to_download:
            self.finish_download()
            return

        for file_type, target_name in files_to_download.items():
            spec = self.config.spec_for(file_type)
            logger.info("Downloading %s from %s", target_name, spec.repo_id)

            self.start_file(file_type, target_name)
            progress_cb = self._make_progress_callback(file_type)

            try:
                resolve_downloading_dir(self.models_dir).mkdir(parents=True, exist_ok=True)

                if spec.is_folder:
                    self._model_downloader.download_snapshot(
                        repo_id=spec.repo_id,
                        local_dir=str(resolve_downloading_path(self.models_dir, self.config.model_download_specs, file_type)),
                        on_progress=progress_cb,
                    )
                else:
                    self._model_downloader.download_file(
                        repo_id=spec.repo_id,
                        filename=spec.name,
                        local_dir=str(resolve_downloading_path(self.models_dir, self.config.model_download_specs, file_type)),
                        on_progress=progress_cb,
                    )

                self._move_to_final(file_type)
            except Exception:
                self.cleanup_downloading_dir()
                raise

        self.finish_download()
        self._models_handler.refresh_available_files()

    def start_model_download(self, model_types: set[ModelFileType]) -> DownloadSessionId | None:
        with self._lock:
            if self.state.downloading_session is not None:
                return None

        files_to_download = self._discover_files_to_download(model_types)
        session_id = self.start_download(set(files_to_download.keys()))

        self._task_runner.run_background(
            lambda: self._download_models_worker(files_to_download),
            task_name="model-download",
            on_error=self._on_background_download_error,
            daemon=True,
        )
        return session_id

    def start_text_encoder_download(self) -> DownloadSessionId | None:
        with self._lock:
            if self.state.downloading_session is not None:
                return None

        text_spec = self.config.spec_for("text_encoder")
        session_id = self.start_download({"text_encoder"})

        def worker() -> None:
            self.start_file("text_encoder", text_spec.name)
            progress_cb = self._make_progress_callback("text_encoder")
            try:
                resolve_downloading_dir(self.models_dir).mkdir(parents=True, exist_ok=True)
                self._model_downloader.download_snapshot(
                    repo_id=text_spec.repo_id,
                    local_dir=str(resolve_downloading_path(self.models_dir, self.config.model_download_specs, "text_encoder")),
                    on_progress=progress_cb,
                )
                self._move_to_final("text_encoder")
            except Exception:
                self.cleanup_downloading_dir()
                raise
            self.finish_download()
            self._models_handler.refresh_available_files()

        self._task_runner.run_background(
            worker,
            task_name="text-encoder-download",
            on_error=self._on_background_download_error,
            daemon=True,
        )
        return session_id
