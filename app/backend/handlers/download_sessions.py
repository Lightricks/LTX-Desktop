"""Single-slot background download session tracker shared by catalog and custom LoRA handlers.

A handler keeps one active session slot in ``AppState`` and a bounded dict of finished
sessions. :class:`DownloadSessionTracker` owns the lock-protected lifecycle for one slot:
opening a session, updating progress, recording completion or failure, and answering the
progress and active-session endpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
import time
from typing import Generic, TypeVar
from uuid import uuid4

from _routes._errors import HTTPError
from api_types import CatalogDownloadStatus
from state.app_state_types import (
    AppState,
    CatalogDownloadSession,
    DownloadSessionComplete,
    DownloadSessionError,
    DownloadSessionId,
    DownloadSessionResult,
)

# Cap retained completed sessions so a long-running app doesn't accrue them forever.
# The client only needs a terminal status once; oldest-out is fine (dicts keep insertion order).
MAX_COMPLETED_SESSIONS = 32

ProgressResponseT = TypeVar("ProgressResponseT")
ActiveResponseT = TypeVar("ActiveResponseT")


def record_completed(
    completed: dict[DownloadSessionId, DownloadSessionResult],
    session_id: DownloadSessionId,
    result: DownloadSessionResult,
) -> None:
    completed[session_id] = result
    while len(completed) > MAX_COMPLETED_SESSIONS:
        completed.pop(next(iter(completed)))


def session_progress_pct(session: CatalogDownloadSession) -> float:
    """Percent done, held below 100 until the session is recorded as complete."""
    if not session.expected_bytes:
        return 0.0
    return min(99.0, session.downloaded_bytes / session.expected_bytes * 100)


class DownloadSessionTracker(Generic[ProgressResponseT, ActiveResponseT]):
    """Lock-safe single-slot download session state machine."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        *,
        session_attr: str,
        completed_attr: str,
        build_progress: Callable[..., ProgressResponseT],
        build_active: Callable[[str | None, str | None, float | None], ActiveResponseT],
        busy_message: str = "DOWNLOAD_ALREADY_RUNNING",
        busy_code: str | None = None,
    ) -> None:
        self._state = state
        self._lock = lock
        self._session_attr = session_attr
        self._completed_attr = completed_attr
        self._build_progress = build_progress
        self._build_active = build_active
        self._busy_message = busy_message
        self._busy_code = busy_code

    def _active(self) -> CatalogDownloadSession | None:
        return getattr(self._state, self._session_attr)

    def _set_active(self, session: CatalogDownloadSession | None) -> None:
        setattr(self._state, self._session_attr, session)

    def _completed(self) -> dict[DownloadSessionId, DownloadSessionResult]:
        return getattr(self._state, self._completed_attr)

    def _busy_error(self) -> HTTPError:
        return HTTPError(409, self._busy_message, code=self._busy_code)

    def open(
        self,
        item_id: str,
        expected_bytes: int,
        *,
        before_open: Callable[[], None] | None = None,
    ) -> DownloadSessionId:
        with self._lock:
            if self._active() is not None:
                raise self._busy_error()
            if before_open is not None:
                before_open()
            session_id = DownloadSessionId(uuid4().hex)
            self._set_active(
                CatalogDownloadSession(
                    id=session_id,
                    item_id=item_id,
                    downloaded_bytes=0,
                    expected_bytes=expected_bytes,
                )
            )
            return session_id

    def refuse_if_item_active(self, item_id: str, *, message: str | None = None) -> None:
        with self._lock:
            session = self._active()
            if session is not None and session.item_id == item_id:
                raise HTTPError(409, message or self._busy_message, code=self._busy_code)

    def update_progress(
        self, session_id: DownloadSessionId, downloaded: int, expected: int = 0
    ) -> None:
        with self._lock:
            session = self._active()
            if session is not None and session.id == session_id:
                now = time.monotonic()
                elapsed = now - session.last_sample_time
                if elapsed >= 0.5 and downloaded >= session.last_sample_bytes:
                    instant = (downloaded - session.last_sample_bytes) / elapsed
                    session.speed_bytes_per_sec = (
                        instant
                        if session.speed_bytes_per_sec == 0.0
                        else 0.3 * instant + 0.7 * session.speed_bytes_per_sec
                    )
                    session.last_sample_time = now
                    session.last_sample_bytes = downloaded
                session.downloaded_bytes = downloaded
                if expected:
                    session.expected_bytes = expected

    def finish(self, session_id: DownloadSessionId) -> None:
        with self._lock:
            record_completed(self._completed(), session_id, DownloadSessionComplete())
            self._set_active(None)

    def fail(self, session_id: DownloadSessionId, error_message: str) -> None:
        with self._lock:
            record_completed(
                self._completed(), session_id, DownloadSessionError(error_message=error_message)
            )
            self._set_active(None)

    def get_progress(self, session_id: str) -> ProgressResponseT:
        with self._lock:
            sid = DownloadSessionId(session_id)
            session = self._active()
            if session is not None and session.id == sid:
                status: CatalogDownloadStatus = "downloading"
                return self._build_progress(
                    status,
                    item_id=session.item_id,
                    downloaded_bytes=session.downloaded_bytes,
                    expected_bytes=session.expected_bytes,
                    progress=session_progress_pct(session),
                    speed_bytes_per_sec=session.speed_bytes_per_sec,
                )
            result = self._completed().get(sid)
            if result is None:
                raise HTTPError(404, "UNKNOWN_DOWNLOAD_SESSION")
            if isinstance(result, DownloadSessionError):
                return self._build_progress("error", error=result.error_message)
            return self._build_progress("complete", progress=100.0)

    def get_active(self) -> ActiveResponseT:
        with self._lock:
            session = self._active()
            if session is None:
                return self._build_active(None, None, None)
            return self._build_active(
                str(session.id), session.item_id, session_progress_pct(session)
            )
