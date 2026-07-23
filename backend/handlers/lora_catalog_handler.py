"""LoRA / IC-LoRA catalog listing + download handler."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Generic, TypeVar
from uuid import uuid4

from _routes._errors import HTTPError
from api_types import (
    CatalogDownloadStartResponse,
    CatalogDownloadStatus,
    IcLoraDownloadProgressResponse,
    IcLoraListItem,
    IcLoraListResponse,
    LoraCatalogItem,
    LoraDownloadProgressResponse,
    LoraListItem,
    LoraListResponse,
)
from handlers.base import StateHandlerBase, with_state_lock
from handlers.hf_auth_utils import optional_hf_token, require_hf_token
from runtime_config.model_download_specs import (
    downloaded_ic_lora_variant_ids,
    downloaded_lora_variant_ids,
    resolve_lora_path,
    resolve_ic_lora_path,
)
from services.interfaces import ModelDownloader, TaskRunner
from services.lora_catalog import LoraCatalogProvider
from services.lora_catalog.media_urls import with_resolved_media
from state.app_state_types import (
    AppState,
    CatalogDownloadSession,
    DownloadSessionComplete,
    DownloadSessionError,
    DownloadSessionId,
    DownloadSessionResult,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

# Cap retained completed sessions so a long-running app doesn't accrue them forever.
# The client only needs a terminal status once; oldest-out is fine (dicts keep insertion order).
_MAX_COMPLETED_SESSIONS = 32

ProgressResponseT = TypeVar("ProgressResponseT", IcLoraDownloadProgressResponse, LoraDownloadProgressResponse)


def _record_completed(
    completed: dict[DownloadSessionId, DownloadSessionResult],
    session_id: DownloadSessionId,
    result: DownloadSessionResult,
) -> None:
    completed[session_id] = result
    while len(completed) > _MAX_COMPLETED_SESSIONS:
        completed.pop(next(iter(completed)))


def _lookup_ic_lora(catalog: LoraCatalogProvider, item_id: str) -> LoraCatalogItem | None:
    return catalog.get_ic_lora(item_id)


def _lookup_lora(catalog: LoraCatalogProvider, item_id: str) -> LoraCatalogItem | None:
    return catalog.get_lora(item_id)


def _ic_lora_progress(
    status: CatalogDownloadStatus,
    *,
    item_id: str | None = None,
    downloaded_bytes: int = 0,
    expected_bytes: int = 0,
    progress: float = 0.0,
    speed_bytes_per_sec: float = 0.0,
    error: str | None = None,
) -> IcLoraDownloadProgressResponse:
    return IcLoraDownloadProgressResponse(
        status=status,
        ic_lora_id=item_id,
        downloaded_bytes=downloaded_bytes,
        expected_bytes=expected_bytes,
        progress=progress,
        speed_bytes_per_sec=speed_bytes_per_sec,
        error=error,
    )


def _lora_progress(
    status: CatalogDownloadStatus,
    *,
    item_id: str | None = None,
    downloaded_bytes: int = 0,
    expected_bytes: int = 0,
    progress: float = 0.0,
    speed_bytes_per_sec: float = 0.0,
    error: str | None = None,
) -> LoraDownloadProgressResponse:
    return LoraDownloadProgressResponse(
        status=status,
        lora_id=item_id,
        downloaded_bytes=downloaded_bytes,
        expected_bytes=expected_bytes,
        progress=progress,
        speed_bytes_per_sec=speed_bytes_per_sec,
        error=error,
    )


@dataclass(frozen=True)
class _CatalogKind(Generic[ProgressResponseT]):
    """Per-kind plumbing so IC-LoRA and plain-LoRA downloads share one implementation.

    Each kind keeps its own state slot (so an IC-LoRA and a plain-LoRA download can run at
    once) and its own wire response type; everything else is identical.
    """

    session_attr: str  # AppState attribute holding the active CatalogDownloadSession | None
    completed_attr: str  # AppState attribute holding the completed-sessions dict
    task_name: str
    not_found: str
    resolve_path: Callable[[Path, str, str], Path]  # (models_dir, item_id, filename) -> final file path
    lookup: Callable[[LoraCatalogProvider, str], LoraCatalogItem | None]
    build_progress: Callable[..., ProgressResponseT]


_IC_LORA: _CatalogKind[IcLoraDownloadProgressResponse] = _CatalogKind(
    session_attr="ic_lora_download_session",
    completed_attr="completed_ic_lora_download_sessions",
    task_name="ic-lora-download",
    not_found="UNKNOWN_IC_LORA",
    resolve_path=resolve_ic_lora_path,
    lookup=_lookup_ic_lora,
    build_progress=_ic_lora_progress,
)

_PLAIN_LORA: _CatalogKind[LoraDownloadProgressResponse] = _CatalogKind(
    session_attr="lora_download_session",
    completed_attr="completed_lora_download_sessions",
    task_name="lora-download",
    not_found="UNKNOWN_LORA",
    resolve_path=resolve_lora_path,
    lookup=_lookup_lora,
    build_progress=_lora_progress,
)


class LoraCatalogHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        catalog: LoraCatalogProvider,
        model_downloader: ModelDownloader,
        task_runner: TaskRunner,
        config: "RuntimeConfig",
    ) -> None:
        super().__init__(state, lock, config)
        self._catalog = catalog
        self._model_downloader = model_downloader
        self._task_runner = task_runner

    # --- listing ---
    # Catalog download.variants is always non-empty; the first variant is the default
    # checkpoint (preferred when listing "downloaded", when generation omits variant_id,
    # and when a download request omits variant_id).
    def list_ic_loras(self) -> IcLoraListResponse:
        catalog = self._catalog.get_catalog()
        items: list[IcLoraListItem] = []
        for r in catalog.ic_loras:
            variant_ids = downloaded_ic_lora_variant_ids(
                self.models_dir, r.id, [(v.id, v.filename) for v in r.download.variants]
            )
            items.append(
                IcLoraListItem(
                    ic_lora=with_resolved_media(r),
                    downloaded=bool(variant_ids),
                    downloaded_variant_ids=variant_ids,
                )
            )
        return IcLoraListResponse(ic_loras=items)

    def list_loras(self) -> LoraListResponse:
        catalog = self._catalog.get_catalog()
        items: list[LoraListItem] = []
        for item in catalog.loras:
            variant_ids = downloaded_lora_variant_ids(
                self.models_dir, item.id, [(v.id, v.filename) for v in item.download.variants]
            )
            items.append(
                LoraListItem(
                    lora=with_resolved_media(item),
                    downloaded=bool(variant_ids),
                    downloaded_variant_ids=variant_ids,
                )
            )
        return LoraListResponse(loras=items)

    # --- download (shared between both kinds) ---
    def _resolve_download_token(self, *, requires_hf_login: bool, use_hf_auth: bool) -> str | None:
        """Token to attach to a catalog download.

        Gated items (a per-entry ``requires_hf_login``) must authenticate — ``require_hf_token``
        raises 403 if not signed in. Otherwise ``use_hf_auth`` is opt-in: attach the in-app token
        when signed in, but still allow anonymous public downloads when signed out (we never force
        sign-in for a public repo).
        """
        if requires_hf_login:
            return require_hf_token(self.state, self._lock)
        if use_hf_auth:
            return optional_hf_token(self.state, self._lock)
        return None

    def _start_download(
        self,
        kind: _CatalogKind[ProgressResponseT],
        item_id: str,
        use_hf_auth: bool,
        variant_id: str | None = None,
    ) -> CatalogDownloadStartResponse:
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_DOWNLOADS_DISABLED_IN_FORCE_API_MODE")
        item = kind.lookup(self._catalog, item_id)
        if item is None:
            raise HTTPError(404, kind.not_found)
        variant = item.download.resolve_variant(variant_id)
        if variant is None:
            raise HTTPError(404, "UNKNOWN_DOWNLOAD_VARIANT")
        # Resolve the token first (before opening a session): a gated item can raise 403 here.
        hf_token = self._resolve_download_token(
            requires_hf_login=item.requires_hf_login, use_hf_auth=use_hf_auth
        )
        with self._lock:
            if getattr(self.state, kind.session_attr) is not None:
                raise HTTPError(409, "DOWNLOAD_ALREADY_RUNNING")
            session_id = DownloadSessionId(uuid4().hex)
            setattr(
                self.state,
                kind.session_attr,
                CatalogDownloadSession(
                    id=session_id,
                    item_id=item_id,
                    downloaded_bytes=0,
                    expected_bytes=variant.size_bytes,
                ),
            )

        dest = kind.resolve_path(self.models_dir, item.id, variant.filename)

        def _work() -> None:
            # Stage into a temp dir and atomically move on success, so an interrupted download
            # never leaves a partial file at the final path (where it would report "downloaded"
            # and get used by generation).
            staging_dir = dest.parent / ".tmp"
            staging_dir.mkdir(parents=True, exist_ok=True)
            try:
                staged = self._model_downloader.download_file(
                    repo_id=item.download.repo_id,
                    filename=variant.filename,
                    local_dir=str(staging_dir),
                    token=hf_token,
                    on_progress=lambda n: self._update_progress(kind, session_id, n),
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                Path(staged).replace(dest)
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
            self._finish(kind, session_id)

        self._task_runner.run_background(
            _work,
            task_name=kind.task_name,
            on_error=lambda exc: self._fail(kind, session_id, str(exc)),
            daemon=True,
        )
        return CatalogDownloadStartResponse(status="started", sessionId=session_id)

    @with_state_lock
    def _update_progress(self, kind: _CatalogKind[ProgressResponseT], session_id: DownloadSessionId, downloaded: int) -> None:
        s: CatalogDownloadSession | None = getattr(self.state, kind.session_attr)
        if s is not None and s.id == session_id:
            s.downloaded_bytes = downloaded

    def _completed(self, kind: _CatalogKind[ProgressResponseT]) -> dict[DownloadSessionId, DownloadSessionResult]:
        return getattr(self.state, kind.completed_attr)

    @with_state_lock
    def _finish(self, kind: _CatalogKind[ProgressResponseT], session_id: DownloadSessionId) -> None:
        _record_completed(self._completed(kind), session_id, DownloadSessionComplete())
        setattr(self.state, kind.session_attr, None)

    @with_state_lock
    def _fail(self, kind: _CatalogKind[ProgressResponseT], session_id: DownloadSessionId, error: str) -> None:
        logger.error("%s failed: %s", kind.task_name, error)
        _record_completed(self._completed(kind), session_id, DownloadSessionError(error_message=error))
        setattr(self.state, kind.session_attr, None)

    @with_state_lock
    def _get_progress(self, kind: _CatalogKind[ProgressResponseT], session_id: str) -> ProgressResponseT:
        sid = DownloadSessionId(session_id)
        s: CatalogDownloadSession | None = getattr(self.state, kind.session_attr)
        if s is not None and s.id == sid:
            pct = min(99.0, s.downloaded_bytes / s.expected_bytes * 100) if s.expected_bytes else 0.0
            return kind.build_progress(
                "downloading",
                item_id=s.item_id,
                downloaded_bytes=s.downloaded_bytes,
                expected_bytes=s.expected_bytes,
                progress=pct,
                speed_bytes_per_sec=s.speed_bytes_per_sec,
            )
        result = self._completed(kind).get(sid)
        if result is None:
            raise HTTPError(404, "UNKNOWN_DOWNLOAD_SESSION")
        if isinstance(result, DownloadSessionError):
            return kind.build_progress("error", error=result.error_message)
        return kind.build_progress("complete", progress=100.0)

    # --- thin per-kind entry points (the routes call these) ---
    def start_ic_lora_download(
        self, ic_lora_id: str, use_hf_auth: bool = False, variant_id: str | None = None
    ) -> CatalogDownloadStartResponse:
        return self._start_download(_IC_LORA, ic_lora_id, use_hf_auth, variant_id=variant_id)

    def get_ic_lora_download_progress(self, session_id: str) -> IcLoraDownloadProgressResponse:
        return self._get_progress(_IC_LORA, session_id)

    def start_lora_download(
        self, lora_id: str, use_hf_auth: bool = False, variant_id: str | None = None
    ) -> CatalogDownloadStartResponse:
        return self._start_download(_PLAIN_LORA, lora_id, use_hf_auth, variant_id=variant_id)

    def get_lora_download_progress(self, session_id: str) -> LoraDownloadProgressResponse:
        return self._get_progress(_PLAIN_LORA, session_id)
