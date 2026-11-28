"""LoRA / IC-LoRA catalog listing + download handler."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Generic, TypeVar

from _routes._errors import HTTPError
from api_types import (
    ActiveIcLoraDownloadResponse,
    ActiveLoraDownloadResponse,
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
from handlers.base import StateHandlerBase
from handlers.checkpoint_files import commit_staged_checkpoint, stage_checkpoint
from handlers.download_sessions import DownloadSessionTracker
from handlers.hf_auth_utils import optional_hf_token, require_hf_token
from runtime_config.ic_lora_job_budget import with_budgeted_duration_controls
from runtime_config.model_download_specs import (
    catalog_item_visible_for_installed_ltx,
    downloaded_ic_lora_variant_ids,
    downloaded_lora_variant_ids,
    get_model_cp_spec,
    missing_checkpoints,
    resolve_catalog_staging_dir,
    resolve_catalog_staging_root,
    resolve_ic_lora_path,
    resolve_lora_path,
)
from services.interfaces import GpuInfo, ModelDownloader, TaskRunner
from services.lora_catalog import LoraCatalogProvider
from services.lora_catalog.media_urls import with_resolved_media
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)

ProgressResponseT = TypeVar(
    "ProgressResponseT", IcLoraDownloadProgressResponse, LoraDownloadProgressResponse
)
ActiveResponseT = TypeVar(
    "ActiveResponseT", ActiveIcLoraDownloadResponse, ActiveLoraDownloadResponse
)


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
class _CatalogKind(Generic[ProgressResponseT, ActiveResponseT]):
    """Per-kind plumbing so IC-LoRA and plain-LoRA downloads share one implementation."""

    task_name: str
    not_found: str
    resolve_path: Callable[[Path, str, str], Path]
    lookup: Callable[[LoraCatalogProvider, str], LoraCatalogItem | None]
    tracker: DownloadSessionTracker[ProgressResponseT, ActiveResponseT]


class LoraCatalogHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        catalog: LoraCatalogProvider,
        model_downloader: ModelDownloader,
        task_runner: TaskRunner,
        config: "RuntimeConfig",
        gpu_info: GpuInfo,
    ) -> None:
        super().__init__(state, lock, config, gpu_info=gpu_info)
        self._catalog = catalog
        self._model_downloader = model_downloader
        self._task_runner = task_runner
        self._ic_lora = _CatalogKind(
            task_name="ic-lora-download",
            not_found="UNKNOWN_IC_LORA",
            resolve_path=resolve_ic_lora_path,
            lookup=_lookup_ic_lora,
            tracker=DownloadSessionTracker(
                state,
                lock,
                session_attr="ic_lora_download_session",
                completed_attr="completed_ic_lora_download_sessions",
                build_progress=_ic_lora_progress,
                build_active=lambda session_id, item_id, progress: ActiveIcLoraDownloadResponse(
                    session_id=session_id, ic_lora_id=item_id, progress=progress
                ),
            ),
        )
        self._plain_lora = _CatalogKind(
            task_name="lora-download",
            not_found="UNKNOWN_LORA",
            resolve_path=resolve_lora_path,
            lookup=_lookup_lora,
            tracker=DownloadSessionTracker(
                state,
                lock,
                session_attr="lora_download_session",
                completed_attr="completed_lora_download_sessions",
                build_progress=_lora_progress,
                build_active=lambda session_id, item_id, progress: ActiveLoraDownloadResponse(
                    session_id=session_id, lora_id=item_id, progress=progress
                ),
            ),
        )
        # A crash leaves the staging folder of a session. No download runs yet, so all of it is stale.
        shutil.rmtree(resolve_catalog_staging_root(self.models_dir), ignore_errors=True)

    # --- listing ---
    def list_ic_loras(self, fresh: bool = False) -> IcLoraListResponse:
        catalog = self._catalog.get_catalog()
        job_filter = self._local_job_filter(total_only=True)
        items: list[IcLoraListItem] = []
        for r in catalog.ic_loras:
            if not fresh and not catalog_item_visible_for_installed_ltx(r.supported_models, self.models_dir):
                continue
            variant_ids = downloaded_ic_lora_variant_ids(
                self.models_dir,
                r.id,
                [(v.id, v.filename) for v in r.download.variants],
                r.requires_checkpoints,
            )
            item = with_budgeted_duration_controls(
                with_resolved_media(r),
                job_filter=job_filter,
            )
            items.append(
                IcLoraListItem(
                    ic_lora=item,
                    downloaded=bool(variant_ids),
                    downloaded_variant_ids=variant_ids,
                )
            )
        return IcLoraListResponse(ic_loras=items, budgetMode=job_filter.budget_mode)

    def list_loras(self, fresh: bool = False) -> LoraListResponse:
        catalog = self._catalog.get_catalog()
        items: list[LoraListItem] = []
        for item in catalog.loras:
            if not fresh and not catalog_item_visible_for_installed_ltx(
                item.supported_models, self.models_dir
            ):
                continue
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
        if requires_hf_login:
            return require_hf_token(self.state, self._lock)
        if use_hf_auth:
            return optional_hf_token(self.state, self._lock)
        return None

    def _start_download(
        self,
        kind: _CatalogKind[ProgressResponseT, ActiveResponseT],
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
        hf_token = self._resolve_download_token(
            requires_hf_login=item.requires_hf_login, use_hf_auth=use_hf_auth
        )
        dest = kind.resolve_path(self.models_dir, item.id, variant.filename)
        fetch_file = not dest.exists()
        missing_cps = missing_checkpoints(self.models_dir, item.requires_checkpoints)
        expected_bytes = (variant.size_bytes if fetch_file else 0) + sum(
            get_model_cp_spec(cp).expected_size_bytes for cp in missing_cps
        )
        session_id = kind.tracker.open(item_id, expected_bytes)

        def _progress_from(base: int) -> Callable[[int], None]:
            return lambda n: kind.tracker.update_progress(session_id, base + n)

        def _work() -> None:
            staging_dir = resolve_catalog_staging_dir(self.models_dir, session_id)
            staging_dir.mkdir(parents=True, exist_ok=True)
            try:
                finished_bytes = 0
                if fetch_file:
                    staged = self._model_downloader.download_file(
                        repo_id=variant.resolved_repo_id(item.download.repo_id),
                        filename=variant.filename,
                        local_dir=str(staging_dir),
                        token=hf_token,
                        on_progress=_progress_from(0),
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    Path(staged).replace(dest)
                    finished_bytes = variant.size_bytes
                for cp in missing_cps:
                    stage_checkpoint(
                        self._model_downloader,
                        staging_dir,
                        cp,
                        hf_token,
                        _progress_from(finished_bytes),
                    )
                    commit_staged_checkpoint(self.models_dir, staging_dir, cp)
                    finished_bytes += get_model_cp_spec(cp).expected_size_bytes
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
            kind.tracker.finish(session_id)

        def _fail(exc: Exception) -> None:
            logger.error("%s failed: %s", kind.task_name, exc)
            kind.tracker.fail(session_id, str(exc))

        self._task_runner.run_background(
            _work,
            task_name=kind.task_name,
            on_error=_fail,
            daemon=True,
        )
        return CatalogDownloadStartResponse(status="started", sessionId=session_id)

    # --- thin per-kind entry points (the routes call these) ---
    def start_ic_lora_download(
        self, ic_lora_id: str, use_hf_auth: bool = False, variant_id: str | None = None
    ) -> CatalogDownloadStartResponse:
        return self._start_download(
            self._ic_lora, ic_lora_id, use_hf_auth, variant_id=variant_id
        )

    def get_ic_lora_download_progress(self, session_id: str) -> IcLoraDownloadProgressResponse:
        return self._ic_lora.tracker.get_progress(session_id)

    def start_lora_download(
        self, lora_id: str, use_hf_auth: bool = False, variant_id: str | None = None
    ) -> CatalogDownloadStartResponse:
        return self._start_download(
            self._plain_lora, lora_id, use_hf_auth, variant_id=variant_id
        )

    def get_lora_download_progress(self, session_id: str) -> LoraDownloadProgressResponse:
        return self._plain_lora.tracker.get_progress(session_id)

    def _delete_installation(
        self,
        kind: _CatalogKind[ProgressResponseT, ActiveResponseT],
        item_id: str,
        variant_id: str | None = None,
    ) -> None:
        if self.config.force_api_generations:
            raise HTTPError(409, "LOCAL_MODEL_DOWNLOADS_DISABLED_IN_FORCE_API_MODE")
        item = kind.lookup(self._catalog, item_id)
        if item is None:
            raise HTTPError(404, kind.not_found)
        variant = item.download.resolve_variant(variant_id)
        if variant is None:
            raise HTTPError(404, "UNKNOWN_DOWNLOAD_VARIANT")
        kind.tracker.refuse_if_item_active(item_id)
        path = kind.resolve_path(self.models_dir, item.id, variant.filename)
        if not path.is_file():
            raise HTTPError(404, "LORA_NOT_INSTALLED")
        path.unlink()
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()

    def delete_lora(self, lora_id: str, variant_id: str | None = None) -> None:
        self._delete_installation(self._plain_lora, lora_id, variant_id)

    def delete_ic_lora(self, ic_lora_id: str, variant_id: str | None = None) -> None:
        self._delete_installation(self._ic_lora, ic_lora_id, variant_id)

    def get_active_lora_download(self) -> ActiveLoraDownloadResponse:
        return self._plain_lora.tracker.get_active()

    def get_active_ic_lora_download(self) -> ActiveIcLoraDownloadResponse:
        return self._ic_lora.tracker.get_active()
