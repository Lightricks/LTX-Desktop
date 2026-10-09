from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import NoReturn

from pydantic import JsonValue

from services.dashboard_stats import DashboardSnapshot, DashboardWindow

from services.records import (
    AssetListPage,
    AssetRecord,
    GenerationErrorCode,
    GenerationRecord,
    MediaKind,
    OutputSpec,
    QueueGenerationRecord,
    UnavailableError,
)


def _unavailable() -> NoReturn:
    raise UnavailableError()


class UnavailableStore:
    def ingest_upload(self, source_path: str) -> AssetRecord:
        _unavailable()

    def get_asset(
        self, asset_id: str, *, tombstone_missing: bool = True
    ) -> AssetRecord | None:
        del asset_id, tombstone_missing
        _unavailable()

    def list_assets(
        self,
        *,
        media_kind: MediaKind | None = None,
        sort: str = "created_at-desc",
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
        tombstone_missing: bool = True,
    ) -> AssetListPage:
        del media_kind, sort, q, cursor, limit, tombstone_missing
        _unavailable()

    def allocate_output_path(
        self, media_kind: MediaKind, mime_type: str
    ) -> tuple[str, str]:
        _unavailable()

    def insert_generation(
        self,
        feature: str,
        spec: dict[str, JsonValue],
        *,
        contract_version: int = 1,
    ) -> GenerationRecord:
        _unavailable()

    def has_queued(self) -> bool:
        _unavailable()

    def claim_next_queued(self) -> GenerationRecord | None:
        _unavailable()

    def dashboard(
        self,
        *,
        window: DashboardWindow,
        tz: str,
        now: datetime | None = None,
    ) -> DashboardSnapshot:
        del window, tz, now
        _unavailable()

    def list_queue_generations(self) -> list[QueueGenerationRecord]:
        _unavailable()

    def reorder_queued_generation(
        self, generation_id: str, before_generation_id: str | None
    ) -> GenerationRecord:
        del generation_id, before_generation_id
        _unavailable()

    def request_generation_cancellation(self, generation_id: str) -> GenerationRecord:
        del generation_id
        _unavailable()

    def retry_generation(self, generation_id: str) -> GenerationRecord:
        _unavailable()

    def mark_succeeded(
        self,
        generation_id: str,
        outputs: list[OutputSpec],
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        _unavailable()

    def mark_failed(
        self,
        generation_id: str,
        error_code: GenerationErrorCode,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        _unavailable()

    def mark_cancelled(
        self,
        generation_id: str,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        _unavailable()

    def get_generation(self, generation_id: str) -> GenerationRecord | None:
        _unavailable()

    def list_generations(self, feature: str) -> list[GenerationRecord]:
        _unavailable()

    def list_recent_features(
        self, *, limit: int, allowed: Collection[str] | None = None
    ) -> list[str]:
        del limit, allowed
        _unavailable()

    def delete_generation(self, generation_id: str) -> bool:
        _unavailable()

    def delete_asset(self, asset_id: str) -> bool:
        _unavailable()

    def fail_running_on_boot(self) -> int:
        return 0

    def unlink_partial_outputs(self, paths: list[str]) -> None:
        del paths
