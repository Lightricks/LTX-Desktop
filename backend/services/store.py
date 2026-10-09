from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import Protocol

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
)


class Store(Protocol):
    def ingest_upload(self, source_path: str) -> AssetRecord: ...
    def get_asset(
        self, asset_id: str, *, tombstone_missing: bool = True
    ) -> AssetRecord | None: ...
    def list_assets(
        self,
        *,
        media_kind: MediaKind | None = None,
        sort: str = "created_at-desc",
        q: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
        tombstone_missing: bool = True,
    ) -> AssetListPage: ...
    def allocate_output_path(
        self, media_kind: MediaKind, mime_type: str
    ) -> tuple[str, str]: ...
    def insert_generation(
        self,
        feature: str,
        spec: dict[str, JsonValue],
        *,
        contract_version: int = 1,
    ) -> GenerationRecord: ...
    def has_queued(self) -> bool: ...
    def claim_next_queued(self) -> GenerationRecord | None: ...
    def dashboard(
        self,
        *,
        window: DashboardWindow,
        tz: str,
        now: datetime | None = None,
    ) -> DashboardSnapshot: ...
    def list_queue_generations(self) -> list[QueueGenerationRecord]: ...
    def reorder_queued_generation(
        self, generation_id: str, before_generation_id: str | None
    ) -> GenerationRecord: ...
    def request_generation_cancellation(self, generation_id: str) -> GenerationRecord: ...
    def retry_generation(self, generation_id: str) -> GenerationRecord: ...
    def mark_succeeded(
        self,
        generation_id: str,
        outputs: list[OutputSpec],
        *,
        attempt_count: int,
    ) -> GenerationRecord: ...
    def mark_failed(
        self,
        generation_id: str,
        error_code: GenerationErrorCode,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord: ...
    def mark_cancelled(
        self,
        generation_id: str,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord: ...
    def get_generation(self, generation_id: str) -> GenerationRecord | None: ...
    def list_generations(self, feature: str) -> list[GenerationRecord]: ...
    def list_recent_features(
        self, *, limit: int, allowed: Collection[str] | None = None
    ) -> list[str]: ...
    def delete_generation(self, generation_id: str) -> bool: ...
    def delete_asset(self, asset_id: str) -> bool:
        """True if deleted or already deleted. False if the id never existed.
        Raises AssetInUseError, StatusError (producer running or cancelling).
        Canonical and thumbnail unlinks are best-effort after commit."""
        ...
    def fail_running_on_boot(self) -> int: ...
    def unlink_partial_outputs(self, paths: list[str]) -> None: ...
