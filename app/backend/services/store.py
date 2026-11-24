from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Protocol

from pydantic import JsonValue

from runtime_config.device_calibration import CalibrationBucket, DeviceIdentity, FrontierPoint
from services.dashboard_stats import DashboardSnapshot, DashboardWindow

from services.records import (
    LOCAL_LANE,
    AssetListPage,
    AssetRecord,
    AssistantQueueRecord,
    AssistantRunRecord,
    GenerationErrorCode,
    GenerationRecord,
    MediaKind,
    OutputSpec,
    ProviderOrphan,
    QueueGenerationRecord,
)


class CalibrationStore(Protocol):
    def load_device_frontier(
        self, identity: DeviceIdentity, epoch: int
    ) -> Mapping[CalibrationBucket, tuple[FrontierPoint, ...]]: ...
    def load_device_machine_factor(
        self, identity: DeviceIdentity, epoch: int
    ) -> float | None: ...
    def widen_device_frontier(
        self,
        identity: DeviceIdentity,
        epoch: int,
        bucket: CalibrationBucket,
        point: FrontierPoint,
        *,
        priced_gib: float,
        generation_id: str | None,
        conv_vae: bool = False,
    ) -> tuple[FrontierPoint, ...] | None:
        """The bucket's frontier after adding ``point``; None when a stored point covers it."""
        ...


class Store(CalibrationStore, Protocol):
    def get_assistant_run(self, run_id: str) -> AssistantRunRecord | None: ...
    def get_latest_assistant_run(self, conversation_id: str) -> AssistantRunRecord | None: ...
    def insert_assistant_run(self, run: AssistantRunRecord, *, title: str, after: str | None, queued: bool = False) -> bool: ...
    def save_assistant_run(self, run: AssistantRunRecord) -> bool: ...
    def stop_assistant_run(self, run_id: str, response: str) -> None: ...
    def list_assistant_runs(self, conversation_id: str) -> list[AssistantRunRecord]: ...
    def list_assistant_job_ids(self, conversation_id: str) -> list[str]: ...
    def list_assistant_conversations(self) -> list[tuple[str, str, int]]: ...
    def delete_assistant_conversation(self, conversation_id: str) -> None: ...
    def list_waiting_assistant_runs(self) -> list[str]: ...
    def enqueue_assistant_message(self, item_id: str, conversation_id: str, payload: str, *, payload_version: int, title: str) -> bool: ...
    def list_assistant_queue(self, conversation_id: str) -> list[AssistantQueueRecord]: ...
    def remove_assistant_queue_item(self, conversation_id: str, item_id: str) -> bool: ...
    def set_assistant_queue_error(self, conversation_id: str, item_id: str, error: str) -> None: ...
    def assistant_queue_errors(self, conversation_id: str) -> dict[str, str]: ...
    def set_assistant_queue_hold(self, conversation_id: str, on_hold: bool) -> None: ...
    def assistant_queue_on_hold(self, conversation_id: str) -> bool: ...
    def hold_queued_assistant_conversations(self) -> None: ...
    def get_asset_producer_id(self, asset_id: str) -> str | None: ...

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
        generation_id: str | None = None,
        lane: str = LOCAL_LANE,
    ) -> GenerationRecord: ...
    def has_queued(self, lane: str = LOCAL_LANE) -> bool: ...
    def claim_next_queued(self, lane: str = LOCAL_LANE) -> GenerationRecord | None: ...
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
    def request_generation_cancellation(
        self, generation_id: str, *, uncancellable_lanes: Collection[str] = ()
    ) -> GenerationRecord:
        """Raises StatusError for a running job on one of uncancellable_lanes."""
        ...
    def retry_generation(
        self, generation_id: str, lane: str | None = None
    ) -> GenerationRecord: ...
    def mark_succeeded(
        self,
        generation_id: str,
        outputs: list[OutputSpec],
        *,
        attempt_count: int,
        written_prompt: str | None = None,
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
    def list_generations(self, feature: str | None = None, *, limit: int | None = None) -> list[GenerationRecord]: ...
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
    def reassign_unknown_lanes(self, known: Collection[str]) -> int: ...
    def set_provider_job_id(
        self, generation_id: str, attempt_count: int, provider_job_id: str
    ) -> bool:
        """Record the id of a running attempt at its provider. False when it is not running."""
        ...
    def clear_provider_job_id(self, generation_id: str, provider_job_id: str) -> bool:
        """Forget a provider job once the provider dropped it. False if nothing holds that id."""
        ...
    def list_provider_orphans(self) -> list[ProviderOrphan]:
        """Jobs that may still run at a provider: an id is stored and not yet cancelled.

        Includes the older attempts that a retry replaced, and deleted rows."""
        ...
    def list_displaced_provider_jobs(self, generation_id: str) -> list[ProviderOrphan]:
        """The ids of older attempts of one job that a retry replaced and nobody cancelled."""
        ...
    def unlink_partial_outputs(self, paths: list[str]) -> None: ...
