from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime
from typing import NoReturn

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
    UnavailableError,
)


def _unavailable() -> NoReturn:
    raise UnavailableError()


class UnavailableStore:
    def list_waiting_assistant_runs(self) -> list[str]:
        _unavailable()

    def enqueue_assistant_message(self, item_id: str, conversation_id: str, payload: str, *, payload_version: int, title: str) -> bool:
        _unavailable()

    def list_assistant_queue(self, conversation_id: str) -> list[AssistantQueueRecord]:
        _unavailable()

    def remove_assistant_queue_item(self, conversation_id: str, item_id: str) -> bool:
        _unavailable()

    def set_assistant_queue_error(self, conversation_id: str, item_id: str, error: str) -> None:
        _unavailable()

    def assistant_queue_errors(self, conversation_id: str) -> dict[str, str]:
        _unavailable()

    def set_assistant_queue_hold(self, conversation_id: str, on_hold: bool) -> None:
        _unavailable()

    def assistant_queue_on_hold(self, conversation_id: str) -> bool:
        _unavailable()

    def hold_queued_assistant_conversations(self) -> None:
        _unavailable()

    def get_asset_producer_id(self, asset_id: str) -> str | None:
        _unavailable()

    def list_assistant_runs(self, conversation_id: str) -> list[AssistantRunRecord]:
        _unavailable()

    def list_assistant_job_ids(self, conversation_id: str) -> list[str]:
        _unavailable()

    def list_assistant_conversations(self) -> list[tuple[str, str, int]]:
        _unavailable()

    def delete_assistant_conversation(self, conversation_id: str) -> None:
        _unavailable()

    def get_latest_assistant_run(self, conversation_id: str) -> AssistantRunRecord | None:
        _unavailable()

    def get_assistant_run(self, run_id: str) -> AssistantRunRecord | None:
        _unavailable()

    def insert_assistant_run(self, run: AssistantRunRecord, *, title: str, after: str | None, queued: bool = False) -> bool:
        _unavailable()

    def save_assistant_run(self, run: AssistantRunRecord) -> bool:
        _unavailable()

    def stop_assistant_run(self, run_id: str, response: str) -> None:
        _unavailable()

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
        generation_id: str | None = None,
        lane: str = LOCAL_LANE,
    ) -> GenerationRecord:
        _unavailable()

    def has_queued(self, lane: str = LOCAL_LANE) -> bool:
        _unavailable()

    def claim_next_queued(self, lane: str = LOCAL_LANE) -> GenerationRecord | None:
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

    def request_generation_cancellation(
        self, generation_id: str, *, uncancellable_lanes: Collection[str] = ()
    ) -> GenerationRecord:
        del generation_id, uncancellable_lanes
        _unavailable()

    def retry_generation(
        self, generation_id: str, lane: str | None = None
    ) -> GenerationRecord:
        _unavailable()

    def mark_succeeded(
        self,
        generation_id: str,
        outputs: list[OutputSpec],
        *,
        attempt_count: int,
        written_prompt: str | None = None,
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

    def list_generations(self, feature: str | None = None, *, limit: int | None = None) -> list[GenerationRecord]:
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

    def load_device_frontier(
        self, identity: DeviceIdentity, epoch: int
    ) -> Mapping[CalibrationBucket, tuple[FrontierPoint, ...]]:
        del identity, epoch
        return {}

    def load_device_machine_factor(
        self, identity: DeviceIdentity, epoch: int
    ) -> float | None:
        del identity, epoch
        return None

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
        del identity, epoch, bucket, point, priced_gib, generation_id, conv_vae
        return None

    def reassign_unknown_lanes(self, known: Collection[str]) -> int:
        del known
        return 0

    def set_provider_job_id(
        self, generation_id: str, attempt_count: int, provider_job_id: str
    ) -> bool:
        del generation_id, attempt_count, provider_job_id
        return False

    def clear_provider_job_id(self, generation_id: str, provider_job_id: str) -> bool:
        del generation_id, provider_job_id
        return False

    def list_provider_orphans(self) -> list[ProviderOrphan]:
        return []

    def list_displaced_provider_jobs(self, generation_id: str) -> list[ProviderOrphan]:
        del generation_id
        return []

    def unlink_partial_outputs(self, paths: list[str]) -> None:
        del paths
