from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
import uuid
from collections.abc import Collection
from datetime import datetime
from pathlib import Path
from typing import cast, get_args

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from services.dashboard_stats import DashboardSnapshot, DashboardWindow, compute_dashboard
from services.migrations import MIGRATIONS_DIR, apply_migrations
from services.media_probe import (
    extension_for_mime,
    probe_file,
    validate_ingest_source,
)
from services.records import (
    AssetInUseError,
    AssetListPage,
    AssetMetadata,
    AssetRecord,
    AttemptError,
    GenerationErrorCode,
    GenerationRecord,
    ListedAsset,
    MarkSucceededError,
    MediaError,
    MediaKind,
    OutputSpec,
    PairedDeviceRecord,
    QueueGenerationRecord,
    StatusError,
    UnavailableError,
)
from services.thumbnails import PillowAvThumbnailWriter, ThumbnailWriter

_RUNNING_STATUSES: set[str] = {"running"}
_CANCELLING_STATUSES: set[str] = {"cancelling"}
_IN_FLIGHT_STATUSES: set[str] = _RUNNING_STATUSES | _CANCELLING_STATUSES
_RETRYABLE_STATUSES: set[str] = {"failed"}
_IN_FLIGHT_DIRNAME = ".in-flight"
_UNLINK_RETRY_ATTEMPTS = 5
_UNLINK_RETRY_BASE_SECONDS = 0.05
_QUEUE_RANK_SPACING = 1_000_000


def _unlink_path(path: Path) -> None:
    """Unlink ``path``, retrying Windows sharing violations after a brief lock."""
    for attempt in range(_UNLINK_RETRY_ATTEMPTS):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == _UNLINK_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_UNLINK_RETRY_BASE_SECONDS * (attempt + 1))


_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_ERROR_CODES = frozenset(get_args(GenerationErrorCode))
logger = logging.getLogger(__name__)


class _AssetMetadataHolder(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    metadata: AssetMetadata


def _parse_json_object(raw: str) -> dict[str, JsonValue]:
    parsed: object = json.loads(raw)
    return _JSON_OBJECT.validate_python(parsed)


def _parse_asset_metadata(raw: str) -> AssetMetadata:
    parsed: object = json.loads(raw)
    return _AssetMetadataHolder.model_validate({"metadata": parsed}).metadata


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _require_int(value)


def _require_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value)!r}")
    return value


def _is_present_file(path: str) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False


def collect_asset_ids(value: object) -> list[str]:
    """Return unique nested spec values whose key is ``assetId``."""
    found: list[str] = []
    seen: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            typed_dict = cast(dict[object, object], node)
            raw = typed_dict.get("assetId")
            if isinstance(raw, str) and raw != "" and raw not in seen:
                seen.add(raw)
                found.append(raw)
            for child in typed_dict.values():
                walk(child)
            return
        if isinstance(node, list):
            typed_list = cast(list[object], node)
            for child in typed_list:
                walk(child)

    walk(value)
    return found


def _derived_path(assets_dir: Path, asset_id: str, mime_type: str) -> Path:
    return assets_dir / f"{asset_id}{extension_for_mime(mime_type)}"


def _encode_cursor(created_at: int, asset_id: str) -> str:
    return f"{created_at}:{asset_id}"


def _decode_cursor(cursor: str) -> tuple[int, str]:
    created_at_s, asset_id = cursor.split(":", 1)
    if asset_id == "":
        raise ValueError("invalid cursor")
    return int(created_at_s), asset_id


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _fetch_asset_list_rows(
    conn: sqlite3.Connection,
    *,
    media_kind: MediaKind | None,
    q: str | None,
    descending: bool,
    after: tuple[int, str] | None,
    fetch_limit: int,
) -> list[sqlite3.Row]:
    clauses = ["deleted_at IS NULL", "origin = 'generated'"]
    params: list[str | int] = []
    if media_kind is not None:
        clauses.append("media_kind = ?")
        params.append(media_kind)
    if q:
        clauses.append("LOWER(name) LIKE '%' || LOWER(?) || '%' ESCAPE '\\'")
        params.append(_like_escape(q))
    if after is not None:
        if descending:
            clauses.append("(created_at, id) < (?, ?)")
        else:
            clauses.append("(created_at, id) > (?, ?)")
        params.append(after[0])
        params.append(after[1])
    order = "DESC" if descending else "ASC"
    params.append(fetch_limit)
    return conn.execute(
        f"""
        SELECT * FROM assets
        WHERE {" AND ".join(clauses)}
        ORDER BY created_at {order}, id {order}
        LIMIT ?
        """,
        params,
    ).fetchall()


class SqliteStore:
    def __init__(
        self,
        app_data_dir: Path,
        thumbnail_writer: ThumbnailWriter | None = None,
    ) -> None:
        try:
            self._assets_dir = app_data_dir / "assets"
            self._in_flight_dir = self._assets_dir / _IN_FLIGHT_DIRNAME
            self._db_path = app_data_dir / "store.sqlite3"
            self._assets_dir.mkdir(parents=True, exist_ok=True)
            conn = self._connect()
            apply_migrations(conn, MIGRATIONS_DIR)
            conn.commit()
            conn.close()
            self._thumbnail_writer: ThumbnailWriter = (
                thumbnail_writer
                if thumbnail_writer is not None
                else PillowAvThumbnailWriter()
            )
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _new_id(self) -> str:
        return str(uuid.uuid4())

    def _next_queued_at(self, conn: sqlite3.Connection, now: int) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(queued_at), 0) AS latest_queued_at FROM generations"
        ).fetchone()
        latest = 0 if row is None else _require_int(row["latest_queued_at"])
        return max(now, latest + 1)

    def _next_queue_rank(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(queue_rank), 0) AS latest_queue_rank
            FROM generations
            WHERE status = 'queued' AND deleted_at IS NULL
            """
        ).fetchone()
        latest = 0 if row is None else _require_int(row["latest_queue_rank"])
        return latest + _QUEUE_RANK_SPACING

    def _rebalance_queued_ranks(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id FROM generations
            WHERE status = 'queued' AND deleted_at IS NULL
            ORDER BY queue_rank ASC, id ASC
            """
        ).fetchall()
        conn.executemany(
            "UPDATE generations SET queue_rank = ? WHERE id = ?",
            [
                (_QUEUE_RANK_SPACING * (index + 1), str(row["id"]))
                for index, row in enumerate(rows)
            ],
        )

    def _asset_record(self, row: sqlite3.Row) -> AssetRecord:
        mime_type = str(row["mime_type"])
        asset_id = str(row["id"])
        raw_thumb = row["thumbnail_path"]
        thumbnail_path = None if raw_thumb is None else str(raw_thumb)
        return AssetRecord.model_validate(
            {
                "id": asset_id,
                "media_kind": str(row["media_kind"]),
                "origin": str(row["origin"]),
                "path": str(_derived_path(self._assets_dir, asset_id, mime_type)),
                "thumbnail_path": thumbnail_path,
                "mime_type": mime_type,
                "name": str(row["name"]),
                "metadata": _parse_asset_metadata(str(row["metadata"])),
                "created_at": _require_int(row["created_at"]),
            }
        )

    def ingest_upload(self, source_path: str) -> AssetRecord:
        source = Path(source_path).resolve()
        try:
            is_file = source.is_file()
        except OSError:
            is_file = False
        if not is_file:
            raise FileNotFoundError(source)
        validate_ingest_source(source)
        asset_id = self._new_id()
        dest = self._assets_dir / f"{asset_id}{source.suffix}"
        created_at = self._now_ms()
        name = source.name
        try:
            shutil.copy2(source, dest)
            media_kind, mime_type, metadata = probe_file(dest)
            canonical = _derived_path(self._assets_dir, asset_id, mime_type)
            if canonical != dest:
                dest.replace(canonical)
                dest = canonical
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO assets (
                        id, media_kind, origin, producer_generation_id,
                        output_ordinal, mime_type, name, metadata, created_at,
                        deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_id,
                        media_kind,
                        "uploaded",
                        None,
                        None,
                        mime_type,
                        name,
                        json.dumps(metadata.model_dump(mode="json")),
                        created_at,
                        None,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except MediaError:
            dest.unlink(missing_ok=True)
            raise
        except Exception as exc:
            dest.unlink(missing_ok=True)
            logger.warning("unreadable media file %s: %s", dest, exc)
            raise MediaError("UNREADABLE_MEDIA", "unreadable media file") from exc
        return self._persist_thumbnail(
            AssetRecord(
                id=asset_id,
                media_kind=media_kind,
                origin="uploaded",
                path=str(dest),
                thumbnail_path=None,
                mime_type=mime_type,
                name=name,
                metadata=metadata,
                created_at=created_at,
            )
        )

    def _persist_thumbnail(self, record: AssetRecord) -> AssetRecord:
        try:
            written = self._thumbnail_writer.write(record.path, record.media_kind)
        except Exception:
            logger.warning("thumbnail writer raised for %s", record.id, exc_info=True)
            return record
        if written is None:
            return record
        persisted = False
        try:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "UPDATE assets SET thumbnail_path = ? WHERE id = ? AND deleted_at IS NULL",
                    (written, record.id),
                )
                conn.commit()
                persisted = cursor.rowcount == 1
            finally:
                conn.close()
        except (sqlite3.Error, OSError):
            logger.warning(
                "thumbnail path persist failed for %s",
                record.id,
                exc_info=True,
            )
        if persisted:
            return record.model_copy(update={"thumbnail_path": written})
        try:
            _unlink_path(Path(written))
        except OSError:
            logger.warning(
                "unpersisted thumbnail unlink failed for %s",
                record.id,
                exc_info=True,
            )
        return record

    def get_asset(
        self, asset_id: str, *, tombstone_missing: bool = True
    ) -> AssetRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT * FROM assets
                WHERE id = ? AND deleted_at IS NULL
                """,
                (asset_id,),
            ).fetchone()
            record = None if row is None else self._asset_record(row)
        finally:
            conn.close()
        if record is None:
            return None
        if _is_present_file(record.path):
            return record
        if tombstone_missing:
            self._tombstone_missing_assets((record.id,))
        return None

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
        if sort not in ("created_at-desc", "created_at-asc"):
            raise ValueError(f"invalid sort: {sort}")
        descending = sort == "created_at-desc"
        after: tuple[int, str] | None = (
            _decode_cursor(cursor) if cursor is not None else None
        )
        conn = self._connect()
        try:
            in_use_ids = self._live_in_use_ids(conn)
            rows = _fetch_asset_list_rows(
                conn,
                media_kind=media_kind,
                q=q,
                descending=descending,
                after=after,
                fetch_limit=limit + 1,
            )
            records = [self._asset_record(row) for row in rows]
        finally:
            conn.close()
        window = records[:limit]
        missing_ids = tuple(
            record.id for record in window if not _is_present_file(record.path)
        )
        if tombstone_missing:
            self._tombstone_missing_assets(missing_ids)
        missing = set(missing_ids)
        items = tuple(
            ListedAsset(asset=record, in_use=record.id in in_use_ids)
            for record in window
            if record.id not in missing
        )
        next_cursor: str | None = None
        if len(records) > limit and window:
            last = window[-1]
            next_cursor = _encode_cursor(last.created_at, last.id)
        return AssetListPage(items=items, next_cursor=next_cursor)

    def _tombstone_missing_assets(self, asset_ids: tuple[str, ...]) -> None:
        """Stamp deleted_at only. Skip live inputs. Do not unlink or hide the producer."""
        if not asset_ids:
            return
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            in_use_ids = self._live_in_use_ids(conn)
            stamped = tuple(
                asset_id for asset_id in asset_ids if asset_id not in in_use_ids
            )
            if stamped:
                now = self._now_ms()
                conn.executemany(
                    """
                    UPDATE assets
                    SET deleted_at = ?
                    WHERE id = ? AND deleted_at IS NULL
                    """,
                    [(now, asset_id) for asset_id in stamped],
                )
            conn.commit()
        finally:
            conn.close()

    def _live_in_use_ids(self, conn: sqlite3.Connection) -> set[str]:
        ids: set[str] = set()
        for row in conn.execute(
            "SELECT spec FROM generations WHERE deleted_at IS NULL"
        ):
            ids.update(collect_asset_ids(_parse_json_object(str(row["spec"]))))
        return ids

    def allocate_output_path(
        self, media_kind: MediaKind, mime_type: str
    ) -> tuple[str, str]:
        if media_kind not in get_args(MediaKind):
            raise ValueError(f"invalid media_kind: {media_kind}")
        asset_id = self._new_id()
        self._in_flight_dir.mkdir(parents=True, exist_ok=True)
        dest = _derived_path(self._in_flight_dir, asset_id, mime_type)
        return (asset_id, str(dest))

    def insert_generation(
        self,
        feature: str,
        spec: dict[str, JsonValue],
        *,
        contract_version: int = 1,
    ) -> GenerationRecord:
        if contract_version < 1:
            raise ValueError("contract_version must be >= 1")
        generation_id = self._new_id()
        now = self._now_ms()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            queued_at = self._next_queued_at(conn, now)
            queue_rank = self._next_queue_rank(conn)
            for asset_id in collect_asset_ids(spec):
                row = conn.execute(
                    "SELECT id FROM assets WHERE id = ? AND deleted_at IS NULL",
                    (asset_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    raise ValueError(f"unknown input asset: {asset_id}")
            conn.execute(
                """
                INSERT INTO generations (
                    id, feature, contract_version, status, spec, error_code,
                    created_at, queued_at, queue_rank, attempt_count, started_at, finished_at,
                    deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    feature,
                    contract_version,
                    "queued",
                    json.dumps(spec),
                    None,
                    now,
                    queued_at,
                    queue_rank,
                    0,
                    None,
                    None,
                    None,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        record = self.get_generation(generation_id)
        if record is None:
            raise RuntimeError(f"generation {generation_id} missing after insert")
        return record

    def has_queued(self) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM generations WHERE status = 'queued' AND deleted_at IS NULL LIMIT 1"
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def list_queue_generations(self) -> list[QueueGenerationRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM generations
                WHERE status IN ('queued', 'running', 'cancelling')
                  AND deleted_at IS NULL
                ORDER BY
                    CASE WHEN status = 'queued' THEN 1 ELSE 0 END ASC,
                    started_at ASC,
                    queue_rank ASC,
                    id ASC
                """
            ).fetchall()
            records: list[QueueGenerationRecord] = []
            for row in rows:
                generation = self._record_from_generation_row(row, [])
                input_ids = collect_asset_ids(generation.spec)
                if input_ids:
                    placeholders = ",".join("?" * len(input_ids))
                    asset_rows = conn.execute(
                        f"""
                        SELECT * FROM assets
                        WHERE id IN ({placeholders}) AND deleted_at IS NULL
                        """,
                        input_ids,
                    ).fetchall()
                    assets_by_id = {
                        str(asset_row["id"]): self._asset_record(asset_row)
                        for asset_row in asset_rows
                    }
                    input_assets = tuple(
                        assets_by_id[asset_id]
                        for asset_id in input_ids
                        if asset_id in assets_by_id
                    )
                else:
                    input_assets = ()
                records.append(
                    QueueGenerationRecord(
                        generation=generation,
                        input_assets=input_assets,
                    )
                )
            return records
        finally:
            conn.close()

    def reorder_queued_generation(
        self, generation_id: str, before_generation_id: str | None
    ) -> GenerationRecord:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._require_status(conn, generation_id, {"queued"})
            if before_generation_id == generation_id:
                return self._commit_loaded(conn, generation_id)
            if before_generation_id is not None:
                self._require_status(conn, before_generation_id, {"queued"})

            rows = conn.execute(
                """
                SELECT id, queue_rank FROM generations
                WHERE status = 'queued' AND deleted_at IS NULL AND id != ?
                ORDER BY queue_rank ASC, id ASC
                """,
                (generation_id,),
            ).fetchall()
            if before_generation_id is None:
                previous_rank = (
                    _require_int(rows[-1]["queue_rank"]) if rows else 0
                )
                new_rank = previous_rank + _QUEUE_RANK_SPACING
            else:
                target_index = next(
                    (
                        index
                        for index, row in enumerate(rows)
                        if str(row["id"]) == before_generation_id
                    ),
                    None,
                )
                if target_index is None:
                    raise RuntimeError("queued reorder target disappeared")
                target_rank = _require_int(rows[target_index]["queue_rank"])
                if target_index == 0:
                    new_rank = target_rank - _QUEUE_RANK_SPACING
                else:
                    previous_rank = _require_int(rows[target_index - 1]["queue_rank"])
                    if target_rank - previous_rank <= 1:
                        self._rebalance_queued_ranks(conn)
                        rows = conn.execute(
                            """
                            SELECT id, queue_rank FROM generations
                            WHERE status = 'queued' AND deleted_at IS NULL AND id != ?
                            ORDER BY queue_rank ASC, id ASC
                            """,
                            (generation_id,),
                        ).fetchall()
                        target_index = next(
                            index
                            for index, row in enumerate(rows)
                            if str(row["id"]) == before_generation_id
                        )
                        target_rank = _require_int(rows[target_index]["queue_rank"])
                        previous_rank = _require_int(
                            rows[target_index - 1]["queue_rank"]
                        )
                    new_rank = previous_rank + (target_rank - previous_rank) // 2

            conn.execute(
                """
                UPDATE generations SET queue_rank = ?
                WHERE id = ? AND status = 'queued' AND deleted_at IS NULL
                """,
                (new_rank, generation_id),
            )
            return self._commit_loaded(conn, generation_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def request_generation_cancellation(self, generation_id: str) -> GenerationRecord:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            status = self._require_status(conn, generation_id, {"queued", "running"})
            now = self._now_ms()
            if status == "queued":
                conn.execute(
                    """
                    UPDATE generations
                    SET status = 'cancelled', queue_rank = NULL, finished_at = ?
                    WHERE id = ? AND status = 'queued' AND deleted_at IS NULL
                    """,
                    (now, generation_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE generations
                    SET status = 'cancelling', queue_rank = NULL, finished_at = NULL
                    WHERE id = ? AND status = 'running' AND deleted_at IS NULL
                    """,
                    (generation_id,),
                )
            return self._commit_loaded(conn, generation_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def claim_next_queued(self) -> GenerationRecord | None:
        while True:
            claimed = self._claim_one_queued()
            if claimed is None:
                return None
            if self._has_missing_input(claimed):
                self.mark_failed(
                    claimed.id,
                    "INPUT_MISSING",
                    attempt_count=claimed.attempt_count,
                )
                continue
            return claimed

    def dashboard(
        self,
        *,
        window: DashboardWindow,
        tz: str,
        now: datetime | None = None,
    ) -> DashboardSnapshot:
        conn = self._connect()
        try:
            return compute_dashboard(
                conn,
                window=window,
                tz=tz,
                now=now,
            )
        finally:
            conn.close()

    def _has_missing_input(self, claimed: GenerationRecord) -> bool:
        conn = self._connect()
        try:
            for asset_id in collect_asset_ids(claimed.spec):
                row = conn.execute(
                    """
                    SELECT id, mime_type FROM assets
                    WHERE id = ? AND deleted_at IS NULL
                    """,
                    (asset_id,),
                ).fetchone()
                if row is None:
                    return True
                path = _derived_path(
                    self._assets_dir, str(row["id"]), str(row["mime_type"])
                )
                if not _is_present_file(str(path)):
                    return True
            return False
        finally:
            conn.close()

    def _claim_one_queued(self) -> GenerationRecord | None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT id
                FROM generations
                WHERE status = 'queued' AND deleted_at IS NULL
                ORDER BY queue_rank ASC, id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                conn.commit()
                return None

            generation_id = str(row["id"])
            now = self._now_ms()
            # started_at is this attempt only. Retry clears it, and the next
            # claim sets it again, so finished_at - started_at skips earlier
            # attempts. Retry also increments attempt_count before this claim
            # increments it again.
            cursor = conn.execute(
                """
                UPDATE generations
                SET status = 'running',
                    queue_rank = NULL,
                    attempt_count = attempt_count + 1,
                    started_at = ?,
                    finished_at = NULL,
                    error_code = NULL
                WHERE id = ? AND status = 'queued' AND deleted_at IS NULL
                """,
                (now, generation_id),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                return None
            return self._commit_loaded(conn, generation_id)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def retry_generation(self, generation_id: str) -> GenerationRecord:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._require_status(conn, generation_id, _RETRYABLE_STATUSES)
            generated_paths = self._delete_generated_assets(conn, generation_id)
            now = self._now_ms()
            queued_at = self._next_queued_at(conn, now)
            queue_rank = self._next_queue_rank(conn)
            conn.execute(
                """
                UPDATE generations
                SET status = 'queued',
                    queued_at = ?,
                    queue_rank = ?,
                    error_code = NULL,
                    started_at = NULL,
                    finished_at = NULL,
                    attempt_count = attempt_count + 1
                WHERE id = ? AND status = 'failed'
                """,
                (queued_at, queue_rank, generation_id),
            )
            record = self._commit_loaded(conn, generation_id)
            self._unlink_under_assets(generated_paths)
            return record
        finally:
            conn.close()

    def _require_status(
        self, conn: sqlite3.Connection, generation_id: str, allowed: set[str]
    ) -> str:
        row = conn.execute(
            """
            SELECT status FROM generations
            WHERE id = ? AND deleted_at IS NULL
            """,
            (generation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(generation_id)
        status = str(row["status"])
        if status not in allowed:
            raise StatusError(generation_id, status)
        return status

    def _require_attempt(
        self,
        conn: sqlite3.Connection,
        generation_id: str,
        attempt_count: int,
        allowed: set[str],
    ) -> None:
        self._require_status(conn, generation_id, allowed)
        row = conn.execute(
            "SELECT attempt_count FROM generations WHERE id = ?",
            (generation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(generation_id)
        actual = _require_int(row["attempt_count"])
        if actual != attempt_count:
            raise AttemptError(generation_id, attempt_count, actual)

    def _load_required(
        self, conn: sqlite3.Connection, generation_id: str
    ) -> GenerationRecord:
        record = self._generation_from_id(conn, generation_id)
        if record is None:
            raise RuntimeError(f"generation {generation_id} missing after update")
        return record

    def _path_under_assets(self, raw: str) -> Path | None:
        try:
            path = Path(raw).resolve()
        except OSError:
            return None
        assets_dir = self._assets_dir.resolve()
        if not path.is_relative_to(assets_dir):
            return None
        return path

    def _is_under_dir(self, path: Path, directory: Path) -> bool:
        try:
            return path.resolve().is_relative_to(directory.resolve())
        except OSError:
            return False

    def _existing_output_path(self, spec: OutputSpec) -> Path | None:
        dest = self._path_under_assets(spec.dest_path)
        if dest is None:
            return None
        if dest.is_file():
            return dest
        if not self._is_under_dir(dest, self._in_flight_dir):
            return dest
        canonical = _derived_path(self._assets_dir, spec.asset_id, spec.mime_type)
        if canonical.is_file():
            return canonical
        return dest

    def _unlink_partials(self, partial_paths: list[str] | None) -> None:
        for raw in partial_paths or []:
            try:
                path = self._path_under_assets(raw)
                if path is None:
                    continue
                _unlink_path(path)
            except Exception as exc:
                logger.warning(
                    "Could not remove partial output %s: %s",
                    raw,
                    exc,
                )
                continue

    def unlink_partial_outputs(self, paths: list[str]) -> None:
        self._unlink_partials(paths)

    def _restore_promoted_outputs(self, moved: list[tuple[Path, Path]]) -> None:
        for canonical, original in reversed(moved):
            try:
                if not self._is_under_dir(canonical, self._assets_dir):
                    continue
                if not canonical.is_file():
                    continue
                if original.exists():
                    canonical.unlink(missing_ok=True)
                    continue
                if not self._is_under_dir(original, self._assets_dir):
                    canonical.unlink(missing_ok=True)
                    continue
                canonical.replace(original)
            except OSError as exc:
                logger.warning(
                    "Could not restore promoted output %s to %s: %s",
                    canonical,
                    original,
                    exc,
                )

    def _delete_generated_assets(
        self, conn: sqlite3.Connection, generation_id: str
    ) -> list[Path]:
        rows = conn.execute(
            """
            SELECT id, mime_type FROM assets
            WHERE producer_generation_id = ?
            """,
            (generation_id,),
        ).fetchall()
        conn.execute(
            "DELETE FROM assets WHERE producer_generation_id = ?",
            (generation_id,),
        )
        return [
            _derived_path(self._assets_dir, str(row["id"]), str(row["mime_type"]))
            for row in rows
        ]

    def _unlink_under_assets(self, paths: list[Path]) -> None:
        for path in paths:
            if not self._is_under_dir(path, self._assets_dir):
                continue
            _unlink_path(path)

    def _commit_loaded(
        self, conn: sqlite3.Connection, generation_id: str
    ) -> GenerationRecord:
        record = self._load_required(conn, generation_id)
        conn.commit()
        return record

    def _require_attempt_unlinking(
        self,
        conn: sqlite3.Connection,
        generation_id: str,
        attempt_count: int,
        allowed: set[str],
        unlink_paths: list[str] | None,
    ) -> None:
        try:
            self._require_attempt(conn, generation_id, attempt_count, allowed)
        except (AttemptError, StatusError, KeyError):
            conn.rollback()
            self._unlink_partials(unlink_paths)
            raise

    def _reclaim_in_flight_files(self) -> None:
        in_flight_dir = self._in_flight_dir
        if not in_flight_dir.is_dir():
            return
        assets_dir = self._assets_dir.resolve()
        try:
            in_flight_resolved = in_flight_dir.resolve()
        except OSError:
            return
        for child in in_flight_dir.iterdir():
            if not child.is_file():
                continue
            try:
                resolved = child.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(in_flight_resolved):
                continue
            if not resolved.is_relative_to(assets_dir):
                continue
            resolved.unlink(missing_ok=True)

    def _mark_failed_on_conn(
        self,
        conn: sqlite3.Connection,
        generation_id: str,
        error_code: GenerationErrorCode,
        partial_paths: list[str] | None,
        attempt_count: int,
    ) -> GenerationRecord:
        if error_code not in _ERROR_CODES:
            raise ValueError(f"invalid error_code: {error_code}")
        generated_paths = self._delete_generated_assets(conn, generation_id)
        now = self._now_ms()
        conn.execute(
            """
            UPDATE generations
            SET status = 'failed', error_code = ?, queue_rank = NULL, finished_at = ?
            WHERE id = ? AND status = 'running' AND attempt_count = ?
            """,
            (error_code, now, generation_id, attempt_count),
        )
        record = self._commit_loaded(conn, generation_id)
        self._unlink_under_assets(generated_paths)
        self._unlink_partials(partial_paths)
        return record

    def mark_succeeded(
        self,
        generation_id: str,
        outputs: list[OutputSpec],
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        if not outputs:
            raise ValueError("mark_succeeded requires at least one output")
        ordinals = [spec.ordinal for spec in outputs]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("mark_succeeded output ordinals must be unique")
        resolved_outputs: list[tuple[OutputSpec, Path]] = []
        for spec in outputs:
            dest = self._existing_output_path(spec)
            if dest is None:
                raise ValueError(f"output path is outside assets: {spec.dest_path}")
            resolved_outputs.append((spec, dest))
        dest_paths = [spec.dest_path for spec in outputs]
        probed: list[tuple[OutputSpec, Path, MediaKind, str, AssetMetadata]] = []
        for spec, dest in resolved_outputs:
            if not dest.is_file():
                raise MarkSucceededError(generation_id, "OUTPUT_MISSING")
            try:
                media_kind, mime_type, metadata = probe_file(dest)
            except Exception:
                raise MarkSucceededError(generation_id, "OUTPUT_UNREADABLE") from None
            probed.append((spec, dest, media_kind, mime_type, metadata))
        conn = self._connect()
        moved: list[tuple[Path, Path]] = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._require_attempt_unlinking(
                conn,
                generation_id,
                attempt_count,
                _RUNNING_STATUSES,
                dest_paths,
            )
            now = self._now_ms()
            for spec, dest, media_kind, mime_type, metadata in probed:
                if not dest.is_file():
                    raise MarkSucceededError(generation_id, "OUTPUT_MISSING")
                canonical = _derived_path(self._assets_dir, spec.asset_id, mime_type)
                if dest.resolve() != canonical.resolve():
                    dest.replace(canonical)
                    moved.append((canonical, dest))
                    dest = canonical
                conn.execute(
                    """
                    INSERT INTO assets (
                        id, media_kind, origin, producer_generation_id,
                        output_ordinal, mime_type, name, metadata, created_at,
                        deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.asset_id,
                        media_kind,
                        "generated",
                        generation_id,
                        spec.ordinal,
                        mime_type,
                        spec.name,
                        json.dumps(metadata.model_dump(mode="json")),
                        now,
                        None,
                    ),
                )
            conn.execute(
                """
                UPDATE generations
                SET status = 'succeeded', queue_rank = NULL, finished_at = ?
                WHERE id = ? AND status = 'running' AND attempt_count = ?
                """,
                (now, generation_id, attempt_count),
            )
            record = self._commit_loaded(conn, generation_id)
        except Exception:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            self._restore_promoted_outputs(moved)
            raise
        finally:
            conn.close()
        persisted = tuple(self._persist_thumbnail(output) for output in record.outputs)
        return record.model_copy(update={"outputs": persisted})

    def mark_failed(
        self,
        generation_id: str,
        error_code: GenerationErrorCode,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._require_attempt_unlinking(
                conn,
                generation_id,
                attempt_count,
                _RUNNING_STATUSES,
                partial_paths,
            )
            return self._mark_failed_on_conn(
                conn,
                generation_id,
                error_code,
                partial_paths,
                attempt_count,
            )
        finally:
            conn.close()

    def mark_cancelled(
        self,
        generation_id: str,
        partial_paths: list[str] | None = None,
        *,
        attempt_count: int,
    ) -> GenerationRecord:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._require_attempt_unlinking(
                conn,
                generation_id,
                attempt_count,
                _CANCELLING_STATUSES,
                partial_paths,
            )
            now = self._now_ms()
            conn.execute(
                """
                UPDATE generations
                SET status = 'cancelled', queue_rank = NULL, finished_at = ?
                WHERE id = ? AND status = 'cancelling'
                  AND attempt_count = ?
                """,
                (now, generation_id, attempt_count),
            )
            record = self._commit_loaded(conn, generation_id)
            self._unlink_partials(partial_paths)
            return record
        finally:
            conn.close()

    def get_generation(self, generation_id: str) -> GenerationRecord | None:
        conn = self._connect()
        try:
            return self._generation_from_id(conn, generation_id)
        finally:
            conn.close()

    def list_recent_features(
        self, *, limit: int, allowed: Collection[str] | None = None
    ) -> list[str]:
        """Distinct features, newest generation first. Deleted rows do not count."""
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if allowed is not None and len(allowed) == 0:
            return []
        conn = self._connect()
        try:
            if allowed is None:
                rows = conn.execute(
                    """
                    SELECT feature
                    FROM generations
                    WHERE deleted_at IS NULL
                    GROUP BY feature
                    ORDER BY MAX(created_at) DESC, feature ASC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                features = tuple(allowed)
                placeholders = ",".join("?" * len(features))
                rows = conn.execute(
                    f"""
                    SELECT feature
                    FROM generations
                    WHERE deleted_at IS NULL
                      AND feature IN ({placeholders})
                    GROUP BY feature
                    ORDER BY MAX(created_at) DESC, feature ASC
                    LIMIT ?
                    """,
                    (*features, limit),
                ).fetchall()
            return [str(row["feature"]) for row in rows]
        finally:
            conn.close()

    def list_generations(self, feature: str) -> list[GenerationRecord]:
        conn = self._connect()
        try:
            gen_rows = conn.execute(
                """
                SELECT *
                FROM generations
                WHERE feature = ? AND deleted_at IS NULL
                ORDER BY created_at DESC, id DESC
                """,
                (feature,),
            ).fetchall()
            if not gen_rows:
                return []
            ids = [str(row["id"]) for row in gen_rows]
            placeholders = ",".join("?" * len(ids))
            output_rows = conn.execute(
                f"""
                SELECT *
                FROM assets
                WHERE producer_generation_id IN ({placeholders})
                  AND deleted_at IS NULL
                ORDER BY output_ordinal ASC, id ASC
                """,
                ids,
            ).fetchall()
            outputs_by_gen: dict[str, list[sqlite3.Row]] = {
                gen_id: [] for gen_id in ids
            }
            for row in output_rows:
                producer = str(row["producer_generation_id"])
                outputs_by_gen[producer].append(row)
            return [
                self._record_from_generation_row(row, outputs_by_gen[str(row["id"])])
                for row in gen_rows
            ]
        finally:
            conn.close()

    def _generation_from_id(
        self, conn: sqlite3.Connection, generation_id: str
    ) -> GenerationRecord | None:
        row = conn.execute(
            """
            SELECT * FROM generations
            WHERE id = ? AND deleted_at IS NULL
            """,
            (generation_id,),
        ).fetchone()
        if row is None:
            return None
        outputs = conn.execute(
            """
            SELECT * FROM assets
            WHERE producer_generation_id = ? AND deleted_at IS NULL
            ORDER BY output_ordinal ASC, id ASC
            """,
            (generation_id,),
        ).fetchall()
        return self._record_from_generation_row(row, outputs)

    def _record_from_generation_row(
        self, row: sqlite3.Row, output_rows: list[sqlite3.Row]
    ) -> GenerationRecord:
        outputs: list[AssetRecord] = []
        for output_row in output_rows:
            record = self._asset_record(output_row)
            if record.origin == "generated" and not _is_present_file(record.path):
                continue
            outputs.append(record)
        return GenerationRecord.model_validate(
            {
                "id": str(row["id"]),
                "feature": str(row["feature"]),
                "contract_version": _require_int(row["contract_version"]),
                "status": str(row["status"]),
                "spec": _parse_json_object(str(row["spec"])),
                "error_code": _optional_str(row["error_code"]),
                "created_at": _require_int(row["created_at"]),
                "queued_at": _require_int(row["queued_at"]),
                "attempt_count": _require_int(row["attempt_count"]),
                "started_at": _optional_int(row["started_at"]),
                "finished_at": _optional_int(row["finished_at"]),
                "outputs": tuple(outputs),
            }
        )

    def delete_generation(self, generation_id: str) -> bool:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status, deleted_at FROM generations WHERE id = ?
                """,
                (generation_id,),
            ).fetchone()
            if row is None or row["deleted_at"] is not None:
                conn.commit()
                return False
            status = str(row["status"])
            if status in _IN_FLIGHT_STATUSES:
                raise StatusError(generation_id, status)
            now = self._now_ms()
            cursor = conn.execute(
                """
                UPDATE generations
                SET deleted_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now, generation_id),
            )
            conn.commit()
            return cursor.rowcount == 1
        finally:
            conn.close()

    def delete_asset(self, asset_id: str) -> bool:
        """True if deleted or already deleted. False if the id never existed.
        Raises AssetInUseError, StatusError (producer running or cancelling).
        Canonical and thumbnail unlinks are best-effort after commit."""
        unlink_paths: list[Path] = []
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM assets WHERE id = ?",
                (asset_id,),
            ).fetchone()
            if row is None:
                conn.commit()
                return False
            if row["deleted_at"] is not None:
                conn.commit()
                return True
            if asset_id in self._live_in_use_ids(conn):
                raise AssetInUseError(asset_id)
            producer_id = _optional_str(row["producer_generation_id"])
            gen_row: sqlite3.Row | None = None
            if producer_id is not None:
                gen_row = conn.execute(
                    "SELECT status, deleted_at FROM generations WHERE id = ?",
                    (producer_id,),
                ).fetchone()
                if gen_row is not None:
                    status = str(gen_row["status"])
                    if status in _IN_FLIGHT_STATUSES:
                        raise StatusError(producer_id, status)
            now = self._now_ms()
            conn.execute(
                """
                UPDATE assets
                SET deleted_at = ?
                WHERE id = ? AND deleted_at IS NULL
                """,
                (now, asset_id),
            )
            if producer_id is not None:
                sibling_row = conn.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM assets
                    WHERE producer_generation_id = ?
                      AND deleted_at IS NULL
                      AND id != ?
                    """,
                    (producer_id, asset_id),
                ).fetchone()
                n = 0 if sibling_row is None else _require_int(sibling_row["n"])
                if n == 0 and gen_row is not None and gen_row["deleted_at"] is None:
                    status = str(gen_row["status"])
                    if status in _IN_FLIGHT_STATUSES:
                        raise StatusError(producer_id, status)
                    conn.execute(
                        """
                        UPDATE generations
                        SET deleted_at = ?
                        WHERE id = ? AND deleted_at IS NULL
                        """,
                        (now, producer_id),
                    )
            unlink_paths.append(
                _derived_path(self._assets_dir, asset_id, str(row["mime_type"]))
            )
            thumbnail_path = _optional_str(row["thumbnail_path"])
            if thumbnail_path is not None:
                unlink_paths.append(Path(thumbnail_path))
            conn.commit()
        finally:
            conn.close()

        try:
            self._unlink_under_assets(unlink_paths)
        except OSError:
            logger.warning(
                "asset unlink failed for %s",
                asset_id,
                exc_info=True,
            )
        return True

    def fail_running_on_boot(self) -> int:
        now = self._now_ms()
        try:
            conn = self._connect()
            try:
                in_flight_ids = [
                    str(row["id"])
                    for row in conn.execute(
                        """
                        SELECT id FROM generations
                        WHERE status IN ('running', 'cancelling') AND deleted_at IS NULL
                        """
                    ).fetchall()
                ]
                generated_paths: list[Path] = []
                for generation_id in in_flight_ids:
                    generated_paths.extend(
                        self._delete_generated_assets(conn, generation_id)
                    )
                failed_cursor = conn.execute(
                    """
                    UPDATE generations
                    SET status = 'failed', error_code = 'INTERRUPTED',
                        queue_rank = NULL, finished_at = ?
                    WHERE status = 'running' AND deleted_at IS NULL
                    """,
                    (now,),
                )
                cancelled_cursor = conn.execute(
                    """
                    UPDATE generations
                    SET status = 'cancelled', error_code = NULL,
                        queue_rank = NULL, finished_at = ?
                    WHERE status = 'cancelling' AND deleted_at IS NULL
                    """,
                    (now,),
                )
                conn.commit()
                self._unlink_under_assets(generated_paths)
                self._reclaim_in_flight_files()
                return int(failed_cursor.rowcount) + int(cancelled_cursor.rowcount)
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc

    def insert_paired_device(self, record: PairedDeviceRecord) -> PairedDeviceRecord:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO paired_devices (
                    id, name, token_hash, created_at, last_seen_at,
                    first_seen_ip, first_seen_user_agent, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.name,
                    record.token_hash,
                    record.created_at,
                    record.last_seen_at,
                    record.first_seen_ip,
                    record.first_seen_user_agent,
                    record.revoked_at,
                ),
            )
            conn.commit()
            return record
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()

    def get_paired_device_by_token_hash(
        self, token_hash: str
    ) -> PairedDeviceRecord | None:
        return self._get_paired_device("token_hash = ?", (token_hash,))

    def get_paired_device(self, device_id: str) -> PairedDeviceRecord | None:
        return self._get_paired_device("id = ?", (device_id,))

    def list_paired_devices(self) -> list[PairedDeviceRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, name, token_hash, created_at, last_seen_at,
                       first_seen_ip, first_seen_user_agent, revoked_at
                FROM paired_devices
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
            return [self._paired_device_from_row(row) for row in rows]
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()

    def revoke_paired_device(self, device_id: str, *, now_ms: int) -> bool:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                UPDATE paired_devices
                SET revoked_at = ?
                WHERE id = ? AND revoked_at IS NULL
                """,
                (now_ms, device_id),
            )
            conn.commit()
            return cursor.rowcount == 1
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()

    def touch_paired_device(
        self,
        device_id: str,
        *,
        now_ms: int | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE paired_devices
                SET last_seen_at = COALESCE(?, last_seen_at),
                    first_seen_ip = COALESCE(NULLIF(NULLIF(first_seen_ip, ''), 'unknown'), ?),
                    first_seen_user_agent = COALESCE(NULLIF(first_seen_user_agent, ''), ?)
                WHERE id = ? AND revoked_at IS NULL
                """,
                (now_ms, ip, user_agent, device_id),
            )
            conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()

    def _get_paired_device(
        self, where: str, params: tuple[object, ...]
    ) -> PairedDeviceRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                f"""
                SELECT id, name, token_hash, created_at, last_seen_at,
                       first_seen_ip, first_seen_user_agent, revoked_at
                FROM paired_devices
                WHERE {where}
                """,
                params,
            ).fetchone()
            if row is None:
                return None
            return self._paired_device_from_row(row)
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()

    def _paired_device_from_row(self, row: sqlite3.Row) -> PairedDeviceRecord:
        return PairedDeviceRecord(
            id=str(row["id"]),
            name=str(row["name"]),
            token_hash=str(row["token_hash"]),
            created_at=_require_int(row["created_at"]),
            last_seen_at=_require_int(row["last_seen_at"]),
            first_seen_ip=_optional_str(row["first_seen_ip"]),
            first_seen_user_agent=_optional_str(row["first_seen_user_agent"]),
            revoked_at=_optional_int(row["revoked_at"]),
        )
