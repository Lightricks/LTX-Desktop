"""SQLite persistence for the per-device calibration Pareto frontier."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from typing import Final, TypeGuard

from pydantic import TypeAdapter

from runtime_config.budget_decision import LoadMode
from runtime_config.device_calibration import (
    CalibrationBucket,
    DeviceIdentity,
    EnvelopePoint,
    FrontierPoint,
    merge_frontier,
)
from runtime_config.eta import (
    implied_envelope_factor,
    implied_machine_factor,
    median_machine_factor,
)
from runtime_config.measured_cells import BudgetFeature
from services.features.ic_lora_recipes import IC_LORA_RECIPES
from services.records import UnavailableError

_BUDGET_FEATURE: TypeAdapter[BudgetFeature] = TypeAdapter(BudgetFeature)
_LOAD_MODE: TypeAdapter[LoadMode] = TypeAdapter(LoadMode)

# Driver updates can shift reported total VRAM by tens of MiB (e.g. 27.2 MiB on
# an RTX 5090 between 591.86 and 595.79), while distinct hardware tiers stay at
# least 4 GiB apart.
TOTAL_MEMORY_TOLERANCE_BYTES: Final = 512 * 1024**2
# 2026-11-02 UTC: skip pre-1.6.0 runs when bootstrapping from ``generations``.
_MIN_HISTORY_CREATED_AT_MS: Final = 1_793_577_600_000
_MIN_HISTORY_SAMPLES: Final = 3

_IDENTITY_WHERE = (
    "epoch = ? AND system = ? AND gpu_model = ? "
    "AND ABS(total_memory_bytes - ?) <= ?"
)
_BUCKET_COLUMNS = (
    "model_id", "feature", "adapter_id", "stage_mode", "load_mode", "tiled", "chunked"
)
_POINT_COLUMNS = ("seq", "canvas_px", "windows", "peak_growth_bytes", "wall_s", "tiles")


def _require_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value)!r}")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _require_int(value)


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _identity_where_params(identity: DeviceIdentity, epoch: int) -> tuple[object, ...]:
    return (
        epoch,
        identity.system,
        identity.gpu_model,
        identity.total_memory_bytes,
        TOTAL_MEMORY_TOLERANCE_BYTES,
    )


def _identity_insert_params(identity: DeviceIdentity, epoch: int) -> tuple[object, ...]:
    return (epoch, identity.system, identity.gpu_model, identity.total_memory_bytes)


def _bucket_params(bucket: CalibrationBucket) -> tuple[object, ...]:
    return (
        bucket.model_id,
        bucket.feature,
        bucket.adapter_id,
        bucket.stage_mode,
        bucket.load_mode,
        int(bucket.tiled),
        int(bucket.chunked),
    )


def _calibration_bucket(row: sqlite3.Row) -> CalibrationBucket:
    return CalibrationBucket(
        model_id=str(row["model_id"]),
        feature=_BUDGET_FEATURE.validate_python(row["feature"]),
        adapter_id=str(row["adapter_id"]),
        stage_mode=str(row["stage_mode"]),
        load_mode=_LOAD_MODE.validate_python(row["load_mode"]),
        tiled=bool(row["tiled"]),
        chunked=bool(row["chunked"]),
    )


def _frontier_point(row: sqlite3.Row) -> FrontierPoint:
    return FrontierPoint(
        point=EnvelopePoint(
            seq=float(row["seq"]),
            canvas_px=_require_int(row["canvas_px"]),
            windows=_require_int(row["windows"]),
            tiles=_require_int(row["tiles"]),
        ),
        peak_growth_bytes=_optional_int(row["peak_growth_bytes"]),
        wall_s=_optional_float(row["wall_s"]),
    )


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def _history_row_factor(system: str, row: sqlite3.Row) -> float | None:
    """The row's machine factor, or None. Odd stored values skip the row instead of raising."""
    try:
        return _unchecked_history_row_factor(system, row)
    except (TypeError, ValueError, OverflowError):
        return None


def _unchecked_history_row_factor(system: str, row: sqlite3.Row) -> float | None:
    started_at = _optional_int(row["started_at"])
    finished_at = _optional_int(row["finished_at"])
    raw_meta = row["output_metadata"]
    if started_at is None or finished_at is None or finished_at <= started_at or raw_meta is None:
        return None
    wall_s = (finished_at - started_at) / 1000.0
    try:
        meta_obj: object = json.loads(str(raw_meta))
        spec_obj: object = json.loads(str(row["spec"])) if row["spec"] is not None else {}
    except (ValueError, TypeError):
        return None
    if not _is_str_object_dict(meta_obj) or meta_obj.get("mediaType") != "video":
        return None
    inner = meta_obj.get("metadata")
    if not _is_str_object_dict(inner):
        return None
    width = _optional_int(inner.get("width"))
    height = _optional_int(inner.get("height"))
    duration_ms = _optional_int(inner.get("durationMs"))
    fps = _optional_float(inner.get("fps")) or 24.0
    if width is None or height is None or duration_ms is None or duration_ms <= 0:
        return None
    frames = max(1, round((duration_ms / 1000.0) * fps))
    feature = str(row["feature"])
    params = spec_obj.get("params") if _is_str_object_dict(spec_obj) else None
    resolution = params.get("resolution") if _is_str_object_dict(params) else None
    ic_lora = feature in IC_LORA_RECIPES
    stage_2 = not (resolution in ("270p", "360p") or ic_lora)
    has_video_src = feature in ("retake", "extend", "motion-track", "revelio", "deblur", "video-guide") or (
        ic_lora and _is_str_object_dict(params) and params.get("duration") is None
    )
    eta_sys = system if system in ("Windows", "Linux", "Darwin") else "Windows"
    return implied_machine_factor(
        wall_s=wall_s,
        width=width,
        height=height,
        frames=frames,
        system=eta_sys,
        stage_2=stage_2,
        ic_lora=ic_lora,
        ic_lora_id=feature if ic_lora else None,
        source_frames=frames if has_video_src else 0,
        reject_clamped=True,
    )



class DeviceCalibrationStore:
    def __init__(
        self,
        connect: Callable[[], sqlite3.Connection],
        now_ms: Callable[[], int],
    ) -> None:
        self._connect = connect
        self._now_ms = now_ms

    def load_device_frontier(
        self, identity: DeviceIdentity, epoch: int
    ) -> Mapping[CalibrationBucket, tuple[FrontierPoint, ...]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT {", ".join(_BUCKET_COLUMNS)}, {", ".join(_POINT_COLUMNS)}
                FROM device_calibration_frontier
                WHERE {_IDENTITY_WHERE}
                ORDER BY id
                """,
                _identity_where_params(identity, epoch),
            ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()
        frontier: dict[CalibrationBucket, tuple[FrontierPoint, ...]] = {}
        for row in rows:
            bucket = _calibration_bucket(row)
            merged = merge_frontier(frontier.get(bucket, ()), _frontier_point(row))
            if merged is not None:
                frontier[bucket] = merged
        return frontier

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
        bucket_params = _bucket_params(bucket)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                f"""
                SELECT id, {", ".join(_POINT_COLUMNS)}
                FROM device_calibration_frontier
                WHERE {_IDENTITY_WHERE}
                  AND {" AND ".join(f"{column} = ?" for column in _BUCKET_COLUMNS)}
                ORDER BY id
                """,
                (*_identity_where_params(identity, epoch), *bucket_params),
            ).fetchall()
            stored: dict[int, FrontierPoint] = {}
            current: tuple[FrontierPoint, ...] = ()
            for row in rows:
                row_id = _require_int(row["id"])
                row_point = _frontier_point(row)
                stored[row_id] = row_point
                deduped = merge_frontier(current, row_point)
                if deduped is not None:
                    current = deduped
            merged = merge_frontier(current, point)
            if merged is None:
                # Refresh the wall time only on a row of the same shape. A dominating row is a
                # bigger job; its factor must not be derived from a smaller job's wall time.
                if point.wall_s is not None:
                    conn.execute(
                        f"""
                        UPDATE device_calibration_frontier
                        SET wall_s = ?, conv_vae = ?, generation_id = ?
                        WHERE id = (
                            SELECT id FROM device_calibration_frontier
                            WHERE {_IDENTITY_WHERE}
                              AND {" AND ".join(f"{column} = ?" for column in _BUCKET_COLUMNS)}
                              AND seq = ? AND canvas_px = ? AND windows = ?
                            ORDER BY id DESC
                            LIMIT 1
                        )
                        """,
                        (
                            point.wall_s,
                            int(conv_vae),
                            generation_id,
                            *_identity_where_params(identity, epoch),
                            *bucket_params,
                            point.point.seq,
                            point.point.canvas_px,
                            point.point.windows,
                        ),
                    )
                conn.commit()
                return None
            conn.executemany(
                "DELETE FROM device_calibration_frontier WHERE id = ?",
                [(row_id,) for row_id, kept in stored.items() if kept not in merged],
            )
            values = (
                *_identity_insert_params(identity, epoch),
                *bucket_params,
                point.point.seq,
                point.point.canvas_px,
                point.point.windows,
                point.peak_growth_bytes,
                point.wall_s,
                point.point.tiles,
                priced_gib,
                generation_id,
                self._now_ms(),
                int(conv_vae),
            )
            conn.execute(
                f"""
                INSERT INTO device_calibration_frontier (
                    epoch, system, gpu_model, total_memory_bytes,
                    {", ".join(_BUCKET_COLUMNS)}, {", ".join(_POINT_COLUMNS)},
                    priced_gib, generation_id, recorded_at, conv_vae
                ) VALUES ({", ".join("?" for _ in values)})
                """,
                values,
            )
            conn.commit()
            return merged
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()

    def load_device_machine_factor(
        self, identity: DeviceIdentity, epoch: int
    ) -> float | None:
        conn = self._connect()
        try:
            frontier_rows = conn.execute(
                f"""
                SELECT {", ".join(_BUCKET_COLUMNS)}, {", ".join(_POINT_COLUMNS)},
                       conv_vae, generation_id
                FROM device_calibration_frontier
                WHERE {_IDENTITY_WHERE} AND wall_s IS NOT NULL
                ORDER BY id DESC
                LIMIT 20
                """,
                _identity_where_params(identity, epoch),
            ).fetchall()
            history_rows = conn.execute(
                """
                SELECT g.id, g.feature, g.spec, g.started_at, g.finished_at,
                       a.metadata AS output_metadata
                FROM generations g
                JOIN assets a ON a.producer_generation_id = g.id AND a.output_ordinal = 0
                WHERE g.status = 'succeeded'
                  AND g.lane = 'local'
                  AND g.provider_job_id IS NULL
                  AND g.started_at IS NOT NULL
                  AND g.finished_at IS NOT NULL
                  AND g.created_at >= ?
                  AND COALESCE(json_extract(g.spec, '$.params.model'), '') NOT IN (
                      'fast', 'pro', 'fast-2.5', 'pro-2.5'
                  )
                  AND json_extract(a.metadata, '$.mediaType') = 'video'
                ORDER BY g.created_at DESC
                LIMIT 20
                """,
                (_MIN_HISTORY_CREATED_AT_MS,),
            ).fetchall()
        except (sqlite3.Error, OSError) as exc:
            raise UnavailableError() from exc
        finally:
            conn.close()

        factors: list[float] = []
        seen_ids: set[str] = set()
        for row in frontier_rows:
            bucket = _calibration_bucket(row)
            pt = _frontier_point(row)
            gen_id = row["generation_id"]
            if isinstance(gen_id, str) and gen_id:
                seen_ids.add(gen_id)
            # NULL rows predate the column and ran the CUDA default, DiffVAE.
            factor = implied_envelope_factor(
                identity.system,
                bucket,
                pt.point,
                pt.wall_s,
                conv_vae=bool(_optional_int(row["conv_vae"])),
                reject_clamped=True,
            )
            if factor is not None:
                factors.append(factor)
        history_factors = [
            factor
            for row in history_rows
            if str(row["id"]) not in seen_ids
            and (factor := _history_row_factor(identity.system, row)) is not None
        ]
        if len(history_factors) >= _MIN_HISTORY_SAMPLES:
            factors.extend(history_factors)
        return median_machine_factor(factors)
