"""Aggregate the generations dashboard from ``generation_stats`` and assets.

Percentiles use linear interpolation between the nearest ranks:
``rank = (n - 1) * p``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

from services.generation_stats_sql import generation_stats_sql

DashboardWindow = Literal["7d", "30d", "all"]
_ACTIVITY_FEATURES = (
    "text-to-video",
    "image-to-video",
    "audio-to-video",
    "lora",
    "retake",
    "extend",
    "ic-lora",
)
_RENDER_GROUPS = frozenset(
    {"text-to-video", "image-to-video", "audio-to-video", "lora"}
)
_EDIT_FEATURES = frozenset({"retake", "extend"})
# Days before today included in a bounded window. 7d is today plus the six days before it.
_WINDOW_LOOKBACK_DAYS = {"7d": 6, "30d": 29}
# All zero-fills at most this many days. Older days appear only when a run finished then.
_MAX_ZERO_FILL_DAYS = 800
_logger = logging.getLogger(__name__)


class InvalidTimezone(ValueError):
    pass


class DashboardKpis(BaseModel):
    model_config = ConfigDict(strict=True)
    content_count: int = Field(ge=0)
    footage_s: float = Field(ge=0)
    gpu_ms: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    success_rate: float | None = None


class ActivityDay(BaseModel):
    model_config = ConfigDict(strict=True)
    day: str
    counts: dict[str, int]


class LoraUsage(BaseModel):
    model_config = ConfigDict(strict=True)
    feature: str
    name: str
    runs: int = Field(ge=0)
    keep_rate: float | None = None


class LoraPanel(BaseModel):
    model_config = ConfigDict(strict=True)
    share: float | None = None
    top: list[LoraUsage]


class RenderCell(BaseModel):
    model_config = ConfigDict(strict=True)
    model: str
    resolution: str
    aspect_ratio: str
    duration_s: float
    median_ms: float
    p90_ms: float
    count: int = Field(ge=1)


class FpsRenderCell(RenderCell):
    """One render cell limited to a single frame rate. The pooled `render` list has no fps."""

    fps: int


class FailureRow(BaseModel):
    model_config = ConfigDict(strict=True)
    error_code: str
    count: int = Field(ge=1)


class KeepRate(BaseModel):
    model_config = ConfigDict(strict=True)
    kept: int = Field(ge=0)
    total: int = Field(ge=0)
    rate: float | None = None


class UsualSettings(BaseModel):
    model_config = ConfigDict(strict=True)
    resolution: str
    aspect_ratio: str
    duration_s: float
    fps: int


class DashboardSnapshot(BaseModel):
    model_config = ConfigDict(strict=True)
    empty: bool
    has_history: bool
    range: DashboardWindow
    tz: str
    kpis: DashboardKpis
    activity: list[ActivityDay]
    loras: LoraPanel
    render: list[RenderCell]
    render_by_fps: list[FpsRenderCell]
    aspect_ratios: list[str]
    fps_values: list[int]
    failures: list[FailureRow]
    keep_rate: KeepRate
    usual: UsualSettings | None = None


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise InvalidTimezone(name) from exc


def _lookback_days(window: Literal["7d", "30d"]) -> int:
    return _WINDOW_LOOKBACK_DAYS[window]


def _window_start_ms(
    window: DashboardWindow, zone: ZoneInfo, now: datetime
) -> int | None:
    if window == "all":
        return None
    start_day = now.astimezone(zone).date() - timedelta(days=_lookback_days(window))
    start = datetime(
        start_day.year, start_day.month, start_day.day, tzinfo=zone
    )
    return int(start.timestamp() * 1000)


def _local_day(finished_at: int, zone: ZoneInfo) -> str:
    return datetime.fromtimestamp(finished_at / 1000, zone).date().isoformat()


def _activity(
    rows: list[sqlite3.Row],
    *,
    window: DashboardWindow,
    zone: ZoneInfo,
    today: date,
) -> list[ActivityDay]:
    """Zero-fill recent days. A huge gap keeps only the days that have runs."""
    if window == "all":
        dates = [
            datetime.fromtimestamp(int(row["finished_at"]) / 1000, zone).date()
            for row in rows
        ]
        first = min(dates) if dates else today
    else:
        first = today - timedelta(days=_lookback_days(window))
    if window == "all" and (today - first).days > _MAX_ZERO_FILL_DAYS:
        fill_from = today - timedelta(days=_MAX_ZERO_FILL_DAYS)
    else:
        fill_from = first
    counts_by_day: dict[str, dict[str, int]] = {
        day.isoformat(): {feature: 0 for feature in _ACTIVITY_FEATURES}
        for day in (
            datetime.fromtimestamp(int(row["finished_at"]) / 1000, zone).date()
            for row in rows
        )
        if day < fill_from
    }
    counts_by_day = dict(sorted(counts_by_day.items()))
    day = fill_from
    while day <= today:
        counts_by_day[day.isoformat()] = {feature: 0 for feature in _ACTIVITY_FEATURES}
        day += timedelta(days=1)
    for row in rows:
        bucket = counts_by_day.setdefault(
            _local_day(int(row["finished_at"]), zone),
            {feature: 0 for feature in _ACTIVITY_FEATURES},
        )
        group = str(row["feature_group"])
        bucket[group] = bucket.get(group, 0) + 1
    return [
        ActivityDay(day=day_key, counts=counts)
        for day_key, counts in sorted(counts_by_day.items())
    ]


def _clip_kept(asset: sqlite3.Row) -> bool:
    """A clip is kept until the asset or its generation is soft-deleted."""
    return asset["asset_deleted_at"] is None and not int(asset["generation_deleted"])


def _footage_s(feature: str, edited_span_s: float | None, metadata: str) -> float:
    if feature in _EDIT_FEATURES:
        return float(edited_span_s or 0)
    try:
        parsed: object = json.loads(metadata)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    inner = cast(dict[str, object], parsed).get("metadata")
    if not isinstance(inner, dict):
        return 0.0
    duration_ms = cast(dict[str, object], inner).get("durationMs")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
        return 0.0
    return float(duration_ms) / 1000


def _render_ready(row: sqlite3.Row) -> bool:
    return (
        row["status"] == "succeeded"
        and row["feature_group"] in _RENDER_GROUPS
        and row["run_ms"] is not None
        and bool(row["resolution_label"])
        and row["render_duration_s"] is not None
        and row["fps"] is not None
    )


def _group_render(rows: list[sqlite3.Row]) -> list[RenderCell]:
    grouped: dict[tuple[str, str, str, float], list[float]] = {}
    for row in rows:
        key = (
            str(row["model"] or ""),
            str(row["resolution_label"]),
            str(row["aspect_ratio"] or ""),
            round(float(row["render_duration_s"]), 4),
        )
        grouped.setdefault(key, []).append(float(row["run_ms"]))
    return [
        RenderCell(
            model=model,
            resolution=resolution,
            aspect_ratio=aspect,
            duration_s=duration_s,
            median_ms=percentile(samples, 0.5),
            p90_ms=percentile(samples, 0.9),
            count=len(samples),
        )
        for (model, resolution, aspect, duration_s), samples in sorted(grouped.items())
    ]


def _usual_ready(row: sqlite3.Row) -> bool:
    if row["status"] != "succeeded" or row["feature"] in _EDIT_FEATURES:
        return False
    return bool(
        row["resolution_label"]
        and row["aspect_ratio"]
        and row["fps"] is not None
        and row["render_duration_s"] is not None
    )


def _log_exclusions(rows: list[sqlite3.Row]) -> None:
    """Count rows dropped from one metric and still counted in the others."""
    render_excluded = sum(
        1
        for row in rows
        if row["status"] == "succeeded"
        and row["feature_group"] in _RENDER_GROUPS
        and not _render_ready(row)
    )
    usual_excluded = sum(
        1
        for row in rows
        if row["status"] == "succeeded"
        and row["feature"] not in _EDIT_FEATURES
        and row["resolution_label"]
        and not _usual_ready(row)
    )
    unrecognized = sum(
        1 for row in rows if str(row["feature_group"]) not in _ACTIVITY_FEATURES
    )
    for metric, count in (
        ("render", render_excluded),
        ("usual_settings", usual_excluded),
        ("feature_group", unrecognized),
    ):
        if count:
            _logger.info("dashboard excluded %s rows from %s", count, metric)


def compute_dashboard(
    conn: sqlite3.Connection,
    *,
    window: DashboardWindow,
    tz: str,
    now: datetime | None = None,
) -> DashboardSnapshot:
    zone = _zone(tz)
    moment = now if now is not None else datetime.now(zone)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=zone)
    start_ms = _window_start_ms(window, zone, moment)
    conn.execute("BEGIN")
    try:
        rows, assets, has_history = _read(conn, start_ms)
    finally:
        conn.rollback()
    return _snapshot(
        rows,
        assets,
        has_history=has_history,
        window=window,
        tz=tz,
        zone=zone,
        moment=moment,
    )


def _read(
    conn: sqlite3.Connection, start_ms: int | None
) -> tuple[list[sqlite3.Row], list[sqlite3.Row], bool]:
    """Both reads share one transaction, so a run that finishes between them cannot skew a request."""
    rows = conn.execute(
        generation_stats_sql(
            """
            SELECT *
            FROM generation_stats
            WHERE finished_at IS NOT NULL
              AND (? IS NULL OR finished_at >= ?)
            """
        ),
        (start_ms, start_ms),
    ).fetchall()
    assets = conn.execute(
        generation_stats_sql(
            """
            SELECT
                a.deleted_at AS asset_deleted_at,
                s.is_deleted AS generation_deleted,
                a.metadata AS metadata,
                s.feature AS feature,
                s.feature_group AS feature_group,
                s.edited_span_s AS edited_span_s,
                s.id AS generation_id
            FROM assets a
            JOIN generation_stats s ON s.id = a.producer_generation_id
            WHERE a.origin = 'generated'
              AND a.media_kind = 'video'
              AND s.status = 'succeeded'
              AND s.finished_at IS NOT NULL
              AND (? IS NULL OR s.finished_at >= ?)
            """
        ),
        (start_ms, start_ms),
    ).fetchall()
    has_history = bool(rows) or (
        start_ms is not None
        and conn.execute(
            generation_stats_sql(
                "SELECT 1 FROM generation_stats WHERE finished_at IS NOT NULL LIMIT 1"
            )
        ).fetchone()
        is not None
    )
    return rows, assets, has_history


def _snapshot(
    rows: list[sqlite3.Row],
    assets: list[sqlite3.Row],
    *,
    has_history: bool,
    window: DashboardWindow,
    tz: str,
    zone: ZoneInfo,
    moment: datetime,
) -> DashboardSnapshot:
    succeeded = sum(1 for row in rows if row["status"] == "succeeded")
    failed = sum(1 for row in rows if row["status"] == "failed")
    cancelled = sum(1 for row in rows if row["status"] == "cancelled")
    decided = succeeded + failed
    gpu_ms = sum(
        int(row["run_ms"])
        for row in rows
        if row["status"] in {"succeeded", "failed"} and row["run_ms"] is not None
    )
    footage_s = 0.0
    kept_assets = 0
    for asset in assets:
        footage_s += _footage_s(
            str(asset["feature"]), asset["edited_span_s"], str(asset["metadata"])
        )
        if _clip_kept(asset):
            kept_assets += 1
    today = moment.astimezone(zone).date()
    activity = _activity(rows, window=window, zone=zone, today=today)

    lora_runs: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        if row["feature_group"] != "lora":
            continue
        lora_runs.setdefault(str(row["feature"]), []).append(row)
    lora_assets: dict[str, list[sqlite3.Row]] = {}
    for asset in assets:
        if asset["feature_group"] != "lora":
            continue
        lora_assets.setdefault(str(asset["feature"]), []).append(asset)
    ranked = sorted(
        lora_runs.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    usages: list[LoraUsage] = []
    for feature, runs in ranked:
        named = sorted(
            (run for run in runs if run["lora_name"]),
            key=lambda run: int(run["finished_at"]),
        )
        feature_assets = lora_assets.get(feature, [])
        kept = sum(1 for asset in feature_assets if _clip_kept(asset))
        usages.append(
            LoraUsage(
                feature=feature,
                name=str(named[-1]["lora_name"]) if named else feature,
                runs=len(runs),
                keep_rate=(kept / len(feature_assets)) if feature_assets else None,
            )
        )
    lora_total = sum(len(runs) for runs in lora_runs.values())

    _log_exclusions(rows)
    render_pool = [row for row in rows if _render_ready(row)]
    aspect_ratios = sorted(
        {str(row["aspect_ratio"]) for row in render_pool if row["aspect_ratio"]}
    )
    fps_values = sorted(
        {int(row["fps"]) for row in render_pool if row["fps"] is not None}
    )
    cells = _group_render(render_pool)
    by_fps = [
        FpsRenderCell(**cell.model_dump(), fps=value)
        for value in fps_values
        for cell in _group_render(
            [row for row in render_pool if int(row["fps"]) == value]
        )
    ]

    failures: dict[str, int] = {}
    for row in rows:
        if row["status"] != "failed":
            continue
        code = str(row["error_code"] or "UNKNOWN")
        failures[code] = failures.get(code, 0) + 1

    usual_counts: dict[tuple[str, str, float, int], tuple[int, int]] = {}
    for row in rows:
        if not _usual_ready(row):
            continue
        key = (
            str(row["resolution_label"]),
            str(row["aspect_ratio"]),
            round(float(row["render_duration_s"]), 4),
            int(row["fps"]),
        )
        count, latest = usual_counts.get(key, (0, 0))
        usual_counts[key] = (count + 1, max(latest, int(row["finished_at"])))
    usual = None
    if usual_counts:
        best = max(usual_counts.items(), key=lambda item: (item[1][0], item[1][1]))
        resolution, ratio, duration_s, setting_fps = best[0]
        usual = UsualSettings(
            resolution=resolution,
            aspect_ratio=ratio,
            duration_s=duration_s,
            fps=setting_fps,
        )

    total_assets = len(assets)
    return DashboardSnapshot(
        empty=len(rows) == 0,
        has_history=has_history,
        range=window,
        tz=tz,
        kpis=DashboardKpis(
            content_count=total_assets,
            footage_s=footage_s,
            gpu_ms=gpu_ms,
            succeeded=succeeded,
            failed=failed,
            cancelled=cancelled,
            success_rate=(succeeded / decided) if decided else None,
        ),
        activity=activity,
        loras=LoraPanel(
            share=(lora_total / len(rows)) if rows else None,
            top=usages,
        ),
        render=cells,
        render_by_fps=by_fps,
        aspect_ratios=aspect_ratios,
        fps_values=fps_values,
        failures=[
            FailureRow(error_code=code, count=count)
            for code, count in sorted(failures.items(), key=lambda item: (-item[1], item[0]))
        ],
        keep_rate=KeepRate(
            kept=kept_assets,
            total=total_assets,
            rate=(kept_assets / total_assets) if total_assets else None,
        ),
        usual=usual,
    )
