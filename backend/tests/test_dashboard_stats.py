"""Dashboard aggregates: timezones, percentiles, soft-deletes, and the HTTP route."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from services.dashboard_stats import percentile
from services.sqlite_store import SqliteStore

NOW = datetime(2026, 3, 10, 12, 0, tzinfo=ZoneInfo("UTC"))
DAY = datetime(2026, 3, 10, 10, 0, tzinfo=ZoneInfo("UTC"))
OLD = datetime(2026, 3, 1, 10, 0, tzinfo=ZoneInfo("UTC"))


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _video_meta(duration_ms: int) -> str:
    return json.dumps(
        {
            "mediaType": "video",
            "metadata": {
                "width": 1920,
                "height": 1080,
                "durationMs": duration_ms,
                "sizeBytes": 10,
                "audioStreamCount": 0,
            },
        }
    )


def _spec(
    *,
    duration: float | None = 5,
    resolution: str | None = "1080p",
    aspect: str | None = "16:9",
    fps: int | None = 24,
    loras: list[dict[str, object]] | None = None,
) -> str:
    params: dict[str, object] = {"model": "ltx-2.3-fast"}
    if resolution is not None:
        params["resolution"] = resolution
    if aspect is not None:
        params["aspectRatio"] = aspect
    if duration is not None:
        params["duration"] = duration
    if fps is not None:
        params["fps"] = fps
    if loras is not None:
        params["loras"] = loras
    return json.dumps({"params": params, "inputs": {}})


def _insert(
    conn: sqlite3.Connection,
    *,
    generation_id: str,
    feature: str,
    status: str,
    finished: datetime,
    run_ms: int,
    spec: str,
    error_code: str | None = None,
    generation_deleted: bool = False,
    asset_duration_ms: int | None = None,
    asset_deleted: bool = False,
) -> None:
    finished_ms = _ms(finished)
    conn.execute(
        """
        INSERT INTO generations (
            id, feature, contract_version, status, spec, error_code,
            created_at, queued_at, queue_rank, attempt_count,
            started_at, finished_at, deleted_at
        ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, NULL, 1, ?, ?, ?)
        """,
        (
            generation_id,
            feature,
            status,
            spec,
            error_code,
            finished_ms - run_ms,
            finished_ms - run_ms,
            finished_ms - run_ms,
            finished_ms,
            finished_ms if generation_deleted else None,
        ),
    )
    if asset_duration_ms is None:
        return
    conn.execute(
        """
        INSERT INTO assets (
            id, media_kind, origin, producer_generation_id, output_ordinal,
            mime_type, name, metadata, created_at, deleted_at
        ) VALUES (?, 'video', 'generated', ?, 0, 'video/mp4', ?, ?, ?, ?)
        """,
        (
            f"asset-{generation_id}",
            generation_id,
            generation_id,
            _video_meta(asset_duration_ms),
            finished_ms,
            finished_ms if asset_deleted else None,
        ),
    )


def _seed(conn: sqlite3.Connection) -> None:
    standard = _spec()
    for index, run_ms in enumerate((60_000, 120_000, 180_000)):
        _insert(
            conn,
            generation_id=f"t2v-{index}",
            feature="text-to-video",
            status="succeeded",
            finished=DAY,
            run_ms=run_ms,
            spec=standard,
            asset_duration_ms=5_000,
            asset_deleted=index == 0,
        )
    _insert(
        conn,
        generation_id="t2v-failed",
        feature="text-to-video",
        status="failed",
        finished=DAY,
        run_ms=30_000,
        spec=standard,
        error_code="EXECUTOR_FAILED",
    )
    _insert(
        conn,
        generation_id="t2v-cancelled",
        feature="text-to-video",
        status="cancelled",
        finished=DAY,
        run_ms=10_000,
        spec=_spec(resolution=None, aspect=None, duration=None, fps=None),
    )
    _insert(
        conn,
        generation_id="retake-1",
        feature="retake",
        status="succeeded",
        finished=DAY,
        run_ms=5_000,
        spec=json.dumps(
            {
                "params": {"model": "ltx-2.3-fast", "duration": 2.5},
                "inputs": {"video": {"assetId": "source"}},
            }
        ),
        asset_duration_ms=20_000,
    )
    lora = _spec(
        duration=6,
        loras=[
            {
                "ref": "",
                "scale": 1,
                "catalogId": "cozy-felt-style",
                "displayName": "Cozy Felt Style",
            }
        ],
    )
    _insert(
        conn,
        generation_id="lora-kept",
        feature="cozy-felt",
        status="succeeded",
        finished=DAY,
        run_ms=90_000,
        spec=lora,
        asset_duration_ms=6_000,
    )
    _insert(
        conn,
        generation_id="lora-dropped",
        feature="cozy-felt",
        status="succeeded",
        finished=DAY,
        run_ms=40_000,
        spec=lora,
        asset_duration_ms=6_000,
        asset_deleted=True,
    )
    _insert(
        conn,
        generation_id="t2v-deleted-gen",
        feature="text-to-video",
        status="succeeded",
        finished=DAY,
        run_ms=7_000,
        spec=standard,
        generation_deleted=True,
        asset_duration_ms=5_000,
    )
    _insert(
        conn,
        generation_id="t2v-old",
        feature="text-to-video",
        status="succeeded",
        finished=OLD,
        run_ms=1_000,
        spec=standard,
        asset_duration_ms=5_000,
    )
    conn.commit()


def _open(tmp_path: Path) -> tuple[SqliteStore, sqlite3.Connection]:
    store = SqliteStore(tmp_path)
    conn = sqlite3.connect(tmp_path / "store.sqlite3")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return store, conn


def test_lora_panel_includes_the_rows_past_the_preview(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    for index in range(11):
        _insert(
            conn,
            generation_id=f"lora-{index}",
            feature=f"recipe-{index}",
            status="succeeded",
            finished=DAY,
            run_ms=1_000,
            spec=_spec(
                loras=[
                    {
                        "ref": "",
                        "scale": 1,
                        "catalogId": f"catalog-{index}",
                        "displayName": f"Style {index}",
                    }
                ]
            ),
            asset_duration_ms=1_000,
        )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert len(snapshot.loras.top) == 11
    assert {lora.name for lora in snapshot.loras.top} == {f"Style {index}" for index in range(11)}


def test_percentile_interpolates_between_ranks() -> None:
    assert percentile([10, 20, 30, 40, 100], 0.5) == 30
    assert percentile([10, 20, 30, 40, 100], 0.9) == 76
    assert percentile([7000, 60000, 120000, 180000], 0.5) == 90_000
    assert percentile([7000, 60000, 120000, 180000], 0.9) == 162_000


def test_seven_day_snapshot_matches_hand_computed_metrics(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    _seed(conn)
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert snapshot.empty is False
    assert snapshot.kpis.gpu_ms == 532_000
    assert snapshot.kpis.succeeded == 7
    assert snapshot.kpis.failed == 1
    assert snapshot.kpis.cancelled == 1
    assert snapshot.kpis.success_rate == 7 / 8
    assert snapshot.kpis.content_count == 7
    assert snapshot.kpis.footage_s == 34.5
    assert snapshot.keep_rate.kept == 4
    assert snapshot.keep_rate.total == 7
    assert snapshot.keep_rate.rate == 4 / 7
    assert snapshot.loras.share == 2 / 9
    assert len(snapshot.loras.top) == 1
    assert snapshot.loras.top[0].feature == "cozy-felt"
    assert snapshot.loras.top[0].name == "Cozy Felt Style"
    assert snapshot.loras.top[0].runs == 2
    assert snapshot.loras.top[0].keep_rate == 0.5
    assert [(row.error_code, row.count) for row in snapshot.failures] == [
        ("EXECUTOR_FAILED", 1)
    ]
    assert snapshot.usual is not None
    assert snapshot.usual.resolution == "1080p"
    assert snapshot.usual.aspect_ratio == "16:9"
    assert snapshot.usual.duration_s == 5
    assert snapshot.usual.fps == 24
    five_seconds = next(cell for cell in snapshot.render if cell.duration_s == 5)
    assert five_seconds.model == "ltx-2.3-fast"
    assert five_seconds.resolution == "1080p"
    assert five_seconds.aspect_ratio == "16:9"
    assert five_seconds.count == 4
    assert five_seconds.median_ms == 90_000
    assert five_seconds.p90_ms == 162_000
    six_seconds = next(cell for cell in snapshot.render if cell.duration_s == 6)
    assert six_seconds.aspect_ratio == "16:9"
    assert six_seconds.count == 2
    assert [day.day for day in snapshot.activity] == [
        "2026-03-04",
        "2026-03-05",
        "2026-03-06",
        "2026-03-07",
        "2026-03-08",
        "2026-03-09",
        "2026-03-10",
    ]
    today = snapshot.activity[-1].counts
    assert today["text-to-video"] == 6
    assert today["lora"] == 2
    assert today["retake"] == 1
    assert today["image-to-video"] == 0

    wider = store.dashboard(window="all", tz="UTC", now=NOW)
    assert wider.kpis.gpu_ms == 533_000
    assert wider.activity[0].day == "2026-03-01"
    assert wider.activity[0].counts["text-to-video"] == 1
    wider_five_seconds = next(cell for cell in wider.render if cell.duration_s == 5)
    assert wider_five_seconds.count == 5


def test_render_grid_is_pooled_and_split_by_fps(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    runs = (("slow-24", 24, 60_000), ("fast-24", 24, 120_000), ("only-30", 30, 90_000))
    for generation_id, fps, run_ms in runs:
        _insert(
            conn,
            generation_id=generation_id,
            feature="text-to-video",
            status="succeeded",
            finished=DAY,
            run_ms=run_ms,
            spec=_spec(fps=fps),
            asset_duration_ms=5_000,
        )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    pooled = snapshot.render
    assert [(cell.aspect_ratio, cell.count) for cell in pooled] == [("16:9", 3)]
    assert {(cell.fps, cell.count) for cell in snapshot.render_by_fps} == {(24, 2), (30, 1)}
    assert snapshot.fps_values == [24, 30]


def test_local_day_follows_the_caller_timezone(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    finished = datetime(2026, 3, 10, 23, 30, tzinfo=ZoneInfo("UTC"))
    _insert(
        conn,
        generation_id="late",
        feature="text-to-video",
        status="succeeded",
        finished=finished,
        run_ms=1_000,
        spec=_spec(),
    )
    conn.commit()
    conn.close()
    moment = datetime(2026, 3, 10, 23, 45, tzinfo=ZoneInfo("UTC"))

    utc = store.dashboard(window="7d", tz="UTC", now=moment)
    tokyo = store.dashboard(window="7d", tz="Asia/Tokyo", now=moment)

    assert utc.activity[-1].day == "2026-03-10"
    assert utc.activity[-1].counts["text-to-video"] == 1
    assert tokyo.activity[-1].day == "2026-03-11"
    assert tokyo.activity[-1].counts["text-to-video"] == 1
    assert tokyo.activity[-2].counts["text-to-video"] == 0


def test_all_range_skips_empty_days_across_a_huge_gap(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    ancient = datetime(1970, 1, 2, tzinfo=ZoneInfo("UTC"))
    _insert(
        conn,
        generation_id="ancient",
        feature="text-to-video",
        status="succeeded",
        finished=ancient,
        run_ms=1_000,
        spec=_spec(),
    )
    _insert(
        conn,
        generation_id="recent",
        feature="text-to-video",
        status="succeeded",
        finished=DAY,
        run_ms=1_000,
        spec=_spec(),
    )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="all", tz="UTC", now=NOW)

    assert snapshot.activity[0].day == "1970-01-02"
    assert snapshot.activity[0].counts["text-to-video"] == 1
    assert snapshot.activity[1].day != "1970-01-03"
    assert snapshot.activity[-1].day == "2026-03-10"
    assert len(snapshot.activity) < 1000


def test_one_render_still_shows_and_aspect_ratios_stay_separate(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    _insert(
        conn,
        generation_id="only",
        feature="text-to-video",
        status="succeeded",
        finished=DAY,
        run_ms=10_000,
        spec=_spec(resolution="720p", aspect="16:9"),
    )
    for aspect in ("4:3", "3:2", "auto"):
        _insert(
            conn,
            generation_id=f"ratio-{aspect}",
            feature="cozy-felt",
            status="succeeded",
            finished=DAY,
            run_ms=20_000,
            spec=_spec(
                resolution="540p",
                aspect=aspect,
                loras=[
                    {
                        "ref": "",
                        "scale": 1,
                        "catalogId": "cozy-felt-style",
                        "displayName": "Cozy Felt Style",
                    }
                ],
            ),
        )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert [(cell.resolution, cell.aspect_ratio, cell.count) for cell in snapshot.render] == [
        ("540p", "3:2", 1),
        ("540p", "4:3", 1),
        ("540p", "auto", 1),
        ("720p", "16:9", 1),
    ]
    assert snapshot.aspect_ratios == ["16:9", "3:2", "4:3", "auto"]


def test_usual_settings_tie_uses_the_most_recent_run(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    earlier = datetime(2026, 3, 8, 12, tzinfo=ZoneInfo("UTC"))
    later = datetime(2026, 3, 9, 12, tzinfo=ZoneInfo("UTC"))
    for index, finished in enumerate((earlier, earlier)):
        _insert(
            conn,
            generation_id=f"old-{index}",
            feature="text-to-video",
            status="succeeded",
            finished=finished,
            run_ms=1_000,
            spec=_spec(resolution="1080p"),
        )
    for index, finished in enumerate((later, later)):
        _insert(
            conn,
            generation_id=f"new-{index}",
            feature="text-to-video",
            status="succeeded",
            finished=finished,
            run_ms=1_000,
            spec=_spec(resolution="720p"),
        )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert snapshot.usual is not None
    assert snapshot.usual.resolution == "720p"


def test_dashboard_route_returns_the_snapshot_and_rejects_a_bad_timezone(client) -> None:
    ok = client.get("/api/stats/dashboard", params={"range": "7d", "tz": "UTC"})
    assert ok.status_code == 200
    assert ok.json()["empty"] is True

    bad = client.get("/api/stats/dashboard", params={"range": "7d", "tz": "Not/AZone"})
    assert bad.status_code == 422
    assert bad.json()["code"] == "INVALID_TIMEZONE"
    for tz in ("/etc/passwd", "../UTC", "a" * 5000):
        escaped = client.get("/api/stats/dashboard", params={"range": "7d", "tz": tz})
        assert escaped.status_code == 422
        assert escaped.json()["code"] == "INVALID_TIMEZONE"


def test_malformed_rows_leave_the_metrics_that_cannot_read_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """provider is not in spec, so its absence drops nothing. A string fps
    drops render time and usual settings only. A recipe with no loras array
    is not a LoRA run. Text-to-video that also carries loras stays text-to-video.
    """
    store, conn = _open(tmp_path)
    _insert(
        conn,
        generation_id="valid",
        feature="text-to-video",
        status="succeeded",
        finished=DAY,
        run_ms=1_000,
        spec=_spec(),
    )
    string_fps = json.dumps(
        {
            "params": {
                "model": "ltx-2.3-fast",
                "resolution": "720p",
                "aspectRatio": "16:9",
                "duration": 5,
                "fps": "24",
            },
            "inputs": {},
        }
    )
    for index in range(3):
        _insert(
            conn,
            generation_id=f"string-fps-{index}",
            feature="text-to-video",
            status="succeeded",
            finished=DAY,
            run_ms=1_000,
            spec=string_fps,
        )
    _insert(
        conn,
        generation_id="recipe-without-loras",
        feature="cozy-felt",
        status="succeeded",
        finished=DAY,
        run_ms=1_000,
        spec=_spec(resolution="540p"),
    )
    _insert(
        conn,
        generation_id="t2v-with-lora",
        feature="text-to-video",
        status="succeeded",
        finished=DAY,
        run_ms=1_000,
        spec=_spec(
            loras=[
                {
                    "ref": "",
                    "scale": 1,
                    "catalogId": "cozy-felt-style",
                    "displayName": "Cozy Felt Style",
                }
            ]
        ),
    )
    conn.commit()
    conn.close()

    with caplog.at_level(logging.INFO, logger="services.dashboard_stats"):
        snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert snapshot.kpis.succeeded == 6
    assert snapshot.kpis.gpu_ms == 6_000
    assert [(cell.resolution, cell.count) for cell in snapshot.render] == [("1080p", 2)]
    assert snapshot.usual is not None
    assert snapshot.usual.resolution == "1080p"
    assert snapshot.loras.top == []
    assert snapshot.loras.share == 0
    assert snapshot.activity[-1].counts["text-to-video"] == 5
    assert snapshot.activity[-1].counts["lora"] == 0
    assert "dashboard excluded 3 rows from render" in caplog.text
    assert "dashboard excluded 3 rows from usual_settings" in caplog.text
    assert "dashboard excluded 1 rows from feature_group" in caplog.text


def test_fifty_thousand_rows_aggregate(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    finished_ms = _ms(DAY)
    spec = _spec()
    conn.executemany(
        """
        INSERT INTO generations (
            id, feature, contract_version, status, spec, error_code,
            created_at, queued_at, queue_rank, attempt_count,
            started_at, finished_at, deleted_at
        ) VALUES (?, 'text-to-video', 1, 'succeeded', ?, NULL, ?, ?, NULL, 1, ?, ?, NULL)
        """,
        [
            (f"bulk-{index}", spec, finished_ms, finished_ms, finished_ms - 1000, finished_ms)
            for index in range(50_000)
        ],
    )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert snapshot.kpis.succeeded == 50_000


def test_usual_settings_ignore_failed_and_cancelled_runs(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    for index in range(3):
        _insert(
            conn,
            generation_id=f"oom-{index}",
            feature="text-to-video",
            status="failed",
            finished=DAY,
            run_ms=1_000,
            spec=_spec(resolution="2160p"),
            error_code="OOM",
        )
    _insert(
        conn,
        generation_id="cancelled",
        feature="text-to-video",
        status="cancelled",
        finished=DAY,
        run_ms=1_000,
        spec=_spec(resolution="2160p"),
    )
    _insert(
        conn,
        generation_id="done",
        feature="text-to-video",
        status="succeeded",
        finished=DAY,
        run_ms=1_000,
        spec=_spec(resolution="720p"),
    )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert snapshot.usual is not None
    assert snapshot.usual.resolution == "720p"


def test_a_quiet_range_is_empty_but_keeps_its_history(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    _insert(
        conn,
        generation_id="old",
        feature="text-to-video",
        status="succeeded",
        finished=datetime(2025, 6, 1, 10, 0, tzinfo=ZoneInfo("UTC")),
        run_ms=1_000,
        spec=_spec(),
    )
    conn.commit()
    conn.close()

    quiet = store.dashboard(window="7d", tz="UTC", now=NOW)
    everything = store.dashboard(window="all", tz="UTC", now=NOW)

    assert quiet.empty is True
    assert quiet.has_history is True
    assert everything.empty is False
    assert everything.has_history is True


def test_a_new_install_has_no_history(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert snapshot.empty is True
    assert snapshot.has_history is False


def test_activity_days_stay_in_date_order_when_a_run_is_dated_after_today(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    for index, day in enumerate((5, 3)):
        _insert(
            conn,
            generation_id=f"future-{index}",
            feature="text-to-video",
            status="succeeded",
            finished=datetime(2026, 3, 10 + day, 10, 0, tzinfo=ZoneInfo("UTC")),
            run_ms=1_000,
            spec=_spec(),
        )
    conn.commit()
    conn.close()

    days = [day.day for day in store.dashboard(window="7d", tz="UTC", now=NOW).activity]

    assert days == sorted(days)


def test_audio_to_video_lengths_share_one_column_per_second(tmp_path: Path) -> None:
    store, conn = _open(tmp_path)
    for index, frames in enumerate((121, 129, 137)):
        spec = json.dumps(
            {
                "params": {
                    "model": "ltx-2.3-fast",
                    "resolution": "540p",
                    "aspectRatio": "16:9",
                    "fps": 24,
                    "numFrames": frames,
                },
                "inputs": {},
            }
        )
        _insert(
            conn,
            generation_id=f"a2v-{index}",
            feature="audio-to-video",
            status="succeeded",
            finished=DAY,
            run_ms=1_000,
            spec=spec,
        )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert sorted({cell.duration_s for cell in snapshot.render}) == [5.0, 6.0]


def test_every_queued_feature_lands_in_a_chart_series(tmp_path: Path) -> None:
    from services.features.ic_lora_recipes import IC_LORA_RECIPES
    from services.features.queued_features import QUEUED_GENERATION_FEATURES

    store, conn = _open(tmp_path)
    plain = {
        "text-to-video",
        "image-to-video",
        "audio-to-video",
        "retake",
        "extend",
    }
    for index, feature in enumerate(sorted(QUEUED_GENERATION_FEATURES)):
        loras = None if feature in plain else [{"catalogId": feature, "displayName": feature}]
        _insert(
            conn,
            generation_id=f"feature-{index}",
            feature=feature,
            status="succeeded",
            finished=DAY,
            run_ms=1_000,
            spec=_spec(loras=loras),
        )
    conn.commit()
    conn.close()

    snapshot = store.dashboard(window="7d", tz="UTC", now=NOW)

    assert sum(snapshot.activity[-1].counts.values()) == len(QUEUED_GENERATION_FEATURES)
    # A stored IC-LoRA recipe spec carries loras. It still counts as ic-lora.
    assert snapshot.activity[-1].counts["ic-lora"] == len(IC_LORA_RECIPES)
    assert set(snapshot.activity[-1].counts) == {
        "text-to-video",
        "image-to-video",
        "audio-to-video",
        "lora",
        "retake",
        "extend",
        "ic-lora",
    }


def test_old_lora_runs_take_their_name_from_the_catalog(tmp_path: Path, fake_services) -> None:
    from handlers.dashboard_handler import DashboardHandler

    store, conn = _open(tmp_path)
    _insert(
        conn,
        generation_id="before-names",
        feature="cozy-felt",
        status="succeeded",
        finished=DAY,
        run_ms=1_000,
        spec=_spec(loras=[{"catalogId": "cozy-felt-style"}]),
    )
    conn.commit()
    conn.close()

    handler = DashboardHandler(store, fake_services.lora_catalog_provider)
    snapshot = handler.get(window="7d", tz="UTC", now=NOW)

    assert [usage.name for usage in snapshot.loras.top] == ["Cozy Felt Style"]
