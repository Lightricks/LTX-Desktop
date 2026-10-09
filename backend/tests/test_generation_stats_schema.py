"""v1 schema stays as shipped. The stats shape is a query CTE."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from services.generation_stats_sql import generation_stats_sql
from services.migrations import MIGRATIONS_DIR, apply_migrations


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _insert_v1(
    conn: sqlite3.Connection,
    *,
    generation_id: str,
    feature: str,
    spec: dict[str, object],
    status: str = "succeeded",
    started_at: int | None = 1_000,
    finished_at: int | None = 4_000,
    deleted_at: int | None = None,
    queue_rank: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO generations (
            id, feature, contract_version, status, spec, error_code,
            created_at, queued_at, queue_rank, attempt_count,
            started_at, finished_at, deleted_at
        ) VALUES (?, ?, 1, ?, ?, NULL, 1, 1, ?, 1, ?, ?, ?)
        """,
        (
            generation_id,
            feature,
            status,
            json.dumps(spec),
            queue_rank,
            started_at,
            finished_at,
            deleted_at,
        ),
    )


def test_v1_database_upgrades_without_dropping_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite3"
    conn = _connect(db_path)
    conn.executescript((MIGRATIONS_DIR / "0001_initial_schema.sql").read_text(encoding="utf-8"))
    _insert_v1(
        conn,
        generation_id="t2v",
        feature="text-to-video",
        spec={
            "params": {
                "model": "ltx-2.3-fast",
                "resolution": "1080p",
                "aspectRatio": "16:9",
                "fps": 24,
                "duration": 5,
                "loras": [{"ref": "", "scale": 1, "catalogId": "cozy-felt-style"}],
            }
        },
    )
    _insert_v1(
        conn,
        generation_id="retake",
        feature="retake",
        spec={
            "params": {
                "model": "ltx-2.3-fast",
                "resolution": {"width": 1280, "height": 720},
                "duration": 3.5,
            }
        },
        started_at=2_000,
        finished_at=8_000,
    )
    _insert_v1(
        conn,
        generation_id="a2v",
        feature="audio-to-video",
        spec={
            "params": {
                "model": "ltx-2.5-fast",
                "resolution": "540p",
                "aspectRatio": "auto",
                "fps": 24,
                "numFrames": 185,
            }
        },
    )
    _insert_v1(
        conn,
        generation_id="recipe",
        feature="cozy-felt",
        spec={
            "params": {
                "model": "ltx-2.3-fast",
                "resolution": "720p",
                "aspectRatio": "16:9",
                "fps": 24,
                "duration": 6,
                "loras": [{"ref": "", "scale": 1, "catalogId": "cozy-felt-style"}],
            }
        },
        deleted_at=9_000,
    )
    _insert_v1(
        conn,
        generation_id="queued",
        feature="text-to-video",
        spec={"params": {"prompt": "still waiting"}},
        status="queued",
        queue_rank=1,
        started_at=None,
        finished_at=None,
    )
    conn.execute(
        """
        INSERT INTO assets (
            id, media_kind, origin, producer_generation_id, output_ordinal,
            mime_type, name, metadata, created_at
        ) VALUES (
            'asset-1', 'video', 'generated', 't2v', 0,
            'video/mp4', 'out.mp4', '{}', 4
        )
        """
    )
    conn.commit()

    apply_migrations(conn, MIGRATIONS_DIR)

    version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert version == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchone() is None
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(generations)").fetchall()
    }
    assert "device" not in columns
    assert "app_version" not in columns
    assert "model" not in columns
    assert (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'view' AND name = 'generation_stats'"
        ).fetchone()
        is None
    )
    assert (
        conn.execute(
            "SELECT producer_generation_id FROM assets WHERE id = 'asset-1'"
        ).fetchone()[0]
        == "t2v"
    )

    rows = {
        row["id"]: row
        for row in conn.execute(
            generation_stats_sql("SELECT * FROM generation_stats")
        ).fetchall()
    }
    assert set(rows) == {"t2v", "retake", "a2v", "recipe"}

    t2v = rows["t2v"]
    assert t2v["model"] == "ltx-2.3-fast"
    assert t2v["resolution_label"] == "1080p"
    assert t2v["aspect_ratio"] == "16:9"
    assert t2v["fps"] == 24
    assert t2v["feature_group"] == "text-to-video"
    assert t2v["lora_catalog_id"] == "cozy-felt-style"
    assert t2v["lora_name"] is None
    assert t2v["render_duration_s"] == 5
    assert t2v["edited_span_s"] is None
    assert t2v["run_ms"] == 3_000
    assert t2v["is_deleted"] == 0

    retake = rows["retake"]
    assert retake["resolution_label"] is None
    assert retake["aspect_ratio"] is None
    assert retake["fps"] is None
    assert retake["render_duration_s"] is None
    assert retake["edited_span_s"] == 3.5
    assert retake["feature_group"] == "retake"
    assert retake["run_ms"] == 6_000

    a2v = rows["a2v"]
    assert a2v["resolution_label"] == "540p"
    assert a2v["aspect_ratio"] == "auto"
    assert a2v["render_duration_s"] == 8
    assert a2v["feature_group"] == "audio-to-video"

    recipe = rows["recipe"]
    assert recipe["feature_group"] == "lora"
    assert recipe["lora_catalog_id"] == "cozy-felt-style"
    assert recipe["is_deleted"] == 1
    assert recipe["render_duration_s"] == 6

    conn.close()


def test_recorded_version_without_a_file_does_not_error(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite3"
    conn = _connect(db_path)
    apply_migrations(conn, MIGRATIONS_DIR)
    conn.execute(
        "INSERT INTO schema_migrations (version, applied_at) VALUES (99, 1)"
    )
    conn.commit()
    apply_migrations(conn, MIGRATIONS_DIR)
    version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert version == 99
    conn.close()
