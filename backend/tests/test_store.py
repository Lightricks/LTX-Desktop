from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue, ValidationError

from api_types import (
    Asset,
    AudioAssetMetadata,
    AudioMeta,
    ImageAssetMetadata,
    ImageMeta,
)
from services.migrations import MIGRATIONS_DIR, apply_migrations
from services.media_probe import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, MAX_VIDEO_BYTES, validate_ingest_source
from services.sqlite_store import (
    SqliteStore,
    _unlink_path,
    collect_asset_ids,
)
from services.records import (
    AssetInUseError,
    AssetRecord,
    AttemptError,
    GenerationRecord,
    MarkSucceededError,
    MediaError,
    OutputSpec,
    StatusError,
    UnavailableError,
)
from services.unavailable_store import UnavailableStore


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def test_fresh_db_applies_initial_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite3"
    conn = _connect(db_path)
    apply_migrations(conn, MIGRATIONS_DIR)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert tables >= {
        "schema_migrations",
        "generations",
        "assets",
        "paired_devices",
    }
    version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert version == 1
    asset_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(assets)").fetchall()
    }
    assert "thumbnail_path" in asset_columns
    asset_indexes = {
        row["name"] for row in conn.execute("PRAGMA index_list(assets)").fetchall()
    }
    assert "idx_assets_created_at" in asset_indexes
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(generations)").fetchall()
    }
    assert {
        "queued_at",
        "attempt_count",
        "error_code",
        "spec",
        "contract_version",
        "deleted_at",
        "queue_rank",
    } <= columns
    assert "error" not in columns
    assert "params" not in columns
    assert (
        conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'generation_assets'"
        ).fetchone()
        is None
    )
    asset_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'assets'"
    ).fetchone()[0]
    assert "producer_generation_id" in asset_sql
    assert "output_ordinal" in asset_sql
    index_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'idx_assets_producer_ordinal'"
    ).fetchone()[0]
    assert "deleted_at IS NULL" in index_sql
    indexes = {
        row["name"] for row in conn.execute("PRAGMA index_list(generations)").fetchall()
    }
    assert "idx_generations_queue" in indexes
    assert "idx_generations_pending_queue_rank" in indexes
    assert "idx_generations_status" not in indexes
    conn.close()


def test_second_open_is_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite3"
    conn = _connect(db_path)
    apply_migrations(conn, MIGRATIONS_DIR)
    apply_migrations(conn, MIGRATIONS_DIR)
    count = conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    assert count == 1
    conn.close()


def test_rerun_after_ddl_without_ledger_row_does_not_brick(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite3"
    conn = _connect(db_path)
    conn.executescript(
        (MIGRATIONS_DIR / "0001_initial_schema.sql").read_text(encoding="utf-8")
    )
    conn.execute("DELETE FROM schema_migrations")
    conn.commit()
    apply_migrations(conn, MIGRATIONS_DIR)
    version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    assert version == 1
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert tables >= {"schema_migrations", "generations", "assets", "paired_devices"}
    conn.close()


def test_failed_migration_rolls_back_ddl_and_skips_ledger_row(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_ok.sql").write_text("CREATE TABLE t1 (id INTEGER PRIMARY KEY);")
    (migrations / "0002_bad.sql").write_text(
        "CREATE TABLE t2 (id INTEGER PRIMARY KEY);\nCREATE TABLE t2 (id INTEGER PRIMARY KEY);"
    )
    conn = _connect(tmp_path / "store.sqlite3")
    try:
        apply_migrations(conn, migrations)
    except sqlite3.OperationalError:
        pass
    else:
        raise AssertionError("expected OperationalError from duplicate CREATE TABLE")
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "t1" in tables
    assert "t2" not in tables
    versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations")]
    assert versions == [1]
    conn.close()


def _png(path: Path, size: tuple[int, int] = (32, 24)) -> Path:
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path


def _mp4(
    path: Path, *, frames: int = 8, width: int = 16, height: int = 16, fps: int = 8
) -> Path:
    import numpy as np
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return path


def _wav(
    path: Path, *, duration_seconds: float = 0.25, sample_rate: int = 8000
) -> Path:
    import wave

    frame_count = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return path


def _claim(store: SqliteStore, generation_id: str):
    claimed = store.claim_next_queued()
    assert claimed is not None
    assert claimed.id == generation_id
    return claimed


def _spec(
    prompt: str | None = None,
    inputs: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    spec: dict[str, JsonValue] = {"params": {}, "inputs": {}}
    if prompt is not None:
        spec["params"] = {"prompt": prompt}
    if inputs is not None:
        spec["inputs"] = inputs
    return spec


def _output(
    asset_id: str,
    dest: str,
    *,
    ordinal: int = 0,
    mime_type: str = "image/png",
    name: str = "out.png",
) -> OutputSpec:
    return OutputSpec(
        asset_id=asset_id,
        dest_path=dest,
        ordinal=ordinal,
        mime_type=mime_type,
        name=name,
    )


def _succeed_png(
    store: SqliteStore,
    *,
    name: str = "out.png",
    prompt: str | None = None,
    size: tuple[int, int] = (8, 8),
) -> AssetRecord:
    gen = store.insert_generation("text-to-video", _spec(prompt=prompt))
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), size)
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest, name=name)],
        attempt_count=claimed.attempt_count,
    )
    return done.outputs[0]


def _succeed_wav(store: SqliteStore, *, name: str = "clip.wav") -> AssetRecord:
    gen = store.insert_generation("audio-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("audio", "audio/wav")
    _wav(Path(dest))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest, mime_type="audio/wav", name=name)],
        attempt_count=claimed.attempt_count,
    )
    return done.outputs[0]


class _FailOnceHydrationDb(SqliteStore):
    def __init__(self, app_data_dir: Path) -> None:
        super().__init__(app_data_dir)
        self._fail_next_hydration = True

    def _load_required(
        self, conn: sqlite3.Connection, generation_id: str
    ) -> GenerationRecord:
        if self._fail_next_hydration:
            self._fail_next_hydration = False
            raise RuntimeError("forced hydration failure")
        return super()._load_required(conn, generation_id)


def test_collect_asset_ids_walks_nested_objects_and_lists() -> None:
    assert collect_asset_ids(
        {
            "params": {"prompt": "x"},
            "inputs": {
                "startFrame": {"assetId": "start"},
                "keyframes": [
                    {"assetId": "k0", "timeMs": 0},
                    {"assetId": "k1", "timeMs": 1000},
                ],
            },
        }
    ) == ["start", "k0", "k1"]


def test_collect_asset_ids_dedupes_and_ignores_non_ids() -> None:
    assert collect_asset_ids(
        {
            "inputs": {
                "startFrame": {"assetId": "same"},
                "endFrame": {"assetId": "same"},
                "empty": {"assetId": ""},
                "number": {"assetId": 12},
                "nested": {"assetId": {"assetId": "inner"}},
                "snake": {"asset_id": "snake-ignored"},
                "pascal": {"AssetId": "pascal-ignored"},
            }
        }
    ) == ["same", "inner"]
    assert collect_asset_ids({}) == []
    assert collect_asset_ids("assetId") == []


def test_ingest_rejects_symlink_disguised_as_video(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("not media")
    disguised = tmp_path / "clip.mp4"
    try:
        disguised.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks not available")
    store = SqliteStore(tmp_path / "app_data")
    try:
        store.ingest_upload(str(disguised))
    except MediaError:
        return
    raise AssertionError("expected MediaError for disguised ingest path")


def test_ingest_upload_copies_file_and_survives_source_delete(tmp_path: Path) -> None:
    app_data = tmp_path / "app_data"
    source = _png(tmp_path / "cat.png", (64, 48))
    store = SqliteStore(app_data)
    asset = store.ingest_upload(str(source))
    source.unlink()
    assert Path(asset.path).is_file()
    assert asset.origin == "uploaded"
    assert asset.media_kind == "image"
    assert asset.mime_type == "image/png"
    assert asset.metadata.mediaType == "image"
    assert asset.metadata.metadata.width == 64
    assert asset.metadata.metadata.height == 48
    copied = Path(app_data / "assets")
    assert Path(asset.path).parent == copied


def test_get_asset_returns_ingested_asset_and_none_for_unknown(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    asset = store.ingest_upload(str(_png(tmp_path / "cat.png", (64, 48))))

    assert store.get_asset(asset.id) == asset
    assert store.get_asset("missing-asset") is None


def test_ingest_video_stores_studio_metadata(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    source = _mp4(tmp_path / "clip.mp4", frames=8, width=16, height=12, fps=8)
    asset = store.ingest_upload(str(source))
    assert asset.media_kind == "video"
    assert asset.metadata.mediaType == "video"
    meta = asset.metadata.metadata
    assert meta.width == 16
    assert meta.height == 12
    assert meta.durationMs == 1000
    assert meta.sizeBytes == source.stat().st_size
    assert meta.audioStreamCount == 0


def test_ingest_audio_stores_studio_metadata(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    source = _wav(tmp_path / "clip.wav", duration_seconds=0.25, sample_rate=8000)
    asset = store.ingest_upload(str(source))
    assert asset.media_kind == "audio"
    assert asset.metadata.mediaType == "audio"
    meta = asset.metadata.metadata
    assert meta.durationMs == 250
    expected_bitrate = round((source.stat().st_size * 8) / 0.25)
    assert meta.bitrate == expected_bitrate


def test_ingest_image_persists_jpeg_thumbnail(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    source = _png(tmp_path / "still.png")
    record = store.ingest_upload(str(source))
    assert record.thumbnail_path is not None
    assert Path(record.thumbnail_path).is_file()
    loaded = store.get_asset(record.id)
    assert loaded is not None
    assert loaded.thumbnail_path == record.thumbnail_path


def test_ingest_audio_leaves_thumbnail_null(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    source = _wav(tmp_path / "clip.wav")
    record = store.ingest_upload(str(source))
    assert record.media_kind == "audio"
    assert record.thumbnail_path is None


def test_ingest_succeeds_when_thumbnail_writer_fails(tmp_path: Path) -> None:
    class Boom:
        def write(self, source_path: str, media_kind: object) -> str | None:
            del source_path, media_kind
            raise RuntimeError("thumb boom")

    store = SqliteStore(tmp_path / "app_data", thumbnail_writer=Boom())
    record = store.ingest_upload(str(_png(tmp_path / "still.png")))
    assert record.id
    assert record.thumbnail_path is None
    assert Path(record.path).is_file()


def test_audio_metadata_bitrate_is_optional() -> None:
    metadata = AudioAssetMetadata(
        mediaType="audio",
        metadata=AudioMeta(durationMs=250),
    )
    assert metadata.metadata.bitrate is None


def test_audio_metadata_without_bitrate_hydrates(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    source = _wav(tmp_path / "clip.wav", duration_seconds=0.25, sample_rate=8000)
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("audio", "audio/wav")
    Path(dest).write_bytes(source.read_bytes())
    store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest, mime_type="audio/wav", name="out.wav")],
        attempt_count=claimed.attempt_count,
    )
    db_path = tmp_path / "app_data" / "store.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE assets SET metadata = ? WHERE id = ?",
        (
            json.dumps({"mediaType": "audio", "metadata": {"durationMs": 250}}),
            asset_id,
        ),
    )
    conn.commit()
    conn.close()
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.outputs[0].metadata.mediaType == "audio"
    assert loaded.outputs[0].metadata.metadata.bitrate is None


def test_asset_rejects_media_kind_metadata_mismatch() -> None:
    image_metadata = ImageAssetMetadata(
        mediaType="image",
        metadata=ImageMeta(width=8, height=8),
    )
    with pytest.raises(ValidationError):
        AssetRecord(
            id="asset-1",
            media_kind="video",
            origin="uploaded",
            path="/tmp/x.png",
            thumbnail_path=None,
            mime_type="image/png",
            name="x.png",
            metadata=image_metadata,
            created_at=1,
        )
    with pytest.raises(ValidationError):
        Asset(
            id="asset-1",
            media_kind="video",
            origin="uploaded",
            path="/tmp/x.png",
            thumbnail_path=None,
            mime_type="image/png",
            name="x.png",
            metadata=image_metadata,
            created_at=1,
        )


def test_ingest_unreadable_video_does_not_leave_dest(tmp_path: Path) -> None:
    junk = tmp_path / "clip.mp4"
    junk.write_bytes(b"not a video")
    store = SqliteStore(tmp_path / "app_data")
    try:
        store.ingest_upload(str(junk))
    except MediaError as exc:
        assert exc.code == "UNREADABLE_MEDIA"
        assets = list((tmp_path / "app_data" / "assets").iterdir())
        assert assets == []
        return
    raise AssertionError("expected MediaError for unreadable video")


def test_allocate_output_path_has_no_row(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    asset_id, dest = store.allocate_output_path("video", "video/mp4")
    dest_path = Path(dest)
    assert asset_id
    assert dest_path.name == f"{asset_id}.mp4"
    assert dest_path.parent.name == ".in-flight"
    assert not dest_path.exists()
    conn = sqlite3.connect(str(tmp_path / "app_data" / "store.sqlite3"))
    count = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    conn.close()
    assert count == 0


def test_insert_and_list_filters_by_feature(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    a = store.ingest_upload(str(_png(tmp_path / "a.png")))
    gen_a = store.insert_generation(
        "text-to-video",
        _spec(prompt="fox", inputs={"startFrame": {"assetId": a.id}}),
    )
    store.insert_generation("cozy-felt", _spec(prompt="felt"))
    listed = store.list_generations("text-to-video")
    assert [g.id for g in listed] == [gen_a.id]
    assert listed[0].status == "queued"
    assert listed[0].queued_at is not None
    assert listed[0].attempt_count == 0
    assert listed[0].started_at is None
    assert listed[0].contract_version == 1
    assert listed[0].spec == {
        "params": {"prompt": "fox"},
        "inputs": {"startFrame": {"assetId": a.id}},
    }
    assert listed[0].outputs == ()


def test_list_recent_features_is_unique_newest_first(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    oldest = store.insert_generation("text-to-video", _spec(prompt="a"))
    middle = store.insert_generation("extend", _spec(prompt="b"))
    newest = store.insert_generation("text-to-video", _spec(prompt="c"))
    dropped = store.insert_generation("retake", _spec(prompt="d"))
    conn = sqlite3.connect(tmp_path / "app_data" / "store.sqlite3")
    conn.executemany(
        "UPDATE generations SET created_at = ? WHERE id = ?",
        [
            (1, oldest.id),
            (2, middle.id),
            (3, newest.id),
            (4, dropped.id),
        ],
    )
    conn.execute(
        "UPDATE generations SET deleted_at = 5 WHERE id = ?",
        (dropped.id,),
    )
    conn.commit()
    conn.close()

    assert store.list_recent_features(limit=4) == ["text-to-video", "extend"]
    assert store.list_recent_features(limit=1) == ["text-to-video"]
    assert store.list_recent_features(
        limit=4, allowed=frozenset({"extend"})
    ) == ["extend"]
    assert store.list_recent_features(limit=4, allowed=frozenset()) == []


def test_list_queue_generations_resolves_generation_inputs(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    asset = store.ingest_upload(str(_png(tmp_path / "input.png")))
    generation = store.insert_generation(
        "image-to-video",
        _spec(inputs={"startFrame": {"assetId": asset.id}}),
    )

    queued = store.list_queue_generations()

    assert len(queued) == 1
    assert queued[0].generation.id == generation.id
    assert queued[0].input_assets == (asset,)


def test_list_queue_generations_includes_active_and_pending_work(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    active = store.insert_generation("text-to-video", _spec(prompt="active"))
    pending = store.insert_generation("text-to-video", _spec(prompt="pending"))
    _claim(store, active.id)

    queue = store.list_queue_generations()

    assert [
        (entry.generation.id, entry.generation.status) for entry in queue
    ] == [(active.id, "running"), (pending.id, "queued")]


def test_insert_generation_stores_contract_version(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation(
        "image-to-video",
        _spec(),
        contract_version=2,
    )
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.contract_version == 2
    with pytest.raises(ValueError, match="contract_version"):
        store.insert_generation("image-to-video", _spec(), contract_version=0)


def test_list_keeps_missing_input_and_groups_generations(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    first = store.ingest_upload(str(_png(tmp_path / "first.png")))
    second = store.ingest_upload(str(_png(tmp_path / "second.png")))
    gen_first = store.insert_generation(
        "text-to-video",
        _spec(prompt="first", inputs={"startFrame": {"assetId": first.id}}),
    )
    gen_second = store.insert_generation(
        "text-to-video",
        _spec(prompt="second", inputs={"startFrame": {"assetId": second.id}}),
    )
    Path(first.path).unlink()
    listed = store.list_generations("text-to-video")
    assert [g.id for g in listed] == [gen_second.id, gen_first.id]
    assert listed[0].spec["inputs"] == {"startFrame": {"assetId": second.id}}
    assert listed[1].spec["inputs"] == {"startFrame": {"assetId": first.id}}


def test_spec_preserves_input_slot_names(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    start = store.ingest_upload(str(_png(tmp_path / "start.png")))
    end = store.ingest_upload(str(_png(tmp_path / "end.png")))
    ref = store.ingest_upload(str(_png(tmp_path / "ref.png")))
    gen = store.insert_generation(
        "image-to-video",
        _spec(
            inputs={
                "startFrame": {"assetId": start.id},
                "endFrame": {"assetId": end.id},
                "ref": {"assetId": ref.id},
            }
        ),
    )
    assert gen.spec["inputs"] == {
        "startFrame": {"assetId": start.id},
        "endFrame": {"assetId": end.id},
        "ref": {"assetId": ref.id},
    }


def test_claim_next_queued_is_fifo_and_increments_attempt(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    first = store.insert_generation("text-to-video", _spec(prompt="first"))
    second = store.insert_generation("text-to-video", _spec(prompt="second"))

    claimed_first = _claim(store, first.id)
    assert claimed_first.status == "running"
    assert claimed_first.attempt_count == 1
    assert claimed_first.started_at is not None
    assert store.claim_next_queued() is not None
    assert store.claim_next_queued() is None

    loaded_second = store.get_generation(second.id)
    assert loaded_second is not None
    assert loaded_second.status == "running"
    assert loaded_second.attempt_count == 1


def test_reorder_moves_queued_generation_before_target(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    first = store.insert_generation("text-to-video", _spec(prompt="first"))
    second = store.insert_generation("text-to-video", _spec(prompt="second"))
    third = store.insert_generation("text-to-video", _spec(prompt="third"))

    reordered = store.reorder_queued_generation(third.id, second.id)

    assert reordered.id == third.id
    assert [entry.generation.id for entry in store.list_queue_generations()] == [
        first.id,
        third.id,
        second.id,
    ]


def test_reordered_generation_is_next_claimed(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    first = store.insert_generation("text-to-video", _spec(prompt="first"))
    second = store.insert_generation("text-to-video", _spec(prompt="second"))
    third = store.insert_generation("text-to-video", _spec(prompt="third"))

    store.reorder_queued_generation(third.id, first.id)

    claimed = store.claim_next_queued()

    assert claimed is not None
    assert claimed.id == third.id
    assert claimed.status == "running"
    assert claimed.attempt_count == 1
    still_first = store.get_generation(first.id)
    still_second = store.get_generation(second.id)
    assert still_first is not None
    assert still_second is not None
    assert still_first.status == "queued"
    assert still_second.status == "queued"


def test_reorder_moves_queued_generation_to_tail(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    first = store.insert_generation("text-to-video", _spec(prompt="first"))
    second = store.insert_generation("text-to-video", _spec(prompt="second"))
    third = store.insert_generation("text-to-video", _spec(prompt="third"))

    reordered = store.reorder_queued_generation(first.id, None)

    assert reordered.id == first.id
    assert [entry.generation.id for entry in store.list_queue_generations()] == [
        second.id,
        third.id,
        first.id,
    ]


def test_reorder_rejects_running_or_cancelled_generation(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    running = store.insert_generation("text-to-video", _spec(prompt="running"))
    target = store.insert_generation("text-to-video", _spec(prompt="target"))
    cancelled = store.insert_generation("text-to-video", _spec(prompt="cancelled"))
    _claim(store, running.id)
    store.request_generation_cancellation(cancelled.id)

    with pytest.raises(StatusError):
        store.reorder_queued_generation(running.id, target.id)
    with pytest.raises(StatusError):
        store.reorder_queued_generation(cancelled.id, target.id)


def test_reorder_rebalances_when_adjacent_ranks_have_no_gap(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    first = store.insert_generation("text-to-video", _spec(prompt="first"))
    second = store.insert_generation("text-to-video", _spec(prompt="second"))
    third = store.insert_generation("text-to-video", _spec(prompt="third"))
    conn = _connect(tmp_path / "app_data" / "store.sqlite3")
    conn.executemany(
        "UPDATE generations SET queue_rank = ? WHERE id = ?",
        [(10, first.id), (11, second.id), (12, third.id)],
    )
    conn.commit()
    conn.close()

    store.reorder_queued_generation(third.id, second.id)

    rows = _connect(tmp_path / "app_data" / "store.sqlite3").execute(
        """
        SELECT id, queue_rank FROM generations
        WHERE status = 'queued'
        ORDER BY queue_rank ASC, id ASC
        """
    ).fetchall()
    assert [row["id"] for row in rows] == [first.id, third.id, second.id]
    assert rows[2]["queue_rank"] - rows[1]["queue_rank"] > 1


def test_request_cancel_queued_finalizes_immediately(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())

    cancelled = store.request_generation_cancellation(generation.id)

    assert cancelled.status == "cancelled"
    assert cancelled.finished_at is not None
    row = _connect(tmp_path / "app_data" / "store.sqlite3").execute(
        "SELECT queue_rank FROM generations WHERE id = ?", (generation.id,)
    ).fetchone()
    assert row["queue_rank"] is None


def test_request_cancel_running_marks_cancelling(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, generation.id)

    cancelling = store.request_generation_cancellation(generation.id)

    assert cancelling.status == "cancelling"
    assert cancelling.attempt_count == claimed.attempt_count
    assert cancelling.finished_at is None


def test_boot_recovery_finalizes_cancelling_generation(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())
    _claim(store, generation.id)
    store.request_generation_cancellation(generation.id)

    assert store.fail_running_on_boot() == 1

    recovered = store.get_generation(generation.id)
    assert recovered is not None
    assert recovered.status == "cancelled"
    assert recovered.finished_at is not None


def test_claim_next_queued_rolls_back_when_hydration_fails(tmp_path: Path) -> None:
    store = _FailOnceHydrationDb(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec(prompt="retry"))

    with pytest.raises(RuntimeError, match="forced hydration failure"):
        store.claim_next_queued()

    loaded = store.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.attempt_count == 0
    assert store.has_queued() is True

    claimed = store.claim_next_queued()
    assert claimed is not None
    assert claimed.id == generation.id
    assert claimed.status == "running"
    assert claimed.attempt_count == 1


def test_has_queued_only_reports_pending_work(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    assert store.has_queued() is False
    generation = store.insert_generation("text-to-video", _spec())
    assert store.has_queued() is True
    store.claim_next_queued()
    assert store.has_queued() is False
    assert generation.status == "queued"


def test_fail_running_on_boot_preserves_queued_generation(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    running = store.insert_generation("text-to-video", _spec())
    queued = store.insert_generation("text-to-video", _spec())
    _claim(store, running.id)

    assert store.fail_running_on_boot() == 1

    failed = store.get_generation(running.id)
    still_queued = store.get_generation(queued.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "INTERRUPTED"
    assert still_queued is not None
    assert still_queued.status == "queued"


def test_retry_requeues_same_generation_id(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, generation.id)
    store.mark_failed(
        generation.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count
    )

    retried = store.retry_generation(generation.id)

    assert retried.id == generation.id
    assert retried.status == "queued"
    assert retried.error_code is None
    assert retried.started_at is None
    assert retried.finished_at is None
    assert retried.attempt_count == 2
    claimed_again = _claim(store, generation.id)
    assert claimed_again.attempt_count == 3


def test_mark_cancelled_rejects_stale_attempt_after_retry(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, generation.id)
    store.mark_failed(
        generation.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count
    )
    retried = store.retry_generation(generation.id)
    current = _claim(store, generation.id)
    store.request_generation_cancellation(generation.id)

    with pytest.raises(AttemptError) as exc_info:
        store.mark_cancelled(
            generation.id,
            attempt_count=claimed.attempt_count,
        )

    assert exc_info.value.expected == claimed.attempt_count
    assert exc_info.value.actual == current.attempt_count
    loaded = store.get_generation(generation.id)
    assert loaded is not None
    assert loaded.status == "cancelling"
    assert loaded.attempt_count == retried.attempt_count + 1


def test_retry_appends_generation_to_queue(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    first = store.insert_generation("text-to-video", _spec(prompt="first"))
    second = store.insert_generation("text-to-video", _spec(prompt="second"))
    first_claim = _claim(store, first.id)
    store.mark_failed(
        first.id, "OUTPUT_UNREADABLE", attempt_count=first_claim.attempt_count
    )
    store.retry_generation(first.id)

    assert _claim(store, second.id).id == second.id
    assert _claim(store, first.id).id == first.id


def test_retry_requires_failed_generation(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())

    with pytest.raises(StatusError):
        store.retry_generation(generation.id)


def test_terminal_update_rejects_stale_attempt(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())
    first_claim = _claim(store, generation.id)
    store.mark_failed(
        generation.id,
        "OUTPUT_UNREADABLE",
        attempt_count=first_claim.attempt_count,
    )
    store.retry_generation(generation.id)
    _claim(store, generation.id)

    with pytest.raises(AttemptError):
        store.mark_failed(
            generation.id,
            "OUTPUT_UNREADABLE",
            attempt_count=first_claim.attempt_count,
        )


def test_unknown_input_asset_rejected(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    with pytest.raises(ValueError, match="unknown input asset"):
        store.insert_generation(
            "image-to-video",
            _spec(inputs={"startFrame": {"assetId": "missing-id"}}),
        )


def test_uploaded_has_no_producer(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    uploaded = store.ingest_upload(str(_png(tmp_path / "in.png")))
    conn = sqlite3.connect(str(tmp_path / "app_data" / "store.sqlite3"))
    row = conn.execute(
        """
        SELECT producer_generation_id, output_ordinal, origin
        FROM assets WHERE id = ?
        """,
        (uploaded.id,),
    ).fetchone()
    conn.close()
    assert row[0] is None
    assert row[1] is None
    assert row[2] == "uploaded"


def test_same_file_as_start_and_end(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    uploaded = store.ingest_upload(str(_png(tmp_path / "in.png")))
    gen = store.insert_generation(
        "image-to-video",
        _spec(
            inputs={
                "startFrame": {"assetId": uploaded.id},
                "endFrame": {"assetId": uploaded.id},
            }
        ),
    )
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest)],
        attempt_count=claimed.attempt_count,
    )
    assert done.status == "succeeded"
    assert done.spec["inputs"] == {
        "startFrame": {"assetId": uploaded.id},
        "endFrame": {"assetId": uploaded.id},
    }
    assert len(done.outputs) == 1
    assert done.outputs[0].origin == "generated"


def test_mark_succeeded_missing_output_raises_without_failing(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    with pytest.raises(MarkSucceededError) as caught:
        store.mark_succeeded(
            gen.id,
                [_output(asset_id, dest)],
            attempt_count=claimed.attempt_count,
        )
    assert caught.value.error_code == "OUTPUT_MISSING"
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.status == "running"
    failed = store.mark_failed(
        gen.id,
        "OUTPUT_MISSING",
        [dest],
        attempt_count=claimed.attempt_count,
    )
    assert failed.status == "failed"
    assert failed.error_code == "OUTPUT_MISSING"


def test_mark_succeeded_missing_output_leaves_partials_for_caller(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    kept_id, kept_dest = store.allocate_output_path("image", "image/png")
    missing_id, missing_dest = store.allocate_output_path("image", "image/png")
    _png(Path(kept_dest), (8, 8))
    with pytest.raises(MarkSucceededError) as caught:
        store.mark_succeeded(
            gen.id,
            [
                _output(kept_id, kept_dest, ordinal=0, name="a.png"),
                _output(missing_id, missing_dest, ordinal=1, name="b.png"),
            ],
            attempt_count=claimed.attempt_count,
        )
    assert caught.value.error_code == "OUTPUT_MISSING"
    assert Path(kept_dest).exists()
    store.mark_failed(
        gen.id,
        "OUTPUT_MISSING",
        [kept_dest, missing_dest],
        attempt_count=claimed.attempt_count,
    )
    assert not Path(kept_dest).exists()
    assert not Path(missing_dest).exists()


def test_mark_succeeded_unreadable_output_raises_without_failing(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("video", "video/mp4")
    Path(dest).write_bytes(b"not a video")
    with pytest.raises(MarkSucceededError) as caught:
        store.mark_succeeded(
            gen.id,
                [_output(asset_id, dest, mime_type="video/mp4", name="out.mp4")],
            attempt_count=claimed.attempt_count,
        )
    assert caught.value.error_code == "OUTPUT_UNREADABLE"
    assert Path(dest).exists()
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.status == "running"
    store.mark_failed(
        gen.id,
        "OUTPUT_UNREADABLE",
        [dest],
        attempt_count=claimed.attempt_count,
    )
    assert not Path(dest).exists()
    retried = store.retry_generation(gen.id)
    assert retried.status == "queued"


def test_mark_failed_rejects_unknown_error_code(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    with pytest.raises(ValueError):
        store.mark_failed(
            gen.id,
            "transient",  # type: ignore[arg-type]
            attempt_count=claimed.attempt_count,
        )


def test_mark_succeeded_requires_output(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    try:
        store.mark_succeeded(gen.id, [], attempt_count=claimed.attempt_count)
    except ValueError:
        loaded = store.get_generation(gen.id)
        assert loaded is not None
        assert loaded.status == "running"
        return
    raise AssertionError("expected ValueError for empty outputs")


def test_mark_succeeded_rejects_path_outside_assets(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    outsider = _png(tmp_path / "outside.png")
    try:
        store.mark_succeeded(
            gen.id,
                [_output("not-used", str(outsider))],
            attempt_count=claimed.attempt_count,
        )
    except ValueError:
        assert outsider.is_file()
        return
    raise AssertionError("expected ValueError for dest outside assets")


def test_mark_succeeded_inserts_output_asset(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec(prompt="x"))
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest)],
        attempt_count=claimed.attempt_count,
    )
    assert done.status == "succeeded"
    assert done.finished_at is not None
    assert len(done.outputs) == 1
    assert done.outputs[0].media_kind == "image"
    assert done.outputs[0].mime_type == "image/png"
    assert done.outputs[0].origin == "generated"
    assert done.outputs[0].id == asset_id
    assert Path(done.outputs[0].path).is_file()
    assert ".in-flight" not in Path(done.outputs[0].path).parts
    assert not Path(dest).exists()
    assert "path" not in done.spec


def test_mark_succeeded_persists_video_thumbnail(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec(prompt="x"))
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("video", "video/mp4")
    _mp4(Path(dest))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest, mime_type="video/mp4", name="out.mp4")],
        attempt_count=claimed.attempt_count,
    )
    assert done.outputs[0].thumbnail_path is not None
    assert Path(done.outputs[0].thumbnail_path).is_file()


def test_mark_succeeded_persists_probed_mime_not_spec(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest, mime_type="video/mp4", name="out.mp4")],
        attempt_count=claimed.attempt_count,
    )
    assert len(done.outputs) == 1
    assert done.outputs[0].media_kind == "image"
    assert done.outputs[0].mime_type == "image/png"


def test_delete_generation_keeps_asset_file(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    uploaded = store.ingest_upload(str(_png(tmp_path / "keep.png")))
    gen = store.insert_generation(
        "text-to-video",
        _spec(inputs={"startFrame": {"assetId": uploaded.id}}),
    )
    assert store.delete_generation(gen.id) is True
    assert store.get_generation(gen.id) is None
    assert Path(uploaded.path).is_file()
    conn = sqlite3.connect(str(tmp_path / "app_data" / "store.sqlite3"))
    remaining = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    deleted = conn.execute(
        "SELECT deleted_at FROM generations WHERE id = ?", (gen.id,)
    ).fetchone()[0]
    conn.close()
    assert remaining == 1
    assert deleted is not None


def test_deleted_queued_generation_is_not_claimed(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    hidden = store.insert_generation("text-to-video", _spec(prompt="hidden"))
    visible = store.insert_generation("text-to-video", _spec(prompt="visible"))
    assert store.delete_generation(hidden.id) is True
    claimed = store.claim_next_queued()
    assert claimed is not None
    assert claimed.id == visible.id
    assert store.get_generation(hidden.id) is None


def test_missing_output_file_omitted_from_get(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest)],
        attempt_count=claimed.attempt_count,
    )
    Path(done.outputs[0].path).unlink()
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.outputs == ()


def test_missing_input_file_stays_on_get(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    uploaded = store.ingest_upload(str(_png(tmp_path / "start.png")))
    gen = store.insert_generation(
        "image-to-video",
        _spec(inputs={"startFrame": {"assetId": uploaded.id}}),
    )
    Path(uploaded.path).unlink()
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.spec["inputs"] == {"startFrame": {"assetId": uploaded.id}}
    assert loaded.outputs == ()


def test_claim_fails_generation_when_input_file_missing_and_claims_next(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    uploaded = store.ingest_upload(str(_png(tmp_path / "start.png")))
    broken = store.insert_generation(
        "image-to-video",
        _spec(inputs={"startFrame": {"assetId": uploaded.id}}),
    )
    healthy = store.insert_generation("text-to-video", _spec(prompt="ok"))
    Path(uploaded.path).unlink()
    claimed = store.claim_next_queued()
    assert claimed is not None
    assert claimed.id == healthy.id
    failed = store.get_generation(broken.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "INPUT_MISSING"
    assert failed.spec["inputs"] == {"startFrame": {"assetId": uploaded.id}}


def test_fail_running_on_boot_marks_only_running_failed(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    _claim(store, gen.id)
    n = store.fail_running_on_boot()
    assert n == 1
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.error_code == "INTERRUPTED"
    assert loaded.finished_at is not None


def test_succeed_after_cancel_raises(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    store.request_generation_cancellation(gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    try:
        store.mark_succeeded(
            gen.id,
                [_output(asset_id, dest)],
            attempt_count=claimed.attempt_count,
        )
    except StatusError:
        return
    raise AssertionError("expected StatusError")


def test_mark_cancelled_deletes_partial_file(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    _, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    store.request_generation_cancellation(gen.id)
    store.mark_cancelled(
        gen.id,
        partial_paths=[dest],
        attempt_count=claimed.attempt_count,
    )
    assert not Path(dest).exists()


def test_mark_cancelled_does_not_unlink_outside_assets(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    outsider = tmp_path / "keep.txt"
    outsider.write_text("x")
    store.request_generation_cancellation(gen.id)
    store.mark_cancelled(
        gen.id,
        partial_paths=[str(outsider)],
        attempt_count=claimed.attempt_count,
    )
    assert outsider.is_file()


def test_mark_cancelled_commits_when_partial_path_is_directory(
    tmp_path: Path, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, generation.id)
    _, partial_path = store.allocate_output_path("video", "video/mp4")
    Path(partial_path).mkdir()
    store.request_generation_cancellation(generation.id)

    cancelled = store.mark_cancelled(
        generation.id,
        partial_paths=[partial_path],
        attempt_count=claimed.attempt_count,
    )

    assert cancelled.status == "cancelled"
    assert Path(partial_path).is_dir()
    assert any(
        "Could not remove partial output" in record.getMessage()
        for record in caplog.records
    )


def test_delete_running_generation_is_rejected(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    generation = store.insert_generation("text-to-video", _spec())
    _claim(store, generation.id)

    with pytest.raises(StatusError):
        store.delete_generation(generation.id)


def test_services_package_does_not_export_store_plumbing() -> None:
    import services

    assert not hasattr(services, "apply_migrations")
    assert not hasattr(services, "MIGRATIONS_DIR")
    assert not hasattr(services, "SqliteStore")


def test_ingest_jpeg_named_png_uses_decoded_mime(tmp_path: Path) -> None:
    source = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), color=(4, 5, 6)).save(source, format="JPEG")
    store = SqliteStore(tmp_path / "app_data")
    asset = store.ingest_upload(str(source))
    assert asset.mime_type == "image/jpeg"
    assert asset.media_kind == "image"
    assert Path(asset.path).suffix in {".jpg", ".jpeg"}
    assert Path(asset.path).is_file()


def test_mark_failed_deletes_generated_assets_so_retry_can_succeed(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    leftover_id, leftover_dest = store.allocate_output_path("image", "image/png")
    leftover_path = Path(leftover_dest).parent.parent / f"{leftover_id}.png"
    _png(Path(leftover_dest), (8, 8))
    Path(leftover_dest).replace(leftover_path)
    db_path = tmp_path / "app_data" / "store.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO assets (
            id, media_kind, origin, producer_generation_id, output_ordinal,
            mime_type, name, metadata, created_at, deleted_at
        ) VALUES (?, 'image', 'generated', ?, 0, 'image/png', 'old.png', ?, 1, NULL)
        """,
        (
            leftover_id,
            gen.id,
            json.dumps(
                {
                    "mediaType": "image",
                    "metadata": {"width": 8, "height": 8},
                }
            ),
        ),
    )
    conn.commit()
    conn.close()
    store.mark_failed(gen.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)
    assert not leftover_path.exists()
    retried = store.retry_generation(gen.id)
    claimed_again = _claim(store, retried.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest)],
        attempt_count=claimed_again.attempt_count,
    )
    assert done.status == "succeeded"
    assert [out.origin for out in done.outputs] == ["generated"]


def test_retry_deletes_generated_assets_left_on_failed_row(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    store.mark_failed(gen.id, "OUTPUT_MISSING", attempt_count=claimed.attempt_count)
    leftover_id, leftover_dest = store.allocate_output_path("image", "image/png")
    leftover_path = Path(leftover_dest).parent.parent / f"{leftover_id}.png"
    _png(Path(leftover_dest), (8, 8))
    Path(leftover_dest).replace(leftover_path)
    db_path = tmp_path / "app_data" / "store.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO assets (
            id, media_kind, origin, producer_generation_id, output_ordinal,
            mime_type, name, metadata, created_at, deleted_at
        ) VALUES (?, 'image', 'generated', ?, 0, 'image/png', 'old.png', ?, 1, NULL)
        """,
        (
            leftover_id,
            gen.id,
            json.dumps(
                {
                    "mediaType": "image",
                    "metadata": {"width": 8, "height": 8},
                }
            ),
        ),
    )
    conn.commit()
    conn.close()
    store.retry_generation(gen.id)
    assert not leftover_path.exists()
    claimed_again = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest)],
        attempt_count=claimed_again.attempt_count,
    )
    assert done.status == "succeeded"


def test_duplicate_output_ordinal_rejected(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    first_id, first_dest = store.allocate_output_path("image", "image/png")
    second_id, second_dest = store.allocate_output_path("image", "image/png")
    _png(Path(first_dest), (8, 8))
    _png(Path(second_dest), (8, 8))
    with pytest.raises(ValueError, match="ordinals must be unique"):
        store.mark_succeeded(
            gen.id,
            [
                _output(first_id, first_dest, ordinal=0),
                _output(second_id, second_dest, ordinal=0),
            ],
            attempt_count=claimed.attempt_count,
        )


def test_mark_failed_return_survives_immediate_delete(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    failed = store.mark_failed(
        gen.id, "OUTPUT_MISSING", attempt_count=claimed.attempt_count
    )
    assert store.delete_generation(gen.id) is True
    assert failed.status == "failed"
    assert failed.error_code == "OUTPUT_MISSING"
    assert store.get_generation(gen.id) is None


def test_mark_succeeded_unlinks_dests_when_attempt_check_fails(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    store.request_generation_cancellation(gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    with pytest.raises(StatusError):
        store.mark_succeeded(
            gen.id,
                [_output(asset_id, dest)],
            attempt_count=claimed.attempt_count,
        )
    assert not Path(dest).exists()


class _CommitFailsStore(SqliteStore):
    def __init__(self, app_data_dir: Path) -> None:
        super().__init__(app_data_dir)
        self.fail_commit = False

    def _commit_loaded(
        self, conn: sqlite3.Connection, generation_id: str
    ) -> GenerationRecord:
        if self.fail_commit:
            raise sqlite3.OperationalError("commit failed")
        return super()._commit_loaded(conn, generation_id)


def test_mark_succeeded_restores_promoted_file_when_commit_fails(
    tmp_path: Path,
) -> None:
    store = _CommitFailsStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    canonical = Path(dest).parent.parent / f"{asset_id}.png"
    store.fail_commit = True
    with pytest.raises(sqlite3.OperationalError, match="commit failed"):
        store.mark_succeeded(
            gen.id,
            [_output(asset_id, dest)],
            attempt_count=claimed.attempt_count,
        )
    assert Path(dest).is_file()
    assert not canonical.exists()
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.status == "running"


def test_mark_failed_unlinks_partials_when_attempt_check_fails(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    _, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    store.request_generation_cancellation(gen.id)
    with pytest.raises(StatusError):
        store.mark_failed(
            gen.id,
            "OUTPUT_MISSING",
            [dest],
            attempt_count=claimed.attempt_count,
        )
    assert not Path(dest).exists()


def test_unlink_path_retries_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "partial.mp4"
    path.write_bytes(b"partial output")
    attempts = {"n": 0}
    real_unlink = Path.unlink

    def flaky(self: Path, *args: object, **kwargs: object) -> None:
        if self == path:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise PermissionError("sharing violation")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky)
    _unlink_path(path)
    assert not path.exists()
    assert attempts["n"] == 2


def test_fail_running_on_boot_reclaims_in_flight_files(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    uploaded = store.ingest_upload(str(_png(tmp_path / "keep.png")))
    gen = store.insert_generation(
        "text-to-video",
        _spec(inputs={"startFrame": {"assetId": uploaded.id}}),
    )
    _claim(store, gen.id)
    _, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    stray = Path(uploaded.path).parent / "stray.png"
    _png(stray, (8, 8))
    assert store.fail_running_on_boot() == 1
    assert not Path(dest).exists()
    assert Path(uploaded.path).is_file()
    assert stray.is_file()
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.error_code == "INTERRUPTED"


def test_fail_running_on_boot_does_not_wipe_assets_if_db_is_recreated(
    tmp_path: Path,
) -> None:
    app_data = tmp_path / "app_data"
    store = SqliteStore(app_data)
    uploaded = store.ingest_upload(str(_png(tmp_path / "keep.png")))
    keep = Path(uploaded.path)
    assert keep.is_file()
    for suffix in ("", "-wal", "-shm"):
        (app_data / f"store.sqlite3{suffix}").unlink(missing_ok=True)
    store = SqliteStore(app_data)
    store.fail_running_on_boot()
    assert keep.is_file()


def test_mark_succeeded_accepts_already_promoted_output(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec())
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    canonical = Path(dest).parent.parent / f"{asset_id}.png"
    Path(dest).replace(canonical)
    done = store.mark_succeeded(
        gen.id,
        [_output(asset_id, dest)],
        attempt_count=claimed.attempt_count,
    )
    assert done.status == "succeeded"
    assert Path(done.outputs[0].path) == canonical
    assert canonical.is_file()
    assert not Path(dest).exists()


def test_ingest_rejects_oversized_image_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "huge.png"
    with source.open("wb") as handle:
        handle.truncate(MAX_IMAGE_BYTES + 1)
    store = SqliteStore(tmp_path / "app_data")
    with pytest.raises(MediaError) as caught:
        store.ingest_upload(str(source))
    assert caught.value.code == "FILE_TOO_LARGE"
    assets_dir = tmp_path / "app_data" / "assets"
    assert list(assets_dir.iterdir()) == []


def test_validate_ingest_source_does_not_cap_video_bytes(tmp_path: Path) -> None:
    source = tmp_path / "huge.mp4"
    with source.open("wb") as handle:
        handle.truncate(MAX_VIDEO_BYTES + 1)
    validate_ingest_source(source)


def test_ingest_rejects_image_dimensions_too_large(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    Image.new("RGB", (MAX_IMAGE_PIXELS + 1, 1), color=(1, 2, 3)).save(source)
    store = SqliteStore(tmp_path / "app_data")
    with pytest.raises(MediaError) as caught:
        store.ingest_upload(str(source))
    assert caught.value.code == "IMAGE_DIMENSIONS_TOO_LARGE"
    assets_dir = tmp_path / "app_data" / "assets"
    assert list(assets_dir.iterdir()) == []


def test_ingest_decompression_bomb_is_unreadable(tmp_path: Path) -> None:
    original = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 1
    try:
        source = _png(tmp_path / "bomb.png", (16, 16))
        store = SqliteStore(tmp_path / "app_data")
        with pytest.raises(MediaError) as caught:
            store.ingest_upload(str(source))
        assert caught.value.code == "UNREADABLE_MEDIA"
        assets_dir = tmp_path / "app_data" / "assets"
        assert list(assets_dir.iterdir()) == []
    finally:
        Image.MAX_IMAGE_PIXELS = original


def test_list_assets_omits_uploaded(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    uploaded = store.ingest_upload(str(_png(tmp_path / "in.png")))
    generated = _succeed_png(store)
    ids = [item.asset.id for item in store.list_assets().items]
    assert uploaded.id not in ids
    assert generated.id in ids
    assert all(item.asset.origin == "generated" for item in store.list_assets().items)


def test_list_assets_filters_kind_q_and_sort(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    img = _succeed_png(store, name="Fox.png")
    _succeed_wav(store, name="clip.wav")
    page = store.list_assets(media_kind="image", q="fox")
    assert [item.asset.id for item in page.items] == [img.id]
    oldest = store.list_assets(sort="created_at-asc")
    newest = store.list_assets(sort="created_at-desc")
    assert oldest.items[0].asset.created_at <= oldest.items[-1].asset.created_at
    assert newest.items[0].asset.id != oldest.items[0].asset.id or len(newest.items) == 1


def test_list_assets_escapes_like_metacharacters(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    _succeed_png(store, name="100%.png")
    _succeed_png(store, name="100x.png")
    page = store.list_assets(q="100%")
    assert [item.asset.name for item in page.items] == ["100%.png"]


def test_list_assets_tombstones_missing_files(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    kept = _succeed_png(store, name="kept.png")
    gone = _succeed_png(store, name="gone.png")
    Path(gone.path).unlink()
    page = store.list_assets()
    ids = [item.asset.id for item in page.items]
    assert kept.id in ids
    assert gone.id not in ids
    assert store.get_asset(gone.id) is None


def test_get_asset_tombstones_missing_file(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gone = store.ingest_upload(str(_png(tmp_path / "gone.png")))
    Path(gone.path).unlink()
    assert store.get_asset(gone.id) is None
    assert gone.id not in [item.asset.id for item in store.list_assets().items]


def test_list_assets_can_omit_missing_files_without_tombstoning(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gone = _succeed_png(store, name="gone.png")
    payload = Path(gone.path).read_bytes()
    Path(gone.path).unlink()
    page = store.list_assets(tombstone_missing=False)
    assert gone.id not in [item.asset.id for item in page.items]
    Path(gone.path).write_bytes(payload)
    assert store.get_asset(gone.id) is not None


def test_get_asset_can_miss_without_tombstoning(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gone = store.ingest_upload(str(_png(tmp_path / "gone.png")))
    payload = Path(gone.path).read_bytes()
    Path(gone.path).unlink()
    assert store.get_asset(gone.id, tombstone_missing=False) is None
    Path(gone.path).write_bytes(payload)
    assert store.get_asset(gone.id) is not None


def test_list_assets_does_not_tombstone_in_use_missing_files(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    start = _succeed_png(store, name="start.png")
    payload = Path(start.path).read_bytes()
    store.insert_generation(
        "image-to-video",
        _spec(inputs={"startFrame": {"assetId": start.id}}),
    )
    Path(start.path).unlink()
    page = store.list_assets()
    assert start.id not in [item.asset.id for item in page.items]
    Path(start.path).write_bytes(payload)
    assert store.get_asset(start.id) is not None


def test_list_assets_marks_nested_in_use(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    start = _succeed_png(store, name="start.png")
    free = _succeed_png(store, name="free.png")
    store.insert_generation(
        "image-to-video",
        _spec(
            inputs={
                "startFrame": {"assetId": start.id},
                "keyframes": [{"assetId": start.id}],
            }
        ),
    )
    by_id = {item.asset.id: item.in_use for item in store.list_assets().items}
    assert by_id[start.id] is True
    assert by_id[free.id] is False


def test_list_assets_keyset_stable_order(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    for i in range(3):
        _succeed_png(store, name=f"{i}.png")
    first = store.list_assets(limit=2)
    assert first.next_cursor is not None
    rest = store.list_assets(limit=2, cursor=first.next_cursor)
    listed = [item.asset.id for item in first.items] + [
        item.asset.id for item in rest.items
    ]
    assert len(listed) == 3
    assert len(set(listed)) == 3


def test_list_assets_keyset_continues_after_tombstoned_head(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    later = _succeed_png(store, name="later.png")
    gone = _succeed_png(store, name="gone.png")
    Path(gone.path).unlink()
    conn = sqlite3.connect(str(tmp_path / "app_data" / "store.sqlite3"))
    conn.execute(
        "UPDATE assets SET created_at = ? WHERE id = ?",
        (later.created_at + 1, gone.id),
    )
    conn.commit()
    conn.close()
    first = store.list_assets(limit=1)
    assert [item.asset.id for item in first.items] == []
    assert first.next_cursor is not None
    rest = store.list_assets(limit=1, cursor=first.next_cursor)
    assert [item.asset.id for item in rest.items] == [later.id]
    assert store.get_asset(gone.id) is None


def test_list_assets_rejects_invalid_cursor(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    _succeed_png(store, name="a.png")
    with pytest.raises(ValueError):
        store.list_assets(cursor="not-a-cursor")
    with pytest.raises(ValueError):
        store.list_assets(cursor="123:")


def test_unavailable_store_list_assets_raises() -> None:
    with pytest.raises(UnavailableError):
        UnavailableStore().list_assets()


def test_delete_unused_upload_unlinks_file(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    record = store.ingest_upload(str(_png(tmp_path / "solo.png")))
    path = Path(record.path)
    assert store.delete_asset(record.id) is True
    assert not path.exists()
    assert store.get_asset(record.id) is None
    assert store.delete_asset(record.id) is True  # idempotent


def test_delete_unknown_asset_returns_false(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    assert store.delete_asset("missing") is False


def test_delete_asset_blocks_cancelling_producer(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec(prompt="x"))
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id, [_output(asset_id, dest)], attempt_count=claimed.attempt_count
    )
    conn = sqlite3.connect(str(tmp_path / "app_data" / "store.sqlite3"))
    conn.execute(
        "UPDATE generations SET status = 'cancelling' WHERE id = ?",
        (gen.id,),
    )
    conn.commit()
    conn.close()
    with pytest.raises(StatusError) as caught:
        store.delete_asset(done.outputs[0].id)
    assert caught.value.status == "cancelling"
    assert store.get_asset(done.outputs[0].id) is not None
    loaded = store.get_generation(gen.id)
    assert loaded is not None
    assert loaded.status == "cancelling"


def test_delete_generated_output_hides_producer(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec(prompt="x"))
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id, [_output(asset_id, dest)], attempt_count=claimed.attempt_count
    )
    out = done.outputs[0]
    assert store.delete_asset(out.id) is True
    assert store.get_generation(gen.id) is None
    assert store.get_asset(out.id) is None


def test_delete_when_producer_already_results_deleted_is_success(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec(prompt="x"))
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id, [_output(asset_id, dest)], attempt_count=claimed.attempt_count
    )
    assert store.delete_generation(gen.id) is True
    assert store.delete_asset(done.outputs[0].id) is True


def test_delete_blocks_when_used_as_nested_input(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    start = store.ingest_upload(str(_png(tmp_path / "start.png")))
    store.insert_generation(
        "image-to-video",
        {"params": {}, "inputs": {"nested": [{"assetId": start.id}]}},
    )
    with pytest.raises(AssetInUseError):
        store.delete_asset(start.id)
    assert Path(start.path).is_file()


def test_delete_blocks_generated_asset_used_as_later_input(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec(prompt="x"))
    claimed = _claim(store, gen.id)
    asset_id, dest = store.allocate_output_path("image", "image/png")
    _png(Path(dest), (8, 8))
    done = store.mark_succeeded(
        gen.id, [_output(asset_id, dest)], attempt_count=claimed.attempt_count
    )
    out_id = done.outputs[0].id
    store.insert_generation(
        "image-to-video",
        {"params": {}, "inputs": {"startFrame": {"assetId": out_id}}},
    )
    with pytest.raises(AssetInUseError):
        store.delete_asset(out_id)


def test_delete_sibling_output_keeps_generation_until_last(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    gen = store.insert_generation("text-to-video", _spec(prompt="x"))
    claimed = _claim(store, gen.id)
    a_id, a_dest = store.allocate_output_path("image", "image/png")
    b_id, b_dest = store.allocate_output_path("image", "image/png")
    _png(Path(a_dest), (8, 8))
    _png(Path(b_dest), (8, 8))
    store.mark_succeeded(
        gen.id,
        [_output(a_id, a_dest, ordinal=0), _output(b_id, b_dest, ordinal=1)],
        attempt_count=claimed.attempt_count,
    )
    assert store.delete_asset(a_id) is True
    assert store.get_generation(gen.id) is not None
    assert store.delete_asset(b_id) is True
    assert store.get_generation(gen.id) is None


def test_delete_already_deleted_succeeds_without_file(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    record = store.ingest_upload(str(_png(tmp_path / "gone.png")))
    path = Path(record.path)
    assert store.delete_asset(record.id) is True
    path.unlink(missing_ok=True)
    assert store.delete_asset(record.id) is True
    assert store.get_asset(record.id) is None


def test_delete_returns_true_when_canonical_unlink_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteStore(tmp_path / "app_data")
    record = store.ingest_upload(str(_png(tmp_path / "busy.png")))
    canonical = Path(record.path)
    real_unlink = Path.unlink

    def boom(self: Path, *args: object, **kwargs: object) -> None:
        if self == canonical:
            raise OSError("busy")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", boom)
    assert store.delete_asset(record.id) is True
    assert store.get_asset(record.id) is None
