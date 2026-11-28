from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from app_handler import AppHandler, open_store
from remote.controller import RemoteExposureController
from services.features.video.reference_sheet import ensure_reference_sheet
from services.records import UnavailableError
from services.unavailable_store import UnavailableStore
from services.sqlite_store import ASSETS_MARKER_FILENAME, SqliteStore
from services.thumbnails import resolve_stored_thumbnail
from tests.test_store import _spec, _succeed_png


def _png(path: Path) -> Path:
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(path)
    return path


def _custom_dir(tmp_path: Path, *, marker: bool = True) -> Path:
    assets_dir = tmp_path / "custom"
    assets_dir.mkdir()
    if marker:
        (assets_dir / ASSETS_MARKER_FILENAME).write_text("")
    return assets_dir


def _stored_thumbnail(app_data: Path, asset_id: str) -> str | None:
    conn = sqlite3.connect(app_data / "store.sqlite3")
    try:
        row = conn.execute(
            "SELECT thumbnail_path FROM assets WHERE id = ?", (asset_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None or row[0] is None else str(row[0])


def test_custom_assets_dir_receives_ingested_files(tmp_path: Path) -> None:
    app_data = tmp_path / "app_data"
    assets_dir = _custom_dir(tmp_path)
    store = SqliteStore(app_data, assets_dir=assets_dir)

    record = store.ingest_upload(str(_png(tmp_path / "cat.png")))

    assert Path(record.path).parent == assets_dir
    assert Path(record.path).is_file()
    assert not (app_data / "assets" / Path(record.path).name).exists()


def test_missing_custom_assets_dir_is_unavailable_and_not_created(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "unplugged" / "assets"

    with pytest.raises(UnavailableError):
        SqliteStore(tmp_path / "app_data", assets_dir=missing)

    assert not missing.exists()
    assert not missing.parent.exists()


def test_custom_assets_dir_without_marker_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(UnavailableError):
        SqliteStore(tmp_path / "app_data", assets_dir=_custom_dir(tmp_path, marker=False))


def test_open_store_degrades_when_the_custom_assets_dir_is_missing(
    tmp_path: Path,
) -> None:
    store = open_store(tmp_path / "app_data", tmp_path / "unplugged")

    assert isinstance(store, UnavailableStore)
    assert not (tmp_path / "unplugged").exists()


def test_open_store_uses_the_custom_assets_dir(tmp_path: Path) -> None:
    assets_dir = _custom_dir(tmp_path)
    store = open_store(tmp_path / "app_data", assets_dir)

    assert isinstance(store, SqliteStore)
    record = store.ingest_upload(str(_png(tmp_path / "cat.png")))
    assert Path(record.path).parent == assets_dir


def test_remote_controller_starts_when_the_custom_assets_dir_is_missing(
    test_state: AppHandler, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The store degrades to UnavailableStore. The remote controller must not crash the server.
    monkeypatch.setattr(test_state.config, "assets_dir", tmp_path / "unplugged")

    RemoteExposureController(test_state)


def test_missing_assets_dir_at_runtime_does_not_tombstone_rows(tmp_path: Path) -> None:
    assets_dir = _custom_dir(tmp_path)
    store = SqliteStore(tmp_path / "app_data", assets_dir=assets_dir)
    record = _succeed_png(store)

    moved = tmp_path / "custom-away"
    assets_dir.rename(moved)
    with pytest.raises(UnavailableError):
        store.list_assets()
    with pytest.raises(UnavailableError):
        store.delete_asset(record.id)
    assert store.get_asset(record.id) is None

    moved.rename(assets_dir)
    page = store.list_assets()
    assert [item.asset.id for item in page.items] == [record.id]


def test_default_assets_dir_still_tombstones_a_missing_file(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data")
    record = _succeed_png(store)
    Path(record.path).unlink()

    assert store.list_assets().items == ()
    assert store.get_asset(record.id) is None
    Path(record.path).write_bytes(b"")
    assert store.get_asset(record.id) is None


def test_generated_output_thumbnail_is_a_name_and_matches_a_later_read(
    tmp_path: Path,
) -> None:
    app_data = tmp_path / "app_data"
    store = SqliteStore(app_data, assets_dir=_custom_dir(tmp_path))
    record = _succeed_png(store)

    assert record.thumbnail_path is not None
    assert _stored_thumbnail(app_data, record.id) == f"{record.id}-thumb.jpg"
    loaded = store.get_asset(record.id)
    assert loaded is not None
    assert loaded.thumbnail_path == record.thumbnail_path
    assert Path(record.thumbnail_path).parent == tmp_path / "custom"


def test_thumbnail_column_holds_the_file_name_only(tmp_path: Path) -> None:
    app_data = tmp_path / "app_data"
    store = SqliteStore(app_data)
    record = store.ingest_upload(str(_png(tmp_path / "cat.png")))

    assert _stored_thumbnail(app_data, record.id) == f"{record.id}-thumb.jpg"


def test_ingest_result_and_later_read_return_the_same_thumbnail_path(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "app_data", assets_dir=_custom_dir(tmp_path))
    record = store.ingest_upload(str(_png(tmp_path / "cat.png")))

    loaded = store.get_asset(record.id)
    assert loaded is not None
    assert record.thumbnail_path is not None
    assert loaded.thumbnail_path == record.thumbnail_path
    assert Path(record.thumbnail_path).parent == tmp_path / "custom"
    assert Path(record.thumbnail_path).is_file()


def test_old_row_with_absolute_thumbnail_path_resolves_in_current_folder(
    tmp_path: Path,
) -> None:
    app_data = tmp_path / "app_data"
    store = SqliteStore(app_data)
    record = store.ingest_upload(str(_png(tmp_path / "cat.png")))
    old_absolute = tmp_path / "old-folder" / f"{record.id}-thumb.jpg"
    conn = sqlite3.connect(app_data / "store.sqlite3")
    try:
        conn.execute(
            "UPDATE assets SET thumbnail_path = ? WHERE id = ?",
            (str(old_absolute), record.id),
        )
        conn.commit()
    finally:
        conn.close()

    loaded = store.get_asset(record.id)

    assert loaded is not None
    assert loaded.thumbnail_path == str(app_data / "assets" / f"{record.id}-thumb.jpg")
    assert Path(loaded.thumbnail_path).is_file()


def test_delete_removes_the_thumbnail_file(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "app_data", assets_dir=_custom_dir(tmp_path))
    record = store.ingest_upload(str(_png(tmp_path / "cat.png")))
    assert record.thumbnail_path is not None

    assert store.delete_asset(record.id) is True

    assert not Path(record.path).exists()
    assert not Path(record.thumbnail_path).exists()


def test_thumbnail_helpers_keep_only_a_safe_file_name(tmp_path: Path) -> None:
    assert resolve_stored_thumbnail(tmp_path, None) is None
    assert resolve_stored_thumbnail(tmp_path, "x-thumb.jpg") == tmp_path / "x-thumb.jpg"
    assert resolve_stored_thumbnail(tmp_path, "../../etc/x.jpg") == tmp_path / "x.jpg"
    assert resolve_stored_thumbnail(tmp_path, "..") is None
    assert resolve_stored_thumbnail(tmp_path, "") is None


def test_marker_name_matches_the_shared_contract() -> None:
    contract = Path(__file__).resolve().parents[2] / "shared" / "assets-marker.json"
    assert json.loads(contract.read_text())["filename"] == ASSETS_MARKER_FILENAME


def test_default_folder_is_recreated_for_a_new_output(tmp_path: Path) -> None:
    app_data = tmp_path / "app_data"
    store = SqliteStore(app_data)
    (app_data / "assets").rmdir()

    _, dest = store.allocate_output_path("image", "image/png")

    assert Path(dest).parent.is_dir()


def test_custom_folder_is_not_recreated_for_a_new_output(tmp_path: Path) -> None:
    custom = _custom_dir(tmp_path)
    store = SqliteStore(tmp_path / "app_data", assets_dir=custom)
    (custom / ASSETS_MARKER_FILENAME).unlink()
    custom.rmdir()

    with pytest.raises(UnavailableError):
        store.allocate_output_path("image", "image/png")

    assert not custom.exists()


def test_missing_custom_folder_does_not_fail_queued_jobs(tmp_path: Path) -> None:
    custom = _custom_dir(tmp_path)
    store = SqliteStore(tmp_path / "app_data", assets_dir=custom)
    uploaded = store.ingest_upload(str(_png(tmp_path / "start.png")))
    job = store.insert_generation(
        "image-to-video", _spec(inputs={"startFrame": {"assetId": uploaded.id}})
    )
    (custom / ASSETS_MARKER_FILENAME).unlink()  # the drive is gone

    assert store.claim_next_queued() is None

    (custom / ASSETS_MARKER_FILENAME).write_text("")
    claimed = store.claim_next_queued()
    assert claimed is not None and claimed.id == job.id


def test_ingest_into_a_missing_custom_folder_is_unavailable(tmp_path: Path) -> None:
    custom = _custom_dir(tmp_path)
    store = SqliteStore(tmp_path / "app_data", assets_dir=custom)
    (custom / ASSETS_MARKER_FILENAME).unlink()

    with pytest.raises(UnavailableError):
        store.ingest_upload(str(_png(tmp_path / "cat.png")))


def test_in_flight_folder_follows_the_custom_assets_folder(tmp_path: Path) -> None:
    from runtime_config.assets_layout import in_flight_folder

    custom = tmp_path / "custom"
    assert in_flight_folder(tmp_path / "app_data", custom) == custom / ".in-flight"
    assert in_flight_folder(tmp_path / "app_data") == tmp_path / "app_data" / "assets" / ".in-flight"


def test_reference_sheet_does_not_create_a_gone_assets_dir(tmp_path: Path) -> None:
    gone = tmp_path / "unplugged"

    with pytest.raises(FileNotFoundError):
        ensure_reference_sheet([], 64, 64, in_flight_dir=gone / ".in-flight")

    assert not gone.exists()
