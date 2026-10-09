from __future__ import annotations

import hashlib
from pathlib import Path

from services.migrations import MIGRATIONS_DIR, apply_migrations
from services.records import PairedDeviceRecord
from services.sqlite_store import SqliteStore
from tests.test_store import _connect


def test_fresh_db_includes_paired_devices(tmp_path: Path) -> None:
    conn = _connect(tmp_path / "store.sqlite3")
    apply_migrations(conn, MIGRATIONS_DIR)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "paired_devices" in tables
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(paired_devices)").fetchall()
    }
    assert columns == {
        "id",
        "name",
        "token_hash",
        "created_at",
        "last_seen_at",
        "first_seen_ip",
        "first_seen_user_agent",
        "revoked_at",
    }
    conn.close()


def test_paired_device_insert_lookup_touch_and_revoke(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    token_hash = hashlib.sha256(b"session").hexdigest()
    record = store.insert_paired_device(
        PairedDeviceRecord(
            id="dev-1",
            name="Phone",
            token_hash=token_hash,
            created_at=10,
            last_seen_at=10,
            first_seen_ip="192.168.1.4",
            first_seen_user_agent="TestUA",
        )
    )
    assert record.id == "dev-1"
    fetched = store.get_paired_device_by_token_hash(token_hash)
    assert fetched is not None
    assert fetched.name == "Phone"
    assert store.get_paired_device("missing") is None

    store.touch_paired_device("dev-1", now_ms=99)
    touched = store.get_paired_device("dev-1")
    assert touched is not None
    assert touched.last_seen_at == 99

    store.touch_paired_device(
        "dev-1",
        now_ms=None,
        ip="10.0.0.8",
        user_agent="Mozilla/5.0 (Macintosh) Chrome/120.0.0.0",
    )
    filled = store.get_paired_device("dev-1")
    assert filled is not None
    assert filled.last_seen_at == 99
    assert filled.first_seen_ip == "192.168.1.4"
    assert filled.first_seen_user_agent == "TestUA"

    assert store.revoke_paired_device("dev-1", now_ms=100) is True
    assert store.revoke_paired_device("dev-1", now_ms=101) is False
    revoked = store.get_paired_device("dev-1")
    assert revoked is not None
    assert revoked.revoked_at == 100
    listed = store.list_paired_devices()
    assert [item.id for item in listed] == ["dev-1"]


def test_touch_replaces_unknown_first_seen_ip(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path)
    token_hash = hashlib.sha256(b"session").hexdigest()
    store.insert_paired_device(
        PairedDeviceRecord(
            id="dev-unknown",
            name="Phone",
            token_hash=token_hash,
            created_at=10,
            last_seen_at=10,
            first_seen_ip="unknown",
            first_seen_user_agent=None,
        )
    )
    store.touch_paired_device("dev-unknown", now_ms=20, ip="10.0.0.8", user_agent="Phone UA")
    filled = store.get_paired_device("dev-unknown")
    assert filled is not None
    assert filled.first_seen_ip == "10.0.0.8"
    assert filled.first_seen_user_agent == "Phone UA"
