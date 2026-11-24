from __future__ import annotations

import sqlite3
import time
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "schema"

# Version 3 was also issued to the device calibration migration on some builds, so a store at
# version 3 can be missing what 0003_generation_lane.sql adds. Keep these in step with that file.
_GENERATION_LANE_COLUMNS = (
    ("lane", "ALTER TABLE generations ADD COLUMN lane TEXT NOT NULL DEFAULT 'local'"),
    ("provider_job_id", "ALTER TABLE generations ADD COLUMN provider_job_id TEXT"),
)
_GENERATION_LANE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_generations_pending_lane_rank
    ON generations(lane, queue_rank ASC, id ASC)
    WHERE status = 'queued' AND deleted_at IS NULL
"""


def _sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    buffer = ""
    for line in script.splitlines(keepends=True):
        buffer += line
        if not sqlite3.complete_statement(buffer):
            continue
        stmt = buffer.strip().rstrip(";").strip()
        buffer = ""
        if stmt:
            statements.append(stmt)
    leftover = buffer.strip()
    if leftover:
        raise ValueError(f"incomplete SQL in migration: {leftover[:120]}")
    return statements


def apply_migrations(conn: sqlite3.Connection, migrations_dir: Path) -> None:
    """Apply numbered ``NNNN_*.sql`` files with version > current.

    Each file is applied in a single transaction together with its
    ``schema_migrations`` row. ``executescript`` is not used: it autocommits
    DDL, so a crash between statements (or before the ledger insert) would
    leave the DB unopenable.

    Pre-ship: editing 0001 and deleting the local DB is allowed. After ship,
    add 0002+ — do not edit 0001 in place once users have the file.
    """
    previous_isolation = conn.isolation_level
    conn.isolation_level = None
    foreign_keys_row = conn.execute("PRAGMA foreign_keys").fetchone()
    restore_foreign_keys = (
        foreign_keys_row is not None and int(foreign_keys_row[0]) == 1
    )
    if restore_foreign_keys:
        conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at INTEGER NOT NULL
            )
            """
        )
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        current = int(row[0]) if row is not None else 0
        files = sorted(migrations_dir.glob("*.sql"))
        for path in files:
            prefix = path.name.split("_", 1)[0]
            if not prefix.isdigit():
                continue
            version = int(prefix)
            if version <= current:
                continue
            statements = _sql_statements(path.read_text(encoding="utf-8"))
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in statements:
                    conn.execute(statement)
                foreign_key_violation = conn.execute(
                    "PRAGMA foreign_key_check"
                ).fetchone()
                if foreign_key_violation is not None:
                    raise sqlite3.IntegrityError(
                        f"foreign key violation in migration {path.name}"
                    )
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, int(time.time() * 1000)),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            current = version
    finally:
        if restore_foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON")
        conn.isolation_level = previous_isolation


def repair_generation_lane(conn: sqlite3.Connection) -> None:
    """Add the lane columns and index of 0003_generation_lane.sql where they are missing."""
    previous_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(generations)").fetchall()
            }
            for name, statement in _GENERATION_LANE_COLUMNS:
                if name not in columns:
                    conn.execute(statement)
            conn.execute(_GENERATION_LANE_INDEX)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.isolation_level = previous_isolation
