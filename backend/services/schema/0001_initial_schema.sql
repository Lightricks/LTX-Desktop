CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    id                TEXT PRIMARY KEY,
    feature           TEXT NOT NULL,
    contract_version  INTEGER NOT NULL CHECK (contract_version > 0),
    status            TEXT NOT NULL CHECK (status IN ('queued','running','cancelling','succeeded','failed','cancelled')),
    spec              TEXT NOT NULL CHECK (json_valid(spec)),
    error_code        TEXT,
    created_at        INTEGER NOT NULL,
    queued_at         INTEGER NOT NULL,
    queue_rank        INTEGER,
    attempt_count     INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    started_at        INTEGER,
    finished_at       INTEGER,
    deleted_at        INTEGER,
    CHECK (
        (status = 'queued' AND queue_rank IS NOT NULL)
        OR (status != 'queued' AND queue_rank IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_generations_feature_created_at
    ON generations(feature, created_at DESC, id DESC)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_generations_queue
    ON generations(status, queued_at ASC, id ASC)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_generations_pending_queue_rank
    ON generations(queue_rank ASC, id ASC)
    WHERE status = 'queued' AND deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS assets (
    id                      TEXT PRIMARY KEY,
    media_kind              TEXT NOT NULL CHECK (media_kind IN ('image','video','audio')),
    origin                  TEXT NOT NULL CHECK (origin IN ('uploaded','generated')),
    producer_generation_id  TEXT REFERENCES generations(id) ON DELETE RESTRICT,
    output_ordinal          INTEGER CHECK (output_ordinal >= 0),
    mime_type               TEXT NOT NULL,
    name                    TEXT NOT NULL,
    metadata                TEXT NOT NULL CHECK (json_valid(metadata)),
    thumbnail_path          TEXT,
    created_at              INTEGER NOT NULL,
    deleted_at              INTEGER,
    CHECK (
        (
            origin = 'uploaded'
            AND producer_generation_id IS NULL
            AND output_ordinal IS NULL
        )
        OR
        (
            origin = 'generated'
            AND producer_generation_id IS NOT NULL
            AND output_ordinal IS NOT NULL
        )
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_assets_producer_ordinal
    ON assets(producer_generation_id, output_ordinal)
    WHERE producer_generation_id IS NOT NULL AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_assets_created_at
    ON assets(created_at DESC, id DESC)
    WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS paired_devices (
    id                     TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    token_hash             TEXT NOT NULL UNIQUE,
    created_at             INTEGER NOT NULL,
    last_seen_at           INTEGER NOT NULL,
    first_seen_ip          TEXT,
    first_seen_user_agent  TEXT,
    revoked_at             INTEGER
);
CREATE INDEX IF NOT EXISTS idx_paired_devices_token_hash
    ON paired_devices(token_hash) WHERE revoked_at IS NULL;
