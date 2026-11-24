-- A deleted chat keeps its row as a tombstone, so a stale client cannot bring it back.
CREATE TABLE assistant_conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    on_hold    INTEGER NOT NULL DEFAULT 0,
    deleted    INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- `payload` is the run's JSON in the shape `payload_version` names; a run in an unknown shape loads as closed.
CREATE TABLE assistant_runs (
    id              TEXT PRIMARY KEY,
    payload         TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    conversation_id TEXT NOT NULL,
    -- The response status (waiting, paused, complete, stopped), NULL before the first response.
    status          TEXT,
    stopped         INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX idx_assistant_runs_conversation ON assistant_runs(conversation_id);
CREATE INDEX idx_assistant_runs_status ON assistant_runs(status) WHERE stopped = 0;

-- Generations the assistant queued. A job belongs to the chat, not to one message:
-- a retry from a later message moves its card there.
CREATE TABLE assistant_jobs (
    generation_id   TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    -- The run that queued it, and its job number there.
    run_id          TEXT NOT NULL,
    job_number      INTEGER NOT NULL,
    -- The run whose message shows its card, and that waits for and stops it.
    shown_in_run_id TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    UNIQUE (run_id, job_number)
);
CREATE INDEX idx_assistant_jobs_conversation ON assistant_jobs(conversation_id);
CREATE INDEX idx_assistant_jobs_shown_in ON assistant_jobs(shown_in_run_id);

CREATE TABLE assistant_queue (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    payload         TEXT NOT NULL,
    payload_version INTEGER NOT NULL,
    -- Run order within the chat, so messages can be reordered later without a migration.
    position        INTEGER NOT NULL,
    -- Why the message could not start, from its last attempt.
    error           TEXT,
    created_at      INTEGER NOT NULL
);
CREATE INDEX idx_assistant_queue_conversation ON assistant_queue(conversation_id, position);
