-- The lane a job runs in: a capacity pool such as local or a provider. Set once at insert.
-- Existing rows run on the local lane.
ALTER TABLE generations ADD COLUMN lane TEXT NOT NULL DEFAULT 'local';

-- The id of the job at the provider, for a job that runs on a provider lane. Set when the
-- provider accepts the job, cleared when the job is retried. NULL for a local job.
ALTER TABLE generations ADD COLUMN provider_job_id TEXT;

CREATE INDEX IF NOT EXISTS idx_generations_pending_lane_rank
    ON generations(lane, queue_rank ASC, id ASC)
    WHERE status = 'queued' AND deleted_at IS NULL;
