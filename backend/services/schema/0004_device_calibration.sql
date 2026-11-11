CREATE TABLE IF NOT EXISTS device_calibration_frontier (
    id                  INTEGER PRIMARY KEY,
    epoch               INTEGER NOT NULL CHECK (epoch > 0),
    system              TEXT    NOT NULL CHECK (system IN ('Windows','Linux','Darwin')),
    gpu_model           TEXT    NOT NULL CHECK (gpu_model <> ''),
    total_memory_bytes  INTEGER NOT NULL CHECK (total_memory_bytes > 0),
    model_id            TEXT    NOT NULL,
    feature             TEXT    NOT NULL CHECK (feature IN ('fast','edit','iclora')),
    adapter_id          TEXT    NOT NULL DEFAULT '',
    stage_mode          TEXT    NOT NULL DEFAULT '',
    load_mode           TEXT    NOT NULL CHECK (load_mode IN ('full_models_loading','streaming_models_loading')),
    tiled               INTEGER NOT NULL CHECK (tiled IN (0,1)),
    chunked             INTEGER NOT NULL CHECK (chunked IN (0,1)),
    seq                 REAL    NOT NULL CHECK (seq > 0),
    canvas_px           INTEGER NOT NULL CHECK (canvas_px > 0),
    windows             INTEGER NOT NULL CHECK (windows >= 1),
    peak_growth_bytes   INTEGER CHECK (peak_growth_bytes IS NULL OR peak_growth_bytes >= 0),
    -- Wall seconds of the run, for the machine factor. NULL when the run was not timed.
    wall_s              REAL    CHECK (wall_s IS NULL OR wall_s > 0),
    -- Fast decode the run used: 1 ConvVAE, 0 DiffVAE.
    conv_vae            INTEGER NOT NULL DEFAULT 0 CHECK (conv_vae IN (0,1)),
    -- Spatial tiles per step of the run. The ETA multiplies the denoise time by it.
    tiles               INTEGER NOT NULL DEFAULT 1 CHECK (tiles >= 1),
    priced_gib          REAL    NOT NULL,
    -- Provenance only; no FK, so deleting history keeps the device fact.
    generation_id       TEXT,
    recorded_at         INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_device_calibration_frontier_bucket
    ON device_calibration_frontier(
        epoch, system, gpu_model, total_memory_bytes, model_id,
        feature, adapter_id, stage_mode, load_mode, tiled, chunked
    );
