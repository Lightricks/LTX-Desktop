# Generation telemetry schema (backend events)

Contract for the data pipeline behind per-hardware-class memory/timing
tables. Two events, joined on `(generation_id, attempt)` on queue rows and
on `generation_id` alone on direct rows (direct never sets `attempt`).
User-facing privacy copy lives in [`TELEMETRY.md`](./TELEMETRY.md).

Envelope on every event: `app_version`, `device_timestamp`,
`installation_id` (random per install, not a user), `platform`
(`mac` | `windows` | `linux`), `event_name`, `eventId`.

`launched` carries `ram_gb`, `gpu_name`, `os_version`, and (Linux only)
`os_distro` in `extra_details`. The sender sets them. A caller cannot.

`os_version` is the numeric part of the OS version from Electron's
`process.getSystemVersion()`: the macOS product version (`26.6.2`), the Windows
version (`10.0.26100`; build 22000 and above is Windows 11), or the Linux kernel
release (`6.8.0`). Any suffix is dropped. An absent key means unknown.

`os_distro` is Linux only. It is `ID-VERSION_ID` from `/etc/os-release`
(`ubuntu-24.04`), or only `ID` for a rolling release (`arch`). Only short
lowercase tokens are accepted. An absent key means unknown.

## `generate_started` — what was requested, on what hardware

Two surfaces emit this event with different vocabularies — the per-surface
columns below are the contract. Queue rows (`surface: home`) come from
`queued_generation_details` (full rows); direct rows (`surface: genspace`)
come from `desktop_generation_details` via `run_tracked_generation` (base
fields only).

| Field              | Type                                            | Queue (`home`)                                | Direct (`genspace`)                                            |
| ------------------ | ----------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------- |
| `generation_id`    | string                                          | Random per generation; join key (with `attempt`) | Random per generation; join key (no `attempt` — see below)  |
| `attempt`          | int                                             | Retry counter; join key                       | Never set                                                      |
| `client`           | `desktop` \| `remote`                           | Control surface (remote = LAN phone)          | Always `desktop`                                               |
| `requested_by`     | `user` \| `assistant`                           | `assistant` when the desktop assistant queued the job (its `feature` names the work), else `user` | Never set                          |
| `surface`          | string                                          | Always `home`                                 | Always `genspace`                                              |
| `feature`          | string                                          | e.g. `text-to-video`, `retake`, `extend`, LoRA recipe id | e.g. `text-to-video`, `retake`, `extend`, `ic-lora`, `text-to-image` |
| `kind`             | string                                          | Always `video`                                | `video`, or `image` on the image route                         |
| `execution`        | string                                          | Always `local`                                | `local`, `ltx_api` (video), or `fal_api` (image)               |
| `model`            | string                                          | Offering id (`ltx-2.5-fast`)                  | Pipeline id (`fast`, `pro`, …)                                 |
| `resolution`       | string                                          | Label passthrough (`720p`) on Fast creates; `WxH` when the create carries a `{width, height}` dict (edits); absent otherwise | Label (`720p`) on `/generate`; `WxH` on edit/image routes (`1280x704`); absent on IC-LoRA |
| `duration_sec`     | number                                          | Requested seconds (derived from frames when the create carries `numFrames`) | Requested seconds                    |
| `fps`              | number                                          | Frames per second                             | Frames per second (`/generate` only)                           |
| `lora_catalog_ids` | string[]                                        | Built-in adapters only                        | Built-in adapters only                                         |
| `has_custom_lora`  | bool                                            | True if a custom LoRA is present (never its path) | True if a custom LoRA is present (never its path)            |
| `prompt_provenance`| `raw` \| `manually-enhanced` \| `auto-enhanced` | Label only, never text                        | Label only, never text                                         |
| `output_count`     | int                                             | Absent (queue is video-only)                  | Image generations only                                         |
| `ram_gb`           | number                                          | Electron-appended total system RAM            | Electron-appended total system RAM                             |
| `gpu_name`         | string?                                         | Electron-appended OS-level GPU name (may be the iGPU on hybrids) | Electron-appended OS-level GPU name                 |
| `os_version`       | string?                                         | Electron-appended OS version (numeric part only)  | Electron-appended OS version (numeric part only)             |
| `os_distro`        | string?                                         | Linux only: Electron-appended `ID-VERSION_ID` (`ubuntu-24.04`) | Linux only: Electron-appended `ID-VERSION_ID`   |
| `gpu_model`          | string?                                       | Backend torch/CUDA device name — authoritative for CUDA class | Absent (direct-path enrichment is out of scope)      |
| `total_vram_bytes` | int?                                          | Discrete VRAM; the hardware-class key         | Absent (direct-path enrichment is out of scope)                |
| `free_vram_bytes`  | int?                                          | Free VRAM at start; capacity context          | Absent (direct-path enrichment is out of scope)                |
| `drive_type`       | `nvme` \| `ssd` \| `hdd`                      | Checkpoint drive; DISK-vs-CPU analysis needs it | Absent (direct-path enrichment is out of scope)              |

## `generate_ended` — how it finished, how long it took

Carries only what is known at finish; never repeats the start payload
(join for the rest).

| Field                      | Type                                                 | Notes                                                                                                                                                                |
| -------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generation_id`, `attempt` | —                                                    | Join keys on queue rows; direct rows carry `generation_id` only (`attempt` is never set there, so direct pairs join on the id alone)                                  |
| `outcome`                  | `succeeded` \| `failed` \| `cancelled`               |                                                                                                                                                                      |
| `runtime_ms`               | int ≥ 0                                              | Wall time, millisecond precision                                                                                                                                     |
| `error_code`               | string?                                              | Fixed label, never a message/path. `CUDA_OOM` is a policy-miss signal (admitted but didn't fit), distinct from other failures                                        |
| `load_mode`                | `full_models_loading` \| `streaming_models_loading`? | Whole model resident vs streamed                                                                                                                                     |
| `offload_mode`             | `NONE` \| `CPU` \| `DISK`?                           | Resident vs pinned-host vs mapped-from-checkpoint streaming                                                                                                          |
| `prompt_rewrite`           | `local` \| `gemini` \| `fal_expansion`?              | Who rewrote the prompt of a fal image run: local Gemma, Gemini, or fal's own expansion. Absent when no rewrite ran                                                  |


Jobs on a lane that does not use the GPU (fal) carry no hardware keys, because the
local GPU does not run them.

Modes ride success rows only: cancelled/failed ends carry null (the summary
never escapes the raise), as do edit rows (the retake pipeline does not
surface its decisions).

## How the tables get built

- **Cell key:** `(feature, model, resolution, duration_sec, fps)` — per
  surface, not uniform: `model` is an offering id on queue rows but a
  pipeline id on direct rows, and `resolution` mixes labels (`720p`) with
  pixel sizes (`1280x704`) depending on the path (see the table above).
  Follow-up: emit resolved WxH + offering id uniformly.
- **Hardware class key:** `(platform, gpu_model, total_vram_bytes bucket, ram_gb bucket, drive_type)`.
- **Wall model:** `runtime_ms` as a function of (cell, `load_mode`, `offload_mode`, class).
- **Policy-miss signal:** `outcome = failed` with `error_code = CUDA_OOM` on an admitted cell.
- **Nulls mean unknown** (best-effort snapshot; telemetry never blocks generation).
  Absent keys and explicit nulls are equivalent — the backend builder omits
  unknowns, Electron drops `undefined`.
- **Coverage:** all direct routes emit via `run_tracked_generation`, plus the
  home-queue runner. Full hardware/mode rows ride queue events only —
  direct-path enrichment is explicitly out of scope (2026-09-28 decision),
  and direct rows never carry `attempt`.
- One pair per queued generation; dev builds and opted-out installs send nothing.

## Code pointers

Python builders: `backend/services/analytics.py` (`queued_generation_details`,
`ENDED_DETAIL_KEYS`); emit sites: `backend/services/generation_queue/runner.py`
(queue, full rows) + `backend/_routes/*.py` via `run_tracked_generation`
(direct, base fields only). Key allowlists live in
`shared/analytics-ended-details.json` + `shared/generation-detail-keys.json`,
imported directly by `electron/analytics-payload.ts`; Python keeps its
runtime tuple pinned to the contract by test. Error codes:
`GenerationErrorCode` in `backend/api_types.py`.
