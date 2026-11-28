# Generation queue lanes

The queue runs jobs in **lanes**. A lane is a pool of capacity for one resource or one provider. It is never tied to a feature.

- Jobs in one lane run in queue order, up to the concurrency of the lane.
- Jobs in different lanes do not wait for each other.
- The lane of a job is set once, when the job is queued. It is stored in `generations.lane`.

| Lane | Concurrency | Uses the GPU | What runs there |
| --- | --- | --- | --- |
| `local` | 1 | yes | Every local job. Every job whose lane is not registered. |
| `fal` | 1 | no | Text-to-image jobs with `provider: "fal"`. |

## Rules

- Every executor must run correctly on the `local` lane. A lane that is not registered falls back to `local`. At boot, `reassign_unknown_lanes` moves the rows of an unregistered lane to `local`.
- The kill switch `LTX_PARALLEL_QUEUE_LANES=0` registers only the `local` lane. All jobs then run one at a time.
- The `local` lane keeps the global `GenerationHandler` slot. Sync routes (GenSpace, Edit) block only this lane.
- An executor reports progress, cancel and provider status through its `JobSession`. It must not read or write the global generation state. Only the `local` lane maps the session onto that state.
- Live facts of a running job (progress, `api_status`, `cancellable`) are never stored. The queue reads them from the running session (`QueueControl.live_state`).
- `QueueSnapshot.active` is a list. It has one entry for each running job.

## The provider job id

A job on a provider lane runs at a third party. That party has its own id for the job.

- The executor calls `JobSession.set_provider_job_id` right after the provider accepts the job.
- The queue stores the id in `generations.provider_job_id`. Only a running attempt takes it. The store checks the attempt, so a late write of an old attempt never lands on a retry.
- A job that failed or was cancelled keeps its id until a provider cancel succeeds (see Provider cancel). A succeeded job keeps it as a record.
- A retry takes the id off the row, so the row shows the newest attempt only. The old id moves to the file `displaced_provider_jobs.json` in the app data folder, and stays there until a provider cancel of it succeeds. The store writes the file before it commits the retry. A crash in between leaves the id in both places, and the readers list it once. If the file cannot be written, the retry fails and the job stays failed with its id. The file is deleted when no id is left. A file that cannot be parsed is moved to `displaced_provider_jobs.json.corrupt`.
- `set_provider_job_id` never raises for a store failure. It returns `False` when the id was not stored (a store error, or the attempt is no longer running). Boot cannot cancel such a job, so the executor must cancel it at the provider.
- An empty id is refused (`ValueError` in the store). It would make the boot orphan list unreadable.
- The id is in the API (`Generation.provider_job_id`), in the Remote API, and in the logs.
- A local job has no provider job id. It stays null.

## Provider cancel

`LaneSpec.cancel_orphan(provider_job_id)` asks the provider to drop a job. It must be safe to repeat, and it must treat "already finished" as success. It must bound its own network time.

- `JobSession.cancel` only signals the executor. It returns at once and does no network I/O, because the queue calls it while it holds the lane lock.
- After the executor returns for an attempt that failed or was cancelled, the runner calls `cancel_orphan` on its own thread. A slow provider never holds a worker.
- When the call succeeds, the id is cleared. When it fails, or the app stops first, the id stays on the row.
- `cancellable_while_running=False` is enforced. A cancel of a running job on such a lane gets `409 INVALID_GENERATION_STATUS`, and the job keeps running. The queue panel hides its Cancel button. The result card still shows one, and the click gets the 409.

A retry never loses an id that was not cancelled yet. The runner of the old attempt cancels it, and a boot cancels it when the call failed or the app stopped first. A deleted row keeps its id in the same way.

## Boot recovery

At boot, a job that was running is marked failed (`fail_running_on_boot`). The provider can still run it and bill it.

1. `QueuedGenerationHandler.recover_on_boot` fails the running jobs and moves the queued rows of unregistered lanes to `local`.
2. It reads every row that still holds a provider job id (`list_provider_orphans`): running, cancelling, failed and cancelled rows, deleted rows too. It also reads the old ids that a retry replaced (`displaced_provider_jobs.json`).
3. For each orphan, it calls `LaneSpec.cancel_orphan(provider_job_id)` of the lane, then clears the id. The calls run on a daemon thread, so a slow provider does not delay the boot.

A failed call is logged and does not stop the others. The id stays, so the next boot tries again. A lane that has no `cancel_orphan`, or is not registered, is skipped with a warning. Its rows keep the id, and the cancel happens at the first boot where the lane is back.

## Logs

Each log line names the lane and the generation id.

- `Generation <id> (<feature>) started on lane <lane>, attempt <n>`
- `Generation <id> left lane <lane> after <seconds>s`
- `Generation <id> accepted by its provider as job <provider job id>`
- `Cancelling provider job <provider job id> of generation <id> (lane <lane>)`
- `Provider job <provider job id> of generation <id> cannot be cancelled: lane <lane> has no cancel. The id stays on the row.`

## The fal lane

- The executor (`QueuedImageHandler`) calls `set_provider_job_id` with the fal request id right after fal accepts the request and before the first status check. If the id is not stored, it cancels the request at fal at once and the job fails.
- `FAL_LANE_SPEC.cancel_orphan` is `QueuedImageHandler.cancel_fal_request`. It sends `PUT` to the cancel URL of the request with the saved fal API key. `202`, `400 ALREADY_COMPLETED` and `404` count as done. Any other status raises, so the id stays for the next boot.
- fal builds its request URLs from the app id (`fal-ai/z-image`), not from the sub-path (`/turbo`). `ZitAPIClientImpl.cancel_queued` derives the URL from `FAL_QUEUE_APP`. A test compares it with the URL in a submit response fixture.
- The boot cancel needs the fal API key. When the key is removed, the cancel fails and the id stays.
- Under the kill switch a fal job runs on the local lane. That lane records no id and has no boot cancel.

## Add a lane

1. Add a `LaneSpec` in `services/generation_queue/lanes.py` and register it in `AppHandler`.
2. Add `lane_for(spec)` to the executor. It must be a pure function of the stored spec.
3. In `AppHandler._lane_runner`, give the lane a slot and a session factory. A lane that does not hold the GPU uses `CountingLaneSlot` and `LaneJobSession`.
4. Set `cancellable_while_running` to `False` only if a running job cannot be stopped.
5. Set `stop_grace_seconds` for a lane whose jobs the app cannot wait for at stop. The queue cancels the running sessions of such a lane.
6. Set `cancel_orphan` so that the boot can drop a job that the provider still runs.
7. Call `session.set_provider_job_id` in the executor, right after the provider accepts the job.

## Borrow the GPU from another lane

A job in another lane that needs the GPU for a short step (for example a Gemma prompt rewrite) must use `GenerationHandler.try_lease_idle_gpu`. The lease never waits. It succeeds only when no local job runs, no local row waits and nobody else waits for the GPU. A granted lease evicts the diffusion stage cache first, so the borrowed step does not load next to a cached transformer. When the lease fails, the job must use a provider-side fallback. A job that already holds the GPU slot (a fal job on the local lane under the kill switch) does not lease: `JobSession.holds_gpu_slot` is true.

## Prompt rewrite of a fal image job

The rewrite runs only when Explore auto-enhance is on and the prompt is not already enhanced. The code is `backend/handlers/image_prompt_plan.py`.

| Gemma downloaded | GPU | Gemini set up | Result |
| --- | --- | --- | --- |
| yes | idle | any | Local Gemma rewrite. `enable_prompt_expansion` is false. |
| yes | busy | any | fal expands (`enable_prompt_expansion` true). |
| no | n/a | yes | Gemini rewrite. `enable_prompt_expansion` is false. |
| no | n/a | no | fal expands. |

A failed rewrite (not a cancel) also falls back to fal expansion. fal bills the expansion.
