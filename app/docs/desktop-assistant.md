# Desktop assistant

The Explore assistant uses Pydantic AI with either a direct Gemini connection and
the user's configured key or local Qwen through app-managed llama.cpp.
Gemma 4 E2B is only used for prompt enhancement under Text Encoding settings.
The assistant does not use a hosted agent service or configure
Logfire. Video jobs use the existing local Explore generation handler, including
its encoding path. There is no cloud video generation tool.

## Flow

Assistant is a desktop navigation destination at `/assistant`. The full-page view
and optional sidebar share one mounted chat, including drafts, attachments, queued
messages and active requests. Asset tiles expose Ask assistant, which opens the
side panel with the asset attached. There is one assistant model choice, saved in
app settings and shared by every chat. Settings and the composer model selector, on
Desktop and Remote, all change it through `PUT /api/assistant/selection`, which
refuses Gemini without a key and local models that are not installed. Desktop leaves
the two fields out of its settings sync so it cannot overwrite a phone change.
`AppSettingsContext` re-reads `GET /api/assistant/selection` on focus and every 10
seconds for the whole app, so Settings and the composer stay current with the
assistant closed; Remote polls it on the same interval.
Each queued message captures its provider and model, so a later change cannot
reroute it. The active request's footer reflects its pinned provider.
Queued messages run in `position` order, so reordering can follow without a migration.
Stored runs and queued messages carry a `payload_version`. A run in an unknown shape,
or one whose JSON no longer parses, loads as a closed "This message can't be loaded."
message, so the rest of the chat still opens. A queued message in an unknown shape is
dropped. Recovery and Stop read the run's `status` column, not its JSON.

`POST /api/assistant/turn` accepts
`{request_id, conversation_id, text, asset_ids, provider, model}`
and runs the model/tool loop. `asset_ids` is optional, with up to eight existing
assets selected in the composer. Their IDs and metadata accompany the prompt, and
image/video thumbnails are shown to the model (see Images); local paths are never
sent. The tools expose installed
models, assets, recipes, recent generations, generation status and all queued
Explore video tools:

- `generate_video`: text-to-video parameters.
- `image_to_video`: image-to-video parameters with start/end asset IDs.
- `audio_to_video`: audio-to-video parameters; audio determines duration.
- `run_lora_recipe`: recipe ID, parameters and optional input assets, using the
  backend recipe registry. Covers the installed Explore styles/motions.
- `run_video_recipe`: a Home IC-LoRA recipe (Day to Night, Restore and the rest of
  `shared/ic-lora-recipes.json`) on a completed generation or a video asset, with an
  optional reference image. `list_video_recipes` reports prompt and image requirements
  and download status; the create handler owns the input-length cap. It runs at
  720p unless the model sets a resolution, and the field tells the model that.
- `retake_video` / `extend_video`: existing `RetakeParams` / `ExtendParams` with
  either a completed generation ID or an imported video asset ID. Retake takes no
  resolution, so it always keeps the source's frame size. Extend keeps it unless the
  user asks for a resolution.
- `run_ingredients`: queues the Ingredients sheet recipe from 1–10 image asset IDs,
  with Home's sheet resolutions, durations and LoRA strength (1.35). Ingredients is
  left out of `list_video_recipes`, and `run_video_recipe` sends it here, because it
  takes images, not a source video. The recipe nudge needs a cue word for it, as
  "ingredients" is an ordinary word.
- `extract_frame`: saves one video frame as a new image asset, inline (not queued),
  through the `AssetHandler` media transforms behind the trim routes. Omitting
  `at_seconds` takes the last frame, so "continue from the end" can feed
  `image_to_video`; the image can also be inspected with `view_asset`.

Generation tools return a queued job immediately. For several independent clips,
the agent calls the same tool repeatedly with `job_number` 1..N; there is no batch
tool or separate queue. One sequence of required job numbers is shared across every
generation tool, including retakes and extensions. Stable numbered IDs plus the
store's input-equality check make retries return the original job, including after
interrupted model calls. Jobs belong to the chat: `assistant_jobs` records the run
that queued each one, its job number, and the run whose message shows its card. A
retry from a later message moves the card there, so a job is never shown twice. The
run's payload and its job rows are saved in one transaction. Deleted jobs
cannot be recreated with the same ID, and changed inputs cannot reuse an existing ID.
A request may submit at most 12 jobs. The instructions and the job number's description
say so, and job 13 is refused with a message telling the model to say how many were not
queued. Only explicitly requested work is allowed.

Dependent edits use `wait_for_generation` and a successful source ID. A failed or
cancelled source blocks its dependent edit, but independent queued jobs continue.
The shared submission helper handles stable action IDs, duplicates and stopped runs.
Existing handlers own media validation, model availability, recipe
downloads/auth/device gates, and the generation pipeline. No model downloads,
filesystem import, settings changes, image generation or project tools are exposed.

`list_assets`, `get_asset` and `list_generations` are read-only discovery tools.
`list_assets` takes `sort: "newest" | "oldest"` (newest first by creation time by default), and
the instructions resolve "my last video" with `list_assets(media_kind="video", limit=1)` rather
than an ID from an older message. A job tool given an asset ID says so instead of "unavailable".
`show_assets` selects up to 30 existing assets for this reply, replacing any earlier
selection. Only those IDs are persisted as `asset_ids` and returned to the panel,
so searching broadly does not display unrelated results. The panel uses the existing asset query, runtime media
URLs and video player to display previews, without parsing model-generated links.
Replies render as Markdown with GitHub tables (`remark-gfm`); images and links stay plain text.
Displayed IDs remain in conversation context for follow-up selections.
The attach button opens a menu: Upload from device (the file chooser for images, audio
or video) or Choose from assets (the asset picker, all kinds). Users can also drop app
assets/files into the composer. Files go through the existing importer before attachment, and
videos stay videos (no audio extraction); ambiguous selections require
clarification. `list_lora_recipes` exposes
recipe requirements and installed-weight status; `get_recipe_guidance` returns one recipe's
catalog prompting rules and examples, the same text the prompt enhancer reads, and names
the tool that queues it. All results remain visible in the conversation,
loaded by generation ID rather than feature-specific lists.

### Tool rules

Assistant jobs follow the Auto-enhance setting, as Home does. With it on they are
marked `typed`, so the queue's enhancer rewrites the model's scene before any recipe
wrap. With it off they are marked `enhanced`: the scene is sent as written, and a
recipe still gets its wrap. The final prompt is logged at debug level. The tools
also apply what the enhancer would otherwise add, so a job is complete without it:

- Style recipes: `run_lora_recipe` adds the LoRA trigger when the recipe template
  lacks it, so guidance omits the trigger line.
- Templated video recipes (Deblur, Colorization, Decompression, Water Simulation):
  guidance shows the template, the model passes `template_values`, and
  `run_video_recipe` fills the template with `fill_prompt_template`, as Home does.

A message that names a recipe sends `generate_video` and `image_to_video` calls,
and for video recipes `extend_video` and `retake_video` calls, back once per step
with the recipe tool to use. A second identical call goes through, for a user who
wants no recipe or a plain job beside the recipe one. Once a recipe job is queued in
the request the nudge stops, so "one Cozy Felt clip and one plain clip" queues both as asked.

Settings are checked before queuing, with a retry that names the rejected value and
lists what is available: resolution, duration, FPS, aspect ratio and camera motion
against `list_video_models`, and the audio-to-video equivalents. For new clips,
`resolution` has no default, so a call without it is sent back rather than queued at
a guessed size.

Submit tools get 3 retries (4 for the tools the recipe nudge can refuse). When a tool
runs out, the turn ends normally with its last error as the reply. That error closes
the open call in the history, and jobs already queued keep running. `list_video_models` marks Home's default model
(the Settings checkpoint, else the first installed) and omits user LoRAs and
multi-keyframe, which no tool can set. When the user names no setting, the
instructions reuse the previous clip's settings in the chat (`_previous_settings`).

Lengths are read from the source asset:

- `extend_video` takes `added_seconds` or `total_seconds`; for a total, the tool
  subtracts the source length.
- `retake_video` refuses a range past the source end and gives the start time that fits.
- `audio_to_video` refuses audio longer than this resolution allows, which the queue
  would cut, and names the resolutions that cover all of it.

Missing recipe weights, a required Hugging Face sign-in and an unsupported model
return a specific message (`_RECIPE_BLOCKERS`) telling the model to ask the user
rather than retry or switch to another recipe.

Waiting defers the agent run. SQLite stores the request, typed model history,
pending wait calls and all submitted job IDs. The panel displays every job using
`generation_ids` and polls `GET /api/assistant/conversations/{id}`.
The backend owns one continuation task per waiting request. The queue's finished
listener wakes it when a job succeeds or fails; a 10-second fallback check covers
cancellations and removals, which the queue does not report. It does not call the
model while jobs are active. Continuations share the request claim with client calls:
a client continuing or replaying a request that a continuation is advancing waits for
that step and receives its result instead of a conflict. Explicit
dependency waits track the specific source ID instead. Once terminal, actual results
return to the model so it can report the outcome or continue requested edits.
A job the user cancels from the queue is final: the request waits for its other jobs,
then closes with a summary built in code, without calling the model. A tool call that
reuses a cancelled job's number is refused.
This continues without a connected frontend, provided the backend is running.
Backend startup restores waiting requests from SQLite; completed, stopped and paused
requests are not restarted. Background failures persist a paused response for explicit
retry. Shutdown cancels assistant tasks before shutting down the local runtime.
If saving a paused state fails, the continuation retries persistence with backoff
without rerunning the model. Status and retry requests report recovery in progress
until it finishes. Invalid saved runs are isolated during startup recovery so other
waiting requests can still resume.
An interrupted first step without a saved waiting response remains manually retryable.

The stop endpoint records a durable stopped state in one store update and cancels all active owned jobs,
without cancelling unrelated jobs or external sources being waited on.
Stop is available during inference. The handler cancels the active agent task;
The local runtime exits before releasing its GPU reservation.
Tool submissions retain their post-enqueue cancellation check to close races.
Stable request and generation IDs prevent duplicate enqueue on request retries.
Each step allows sixteen model requests and twenty-four tool calls. A new step only
starts when jobs finish, and a request has at most twelve jobs, so a run needs no
separate step count. Argument correction is bounded. Only INTERRUPTED jobs have an
automatic retry tool, limited to one retry with unchanged parameters. When the user
asks, `retry_failed_job` reruns a failed job from any message in the chat, in place
(same job and card), at most once per job per request. A job that failed in the request
that queued it is refused: the user has not seen that failure yet, so cannot have asked.

The model receives conversation text, model capabilities and generation
metadata, including asset names, kinds, dimensions/duration and IDs. A successful
generation confirms production, not visual quality.

### Images

Gemini and Qwen both receive images, the same way. History stores asset IDs only;
`PreviewModel` (`services/assistant_media.py`) adds pixels to a copy of each
outgoing request, so saved runs and the history byte budget never contain them.
Each image is the asset's app thumbnail re-encoded as a JPEG of at most 512 pixels
per side and 64 KB, never the full file. Audio has no preview and keeps metadata only.

- Attachments on the current message are shown on every model request of that run,
  including resumes after jobs finish.
- Earlier attachments and generated results are not shown automatically. The model
  calls `view_asset(asset_id)`, and the preview follows that tool result for the rest
  of the current step. A later step can view it again.

`GET /api/assistant/runs/{id}/events` streams SSE snapshots of tool activity and
partial replies. Gemini and Qwen stream reply text. Reconnection receives the current snapshot.
The stream is pushed on change (with a 15-second heartbeat), not polled. It ends with a
`done: true` event once the run is complete, stopped or paused; waiting runs keep it open
because they resume on their own. The panel reopens it when a paused run is continued.
The async turn/continue endpoints do not occupy a worker thread while awaiting model
IO. SQLite, asset and preview work before and after each model call runs in worker
threads, not on the event loop. Accepted model steps are shielded from HTTP request cancellation; disconnecting
the request or event stream does not cancel the work. Stop does.

## Conversation context

SQLite is the source of truth for visible chat and structured model history.
`GET /api/assistant/conversations` lists the 100 most recently updated conversations,
titled by their first message, and `GET /api/assistant/conversations/{id}` restores
their turns. The panel's chats menu switches between them, starts a new chat, or
deletes the current one. Switching leaves runs going in the background. The menu
disables only while a delete is in progress. Deleting
removes the chat's runs, not the videos they made, and is refused with 409 while any
of its runs is still working or waiting. Each chat has a row in `assistant_conversations`
(title, queue hold, last update). Deleting keeps that row as an untitled tombstone, so a
message queued from a stale client gets 404 instead of bringing the chat back.
The full page URL is `/assistant/{chatId}`.
The docked panel opens the most recent chat. The browser does not keep a copy of the
chat or its queue. Each turn sends
that ID, a fresh request ID, the new user message and optional attachment IDs.
The backend uses an indexed lookup for the latest saved run and carries forward
Pydantic AI's structured history, including tool arguments and results. A fresh
request still gets its own job identities, retry budget and cancellation scope.
Resuming or retrying a request uses its saved history, not a later conversation head.

Follow-up messages queue in SQLite and are sent one at a time. Enter sends or queues;
Shift+Enter adds a newline. Stop holds the queue until it is released, which is separate
from a paused run. A queued message that cannot start (a deleted attachment, a missing
key) also holds the queue; its reason is stored in `assistant_queue.error` and shown
under the message. The user can release the queue or remove a queued message. Attachments and queued messages
travel with their respective request; importing files blocks sending until finished.
Stopped-run metadata is input context, never a synthetic assistant reply.

Only one request may advance per conversation at a time. A waiting, paused or failed
request must finish or be stopped before the next turn. SQLite enforces this when the
next run is inserted: the insert only succeeds if the conversation's latest run is
still the one its history was built from. In process, one task registry per run
allows one step at a time. Stopped deferred tool calls receive
an interrupted result before the next message, so the history remains well-formed.
New chat clears the ID and starts without earlier context.

`assistant_parallel_gemini` (off by default, no UI; set it in `settings.json`) frees
the chat sooner with Gemini, which uses no GPU. A Gemini run whose reply is written
completes at once instead of waiting for its jobs, so the next message starts. No
continuation is scheduled, so the history stays one chain. Its jobs report on their
job cards, with no closing message and no automatic retry after a restart. Local runs,
and runs that called `wait_for_generation`, still wait.

Before every model call, including tool-loop steps and resumes, the history processor
limits serialized history to 32 KB for local models and 128 KB for Gemini. These are
payload bounds, not exact token counts; instructions, tool schemas and response
headroom remain separate. Older completed tool calls and results are reduced in
place to identities and outcomes first, preserving call/result pairs and a valid
final request. If necessary, complete older exchanges are dropped while retaining
compact references to the previous subject and current jobs. The current user request,
unanswered calls and submitted job-number references remain. The two latest tool
results stay intact when space allows; under heavier pressure, they also become
compact references. Repeated identical tool results are deduplicated.
Details can be fetched again by ID; the durable job map and submission idempotency
are unchanged. A current message or unanswered call that cannot fit even after
compaction receives an input-size error; already-created jobs remain saved.
Discovery pages return at most ten assets, and recipe listings exclude prompt
templates; only guidance for a templated video recipe shows one. Earlier run records remain in SQLite; compaction does not add
long-term memory.
Retry sends the request saved with the conversation. A retry of the exact call that is still
running (for example after a reload during a long request) reattaches and waits for
its result; a different call on the same run gets 409. Backend run records
are retained because they also protect retries; no automatic retention deletion is enabled.

## Adding a tool

1. Add a typed function decorated with `@tools.tool` in
   `backend/handlers/assistant_tools.py`. Its docstring describes its purpose.
2. Reuse shared enums, input references and existing handlers. Tool parameter models
   in `assistant_params.py` offer only values local models run, and omit arbitrary
   LoRAs and provenance; `api_params` converts them to the existing API models.
   Add dependencies to `AssistantDeps` only when necessary.
3. Use `_submit` with the existing create handler and job number for queued work.
   It passes the request's `origin` (`desktop`, or `remote` for the phone routes) as
   `analytics_client` through the existing consent-aware path.
   The shared tool hook checks stopped state and waits for in-flight submissions
   during cancellation. Use explicit waits for dependencies; long-running
   work must not hold a model connection.
4. Add an integration test in `backend/tests/test_assistant.py`. The fake model
   scripts SDK messages; the real agent loop and tools execute against service
   fakes. No separate JSON schema or frontend tool switch is needed.

## Assistant provider

Settings > General > Assistant selects Gemini API (the default) or a local model
(Qwen 3.5 9B). It is the same shared choice the composers change.
This is independent of text encoding and prompt enhancement. Gemini reuses the
existing API key and model selection; local mode needs no key and never falls back
to the cloud. All models use exactly the same tools, agent loop and queue.

The LAN remote mounts the assistant chat routes. Model download and delete stay on the Desktop.

Gemini requests retry up to 3 times with a short backoff on 429 and 5xx responses
(for example, "high demand" 503s). If they still fail, the step reports a specific
error: `GEMINI_UNAVAILABLE`, `GEMINI_RATE_LIMITED`, `GEMINI_TIMEOUT` or
`GEMINI_AUTH_FAILED`. Only the failed model request is retried, so tool calls
never run twice.

For Qwen, download the 9B from
Settings > Models > Assistant models. The catalog uses the same table, download
controls, progress and installed indicator as the other model sections. Downloads
run in the backend and reattach when Settings is reopened. The existing Hugging
Face downloader handles model transfers. SHA-256 and size verification must pass
before a checkpoint is installed. Downloads include the matching F16 vision projector
and the required runtime. Existing verified language-model weights are reused.
A model is selectable only once both weights and projector are verified; deleting
it removes both files. The two Qwen projectors have separate model-specific folders.
Only installed models can be selected in General > Assistant. The backend checks
the model before each local assistant step, since files can be removed outside the app.

The app installs a pinned, checksum-verified llama.cpp release under app data
(`runtimes/`): Metal on Apple Silicon, Vulkan on Windows x64 and Linux x64. Linux
needs the system Vulkan loader (`libvulkan1`, a dependency of the deb). NVIDIA on Linux also
uses Vulkan: llama.cpp's Linux CUDA build needs a separate ~600 MB CUDA runtime archive. No external app,
Python package compilation, or user-managed server is required. Model files are
Q4_K_M GGUFs under the configured models directory (`assistant/`), shared across
platforms. Runtime/model versions and hashes live in `llama_runtime.py` and
`assistant_models.py`; updates must be verified together, including tool calling.
Runtime support does not replace platform testing: verify local inference and
cancellation on Apple Silicon, Windows Vulkan and Linux Vulkan before release.

Inference binds only to a temporary loopback port with a per-process random API
key. Cloud credentials and inherited llama configuration are not forwarded.
The runtime loads once per agent step and remains resident through its tool loop.
Qwen holds the shared GPU reservation and evicts any resident video
pipeline before loading. Queued videos start after the step releases its weights
and reservation. If a video is running, Qwen waits for it with no deadline, but only
for that video: the queue holds further jobs at the next boundary so Qwen goes next.
Stop ends the wait. The reservation does not expire during active inference or
cleanup. Model runtime, download information and display names are catalogued in
`assistant_models.py`; a new local model is one more entry there and in `LocalAssistantModel`.
Qwen thinking is disabled, context is bounded to 16K, and the existing request,
run and tool-call limits still apply. A small parent-pipe watcher stops the runtime
on normal shutdown, backend crashes, and forced backend termination.

The runtime receives the matching projector with `--mmproj`. Pydantic AI sends
preview bytes as inline image content to the authenticated loopback endpoint.
Video inspection is currently a single thumbnail, not temporal understanding or
audio analysis.

Provider and model are saved with each run. Resumes always use that saved selection,
even after settings change, preventing an accidental local-to-cloud switch. New
requests explicitly capture the selected provider and model and
include the saved conversation history and tool results, so switching to Gemini
intentionally sends that context to Gemini. Start a new conversation to leave it behind.

`AssistantProvider` remains the sole model transport boundary. There is no separate
local tool implementation or MCP layer.

Shared backend limits live in `backend/assistant_limits.py`. Conversation lookup and stopped-state guards use indexed
SQLite columns. Run records are retained for retry safety; no automatic deletion
policy is enabled yet.

Model-call and tool-call ceilings pause the current request. Polling
does not restart a paused run: Continue explicitly grants another bounded step via
`POST /api/assistant/runs/{id}/continue`. Submitted job IDs and tool
history survive both limits and failed model calls. An input that cannot fit after
compaction ends the current turn with a shorter-follow-up message, leaving the
conversation usable and submitted jobs intact. Stop can be repeated to retry failed
job cancellations; it still attempts the other owned jobs if one cancellation fails.
