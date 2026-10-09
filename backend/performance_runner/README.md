# Performance Runner

A performance & validation harness for the LTX Desktop backend. It drives real
generations over HTTP, samples GPU VRAM (and host RAM) via `nvidia-smi` / psutil,
and reports on memory, VRAM fit per GPU tier, VRAM headroom (OOM margin), throughput
(realtime factor), thermal/throttle + energy per generation, cold-start latency, output
integrity (resolution/duration/fps/audio + file size/bitrate), and correctness — all from
a local dashboard (or the CLI).

Everything runs against the same backend the app uses, so what you measure is
what ships. Because LTX Desktop runs on the _user's own_ GPU and shares their
machine, the harness leans on desktop-shaped questions: will a config fit a 12 GB
card, does it leak into system RAM, how slow is the first generation after launch.

## Platforms

The harness adapts to the GPU it finds; the dashboard detects the platform (via the
backend's `/api/gpu-info`) and hides whatever isn't measurable on your machine.

- **CUDA (Windows / Linux, discrete NVIDIA)** — the full suite. Memory is discrete
  **VRAM** sampled with `nvidia-smi`: the soak's VRAM-floor leak gate, per-tier fit +
  headroom (OOM margin), and temp/clock throttle. The cache-patch and Torch-Compile
  toggles and the runs that depend on them (soak, pair, control, A/B, decompose) apply.
- **Apple Silicon (MPS)** — Torch Compile and the cache patch are no-ops here, so those
  toggles and the patch/compile-specific runs (and the soak-trend chart) don't appear.
  What runs: the **scenario sweeps** and **cold-start**. There's no discrete VRAM, so the
  memory signal is **unified / host RAM** — total RAM (the SKU number that gates
  local generation at server start), free RAM, swap pressure, and how close the
  process sits to the MPS memory ceiling (driver-allocated vs recommended-max).
  VRAM fit tiers, headroom %, and the throttle heuristic need `nvidia-smi` and
  are CUDA-only.

## What it does

| Tool                                            | Question it answers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Soak** (`soak_test.py`)                       | Does memory leak across many back-to-back generations? Gates on a flat inter-generation VRAM **floor**; also tracks the host-RAM floor (spill), GPU temp/clock (throttle), throughput (× realtime), and energy per gen.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Correctness A/B** (`output_ab.py`)            | Fixed seed, a setting OFF vs ON — do the outputs match? (decoded-frame PSNR).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Decompose** (`decompose.py`)                  | How much wall-time does a given optimization save per generation?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Cold-start** (`coldstart.py`)                 | How long is the first generation after a fresh backend launch (model load / disk read / build) vs the warm steady state?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Sanity sweep** (`sanity.py` + `scenarios.py`) | Does the whole feature surface (t2v/i2v/a2v/ia2v/t2i/extend/retake/LoRA/IC-LoRA) still generate end-to-end? Records each scenario's **peak VRAM** (which card tier it fits + headroom %), **throughput** (× realtime), **energy**, and the output's resolution/size/bitrate, and reports an advisory **integrity** check (output vs request — never fails the sweep). Scenarios that need a LoRA/IC-LoRA are **skipped if it isn't downloaded** — availability is read from the app's own library (`/api/loras` + `/api/ic-loras`, the same source as `lora_catalog.json`), so a scenario references a catalog **id** and its weight path resolves from that listing. `--fast` caps to 540p/5s for a quick surface check. Collects each scenario's input + output into a viewer. |
| **Dashboard** (`dashboard/`)                    | One local page to launch all of the above, flip the same Settings controls (active LTX version, Fast decode, cache, Torch Compile), watch logs, and chart trends.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

Every gate tool exits non-zero on FAIL, so the same scripts can gate CI.

## Quick start

### Headless (recommended for perf) — one command

```bash
pnpm perf:dev
```

Starts the backend **headless** (no Electron/frontend, so nothing else touches the
GPU), injects an auth token into both the backend and the dashboard, and opens the
dashboard in a browser. Ctrl+C stops both.

> Headless reuses the models the app has already downloaded/activated. If none is
> loaded, open the app once to fetch a model, then re-run.

> The runner measures **local** generation. The backend decides local-vs-API at
> startup from the hardware available _at that moment_ — on Apple Silicon, **total**
> unified RAM (32 GiB minimum), not free RAM. If it comes up in API-only mode,
> `perf:dev` stops because this machine is below the local-gen floor.

> **`perf:dev` measures the dev backend** — the same interpreter `pnpm dev` uses, with
> the app's generation env mirrored (auth, app-data dir, MPS CPU-fallback). On CUDA that's
> representative of what ships. On Apple Silicon the packaged app loads a prebuilt zero-copy
> mps-sdpa extension that a dev run JIT-builds instead, so the attention path — and its
> memory profile — can differ from the released Mac build; treat Mac `perf:dev` numbers as
> indicative, and point `run_dashboard.py` at the **running app** for release-accurate Mac
> readings (at the cost of the app's own GPU/RAM contention).

### Starting the backend by hand (Apple Silicon)

`pnpm perf:dev` and the app already do the right thing. If you launch `ltx2_server.py`
yourself, put the venv's `bin/` first on `PATH`:

```bash
cd backend
PATH="$PWD/.venv/bin:$PATH" LTX_APP_DATA_DIR=<dir> LTX_PORT=41954 LTX_AUTH_TOKEN=<token> \
  .venv/bin/python ltx2_server.py
```

Without it `ninja` is not found, the zero-copy attention backend (`mpsgraph_zc`) fails to
build, and the leaking `mpsgraph` fallback runs: driver memory climbs ~2 GB / 15 s and a
48 GB Mac is killed within a couple of generations. Check the first log line says
`picked=mpsgraph_zc active=True`. Details: `docs/mps-attention-memory-leak.md`.

### Against the running app

If the desktop app is already running, just launch the dashboard:

```bash
cd backend/performance_runner
uv run python run_dashboard.py          # cross-platform; also run_dashboard.sh / .ps1
# -> http://127.0.0.1:8750
```

Then paste the backend token (copy it from the app's **Logs** footer — the faint
`•••••`) into the dashboard's token field.

## The dashboard

- **Header** — backend/token status and live memory telemetry: on CUDA, GPU
  util / VRAM / temp / clock / power; on Apple Silicon, GPU name + unified-memory use,
  free RAM, swap, and the MPS ceiling.
- **Token** — paste it once; the header shows `token: set`.
- **Toggles** — reflect the backend's live settings (cache patch, Torch Compile);
  disabled while a run is in progress so they can't be flipped mid-run. Off CUDA
  (e.g. Apple Silicon), Torch Compile and the cache patch have no effect, so these
  toggles and the patch/compile-specific runs (soak, control, A/B, decompose) are
  hidden — only the scenario sweeps and cold-start apply there.
- **Run** — soak (cache on / control off / pair), A/B, decompose, cold-start,
  sanity (wired / all / fast), or a single scenario from the dropdown. Each asks
  for confirmation first; launch buttons disable while a run is active.
- **Scenarios** — a scenario whose LoRA/IC-LoRA isn't downloaded is greyed out and
  its `run scenario` button disabled, with a tooltip pointing at the app's library.
  Download it there, then hit **↻ refresh** to re-check without restarting.
- **Runs table** — live status, `N/50` progress, verdict, elapsed, and a **result**
  column linking a run to its input/output viewer once ready. Per-row stop +
  stop-all (confirmed). Survives a dashboard restart.
- **Log + chart** — polled log pane (run log or backend session log) and a
  scrollable soak floor/wall chart.

All artifacts (logs, soak CSVs, sanity results) land in `dashboard_runs/`.

## Interpreting a soak

> CUDA only — the soak, VRAM fit tiers, headroom %, and thermal throttle all need
> `nvidia-smi`. On Apple Silicon, use the scenario sweeps + cold-start and read memory
> from the header's unified / free-RAM + MPS-ceiling telemetry.

- **Floor slope ≤ 50 MiB/gen = PASS.** The inter-generation VRAM floor is the leak
  signal; flat/plateauing = the evict reclaims correctly, rising = a leak.
- **Wall-time drift is advisory, not a failure.** On a shared box (other GPU users)
  wall time inflates with no leak at all, and it can't be told apart from a real
  Windows shared-memory spill by wall time alone. If it drifts, re-run on an **idle**
  box; if it persists with the GPU otherwise idle, suspect a spill.
- **Host-RAM floor is advisory too.** A rising system-RAM floor across gens points at
  a leak/spill into host memory (affects the whole machine, not just VRAM) — the
  direct signal a wall-time drift only hints at.
- **Thermal throttle is advisory.** If the SM clock trends down while the GPU is hot,
  the summary flags `THROTTLING?` — another way wall-time can drift with no leak
  (common on laptops / small-form desktops). The floor is still the gate.
- Run a **cache-OFF control** of equal length to compare against.

## CLI

```bash
cd backend/performance_runner
uv run python soak_test.py --n 50 --label cache_on  --set-cache on
uv run python soak_test.py --n 50 --label cache_off --set-cache off   # control
uv run python soak_test.py --n 50 --pair                              # on + off + compare
uv run python output_ab.py --seed 42 --run
uv run python decompose.py --reps 5
uv run python coldstart.py --warm 3        # run right after a fresh backend start
uv run python sanity.py                    # wired scenarios; --only <key> / --include-unwired / --fast
uv run python sanity.py --model-id 2.5 --use-conv-vae   # same as Settings → Base model + Fast decode
```

Set `PERF_AUTH_TOKEN` (and optionally `LTX_PORT`) in the environment first. Each
gate script (`soak_test`, `output_ab`, `sanity`) exits non-zero on FAIL for CI use.

`ffprobe` (output-integrity checks) and `psutil` (host-RAM sampling) are optional:
if either is missing the harness degrades gracefully — that metric is just omitted.

## Configuration

All backend-specific wiring lives in **`perf_config.py`** (base URL/port, auth
token, the generation payload, and the settings toggles). Point the suite at a
different backend by editing that one file. `nvidia-smi` VRAM sampling and the
verdict logic are backend-agnostic.

### Adding a scenario

Add one `Scenario(...)` entry in **`scenarios.py`**; the sweep and the dashboard
pick it up automatically. For a scenario that needs a catalog LoRA/IC-LoRA, name its
**id** (from `lora_catalog.json`) in `required_loras` / `required_ic_loras` — the
`loras[].ref`, downloaded status, and UI availability all resolve from the app's
library via **`catalog.py`**. Non-catalog control weights (canny/depth union-control,
depth processor) go in `required_files` as paths relative to `models_dir`, checked on
disk.

## Live regression suite

Run this by hand after any big change (LTX-2 bump, pipeline or runtime refactor,
dependency upgrade) to confirm every core flow still generates **and still behaves
correctly**. It needs real weights, a GPU/Apple Silicon box and a running backend, so it
is not part of CI. Run it on each platform you ship (Mac, Windows).

```bash
cd backend/performance_runner
uv run python sanity.py --tags smoke                # one small run per flow and code path
uv run python sanity.py --tags smoke full bump      # + LoRAs, portrait, cancel on every route, full-size bump gate
uv run python sanity.py --only smoke_retake_silent_source   # any single scenario
```

The dashboard has **regression: smoke** and **regression: full** buttons, and every
scenario is in the single-scenario dropdown. Scenarios whose weights or model offering are
missing are reported as SKIP, never as PASS.

**Why smoke is small.** Local t2v at 270p/360p runs stage 1 only (the skip-stage-2 path);
540p and up runs the two-stage pipeline, so smoke has one of each, and keyframes on both.
Small canvases hide tiling/VRAM problems; the full-size `bump` scenarios cover those.

**What passes means** (`checks.py`, on top of "the file exists"): it decodes end to end
with the expected duration and audio, no frame is black, blown out, or frozen, the first
and last keyframes appear in the output, A2V and video-only retakes carry the source
soundtrack, retakes/extends keep the parts of the source they must keep, a different seed
gives a different video, and the **same request re-run at the very end reproduces the
first run** (catches state leaking from any flow in between: adapters, caches, cancel).

| Flow | Scenarios |
| --- | --- |
| t2v, both pipeline paths, seed handling | `smoke_t2v_270p`, `smoke_t2v_270p_seed_b`, `smoke_t2v_540p_2s`, `smoke_t2v_270p_repeat` |
| plain LoRA (`loras` in the request) | `smoke_lora_t2v` (Cozy Felt; skipped if not downloaded). Same request as the reference, so it must differ from it, and the final `_repeat` proves the adapter did not leak into later runs. Full: `lora_cozy_felt`, `lora_cozy_felt_openwheel` |
| i2v, keyframes (each path) | `smoke_i2v_270p`, `smoke_mkf_270p`, `smoke_mkf_540p` |
| audio-to-video | `smoke_a2v_270p`, `smoke_ia2v_270p` |
| image | `smoke_t2i` |
| extend / prepend | `smoke_extend_end`, `smoke_extend_start` |
| retake: every mode, silent source | `smoke_retake_av`, `_video_only`, `_audio_only`, `_silent_source` |
| IC-LoRA | `smoke_iclora_canny`, `_depth`, `_day_to_night` |
| cancel and recovery | `smoke_cancel_extend`; full: `full_cancel_a2v`, `full_cancel_iclora` |
| queued surface (smoke) | `smoke_q_t2v_270p`, `_i2v_end_270p`, `_a2v_270p`, `_extend_end`, `_retake_av`, `_batch_seeds` (3 jobs queued together), `_cancel_extend` |
| queued surface (full) | `full_q_t2v_540p_2s`, `_i2v_270p`, `_i2v_end_540p`, `_ia2v_270p`, `_extend_start`, `_retake_video`, `_retake_audio`, `_retake_silent_source` |

Notes:

- **Sleep:** `sanity.py` keeps the machine and its display on for the whole sweep, so no sleep, screen saver or idle lock screen (macOS `caffeinate -d -i`, Windows `SetThreadExecutionState`; released when the process ends, even if killed). Closing a laptop lid still sleeps it; keep the lid open or use power plus an external display.
- The thresholds in `checks.py` (frame/audio correlation, same-seed PSNR) are loose
  heuristics. The first Windows (RTX 5090) and macOS (M5 Max) sweeps cleared them with
  margin (see the `checks.py` docstring for the measured values). Each check reports its
  measured value; tighten the constants if a regression slips through. A failure may be a
  threshold, so read the detail before suspecting the app.
- `reproduces` / `differs_from` compare against the run recorded earlier **in the same
  sweep**. Run alone they **fail** ("no reference recorded"): an unverified comparison must not look
  green. Run them together with the scenario that records the reference.
- **Two surfaces.** The plain scenarios drive the synchronous GenSpace routes
  (`/api/generate`, `/api/extend`, `/api/retake`, `/api/ic-lora/generate`,
  `/api/generate-image`). The `*_q_*` scenarios (tag `queued`) drive the queued Home/Explore
  surface (`/api/generations/...`) via `queued.py`: input files are ingested as assets, the
  job is polled through the ledger, and cancel goes through `/generations/{id}/cancel`.
  Both end in the same pipelines but not the same request schemas, edit executors or ledger.
  Queued IC-LoRA and text-to-image have no queued route, so they are GenSpace-only. Run the
  queued surface alone with `--tags queued`. Feature tags (`retake`, `a2v`, ...) also select the
  older wired scenarios that carry them, not only the regression ones.
- The queued and GenSpace surfaces are not compared pixel for pixel: with local text
  encoding GenSpace rewrites the prompt and Explore sends it as typed (auto-enhance off), so
  the same seed yields different videos by design. Queued same-seed determinism is covered by
  `smoke_q_batch_seeds` (seed A, B, A).
- Queued scenario bodies are validated against the backend's `Create*Request` models in CI
  (`tests/test_perf_runner_queued.py`), so a schema change fails there first.
- Derived assets: `reference_video_silent.mp4` (`ffmpeg -an`), `reference_video_2s.mp4`
  (first 2 s), `reference_keyframe_b.jpg` (frame at 3.5 s), all from `reference_video.mp4`.

## LTX-2 bump gate

Use this sweep when bumping `ltx-core` / `ltx-pipelines`. `--fast` is **not** the
bump gate: it forces 540p/5s and IC-LoRA `resolution_factor=1.0`, which hides the
tiling/VRAM bugs these scenarios exist to catch.

```bash
cd backend/performance_runner
uv run python sanity.py --tags bump --gate-integrity
uv run python sanity.py --tags bump --model-id 2.5 --use-conv-vae --gate-integrity
uv run python soak_test.py --n 3 --label bump_e4 --set-cache on   # E4 stage-cache floor
```

| Tag / key                           | Experiment                  | Watch                                  |
| ----------------------------------- | --------------------------- | -------------------------------------- |
| `t2v_540p_8s`, `t2v_1080p_5s`       | E1 decode VRAM              | completes (no hang at device capacity) |
| those plus `last_frames_not_black`  | E2 MPS tail-frame zeroing   | each of last ~8 frames has non-zero luma (fails closed without ffmpeg) |
| `iclora_stage2_on`                  | E3 two-stage IC-LoRA        | completes with `skip_stage_2=false` (stage 2 is not forced onto streaming) |
| `soak_test.py --n 3 --set-cache on` | E4 stage-cache soak         | flat reserved-VRAM floor               |
| `mkf_interpolation`                 | guiding-latent swap         | output exists                          |
| `cancel_mid_denoise`                | interrupt hook              | generate returns `cancelled` after `phase=inference` |
| `iclora_day_to_night`               | shipped skip_stage_2 tiling | no OOM / crawl                         |

`--gate-integrity` fails a default-route scenario whose output duration/fps/requested
audio does not match the request (those diffs are advisory on a normal sweep).

## Test assets

`test_assets/` holds small, self-made fixtures used by the modality scenarios: a
cartoon reference image (i2v/ia2v), a short TTS speech clip (a2v/ia2v), and a
short reference video (extend/retake/IC-LoRA). Swap them freely.

## Security

The dashboard binds `127.0.0.1` only — it spawns processes and flips app settings,
so never expose it on a network interface.
