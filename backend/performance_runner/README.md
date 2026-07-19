# Performance Runner

A performance & validation harness for the LTX Desktop backend. It drives real
generations over HTTP, samples GPU VRAM (and host RAM) via `nvidia-smi` / psutil,
and reports on memory, VRAM fit per GPU tier, VRAM headroom (OOM margin), throughput
(realtime factor), thermal/throttle + energy per generation, cold-start latency, output
integrity (resolution/duration/fps/audio + file size/bitrate), and correctness — all from
a local dashboard (or the CLI).

Everything runs against the same backend the app uses, so what you measure is
what ships. Because LTX Desktop runs on the *user's own* GPU and shares their
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
  memory signal is **unified / host RAM** — free RAM (the number that gates local
  generation at server start), swap pressure, and how close the process sits to the MPS
  memory ceiling (driver-allocated vs recommended-max). VRAM fit tiers, headroom %, and
  the throttle heuristic need `nvidia-smi` and are CUDA-only.

## What it does

| Tool | Question it answers |
|---|---|
| **Soak** (`soak_test.py`) | Does memory leak across many back-to-back generations? Gates on a flat inter-generation VRAM **floor**; also tracks the host-RAM floor (spill), GPU temp/clock (throttle), throughput (× realtime), and energy per gen. |
| **Correctness A/B** (`output_ab.py`) | Fixed seed, a setting OFF vs ON — do the outputs match? (decoded-frame PSNR). |
| **Decompose** (`decompose.py`) | How much wall-time does a given optimization save per generation? |
| **Cold-start** (`coldstart.py`) | How long is the first generation after a fresh backend launch (model load / disk read / build) vs the warm steady state? |
| **Sanity sweep** (`sanity.py` + `scenarios.py`) | Does the whole feature surface (t2v/i2v/a2v/ia2v/t2i/extend/retake/LoRA/IC-LoRA) still generate end-to-end? Records each scenario's **peak VRAM** (which card tier it fits + headroom %), **throughput** (× realtime), **energy**, and the output's resolution/size/bitrate, and reports an advisory **integrity** check (output vs request — never fails the sweep). Scenarios that need a LoRA/IC-LoRA are **skipped if it isn't downloaded** — availability is read from the app's own library (`/api/loras` + `/api/ic-loras`, the same source as `lora_catalog.json`), so a scenario references a catalog **id** and its weight path resolves from that listing. `--fast` caps to 540p/5s for a quick surface check. Collects each scenario's input + output into a viewer. |
| **Dashboard** (`dashboard/`) | One local page to launch all of the above, flip toggles, watch logs, and chart trends. |

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
> startup from the hardware available *at that moment* — on Apple Silicon, the **free**
> (not total) unified RAM, since the GPU shares system memory. If it comes up in
> API-only mode, `perf:dev` stops and asks you to free RAM (close memory-heavy apps)
> and re-run, rather than opening a dashboard with no local work to measure.

> **`perf:dev` measures the dev backend** — the same interpreter `pnpm dev` uses, with
> the app's generation env mirrored (auth, app-data dir, MPS CPU-fallback). On CUDA that's
> representative of what ships. On Apple Silicon the packaged app loads a prebuilt zero-copy
> mps-sdpa extension that a dev run JIT-builds instead, so the attention path — and its
> memory profile — can differ from the released Mac build; treat Mac `perf:dev` numbers as
> indicative, and point `run_dashboard.py` at the **running app** for release-accurate Mac
> readings (at the cost of the app's own GPU/RAM contention).

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
  free RAM (the local-gen gate), swap, and the MPS ceiling.
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

## Test assets

`test_assets/` holds small, self-made fixtures used by the modality scenarios: a
cartoon reference image (i2v/ia2v), a short TTS speech clip (a2v/ia2v), and a
short reference video (extend/retake/IC-LoRA). Swap them freely.

## Security

The dashboard binds `127.0.0.1` only — it spawns processes and flips app settings,
so never expose it on a network interface.
