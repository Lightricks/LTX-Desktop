# LTX Desktop

LTX Desktop is an open-source desktop app for generating videos with LTX models — locally on supported Windows NVIDIA GPUs, with an API mode for unsupported hardware and macOS.

> **Status: Beta.** Expect breaking changes.
> Frontend architecture is under active refactor; large UI PRs may be declined for now (see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md)).

<p align="center">
  <img src="images/gen-space.png" alt="Gen Space" width="70%">
</p>

<p align="center">
  <img src="images/video-editor.png" alt="Video Editor" width="70%">
</p>

<p align="center">
  <img src="images/timeline-gap-fill.png" alt="Timeline gap fill" width="70%">
</p>

## Features

### Generation
- **Text-to-video** — generate video clips from text prompts
- **Image-to-video** — animate a still image into video
- **Audio-to-video** — drive video generation from an audio track
- **Image generation** — create images with ZIT (local) or fal API
- **Image editing (img2img)** — edit existing images with ZIT Edit
- **Video Retake** — re-generate portions of an existing video
- **IC-LoRA** — identity-consistent generation with LoRA weights
- **Video Extend** — continue generating from the last frame of a video
- **Prompt Enhancement** — AI-powered prompt rewriting (via LTX API or Palette)

### Batch Generation
- **Batch Builder** — queue multiple generation jobs at once
- **List mode** — add prompts one-by-one with per-job settings
- **Import mode** — bulk-import prompts from CSV, JSON, or plain text files
- **Grid Sweep mode** — combinatorial parameter sweeps (prompts × seeds × models)
- **Timeline import** — import an edited timeline as a batch to re-generate all segments

### Library & Organization
- **Gallery** — browse, filter, and manage all generated images and videos
- **Prompt Library** — save, tag, and reuse favorite prompts
- **Characters** — store character descriptions for consistent generation
- **Styles** — save and apply visual style presets
- **References** — manage reference images for guided generation
- **Wildcards** — define placeholder tokens that expand to random values

### Palette Cloud Integration
- **Directors Palette sync** — connect to [Directors Palette](https://directorspal.com) for cloud-synced library content
- **Email/password login** — authenticate directly or via deep link
- **Credit balance & cost tracking** — view remaining credits in the header, see estimated cost on Generate buttons before submitting
- **Automatic credit deduction** — API-slot jobs automatically deduct credits after successful generation
- **Seedance video generation** — generate videos via Seedance 1.5 Pro through the Replicate API

### Editor & Export
- **Video Editor** — multi-track timeline editor with clips, transitions, and keyframes
- **Video Projects** — save and reopen editing sessions
- **FFmpeg export** — export final videos with configurable codec and quality settings

## Local vs API mode

| Platform / hardware | Generation mode | Notes |
| --- | --- | --- |
| Windows + CUDA GPU with **≥32GB VRAM** | Local generation | Downloads model weights locally |
| Windows (no CUDA, <32GB VRAM, or unknown VRAM) | API-only | **LTX API key required** |
| macOS (Apple Silicon builds) | API-only | **LTX API key required** |
| Linux | Not officially supported | No official builds |

In API-only mode, available resolutions/durations may be limited to what the API supports.

## System requirements

### Windows (local generation)

- Windows 10/11 (x64)
- NVIDIA GPU with CUDA support and **≥32GB VRAM** (more is better)
- 16GB+ RAM (32GB recommended)
- Plenty of free disk space for model weights and outputs

### macOS (API-only)

- Apple Silicon (arm64)
- macOS 13+ (Ventura)
- Stable internet connection

## Install

1. Download the latest installer from GitHub Releases: [Releases](../../releases)
2. Install and launch **LTX Desktop**
3. Complete first-run setup

## First run & data locations

LTX Desktop stores app data (settings, models, logs) in:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`

Model weights are downloaded into the `models/` subfolder (this can be large and may take time).

On first launch you may be prompted to review/accept model license terms (license text is fetched from Hugging Face; requires internet).

Text encoding: to generate videos you must configure text encoding:

- **LTX API key** (cloud text encoding) — **text encoding via the API is completely FREE** and highly recommended to speed up inference and save memory. Generate a free API key at the [LTX Console](https://console.ltx.video/). [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).
- **Local Text Encoder** (extra download; enables fully-local operation on supported Windows hardware) — if you don't wish to generate an API key, you can encode text locally via the settings menu.

## API keys, cost, and privacy

### LTX API key

The LTX API is used for:

- **Cloud text encoding and prompt enhancement** — **FREE**; text encoding is highly recommended to speed up inference and save memory
- API-based video generations (required on macOS and on unsupported Windows hardware) — paid
- Retake — paid

An LTX API key is required in API-only mode, but optional on Windows local mode if you enable the Local Text Encoder.

Generate a FREE API key at the [LTX Console](https://console.ltx.video/). Text encoding is free; video generation API usage is paid. [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).

When you use API-backed features, prompts and media inputs are sent to the API service. Your API key is stored locally in your app data folder — treat it like a secret.

### fal API key (optional)

Used for Z Image Turbo text-to-image generation in API mode. When enabled, image generation requests are sent to fal.ai.

Create an API key in the [fal dashboard](https://fal.ai/dashboard/keys).

### Replicate API key (optional)

Used for Seedance 1.5 Pro video generation. When enabled, video generation requests are sent to Replicate.

Create an API key in the [Replicate dashboard](https://replicate.com/account/api-tokens).

### Gemini API key (optional)

Used for AI prompt suggestions. When enabled, prompt context and frames may be sent to Google Gemini.

### Directors Palette account (optional)

Connect to [Directors Palette](https://directorspal.com) to sync library content (characters, styles, references), use cloud-based prompt enhancement, and track credit usage. Sign in via email/password in **Settings > Palette Connection**.

Credits are consumed when generating via API-backed models (cloud video, Seedance, cloud image). Local GPU generations are free. Credit balance and per-generation costs are displayed in the UI.

## Architecture

LTX Desktop is split into three main layers:

- **Renderer (`frontend/`)**: TypeScript + React UI.
  - Calls the local backend over HTTP at `http://localhost:8000`.
  - Talks to Electron via the preload bridge (`window.electronAPI`).
- **Electron (`electron/`)**: TypeScript main process + preload.
  - Owns app lifecycle and OS integration (file dialogs, native export via ffmpeg, starting/managing the Python backend).
  - Security: renderer is sandboxed (`contextIsolation: true`, `nodeIntegration: false`).
- **Backend (`backend/`)**: Python + FastAPI local server.
  - Orchestrates generation, model downloads, and GPU execution.
  - Calls external APIs only when API-backed features are used.
  - Output files follow the naming convention `dd_{model}_{prompt_slug}_{timestamp}.{ext}` for easy identification.

```mermaid
graph TD
  UI["Renderer (React + TS)"] -->|HTTP: localhost:8000| BE["Backend (FastAPI + Python)"]
  UI -->|IPC via preload: window.electronAPI| EL["Electron main (TS)"]
  EL --> OS["OS integration (files, dialogs, ffmpeg, process mgmt)"]
  BE --> GPU["Local models + GPU (when supported)"]
  BE --> EXT["External APIs (only for API-backed features)"]
  EL --> DATA["App data folder (settings/models/logs)"]
  BE --> DATA
```

## Development (quickstart)

Prereqs:

- Node.js
- `uv` (Python package manager)
- Python 3.12+
- Git

Setup:

```bash
# macOS
pnpm setup:dev:mac

# Windows
pnpm setup:dev:win
```

Run:

```bash
pnpm dev
```

Debug:

```bash
pnpm dev:debug
```

`dev:debug` starts Electron with inspector enabled and starts the Python backend with `debugpy`.

Typecheck:

```bash
pnpm typecheck
```

Backend tests:

```bash
pnpm backend:test
```

Building installers:
- See [`INSTALLER.md`](docs/INSTALLER.md)

## Telemetry

LTX Desktop collects minimal, anonymous usage analytics (app version, platform, and a random installation ID) to help prioritize development. No personal information or generated content is collected. Analytics is enabled by default and can be disabled in **Settings > General > Anonymous Analytics**. See [`TELEMETRY.md`](docs/TELEMETRY.md) for details.

## Docs

- [`INSTALLER.md`](docs/INSTALLER.md) — building installers
- [`TELEMETRY.md`](docs/TELEMETRY.md) — telemetry and privacy
- [`backend/architecture.md`](backend/architecture.md) — backend architecture

## Contributing

See [`CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## License

Apache-2.0 — see [`LICENSE.txt`](LICENSE.txt).

Third-party notices (including model licenses/terms): [`NOTICES.md`](NOTICES.md).

Model weights are downloaded separately and may be governed by additional licenses/terms.
