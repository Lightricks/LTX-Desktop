# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LTX Desktop is an Electron app for AI video generation using LTX models. Three-layer architecture:

- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind CSS renderer
- **Electron** (`electron/`): Main process managing app lifecycle, IPC, Python backend process, ffmpeg export
- **Backend** (`backend/`): Python FastAPI server (port 8000) handling ML model orchestration and generation

## Common Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server (Vite + Electron + Python backend) |
| `pnpm dev:debug` | Dev with Electron inspector + Python debugpy |
| `pnpm typecheck` | Run TypeScript (`tsc --noEmit`) and Python (`pyright`) type checks |
| `pnpm typecheck:ts` | TypeScript only |
| `pnpm typecheck:py` | Python pyright only |
| `pnpm backend:test` | Run Python pytest tests |
| `pnpm build:frontend` | Vite frontend build only |
| `pnpm build` | Full platform build (auto-detects platform) |
| `pnpm setup:dev` | One-time dev environment setup (auto-detects platform) |

Run a single backend test file via pnpm: `pnpm backend:test -- tests/test_ic_lora.py`

## CI Checks

PRs must pass: `pnpm typecheck` + `pnpm backend:test` + frontend Vite build.

## Frontend Architecture

- **Path alias**: `@/*` maps to `frontend/*`
- **State management**: React contexts only (`ProjectContext`, `AppSettingsContext`, `KeyboardShortcutsContext`) — no Redux/Zustand
- **Routing**: View-based via `ProjectContext` with views: `home`, `project`
- **IPC bridge**: All Electron communication through `window.electronAPI` (defined in `electron/preload.ts`)
- **Backend calls**: Always use `backendFetch` from `frontend/lib/backend.ts` for app backend HTTP requests (it attaches auth/session details). Do not call `fetch` directly for backend endpoints.
- **Styling**: Tailwind with custom semantic color tokens via CSS variables; utilities from `class-variance-authority` + `clsx` + `tailwind-merge`
- Frontend, Electron, and shared tests are `node:test` files. `pnpm run scripts:test` runs them. Which ones are worth writing is in [Tests](#tests).

## Backend Architecture

Request flow: `_routes/* (thin) → AppHandler → handlers/* (logic) → services/* (side effects) + state/* (mutations)`

Key patterns:
- **Routes** (`_routes/`): Thin plumbing only — parse input, call handler, return typed output. No business logic.
- **AppHandler** (`app_handler.py`): Single composition root owning all sub-handlers, state, and lock
- **State** (`state/`): Centralized `AppState` using discriminated union types for state machines (e.g., `GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled`)
- **Services** (`services/`): Protocol interfaces with real implementations and fake test implementations. The test boundary for heavy side effects (GPU, network).
- **Concurrency**: Thread pool with shared `RLock`. Pattern: lock→read/validate→unlock→heavy work→lock→write. Never hold lock during heavy compute/IO.
- **Exception handling**: Boundary-owned traceback policy. Handlers raise `HTTPError` with `from exc` chaining; `app_factory.py` owns logging. Don't `logger.exception()` then rethrow.
- **Naming**: `*Payload` for DTOs/TypedDicts, `*Like` for structural wrappers, `Fake*` for test implementations

### Backend Testing

- Integration-first using Starlette `TestClient` against real FastAPI app
- **No mocks**: `test_no_mock_usage.py` enforces no `unittest.mock`. Swap services via `ServiceBundle` fakes only.
- Assert the status, the body, the resulting state, or the persisted row. A call assertion on a fake is out.
- Fakes live in `tests/fakes/`; `conftest.py` wires fresh `AppHandler` per test
- Pyright strict mode runs in the CI typecheck job (`uv run pyright`). Pytest does not run it again.
- Which tests are worth writing is in [Tests](#tests)

### Adding a Backend Feature

1. Define request/response models in `api_types.py`
2. Add endpoint in `_routes/<domain>.py` delegating to handler
3. Implement logic in `handlers/<domain>_handler.py` with lock-aware state transitions
4. If new heavy side effect needed, add service in `services/` with Protocol + real + fake implementations
5. Add integration test in `tests/` using fake services

## TypeScript Config

- Strict mode with `noUnusedLocals`, `noUnusedParameters`
- Frontend: ES2020 target, React JSX
- Electron main process: ESNext, compiled to `dist-electron/`
- Preload script must be CommonJS

## Python Config

- Python 3.13+ (per `.python-version`), managed with `uv`
- Pyright strict mode (`backend/pyrightconfig.json`)
- Dependencies in `backend/pyproject.toml`
- Bumping `ltx-core` / `ltx-pipelines`: re-verify every `services.patches` import in `backend/ltx2_server.py` (private upstream monkey-patches). Drop a patch only if upstream now includes the fix.

## Key File Locations

- Backend architecture doc: `backend/architecture.md`
- Default app settings schema: `settings.json`
- Electron builder config: `electron-builder.yml`
- Video editor (largest frontend file): `frontend/views/VideoEditor.tsx`
- Project types: `frontend/types/project.ts`

## Tests

Before writing tests, list the cases you intend to cover and wait for approval.
Each line names one bug. Do not create or edit a test file until that list is approved.

Each test must correspond to a specific bug a reasonable developer could
introduce. Put that bug in the test name. Use a one-line comment only when the
name would be unreadable. If you can't name the bug, don't write the test.

Do NOT write tests that:

- assert a component renders without throwing
- assert a pure function returns its own obvious output (add(1, 2) === 3). A cap, an alignment grid, or None when over the cap is in scope
- verify that a mock you just configured was called with what you configured
- re-assert types the compiler already guarantees
- re-assert a Pydantic model, dataclass, or TypedDict field type that Pyright or the schema already enforces. A wire alias (currentStep present, current_step absent) and the error envelope (code, message) are in scope
- snapshot entire component trees
- test library behaviour (Zustand's set, React's useState, Radix internals)
- exist to raise coverage
- fail only when the entire module is deleted

DO write tests for:

- branch and boundary conditions (empty, single, off-by-one, null vs undefined)
- error paths and failure modes
- state transitions and their ordering
- anything with non-obvious business rules
- regressions: bugs previously fixed

Prefer fewer tests with real assertions over many shallow ones.
Test observable behaviour through the public interface.

When you touch a file that already has tests, delete any that break the rules
above. On legacy surfaces (GenSpace and similar), keep the smallest set that
still catches the cases above. Do not grow those suites.
