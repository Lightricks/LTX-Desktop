# Windows Setup And Run

This note captures public, repository-safe setup findings from a local Windows validation run.

## Verified workflow

- Node dependencies installed successfully.
- Backend Python dependencies installed successfully with `uv`.
- TypeScript typecheck passed.
- Python typecheck passed.
- Backend tests passed.
- Frontend and Electron production bundles built successfully.
- A runnable unpacked Windows app was produced and launched locally.

## Commands used successfully

```powershell
npx pnpm@10.30.3 install

Set-Location backend
uv sync --extra dev

Set-Location ..
npx pnpm@10.30.3 run typecheck:ts
npx pnpm@10.30.3 run typecheck:py
npx pnpm@10.30.3 run backend:test
npx pnpm@10.30.3 run build:frontend

& .\scripts\prepare-python.ps1
Set-Location release\win-unpacked
& '.\LTX Desktop.exe'
```

## Model directory behavior

The application stores a single effective `models_dir` setting and resolves all required model files relative to that root.

- Existing models are detected only when the expected canonical file or directory names are present under the configured root.
- Alternate filenames are not treated as equivalents.
- If a required canonical file is missing, the app downloads it into the configured root.
- If the exact canonical target already exists, the app skips downloading that target.

The current implementation is based on one canonical model root, not a recursive multi-layout search.

## Local build notes

### `pnpm` availability

In this validation environment, a global `pnpm` binary was not available on `PATH`.

Commands succeeded via:

```powershell
npx pnpm@10.30.3 <command>
```

### Windows packaging and signing

The checked-in Windows packaging configuration is oriented toward signed release builds.

During local validation:

- local packaging can fail if release signing credentials are not configured
- an unpacked Windows build can still be produced and run locally

### Updater noise in local unpacked runs

Unpacked local runs may log updater-related errors because release update metadata is not present in the local package. This did not prevent the application from launching during validation.

### Packaged icon path

The local unpacked build may log a missing icon path during startup. This appeared to be a packaging/runtime polish issue rather than a core application blocker during validation.