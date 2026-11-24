# Backend Architecture (FastAPI + AppHandler + State + Services)

This document describes the **Python backend** architecture and the contracts we rely on for:

- Distributing business logic and state management consistently.
- Keeping the codebase readable and easy to navigate.
- Preserving high-confidence testability through integration-style tests.

## Big picture

At runtime, the backend is a **local FastAPI server**. Endpoints are intentionally thin and delegate all work to a
single, shared **`AppHandler`** instance which owns:

- `RuntimeConfig`: constant environment configuration (treat as immutable).
- `AppState`: a centralized, typed, mutable state object.
- A shared lock used for safe state access/mutation under concurrency.
- A collection of domain-specific sub-handlers (the “business logic modules”).

High-level request flow:

```
HTTP request
  -> _routes/* (FastAPI endpoint function; minimal logic)
    -> AppHandler (injected via FastAPI Depends)
      -> handlers/* (domain logic + state transitions)
        -> services/* (side effects boundary: GPU, IO, network, etc.)
        -> state/* (AppState read/mutation under a shared lock)
```

## Composition roots and wiring

- `ltx2_server.py` is the **runtime composition root**:
  - Builds `RuntimeConfig`.
  - Builds the `AppHandler` via `build_initial_state(...)`.
  - Creates the FastAPI app via `create_app(handler=...)`.
  - Starts `uvicorn`.
- `app_factory.py` is the **FastAPI app factory** (importable from tests):
  - Registers exception handlers and CORS.
  - Includes routers from `_routes/`.
  - Calls `state.init_state_service(handler)` so routes can depend on the shared `AppHandler`.
- `state/deps.py` holds the FastAPI dependency hook:
  - `get_state_service()` returns the shared `AppHandler`.
  - Tests override it via `set_state_service_for_tests(...)`.

## Routes: API surface only

Routes live in `_routes/` and define the HTTP API (paths, request parsing, response models).

**Contract**

- Routes should be “plumbing”: parse typed inputs, call one handler method, return a typed output.
- Business logic and state mutation belong in `handlers/`, not in `_routes/`.
- Requests/responses should be strictly typed via Pydantic models from `api_types.py`.

Example pattern (using the stable health endpoint):

```py
@router.get("/health", response_model=HealthResponse)
def route_health(handler: AppHandler = Depends(get_state_service)) -> HealthResponse:
    return handler.health.get_health()
```

## AppHandler: the root “application service”

`AppHandler` (in `app_handler.py`) is the single object routes depend on. It is responsible for:

1. Owning the single `AppState` instance (centralized mutable state).
2. Owning a single shared lock used to protect state access/mutation.
3. Wiring sub-handlers and injecting them with:
   - `state` + `lock`
   - `RuntimeConfig` where relevant
   - Services required for side effects

`AppHandler` holds sub-handlers in composition (e.g. `handler.health`, `handler.models`, `handler.downloads`, ...).
Each sub-handler is responsible for one cohesive domain of operations.

### Service wiring and import-safety

To keep tests lightweight and imports safe:

- `build_default_service_bundle(...)` performs **lazy imports** of heavy runtime implementations.
- Tests pass a `ServiceBundle` containing fakes instead of importing GPU/network implementations.

## AppState: normalized and type-driven

`AppState` lives in `state/app_state_types.py` and is the canonical model of **mutable runtime state**.

**Design goals**

- Keep the state “normalized”: represent important state machines explicitly via *unions* rather than loose dicts.
- Maximize static integrity: prefer `Literal`, `Enum`, `Protocol`, and union types over dynamic runtime checks.
- Minimize runtime dependencies from state types (use `TYPE_CHECKING` + lightweight structural types where needed).

In practice, a large portion of the state is about **limited resource management** (e.g. downloads, which pipeline owns
the GPU, which generation is running, etc.).

Example: a state machine represented as a union of small dataclasses:

```py
GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled
```

This enables exhaustive matching and makes illegal states harder to represent.

## Concurrency model and the locking contract

### Why threads

This backend is optimized for a **single local client** with **heavy requests** (GPU/CPU work), rather than a high-QPS,
multi-tenant server.

Most endpoints are defined as synchronous `def` route handlers, which FastAPI/Starlette executes via a **thread pool**.
That means multiple requests can overlap in time even in a “single client” setting (e.g. progress polling, cancels,
downloads, settings updates, etc.).

### Locking rules

**All state access/mutation must be done with extra care.** The shared lock exists to prevent race conditions and torn
reads/writes.

**Rules of thumb**

- Any `AppState` read/write that influences decisions should happen under the shared lock (especially read-modify-write).
- Prefer handler methods decorated with `handlers.base.with_state_lock` for consistent locking.
- Do **not** hold the lock while doing heavy compute or slow IO.

#### Lock scope and “heavy work”

The most fragile point when implementing handlers is choosing the *locking scope* correctly:

- Locking too little risks inconsistent state transitions.
- Locking too much can serialize the whole server and block other endpoints for long periods.

When heavy work is involved, prefer this pattern:

1. **Lock** → read/validate state + compute a small “plan” / snapshot.
2. **Unlock** → perform heavy compute / IO using the snapshot.
3. **Lock** → re-check that the state is still compatible, then apply mutations.

Never assume the state stayed the same across `lock → heavy work → lock`.

## Handlers: business logic + state transitions

Handlers live in `handlers/`. They are the primary home for:

- High-level business logic.
- State transitions and resource management.
- Coordinating services (side effects) in a testable way.

**Contract**

- Handlers may mutate `AppState` (with locking).
- Routes should not mutate `AppState` directly.
- Handlers should not “fake” side effects; they should call services.

## Services: the test boundary for heavy side effects

Some side effects are not suitable for integration tests (GPU-heavy compute, network IO, etc.). We isolate those behind
**services** in `services/`.

**Contract**

- Services are the boundary between:
  - The runtime app (real implementations), and
  - The tested app (fake implementations).
- Services should be narrowly scoped to their side effect and avoid business/state logic.
- If a heavy side effect must be avoided in tests, it should be avoided **only** by introducing (or using) a service.

### Service interfaces and naming conventions

Each service should have:

- A Protocol interface (e.g. `HTTPClient`, `GpuInfo`, `FastVideoPipeline`)
- A real runtime implementation:
  - Use a concrete name when coupled to a specific implementation (e.g. `LTXFastVideoPipeline`)
  - Otherwise use an `*Impl` suffix (e.g. `HTTPClientImpl`)
- A fake implementation for tests: `Fake<ServiceName>` (e.g. `FakeHTTPClient`)

### Patches: temporary upstream monkey-patches

`services/patches/` holds modules that monkey-patch `ltx_core`/`ltx_pipelines` at import time to work around upstream
bugs or missing features (e.g. `safetensors_metadata_fix.py`, `pinned_pool_fix.py`). They are wired via side-effecting
imports in `ltx2_server.py` (`import services.patches.foo_fix  # pyright: ignore[reportUnusedImport]`), not through
`ServiceBundle`/DI — patching has to happen once, at process startup, before any pipeline code runs.

**Contract**

- Each import comment states the removal condition (e.g. `# Remove once ltx-core includes the fix`) — patches are
  meant to be temporary.
- Assert the patched symbol exists before rebinding it, so an upstream refactor fails loudly at startup instead of
  silently no-op'ing.

### “Payload” and “Like” conventions

- Report/DTO-like shapes commonly use a `*Payload` suffix (often `TypedDict`) to make “this is a data payload” obvious
  (e.g. `GpuTelemetryPayload`, `VideoInfoPayload`).
- To avoid heavy import dependencies, structural wrappers use a `*Like` suffix (e.g. `HttpResponseLike`,
  `VideoCaptureLike`).

## Exception Handling and Logging Policy

The backend uses a **boundary-owned traceback policy** to avoid duplicate stack traces and fragile per-handler
decisions.

### Request path policy

- `app_factory.py` owns request exception logging via centralized helpers in `logging_policy.py`.
- `HTTPError` with status `4xx` is logged as message-only (no traceback).
- `HTTPError` with status `5xx` is logged with full traceback.
- Unhandled `Exception` is logged with full traceback.
- Handlers should not call `logger.exception(...)` and then rethrow to the request boundary.

### Background task policy

- `ThreadingRunner` owns uncaught background exception logging via `logging_policy.log_background_exception(...)`.
- Background callers should pass `on_error` callbacks for state transitions and avoid duplicate traceback logging.

### Wrapping rules

- When converting one exception into another for propagation, use exception chaining:
  - `raise HTTPError(500, "...") from exc`
- This preserves causal stacks while keeping traceback emission centralized.

### Reference patterns

Request handler pattern:

```py
try:
    do_work()
except Exception as exc:
    raise HTTPError(500, "Operation failed") from exc
```

Background task pattern:

```py
task_runner.run_background(
    target=worker,
    task_name="model-download",
    on_error=lambda exc: set_error_state(str(exc)),
)
```

## Testing strategy: integration-first, mock-free

The backend aims for maximal coverage via **integration-style tests** in `tests/`:

- Tests create a real FastAPI app via `create_app(handler=...)`.
- Tests inject fakes via `ServiceBundle` when building the `AppHandler`.
- Tests call real routes using `TestClient` and assert on final outcomes.

**Contract**

- Do not mock/patch routes, `AppHandler`, or handlers.
- Fake only through services (by swapping service implementations).
- Prefer “behavioral” fakes that mimic the real contract in a lightweight way over `MagicMock`-style call assertions.

## File map (backend)

Primary entities:

- `ltx2_server.py`: runtime bootstrap (logging, `RuntimeConfig`, `AppHandler`, `uvicorn`)
- `app_factory.py`: FastAPI app factory (routers, DI init, exception handling)
- `_routes/*`: endpoint definitions (API surface)
- `api_types.py`: Pydantic request/response models (API typing contract)
- `runtime_config/`: immutable-ish runtime configuration models and constants
- `state/`: `AppState` and DI helpers (`get_state_service`, test overrides)
- `app_handler.py`: `AppHandler` composition root + service wiring (`ServiceBundle`)
- `handlers/*`: domain business logic + state transitions (sub-handlers)
- `services/*`: side-effect services (protocols + real implementations)
- `services/patches/*`: temporary upstream monkey-patches, imported for side effect in `ltx2_server.py`
- `tests/*`: integration-style tests + service fakes

**Modularity convention:** prefer many small files over umbrella modules (one route per `_routes/*.py`, one handler per
`handlers/*_handler.py`, one service per `services/<service>/...`).

## Adding a new feature (checklist)

1. Define request/response models in `api_types.py` (avoid `Any`/dynamic dicts when possible).
2. Add/extend an endpoint in `_routes/<domain>.py` that delegates to `handler.<domain>`.
3. Implement the domain logic in `handlers/<domain>_handler.py`.
   - Use the shared lock for state interactions.
   - Keep lock scope small around heavy work.
4. If you need a new heavy side effect, add a new service in `services/` and inject it via `ServiceBundle`.
5. Add an integration-style test in `tests/` using a fake service implementation (no mocking/patching).
6. Does this change follow boundary-owned traceback logging policy?
