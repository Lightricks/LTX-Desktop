# Implementation Plan: ComfyUI Integration

This plan outlines the steps required to implement the ComfyUI integration described in the High-Level Design (HLD), adhering to the principles of "Additive Isolation" to minimize fork impact.

## Phase 1: Foundation (Settings & API Types)

**Goal:** Extend the core data structures to support routing and dynamic payload parameters without breaking existing schemas.

1.  **Update Settings:**
    *   Modify `settings.json` and `backend/state/app_settings.py` to add `generation_backend` (defaulting to `"local"`).
2.  **Update API Types:**
    *   In `backend/api_types.py`, add `workflow_params: dict[str, Any] | None = None` to generation requests (e.g., `VideoGenerationRequest`).
3.  **Update State Types:**
    *   In `backend/state/app_state_types.py`, define a new `ComfyUIJobSlot` (tracking status, progress, current job ID).
    *   Add `comfyui_job: ComfyUIJobSlot | None` to the root `AppState` definition.

## Phase 2: Core ComfyUI Services

**Goal:** Create the isolated module (`backend/services/comfyui/`) for parsing workflows and communicating with the ComfyUI server.

1.  **Create Service Directory:** Initialize `backend/services/comfyui/`.
2.  **Implement `WorkflowParser`:**
    *   Create logic to read JSON workflows from a designated directory.
    *   Extract `proxyWidgets` metadata from node properties.
3.  **Implement `ComfyUIClient`:**
    *   Create an asynchronous HTTP client to communicate with the ComfyUI API (`/prompt`, `/upload/image`, `/history`, `/view`).
4.  **Expose Workflows Endpoint:**
    *   Create a new route in `backend/_routes/` (e.g., `workflows.py`) to expose the parsed workflows and their configurable parameters to the frontend.
    *   Wire the route into `app_factory.py`.

## Phase 3: Pipeline Adapters and Progress Tracking

**Goal:** Implement the adapter that bridges the strictly typed LTX-Desktop protocols with the dynamic ComfyUI engine.

1.  **Implement `ComfyUIVideoPipeline`:**
    *   Create a class implementing the `FastVideoPipeline` (or relevant) protocol.
    *   Implement graph construction: merge the base JSON workflow with the incoming `workflow_params`.
2.  **Implement Background Polling:**
    *   Use the existing `TaskRunner` to spawn a background task upon job submission.
    *   Poll the ComfyUI server for progress on the submitted `prompt_id`.
    *   Translate the progress into standard `GenerationProgress` state objects.

## Phase 4: Handler Routing & Locking Integration

**Goal:** Update the centralized handler to securely route tasks based on the active backend.

1.  **Update `GenerationHandler`:**
    *   In `backend/handlers/generation_handler.py`, read `generation_backend` from settings.
    *   Implement branching logic:
        *   If `"local"`: Use `GpuSlot` and standard pipeline logic (existing code).
        *   If `"comfyui"`: Acquire lock, validate `ComfyUIJobSlot` is idle, set to running, release lock, and dispatch to `ComfyUIVideoPipeline`.
2.  **Ensure Lock Safety:**
    *   Verify the "lock -> check -> unlock -> heavy work -> lock -> update" pattern is strictly followed for the new `ComfyUIJobSlot`.

## Phase 5: Frontend Integration

**Goal:** Update the React frontend to dynamically render UI elements based on the parsed ComfyUI workflows.

1.  **Backend Toggle:** Add a UI toggle in the settings to switch between Local and ComfyUI backends.
2.  **Fetch Workflows:** On mount (if ComfyUI is active), fetch the available workflows from the new backend endpoint.
3.  **Dynamic Rendering:** 
    *   Parse the returned proxy widget schemas.
    *   Dynamically render sliders, dropdowns, and text inputs based on the expected types of the proxy widgets.
4.  **Submission Logic:** Update the `backendFetch` calls for generation to include the user-configured `workflow_params` dictionary.

## Phase 6: Testing and Validation

**Goal:** Ensure the integration is robust and the local pipeline remains unaffected.

1.  **Backend Integration Tests:**
    *   Create new tests in `backend/tests/` using fakes for the `ComfyUIClient`.
    *   Verify routing logic works correctly based on settings.
2.  **Type Checking:**
    *   Run `pnpm typecheck` to ensure the new dynamic dictionaries haven't violated strict mode rules elsewhere.
3.  **Local Regression:**
    *   Run existing `backend:test` suite to guarantee standard local generation is completely isolated and functional.