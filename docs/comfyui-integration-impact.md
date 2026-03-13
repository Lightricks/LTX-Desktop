# Impact Assessment: Replacing Backend Generation with Configurable ComfyUI Workflows

## 1. Executive Summary

This document assesses the architectural impact of integrating a ComfyUI backend alongside the current local generative pipelines. The goal is to allow dynamic, runtime switching between the existing local GPU implementations and a configurable ComfyUI workflow engine.

The integration assumes that a ComfyUI adapter already exists and that necessary generation workflows (in JSON format) are supplied with the application. The primary challenge is adapting the existing FastAPI + `AppHandler` architecture to support a modular, parameterized workflow engine without severely disrupting the strictly typed, centralized `AppState` and resource locking mechanisms.

---

## 2. Current Architecture Overview

The backend uses a local FastAPI server where endpoints delegate business logic to a centralized `AppHandler`.

- **State Management:** A highly normalized, typed `AppState` manages limited resources (e.g., `GpuSlot`, `CpuSlot`, `DownloadingSession`).
- **Concurrency & Locking:** A single shared `RLock` protects `AppState`. Handlers follow a strict "lock -> check -> unlock -> heavy work -> lock -> update" pattern to prevent blocking the server during generation.
- **Service Boundaries:** Heavy generative tasks are isolated behind strictly typed Python Protocols in `backend/services/` (e.g., `FastVideoPipeline`, `ImageGenerationPipeline`). These protocols dictate exact Python method signatures (e.g., `generate(...)` with specific arguments).
- **Generation Lifecycle:** `GenerationHandler` tracks progress using normalized state machines (`GenerationRunning`, `GenerationComplete`, etc.).

---

## 3. Proposed Architecture Options

We must expose supplied ComfyUI workflows dynamically, allowing the UI to configure parameters (models, LoRAs) and the user to switch generation backends at runtime.

### Option A: Adapter Service Implementations (The "Facade" Approach)

In this approach, we retain the exact existing Service Protocols (like `FastVideoPipeline`) and introduce new implementations (e.g., `ComfyUIFastVideoPipeline`) that wrap the ComfyUI adapter and hardcode the mapping to the JSON workflows.

*   **Pros:**
    *   **Minimal Blast Radius:** Zero changes to `PipelinesHandler`, `GenerationHandler`, or `_routes`.
    *   **Maintains Strong Typing:** Preserves the existing rigid structural typing of the Python backend.
*   **Cons:**
    *   **Hides Dynamism:** Fails the requirement of "configurable ComfyUI workflows" because adding a new workflow or exposing a new parameter (like a new LoRA node) would require modifying the Python protocol and every implementation.
    *   **Duplication:** We would have to maintain a 1:1 mapping of rigid Python interfaces to dynamic JSON workflows.

### Option B: Generic Workflow Engine Service (Recommended)

In this approach, we introduce a new generic service, `ComfyUIWorkflowEngine`, and update the `GenerationHandler` and endpoints to route requests based on the user's active configuration.

*   **Core Concept:** The backend loads JSON workflows from disk at startup (or dynamically). These workflows are parsed to identify configurable nodes (parameters, models, LoRAs).
*   **Runtime Config:** A new setting in `AppSettings` (e.g., `generation_backend: Literal["local", "comfyui"]`) dictates the routing in the handlers.
*   **Pros:**
    *   **Highly Extensible:** Adding a new workflow or parameter only requires updating the JSON file and the UI; the backend simply passes through the configuration to the ComfyUI adapter.
    *   **Modular:** Clearly separates local GPU logic from ComfyUI external process logic.
*   **Cons:**
    *   Requires modifications to `AppState` to track ComfyUI jobs.
    *   Requires updates to request/response models (`api_types.py`) to accept dynamic key-value parameters for ComfyUI.

---

## 4. Architectural Impact & Necessary Changes (Based on Option B)

To achieve a modular integration that fulfills the dynamic configuration requirements, the following architectural changes are required.

### 4.1. AppSettings and API Types

*   **`app_settings.py`:** Add a `generation_backend` property to allow runtime switching.
*   **`api_types.py`:** Update generation request payloads (e.g., `VideoGenerationRequest`) to include an optional dictionary for dynamic workflow parameters: `workflow_params: dict[str, Any] | None = None`.
*   *Note: While the backend generally avoids dynamic dicts, it is necessary here as a pass-through layer for the ComfyUI JSON schema.*

### 4.2. Workflow Parsing and Exposure

*   **New Service:** Introduce a `WorkflowParserService` that reads the provided JSON workflows from a designated directory (e.g., `resources/workflows/`).
*   **New Endpoint:** Create `GET /api/workflows` to serve the parsed parameters to the frontend, allowing the UI to dynamically render dropdowns (for models, LoRAs) and sliders based on the workflow's exposed nodes.

### 4.3. State Management (`AppState` & `GpuSlot`)

Currently, `GpuSlot` is strictly typed to local pipelines (e.g., `VideoPipelineState | ICLoraState | ...`).

*   **Impact:** Offloading to ComfyUI means the local GPU is *not* occupied by the FastAPI process.
*   **Change:** `AppState` must be expanded. `GpuSlot` can remain for local models, but a new slot (e.g., `ExternalSlot` or `ComfyUISlot`) must be added to track the state of the ComfyUI process.
*   **Locking:** The `AppHandler` lock will still protect this new slot. The lock scope remains identical: Lock -> update `ComfyUISlot` to running -> Unlock -> send HTTP request to ComfyUI -> Lock -> update progress.

### 4.4. Generation Handler and Progress Tracking

*   **Polling vs. Callbacks:** ComfyUI typically operates asynchronously. The backend's `GenerationHandler` currently assumes it is the active runner updating progress continuously.
*   **Change:** If using ComfyUI, the backend must initiate a background task (using the existing `TaskRunner`) to poll the ComfyUI adapter for progress, mapping ComfyUI's step data to the existing `GenerationProgress` state model. This ensures the frontend's `/api/generation/progress` polling endpoint remains unbroken.

### 4.5. Model Management (Out of Scope)

*   Currently, `ModelDownloader` fetches models to a local directory. For this iteration, ComfyUI's model management is considered out of scope. ComfyUI will be responsible for locating its own models and LoRAs as specified by the parameterized JSON workflows.

---

## 5. Conclusion

To support configurable, parameterized ComfyUI workflows alongside the existing local GPU pipelines, the backend must shift slightly from its strictly typed generative protocols to a generic workflow engine.

By introducing a generic `ComfyUIWorkflowEngine` service, expanding `AppState` to include an `ExternalSlot` for job tracking, and updating `GenerationHandler` to poll the ComfyUI adapter in the background, the application can remain modular. This approach keeps the heavy lifting in the respective processes while maintaining the centralized locking and state guarantees that the FastAPI architecture relies upon.