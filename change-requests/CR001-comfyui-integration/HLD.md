# High-Level Design (HLD): ComfyUI Integration

## 1. Overview

This High-Level Design details the integration of a ComfyUI engine into the LTX-Desktop application. As determined in the Impact Assessment, this integration follows **Option B: Proxy-Based Metadata Mapping**. 

The core architectural philosophy is **Additive Isolation**: the ComfyUI integration will be built *on top* of the existing local generation capabilities without modifying their underlying logic. This minimizes merge conflicts with the upstream fork and ensures the default native GPU experience is preserved.

## 2. Core Components

### 2.1. App Settings and API Types (Additive)

*   **`AppSettings`**: A new setting, `generation_backend` (Literal: `"local" | "comfyui"`), will be added to dictate the routing logic.
*   **`api_types.py`**: Generation request payloads (e.g., `VideoGenerationRequest`) will be extended with an optional `workflow_params: dict[str, Any] | None` to pass dynamic proxy widget values from the UI to the backend.

### 2.2. State Management (`AppState`)

To avoid disrupting the highly tuned local `GpuSlot` management:
*   A new state slot, **`ComfyUIJobSlot`**, will be introduced in `AppState`.
*   The centralized `AppHandler` lock will protect this new slot exactly as it protects the `GpuSlot`. 

### 2.3. ComfyUI Service Module

A new, isolated module (`backend/services/comfyui/`) will encapsulate all ComfyUI-specific logic:

1.  **`WorkflowParser`**: 
    *   Reads predefined ComfyUI JSON workflows.
    *   Extracts the `proxyWidgets` metadata to identify which internal node parameters are exposed to the UI.
2.  **`ComfyUIClient`**:
    *   Handles HTTP communication with the ComfyUI server (e.g., `/prompt`, `/upload/image`, `/history`).
    *   Manages WebSocket connections (if required) for real-time progress updates.
3.  **`ComfyUIPipelineAdapters`**:
    *   Implements the existing strictly-typed protocols (e.g., `FastVideoPipeline`).
    *   Translates the incoming `VideoGenerationRequest` (including `workflow_params`) into the final execution graph JSON.

### 2.4. Generation Handler Routing

The `GenerationHandler` will act as a router based on the `generation_backend` setting:

*   **If `"local"`**: The handler proceeds normally, acquiring the `GpuSlot` and delegating to the native `services.video_processor`.
*   **If `"comfyui"`**: The handler bypasses the `GpuSlot`, acquires the `ComfyUIJobSlot`, and delegates to the `ComfyUIPipelineAdapter`.

### 2.5. Progress Translation

To ensure the frontend requires zero changes to its progress tracking logic:
*   The `ComfyUIPipelineAdapter` will spawn a background polling task (using the existing `TaskRunner`).
*   This task will translate ComfyUI's native execution progress into the exact `GenerationProgress` (e.g., `GenerationRunning`, `GenerationComplete`) state objects expected by `AppState`.

## 3. Architectural Flow (ComfyUI Active)

1.  **UI Configuration**: Frontend fetches available workflows via a new endpoint (parsed by `WorkflowParser`) and dynamically renders controls for the exposed `proxyWidgets`.
2.  **Submission**: User clicks generate. Frontend sends `VideoGenerationRequest` including `workflow_params`.
3.  **Routing**: `GenerationHandler` sees `generation_backend == "comfyui"`.
4.  **Locking**: Handler acquires lock -> sets `ComfyUIJobSlot` to running -> unlocks.
5.  **Execution**: `ComfyUIPipelineAdapter` constructs the final JSON graph and sends it to the `ComfyUIClient`.
6.  **Progress**: Background task polls ComfyUI, locking briefly to update `ComfyUIJobSlot` progress.
7.  **Completion**: Adapter retrieves the final media from ComfyUI, saves it locally, and updates state to `GenerationComplete`.
