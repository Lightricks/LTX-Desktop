# Impact Assessment: Replacing Backend Generation with Configurable ComfyUI Workflows

## 1. Executive Summary

This document assesses the architectural impact of integrating a ComfyUI backend alongside the current local generative pipelines. The goal is to allow dynamic, runtime switching between the existing local GPU implementations and a configurable ComfyUI workflow engine.

Based on recent analysis of reference projects (Krita AI Diffusion and Vlo), this document focuses on clarifying the architectural options for procedural node discovery and UI mapping. The objective of this phase is to review these options prior to committing to a final architectural decision.

---

## 2. Current Architecture Overview

The backend uses a local FastAPI server where endpoints delegate business logic to a centralized `AppHandler`.

- **State Management:** A highly normalized, typed `AppState` manages limited resources (e.g., `GpuSlot`, `CpuSlot`, `DownloadingSession`).
- **Concurrency & Locking:** A single shared `RLock` protects `AppState`. Handlers follow a strict "lock -> check -> unlock -> heavy work -> lock -> update" pattern to prevent blocking the server during generation.
- **Service Boundaries:** Heavy generative tasks are isolated behind strictly typed Python Protocols in `backend/services/` (e.g., `FastVideoPipeline`).
- **Generation Lifecycle:** `GenerationHandler` tracks progress using normalized state machines (`GenerationRunning`, `GenerationComplete`, etc.).

---

## 3. Proposed Architecture Options for UI-to-Node Mapping

A core challenge is how the frontend UI (sliders, dropdowns for models/LoRAs) dynamically maps to and controls the underlying ComfyUI node graph. Two distinct architectural options have been identified based on industry reference projects.

### Option A: Functional Mapping via Dedicated Custom Nodes (The Krita Approach)

In this approach, the integration relies on abstracting the low-level node graph into a high-level Python API, tightly coupled with a dedicated suite of custom ComfyUI nodes.

*   **Mechanism:**
    *   Relies on the ComfyUI `/object_info` API to discover available nodes and their schemas.
    *   The backend implements a "Builder Pattern" (`ComfyWorkflow`) that translates high-level UI requests (e.g., `generate_video(prompt, seed)`) into specific node instantiations.
    *   **Crucial Prerequisite:** Requires installing a dedicated ComfyUI extension with custom nodes (e.g., `LTX_LoadImage`, `LTX_InjectVideo`) designed specifically to bridge the communication gap (e.g., handling in-memory transfers or specific app logic).
*   **Pros:**
    *   **Strong Type Safety:** The builder validates inputs/outputs against the `object_info` schema before execution.
    *   **Simpler UI Logic:** The UI only interacts with high-level parameters; the backend handles the complex graph construction.
*   **Cons:**
    *   **High Maintenance:** Tightly coupled to specific custom nodes. Any changes to the node logic require updating the backend builder.
    *   **Less Flexible:** Harder for users to drop in arbitrary, wildly different ComfyUI workflows without backend updates.

### Option B: Proxy-Based Metadata Mapping (The Vlo Approach)

This approach is workflow-agnostic and relies on embedding UI mapping rules directly within the ComfyUI workflow JSON metadata.

*   **Mechanism:**
    *   Relies on a custom metadata field, specifically the established ComfyUI convention `proxyWidgets`, located within the `properties` dictionary of Subgraphs (Group Nodes) or individual nodes.
    *   The workflow JSON explicitly defines mapping tuples (e.g., `["node_id_31", "seed"]`).
    *   The backend parses these JSON files at startup, discovers the exposed parameters, and serves this dynamic schema to the frontend.
    *   **Crucial Prerequisite:** Requires zero dedicated custom nodes. It interfaces with standard ComfyUI nodes and relies entirely on the JSON metadata structure.
*   **Pros:**
    *   **Highly Decoupled & Workflow Agnostic:** The UI structure is defined *by* the graph. Users can drop in entirely new workflows (using standard nodes) as long as they tag the inputs with `proxyWidgets`.
    *   **Minimal Backend Logic:** The backend acts primarily as a pass-through and normalizer, rather than a complex graph builder.
*   **Cons:**
    *   **Weaker Typing:** Relies on dynamic dictionaries passing through the backend, requiring careful validation logic.
    *   **Complex JSON Maintenance:** The burden of defining the UI shifts to whoever authors the ComfyUI JSON workflows; they must correctly set up the `proxyWidgets` arrays.

---

## 4. Architectural Impact on LTX-Desktop Backend

Regardless of the option chosen, migrating to a ComfyUI engine impacts the core architecture:

### 4.1. AppSettings and API Types
*   **Runtime Config:** `app_settings.py` needs a `generation_backend` flag.
*   **Dynamic Payloads:** `api_types.py` must be updated. Option B requires highly dynamic `workflow_params: dict[str, Any]` to accommodate arbitrary proxy widgets, whereas Option A might allow slightly more structured, high-level requests.

### 4.2. State Management (`AppState` & `GpuSlot`)
*   Currently, `GpuSlot` is strictly typed to local pipelines.
*   A new slot (e.g., `ComfyUISlot`) must be added to track the state of the external ComfyUI process. The `AppHandler` lock will protect this slot similarly to local execution.

### 4.3. Generation Handler and Progress Tracking
*   ComfyUI operates asynchronously. The backend must introduce a background polling mechanism (via `TaskRunner`) or websocket listener to track progress and map ComfyUI's step data to the existing `GenerationProgress` state model, keeping the frontend polling endpoint intact.

---

## 5. Conclusion and Next Steps

The decision between **Option A (Functional/Custom Nodes)** and **Option B (Proxy Metadata/Agnostic)** dictates the entire trajectory of the backend integration. 

*   Option A favors strict control, type safety, and specialized custom node logic but sacrifices user workflow flexibility.
*   Option B favors extreme flexibility and relies on established ComfyUI UI conventions (`proxyWidgets`), pushing the configuration burden to the workflow JSON authors.

**Next Step:** This document serves as the basis for an architectural review. A final decision on Option A vs. Option B must be made before implementation of the ComfyUI adapter service begins.