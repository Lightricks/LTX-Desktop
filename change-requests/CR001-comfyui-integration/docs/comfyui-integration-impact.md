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

## 4. Architectural Impact on LTX-Desktop Backend (Option B Selected)

Following review, **Option B (Proxy-Based Metadata Mapping)** has been selected. The primary directive for this integration is to **maximize out-of-the-box compatibility** while ensuring a **minimal fork update impact**. 

Crucially, the ComfyUI integration must be added *on top* of existing services. Current LTX Desktop functionality (local, native GPU generation) must be fully retained and operate exactly as before when ComfyUI is not active. By isolating the new ComfyUI logic, we ensure that upstream merges from the original `LTX-Desktop` repository remain trivial.

### 4.1. Core Principle: Isolation for Minimal Merge Conflicts
To minimize merge conflicts when pulling from the upstream fork, the ComfyUI integration will avoid heavily modifying existing core files (like `app_handler.py` or complex state machines) wherever possible. Instead, it will rely on new interface implementations and isolated modules.

*   **New Modules:** All ComfyUI-specific parsing, proxy mapping, and network communication will live in a strictly separated directory (e.g., `backend/services/comfyui/`).
*   **Interface Implementation:** The ComfyUI engine will implement the existing pipeline interfaces (e.g., creating a `ComfyUIVideoPipeline` that adheres to the same Protocol as `FastVideoPipeline`). This allows the core `GenerationHandler` to treat it as just another backend without knowing its internal complexities.

### 4.2. AppSettings and API Types (Additive Changes)
Changes to core API files will be strictly additive, preserving all existing schemas.
*   **`app_settings.py`:** Add a new, optional `generation_backend` flag (defaulting to `local`).
*   **`api_types.py`:** Add a new, optional `workflow_params: dict[str, Any] | None = None` to the generation request models to handle the dynamic `proxyWidgets` inputs. Existing strictly-typed fields (prompt, seed, etc.) remain untouched and can be mapped internally if ComfyUI is the active backend.

### 4.3. State Management (`AppState` & `GpuSlot`)
The existing `GpuSlot` logic, which carefully manages local VRAM, must remain untouched to ensure the default local generation experience is not compromised.
*   **Additive State:** A new, distinct slot (e.g., `ExternalSlot` or `ComfyUIJobSlot`) will be introduced to `AppState`.
*   **Locking:** The `AppHandler` will use the same locking mechanism to protect this new slot, ensuring thread safety without needing to rewrite the existing local GPU locking logic. If the active backend is ComfyUI, the handler checks the `ExternalSlot` instead of the `GpuSlot`.

### 4.4. Generation Handler and Progress Tracking
The existing `GenerationHandler` relies heavily on the local process actively reporting progress.
*   **Adapter Pattern:** A `ComfyUIAdapter` service will act as a bridge. When a generation task is dispatched to ComfyUI, the adapter will use the `TaskRunner` to spawn a background polling task.
*   **State Translation:** This polling task will fetch ComfyUI's native progress (using its API/Websocket) and translate it into the *exact same* `GenerationProgress` state objects currently expected by the frontend. This ensures the frontend UI requires minimal, if any, modifications to display progress bars.

---

## 5. Conclusion

By selecting **Option B (Proxy-Based Metadata Mapping)**, we achieve a highly flexible, workflow-agnostic system. 

By applying a strict **"Additive Isolation"** architectural lens, we ensure that:
1.  **Original functionality is preserved:** Local generation remains completely unaffected and acts as the default.
2.  **Upstream merges are trivial:** By avoiding modifications to core logic loops and instead adding new interface implementations and isolated service folders, we minimize the "fork divergence," allowing the project to easily consume future updates from the original Lightricks repository.