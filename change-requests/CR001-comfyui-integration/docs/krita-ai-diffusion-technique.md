# Procedural ComfyUI Node Discovery and UI Mapping in Krita AI Diffusion

## Overview

Krita AI Diffusion manages ComfyUI integration by abstracting the low-level node graph into a high-level Python API. This approach ensures type safety and simplifies the construction of complex workflows while remaining flexible enough to support custom nodes.

## Prerequisites

This technique requires the installation of a **dedicated ComfyUI extension** (`ComfyUI-Krita-AI-Diffusion`). 

- **Custom Nodes**: The system relies on specific nodes (prefixed with `ETN_`, such as `ETN_LoadImageCache`, `ETN_InjectImage`, and `ETN_LoadMaskBase64`) to handle specialized tasks like in-memory image transfer and precise canvas synchronization.
- **Fixed API Contract**: The Python backend is tightly coupled to these custom nodes. Without them, the high-level builder cannot generate the necessary graph structure to communicate with Krita.

## Node Discovery Technique

The system relies on the `/object_info` endpoint of the ComfyUI API to procedurally discover available nodes and their respective input/output schemas.

1. **Schema Retrieval**: Upon connection, the client fetches the full `object_info` dictionary.
2. **Type Mapping**: It maps ComfyUI types (e.g., `IMAGE`, `LATENT`, `MODEL`) to internal Python classes.
3. **Validation**: The `ComfyWorkflow` builder uses this schema to validate inputs and outputs at construction time, preventing the creation of invalid graphs before they are even sent to the server.

## UI-to-Node Mapping

Unlike systems that use a 1:1 visual mapping, Krita AI Diffusion uses a **Functional Mapping** approach:

1. **Abstract Workflows**: The UI interacts with a `Workflow` class that defines high-level operations (e.g., `sampling`, `upscaling`).
2. **Builder Pattern**: These high-level operations are translated into a sequence of `ComfyWorkflow` method calls (e.g., `w.load_checkpoint()`, `w.ksampler()`).
3. **Dynamic Node IDs**: Node IDs are generated procedurally during the building process. The UI does not need to know specific node IDs; it only cares about the parameters (e.g., `seed`, `steps`, `denoise`) passed to the high-level API.
4. **Custom Workflows**: For user-provided workflows, the system imports the JSON graph and uses the `object_info` schema to identify "input" nodes (e.g., `PrimitiveNode`, `CheckpointLoaderSimple`) that should be exposed as UI widgets based on their titles or specific metadata.

## Key Advantages

- **Type Safety**: Prevents common "wrong input type" errors.
- **Maintainability**: Changes in the underlying ComfyUI node names can be handled by updating the mapping logic in one place.
- **User Experience**: Simplifies complex graphs into a familiar, tool-like interface for artists.
