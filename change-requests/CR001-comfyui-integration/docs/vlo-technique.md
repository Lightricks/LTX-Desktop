# Procedural ComfyUI Node Discovery and UI Mapping in Vlo

## Overview

Vlo implements a "Proxy-Based" approach to ComfyUI node management. It allows for a highly flexible UI that can dynamically adapt to various underlying ComfyUI workflows by explicitly defining links between UI elements and backend nodes within the workflow metadata itself.

## Prerequisites

Vlo is **workflow-agnostic** and does not require dedicated custom nodes for its core mapping functionality.

- **Standard Nodes**: It interfaces with standard ComfyUI nodes (e.g., `LoadImage`, `CLIPTextEncode`, `KSampler`) and popular community extensions (e.g., `VideoHelperSuite`) using their default schemas.
- **Metadata-Driven**: The only requirement is the inclusion of specific metadata (`proxyWidgets`) within the JSON workflow file's `properties` field. This allows the system to remain compatible with any node as long as the mapping is explicitly defined in the graph.

## Node Discovery Technique

Vlo's discovery is centered around the **Workflow Metadata** and the `properties` field of ComfyUI nodes.

1. **Graph Introspection**: The system parses the ComfyUI JSON workflow (both API and Graph formats).
2. **Property Extraction**: It specifically looks for a custom `proxyWidgets` field within the `properties` dictionary of a node.
3. **State Synchronization**: The backend maintains a cache of the current graph state and uses it to resolve which UI-exposed parameters correspond to which internal node inputs.

## UI-to-Node Mapping

Vlo uses a **Proxy Widget Mapping** technique to bridge the UI and the backend:

1. **Explicit Mapping**: The `proxyWidgets` property contains a list of mappings. Each entry defines a `target_node_id` and a `target_param_name`.
2. **Virtual Groups**: UI elements are organized into "Groups" (e.g., "Sampling Settings", "Model Selection") based on the `group_id` and `group_title` associated with the proxy mapping.
3. **Procedural UI Generation**: The frontend receives a list of these groups and their associated controls. When a user modifies a value in the UI, the backend uses the mapping to update the specific node and parameter in the graph before submission.
4. **Node Normalization**: Before sending the final graph to ComfyUI, the system "normalizes" it, ensuring that all proxy-driven changes are correctly applied to the final execution graph.

## Key Advantages

- **Decoupling**: The UI structure is not hardcoded to a specific node graph; it is defined *by* the graph.
- **Granular Control**: Specific parameters from different nodes can be grouped together in the UI, even if they are far apart in the actual graph.
- **Workflow Agnostic**: Any ComfyUI workflow can be "UI-enabled" simply by adding the appropriate `proxyWidgets` metadata to the nodes.

## Findings on `proxyWidgets` Convention

Research indicates that `proxyWidgets` is an established **ComfyUI metadata convention** used specifically for **Subgraphs (Group Nodes)** and **Blueprints**. 

1. **Standard Purpose**: In the broader ComfyUI ecosystem, `proxyWidgets` allows a "Group Node" or "Subgraph" to expose internal parameters (like `seed`, `steps`, or `prompt`) to the top-level UI of the collapsed group.
2. **Implementation**: It is stored in the `properties` field of a node in the workflow JSON as an array of tuples: `["internal_node_id", "widget_name"]`.
3. **Vlo's Leverage**: Vlo utilizes this standard mapping logic to drive its entire dynamic UI generation. By following this convention, Vlo can interface with any complex workflow that has been organized into logical groups/subgraphs without needing custom backend code for every new node type.
