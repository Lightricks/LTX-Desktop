---
name: ltx-video-editor-development
description: Use this skill when implementing, refactoring, or reviewing code in the LTX Desktop video editor, especially files under frontend/views/VideoEditor.tsx and frontend/views/editor/** that touch the editor store, selectors, actions, undo/history, persistence, keyboard/menu wiring, playback, timeline interactions, or other editor-specific UI behavior.
---

# LTX Video Editor Development

## Overview

The LTX Desktop video editor uses a dedicated editor store with domain selectors and actions. Use this skill for any work inside the editor runtime so changes stay aligned with the current architecture and do not regress undo, persistence, keyboard behavior, timeline semantics, or hot-path performance.

## When To Use

Use this skill when the task touches any of:

- `frontend/views/VideoEditor.tsx`
- `frontend/views/editor/**`
- Editor store shape
- Editor selectors or actions
- Undo/redo behavior
- Project persistence bridge
- Keyboard or menu behavior
- Playback or timeline interaction behavior
- Gen Space integration from the video editor

Do not use this skill for unrelated frontend work, backend work, Electron-only work, or generic React refactors outside the video editor.

## First Pass

Before changing behavior, inspect the current contracts first:

- [frontend/views/editor/editor-state.ts](../../../frontend/views/editor/editor-state.ts)
- [frontend/views/editor/editor-store.tsx](../../../frontend/views/editor/editor-store.tsx)
- [frontend/views/editor/editor-selectors.ts](../../../frontend/views/editor/editor-selectors.ts)
- [frontend/views/editor/editor-actions.ts](../../../frontend/views/editor/editor-actions.ts)
- [frontend/views/editor/editor-project-bridging.ts](../../../frontend/views/editor/editor-project-bridging.ts)
- [frontend/views/VideoEditor.tsx](../../../frontend/views/VideoEditor.tsx)

If the task touches imperative integrations, also inspect the relevant hook or component first, for example keyboard, menu, playback, timeline drag, source monitor, or regeneration.

## State Model

Treat the editor as four separate concerns:

- `editorModel`: persistent document state. This is the editor content.
- `session`: ephemeral editor session state such as selection, transport, tools, UI, clipboard, and regeneration.
- `history`: undo/redo storage, not product state.
- `projectSync`: sync bookkeeping only.

Use this split when deciding where a new field belongs.

- If the field is shared across editor components or hooks, or needed by editor actions and invariants, it usually belongs in the store.
- If the field is only local widget UI state, keep it local to that component or hook.
- If the field is only meaningful during a high-frequency interaction, prefer refs or imperative handling instead of pushing every frame through the store.

## Selector Rules

- Components and hooks should read editor state through `useEditorStore(selector)`.
- Prefer selectors phrased in domain semantics, not raw field plumbing.
- Before adding a selector, check whether one already exists.
- If a component needs a composite view model, create a selector that matches that component's read surface.
- Selectors must not allocate fresh default arrays or objects on fallback paths. Use shared constants instead.
- If a selector intentionally assembles a new object or array from stable child references and primitives, subscribe with `useShallow(...)` when shallow top-level equality is the right contract.
- `useShallow(...)` is not a fix for unstable selectors. Fix referential instability at the selector level first.
- Prefer primitive selectors or stable references when possible.

## Action Rules

- Components and hooks should mutate editor state through `useEditorActions()`.
- Prefer intention-driven actions over low-level field setters.
- Before adding an action, check whether one already exists.
- Actions own editor invariants and behaviorally significant transitions.
- Do not hide product rules in `useEffect` when they really belong in an action.

Examples of the intended pattern:

- timeline switching behavior belongs in `switchActiveTimeline(...)`
- modal visibility should use explicit actions like `openExportModal()` and `closeExportModal()`
- component event handlers should compose actions, not manually rebuild state

## Undo And Persistence

Undo is snapshot-based, not action-metadata-based.

Current undo snapshot:

```ts
type EditorUndoSnapshot = {
  assets: Asset[]
  timelines: Timeline[]
}
```

Rules:

- If a new field should be undoable, it must be represented in the undo snapshot derivation and apply path.
- If a field is not in the undo snapshot, changing it will not participate in undo/redo by design.
- `undo` and `redo` are special-cased to bypass history recording. Normal mutating actions go through history recording.

Persistence is derived from `editorModel`, not from the entire store.

- If a new field should persist into the project, update [frontend/views/editor/editor-project-bridging.ts](../../../frontend/views/editor/editor-project-bridging.ts).
- In practice that means updating both `getEditorModel(...)` and `updatedProject(...)`.
- Session and UI fields should not be pushed into project persistence unless there is an explicit product decision to persist them.

## Imperative Hot Paths

Do not route high-frequency interaction updates through the store on every frame.

Current examples:

- playback uses imperative refs and DOM synchronization on the hot path
- clip drag preview uses DOM takeover on the active clip elements and commits once at the end

Use this pattern for future hot paths too:

1. Keep the in-gesture state imperative.
2. Update only the actively manipulated DOM elements if needed.
3. Commit the final meaningful result into the store once the interaction ends.

If the result remains relevant after the interaction, sync the final state back into the store at the appropriate boundary.

## Reading State Outside React

If code runs outside the React render cycle, read the store lazily through `useEditorGetState()`.

This is justified for:

- window event listeners
- keyboard handlers
- `requestAnimationFrame` loops
- other imperative callbacks that need the latest editor state

Do not use `getState()` for render-time UI derivation. Use selectors for that.

Prefer sampling fresh state at execution time over mirroring large store slices into manual refs, unless there is a concrete hot-path reason.

## External Integrations

Not all editor-triggered behavior belongs in the store.

Keep external side effects outside store actions unless there is a strong reason otherwise. This includes:

- Gen Space navigation and payload wiring
- Electron APIs
- file picker flows
- import/export orchestration
- other application-shell integrations

The pattern is:

- use editor actions for editor state mutation
- perform external side effects explicitly in the component or hook that owns the integration

Gen Space communication currently happens through `ProjectContext` setters from [frontend/views/VideoEditor.tsx](../../../frontend/views/VideoEditor.tsx), not through editor store actions.

## VideoEditor Role

`VideoEditor.tsx` should stay focused on:

- store provisioning
- layout and composition
- cross-panel orchestration
- external integrations

It should not accumulate low-level state mutation logic or broad raw derivation logic. If a panel or hook can own its selectors and actions internally, let it.

If you find yourself wanting to prop-drill shared editor state through `VideoEditor`, that is usually a sign the state belongs in the store.

## Behavior-Sensitive Changes

When changing behavior-sensitive areas such as timeline actions, keyboard shortcuts, menus, playback, drag/drop, or import/export:

- verify the existing user-visible behavior first
- preserve behavior intentionally in the new selector or action shape
- compare the migrated implementation against the previous semantics, not just against the current code shape

State architecture changes in this codebase have repeatedly caused regressions when product rules were only implicit.

## Working Checklist

Use this checklist before merging editor changes:

- Does the new field belong in `editorModel`, `session`, `history`, `projectSync`, or local component state?
- Does an appropriate selector already exist?
- Does an appropriate action already exist?
- Are selectors returning stable references on fallback paths?
- Is `useShallow(...)` needed for any composite selector subscription?
- Should this field be undoable?
- Should this field persist to the project bridge?
- Is this a hot path that should remain imperative?
- Is there an external side effect that should stay outside the store?
- Should this logic be extracted into a self-contained hook with a single responsibility?

## File Map

Use these files as the primary source of truth:

- [frontend/views/editor/editor-state.ts](../../../frontend/views/editor/editor-state.ts)
- [frontend/views/editor/editor-store.tsx](../../../frontend/views/editor/editor-store.tsx)
- [frontend/views/editor/editor-selectors.ts](../../../frontend/views/editor/editor-selectors.ts)
- [frontend/views/editor/editor-actions.ts](../../../frontend/views/editor/editor-actions.ts)
- [frontend/views/editor/editor-project-bridging.ts](../../../frontend/views/editor/editor-project-bridging.ts)
- [frontend/views/VideoEditor.tsx](../../../frontend/views/VideoEditor.tsx)
