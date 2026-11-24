import { useSyncExternalStore } from 'react'

// Local, live "is a generation running?" signal for the renderer UI.
// Ref-counted because generations can overlap.
let activeCount = 0
const listeners = new Set<() => void>()
function emit() { for (const l of listeners) l() }

function subscribe(cb: () => void): () => void {
  listeners.add(cb)
  return () => { listeners.delete(cb) }
}
function getSnapshot(): boolean {
  return activeCount > 0
}

export function useIsGenerationActive(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
}

// Local generation can starve the backend's event loop; withGenerationActive tells main so the
// liveness monitor doesn't kill a busy backend. We also keep a local count for the UI signal above.
export async function withGenerationActive<T>(fn: () => Promise<T>): Promise<T> {
  activeCount += 1; emit()
  void window.electronAPI.notifyGenerationActive({ active: true })
  try {
    return await fn()
  } finally {
    activeCount = Math.max(0, activeCount - 1); emit()
    void window.electronAPI.notifyGenerationActive({ active: false })
  }
}

/** Evaluate at job start and freeze — live Settings must not flip Stop mid-request. */
export function canCancelLocalJob(
  kind: 'video' | 'image',
  videoUsesLtxApi: boolean,
  imageUsesFalApi: boolean,
): boolean {
  if (kind === 'image') return !imageUsesFalApi
  if (kind === 'video') return !videoUsesLtxApi
  return false
}
