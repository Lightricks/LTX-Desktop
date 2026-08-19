import { useEffect, useState } from 'react'
import { subscribeWhileGenerationMayBeActive } from '../lib/generation-progress-poll'
import { GENERATION_RECOVERY_KEY, readRecoveryMarkerCanCancel } from './use-generation'

export interface GlobalGenerationLock {
  // Fail-closed Generate disable: a recovery marker, a running poll, or an unconfirmed
  // poll error all mean the single global slot must be treated as busy.
  isRunning: boolean
  // Same poll as isRunning, gated on the backend's frozen local-GPU vs API slot.
  // Never derived from live Settings — switching to local mid-API-job must not reveal Stop.
  canCancel: boolean
  // Slot still busy after Stop (`status=running`, `phase=cancelled`) until generate() unwinds.
  isCancelling: boolean
}

// Only one generation can run at a time across the whole app (single global backend slot), but
// each project's GenSpace only tracks its OWN local isGenerating-style state — it has no idea a
// different project (or the same one, reconnected via a stale click) is already occupying that
// slot. Without this, Generate stays clickable in project B while project A is mid-generation;
// the request 409s, but only after writeRecoveryContext already overwrote A's in-flight recovery
// marker with B's (now-failed) one. Polling here lets Generate disable proactively instead.
// Stop uses the same sources: the recovery marker (immediate, frozen at job start) and
// GET /generation/progress.cancellable (authoritative once a poll lands). Hook-local canCancel
// dies on UI refresh; these do not. Live Settings must not flip Stop mid-job.
// No marker anywhere means nothing CAN be running (see subscribeWhileGenerationMayBeActive), so
// idle starts unlocked and costs no network call; once a marker exists and we're actually
// polling, an unconfirmed/failed poll is treated as locked rather than silently trusting "not
// running" — that unconfirmed-failure gap is exactly what previously let Generate stay clickable
// during another project's generation. The initial state has to check the marker too, not just
// hardcode false: a page refresh resets this hook's React state from scratch while another
// project's marker (and its still-running backend generation) survives in localStorage, and the
// first poll takes a network round trip to resolve — that gap is otherwise the same unconfirmed
// window all over again, just re-opened on every reload instead of only at first app launch.
export function useGlobalGenerationLock(): GlobalGenerationLock {
  const [lock, setLock] = useState<GlobalGenerationLock>(() => ({
    isRunning: localStorage.getItem(GENERATION_RECOVERY_KEY) != null,
    canCancel: readRecoveryMarkerCanCancel(),
    isCancelling: false,
  }))

  useEffect(() => subscribeWhileGenerationMayBeActive(result => {
    if (!result.ok) {
      // Same fail-closed Generate lock as before. Stop keeps the frozen marker bit rather
      // than guessing — a starved poll during local GPU work must not hide Stop, and a
      // failed poll during an API job must not reveal it.
      setLock({
        isRunning: true,
        canCancel: readRecoveryMarkerCanCancel(),
        isCancelling: false,
      })
      return
    }
    const isRunning = result.data.status === 'running'
    setLock({
      isRunning,
      canCancel: isRunning && result.data.cancellable,
      isCancelling: isRunning && result.data.phase === 'cancelled',
    })
  }), [])

  return lock
}
