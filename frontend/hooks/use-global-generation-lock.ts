import { useEffect, useState } from 'react'
import { subscribeWhileGenerationMayBeActive } from '../lib/generation-progress-poll'
import { GENERATION_RECOVERY_KEY } from './use-generation'

// Only one generation can run at a time across the whole app (single global backend slot), but
// each project's GenSpace only tracks its OWN local isGenerating-style state — it has no idea a
// different project (or the same one, reconnected via a stale click) is already occupying that
// slot. Without this, Generate stays clickable in project B while project A is mid-generation;
// the request 409s, but only after writeRecoveryContext already overwrote A's in-flight recovery
// marker with B's (now-failed) one. Polling here lets Generate disable proactively instead.
// No marker anywhere means nothing CAN be running (see subscribeWhileGenerationMayBeActive), so
// idle starts unlocked and costs no network call; once a marker exists and we're actually
// polling, an unconfirmed/failed poll is treated as locked rather than silently trusting "not
// running" — that unconfirmed-failure gap is exactly what previously let Generate stay clickable
// during another project's generation. The initial state has to check the marker too, not just
// hardcode false: a page refresh resets this hook's React state from scratch while another
// project's marker (and its still-running backend generation) survives in localStorage, and the
// first poll takes a network round trip to resolve — that gap is otherwise the same unconfirmed
// window all over again, just re-opened on every reload instead of only at first app launch.
export function useGlobalGenerationLock(): boolean {
  const [isRunning, setIsRunning] = useState(() => localStorage.getItem(GENERATION_RECOVERY_KEY) != null)

  useEffect(() => subscribeWhileGenerationMayBeActive(result => {
    setIsRunning(result.ok ? result.data.status === 'running' : true)
  }), [])

  return isRunning
}
