import { ApiClient } from './api-client'
import { GENERATION_RECOVERY_KEY } from '../hooks/use-generation'

type ProgressResult = Awaited<ReturnType<typeof ApiClient.getGenerationProgress>>
type Listener = (result: ProgressResult) => void

const POLL_INTERVAL_MS = 3000
const MARKER_CHECK_INTERVAL_MS = 3000

const listeners = new Set<Listener>()
let interval: ReturnType<typeof setInterval> | null = null
let lastResult: ProgressResult | null = null
let pollInFlight = false

async function poll(): Promise<void> {
  // getGenerationProgress can be slow under the exact event-loop-starvation this PR exists to
  // handle — without this guard, a slow response plus this interval's next tick queues a second
  // overlapping request, and whichever resolves last "wins" regardless of which was actually
  // more recent, dispatching stale results out of order.
  if (pollInFlight) return
  pollInFlight = true
  try {
    const result = await ApiClient.getGenerationProgress()
    lastResult = result
    listeners.forEach(listener => listener(result))
  } finally {
    pollInFlight = false
  }
}

// The global generation lock and the background recovery watcher both need the same
// generation-progress poll while mounted together (any project open); sharing one interval
// instead of one per subscriber halves that network chatter.
export function subscribeToGenerationProgress(listener: Listener): () => void {
  listeners.add(listener)
  if (lastResult) listener(lastResult)
  if (!interval) {
    void poll()
    interval = setInterval(poll, POLL_INTERVAL_MS)
  }
  return () => {
    listeners.delete(listener)
    if (listeners.size === 0 && interval) {
      clearInterval(interval)
      interval = null
      // Otherwise a later generation's fresh subscribe (see subscribeWhileGenerationMayBeActive)
      // immediately replays this now-unrelated snapshot before its own first poll ever completes
      // - e.g. reporting "running" from a previous, unrelated session for a moment.
      lastResult = null
    }
  }
}

// Every generate-starting call site (see GenSpace.tsx's writeRecoveryContext calls) writes a
// recovery marker into localStorage BEFORE it starts, in this same renderer process — so no
// marker anywhere is proof nothing here could be running, checkable with a local read instead of
// a network call. Gates the shared poll on that: idle app (no generation ever started, or one
// that already finished and was consumed) costs zero network calls, not just at startup but for
// the whole session, since this re-checks on the same cadence as the poll it gates.
export function subscribeWhileGenerationMayBeActive(listener: Listener): () => void {
  let unsubscribe: (() => void) | null = null
  let latest: ProgressResult | null = null
  const trackingListener: Listener = result => {
    latest = result
    listener(result)
  }
  const sync = () => {
    const hasMarker = localStorage.getItem(GENERATION_RECOVERY_KEY) != null
    if (hasMarker) {
      if (!unsubscribe) unsubscribe = subscribeToGenerationProgress(trackingListener)
      return
    }
    // No marker doesn't mean safe to stop yet: a marker's writer can clear it eagerly, straight
    // off its own HTTP response (Enhance does — see GenSpace.tsx's runEnhance), without ever
    // going through a poll cycle first. If we unsubscribed immediately here, a shared poll tick
    // still mid-flight (or one that simply hasn't fired again yet) never gets to redeliver the
    // real terminal status, and the listener (e.g. useGlobalGenerationLock's isRunning) is stuck
    // on whatever "running" snapshot it last saw — forever, since nothing calls it again. Only
    // stop once a poll has actually caught up and confirmed we're not running.
    if (unsubscribe && latest && (!latest.ok || latest.data.status !== 'running')) {
      unsubscribe()
      unsubscribe = null
    }
  }
  sync()
  const interval = setInterval(sync, MARKER_CHECK_INTERVAL_MS)
  return () => {
    clearInterval(interval)
    unsubscribe?.()
  }
}
