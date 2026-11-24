import { useEffect, useState } from 'react'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import {
  nextRemoteStatus,
  REMOTE_STATUS_SETTLED_POLL_MS,
  remoteStatusEventFromPoll,
  remoteStatusPollDelayMs,
} from '../lib/remote-keep-awake-signal'

export type RemoteStatus = ApiSuccessOf<'getRemoteStatus'>

export function useRemoteStatus(
  active: boolean,
  settledMs: number = REMOTE_STATUS_SETTLED_POLL_MS,
): {
  status: RemoteStatus | null
  unreachable: boolean
} {
  const [status, setStatus] = useState<RemoteStatus | null>(null)
  const [unreachable, setUnreachable] = useState(false)

  useEffect(() => {
    if (!active) {
      setStatus(nextRemoteStatus({ type: 'deactivate' }))
      setUnreachable(false)
      return
    }
    let cancelled = false
    let timer: number | undefined
    let inFlight = false
    const poll = async () => {
      if (inFlight) return
      inFlight = true
      window.clearTimeout(timer)
      let next: RemoteStatus | null = null
      try {
        const result = await ApiClient.getRemoteStatus()
        if (cancelled) return
        next = nextRemoteStatus(remoteStatusEventFromPoll(result))
        setStatus(next)
        setUnreachable(!result.ok)
      } catch {
        // Drop stale serving so keep-awake can release after a crash or unreachable backend.
        if (cancelled) return
        setStatus(nextRemoteStatus({ type: 'failure' }))
        setUnreachable(true)
      } finally {
        inFlight = false
      }
      timer = window.setTimeout(() => { void poll() }, remoteStatusPollDelayMs(next, settledMs))
    }
    const onFocus = () => { void poll() }
    void poll()
    window.addEventListener('focus', onFocus)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
      window.removeEventListener('focus', onFocus)
    }
  }, [active, settledMs])

  return { status, unreachable }
}
