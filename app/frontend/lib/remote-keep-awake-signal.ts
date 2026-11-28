export type RemoteKeepAwakeStatus = {
  permitted: boolean
  serving: boolean
  reason?: string | null
}

export type RemoteStatusEvent<TStatus> =
  | { type: 'deactivate' }
  | { type: 'failure' }
  | { type: 'success'; status: TStatus }

export function nextRemoteStatus<TStatus>(
  event: RemoteStatusEvent<TStatus>,
): TStatus | null {
  switch (event.type) {
    case 'deactivate':
    case 'failure':
      return null
    case 'success':
      return event.status
    default: {
      const exhaustive: never = event
      return exhaustive
    }
  }
}

export function remoteStatusEventFromPoll<TStatus>(
  result: { ok: true; data: TStatus } | { ok: false },
): RemoteStatusEvent<TStatus> {
  return result.ok
    ? { type: 'success', status: result.data }
    : { type: 'failure' }
}

export function shouldNotifyRemoteKeepAwake(input: {
  remoteEnabled: boolean
  backendAlive: boolean
  status: RemoteKeepAwakeStatus | null
}): boolean {
  if (!input.remoteEnabled || !input.backendAlive || input.status === null) {
    return false
  }
  if (!input.status.permitted) {
    return false
  }
  return input.status.serving || input.status.reason === 'starting'
}

/** Fast while the server is starting or unreachable, so state changes show up promptly. */
export const REMOTE_STATUS_UNSETTLED_POLL_MS = 2000
/** Once serving, polling only has to notice a crash or a revoked permit. */
export const REMOTE_STATUS_SETTLED_POLL_MS = 30_000

export function remoteStatusPollDelayMs(
  status: Pick<RemoteKeepAwakeStatus, 'serving'> | null,
  settledMs: number,
): number {
  return status?.serving ? settledMs : REMOTE_STATUS_UNSETTLED_POLL_MS
}
