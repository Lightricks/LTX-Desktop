import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import {
  nextRemoteStatus,
  REMOTE_STATUS_SETTLED_POLL_MS,
  REMOTE_STATUS_UNSETTLED_POLL_MS,
  remoteStatusEventFromPoll,
  remoteStatusPollDelayMs,
  shouldNotifyRemoteKeepAwake,
  type RemoteKeepAwakeStatus,
} from './remote-keep-awake-signal.ts'

const serving: RemoteKeepAwakeStatus = {
  permitted: true,
  serving: true,
  reason: null,
}

const starting: RemoteKeepAwakeStatus = {
  permitted: true,
  serving: false,
  reason: 'starting',
}

describe('shouldNotifyRemoteKeepAwake', () => {
  it('holds only while Remote is enabled, the backend is alive, and status is permitted starting or serving', () => {
    assert.equal(
      shouldNotifyRemoteKeepAwake({ remoteEnabled: true, backendAlive: true, status: serving }),
      true,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({ remoteEnabled: true, backendAlive: true, status: starting }),
      true,
    )
  })

  it('releases when Remote is disabled or the backend is not alive, even with a stale serving status', () => {
    assert.equal(
      shouldNotifyRemoteKeepAwake({ remoteEnabled: false, backendAlive: true, status: serving }),
      false,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({ remoteEnabled: true, backendAlive: false, status: serving }),
      false,
    )
  })

  it('releases when status is missing, not permitted, failed, stopped, or otherwise not serving', () => {
    assert.equal(
      shouldNotifyRemoteKeepAwake({ remoteEnabled: true, backendAlive: true, status: null }),
      false,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: { permitted: false, serving: true, reason: 'not permitted' },
      }),
      false,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: { permitted: false, serving: false, reason: 'starting' },
      }),
      false,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: { permitted: true, serving: false, reason: 'remote server failed to start' },
      }),
      false,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: { permitted: true, serving: false, reason: 'stopped' },
      }),
      false,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: { permitted: true, serving: false, reason: 'remote server stopped unexpectedly' },
      }),
      false,
    )
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: { permitted: true, serving: false, reason: null },
      }),
      false,
    )
  })
})

describe('nextRemoteStatus', () => {
  it('replaces status on a successful poll', () => {
    assert.deepEqual(nextRemoteStatus({ type: 'success', status: serving }), serving)
    assert.deepEqual(
      nextRemoteStatus(remoteStatusEventFromPoll({ ok: true, data: starting })),
      starting,
    )
  })

  it('clears a stale serving status on deactivate', () => {
    assert.equal(nextRemoteStatus({ type: 'deactivate' }), null)
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: nextRemoteStatus({ type: 'deactivate' }),
      }),
      false,
    )
  })

  it('clears a stale serving status on polling, network, or 401 failure', () => {
    assert.deepEqual(remoteStatusEventFromPoll({ ok: false }), { type: 'failure' })
    assert.equal(nextRemoteStatus({ type: 'failure' }), null)
    assert.equal(nextRemoteStatus(remoteStatusEventFromPoll({ ok: false })), null)
    assert.equal(
      shouldNotifyRemoteKeepAwake({
        remoteEnabled: true,
        backendAlive: true,
        status: nextRemoteStatus(remoteStatusEventFromPoll({ ok: false })),
      }),
      false,
    )
  })
})

describe('remote status poll delay', () => {
  it('polls fast until the server is serving, then slowly', () => {
    const settled = REMOTE_STATUS_SETTLED_POLL_MS
    assert.equal(remoteStatusPollDelayMs(null, settled), REMOTE_STATUS_UNSETTLED_POLL_MS)
    assert.equal(
      remoteStatusPollDelayMs({ serving: false }, settled),
      REMOTE_STATUS_UNSETTLED_POLL_MS,
    )
    assert.equal(remoteStatusPollDelayMs({ serving: true }, settled), settled)
    assert.equal(remoteStatusPollDelayMs({ serving: true }, 5000), 5000)
  })
})
