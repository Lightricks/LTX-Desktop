import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { createBackendWsUrl } from './backend.ts'

describe('createBackendWsUrl', () => {
  it('binds WebSocket URLs to host credentials without a process global', async () => {
    const desktop = createBackendWsUrl(async () => ({
      url: 'http://127.0.0.1:41954',
      token: 'desk',
    }))
    const remote = createBackendWsUrl(async () => ({
      url: 'http://192.168.1.4:41955',
      token: 'phone',
    }))

    assert.equal(
      await desktop('/ws/progress'),
      'ws://127.0.0.1:41954/ws/progress?token=desk',
    )
    assert.equal(
      await remote('/ws/progress?x=1'),
      'ws://192.168.1.4:41955/ws/progress?x=1&token=phone',
    )
  })

  it('maps https origins to wss', async () => {
    const wsUrl = createBackendWsUrl(async () => ({
      url: 'https://example.test',
      token: 't',
    }))
    assert.equal(await wsUrl('/ws'), 'wss://example.test/ws?token=t')
  })
})
