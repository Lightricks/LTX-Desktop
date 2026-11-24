import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { createAnalyticsOptOutWrites } from './analytics-opt-out-write.ts'

describe('createAnalyticsOptOutWrites', () => {
  it('lets Install wait until the opt-out write finishes', async () => {
    let releaseWrite: () => void = () => {}
    const writeGate = new Promise<void>((resolve) => {
      releaseWrite = resolve
    })
    const order: string[] = []
    const writes = createAnalyticsOptOutWrites(async (enabled) => {
      order.push(`write:${enabled}`)
      await writeGate
    })

    writes.enqueue(false, () => {})
    const install = (async () => {
      await writes.flush()
      order.push('install-signaled')
    })()

    await Promise.resolve()
    assert.deepEqual(order, ['write:false'])

    releaseWrite()
    await install

    assert.deepEqual(order, ['write:false', 'install-signaled'])
  })

  it('waits for a toggle that arrives while an earlier write is still in flight', async () => {
    let releaseFirst: () => void = () => {}
    let releaseSecond: () => void = () => {}
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve
    })
    const secondGate = new Promise<void>((resolve) => {
      releaseSecond = resolve
    })
    let calls = 0
    const writes = createAnalyticsOptOutWrites(async () => {
      calls += 1
      await (calls === 1 ? firstGate : secondGate)
    })

    writes.enqueue(false, () => {})
    let flushed = false
    const done = writes.flush().then(() => {
      flushed = true
    })

    await Promise.resolve()
    writes.enqueue(true, () => {})
    releaseFirst()
    await Promise.resolve()
    assert.equal(flushed, false)

    releaseSecond()
    await done
    assert.equal(flushed, true)
  })

  it('rejects flush when the latest write fails, then accepts a later success', async () => {
    let failNext = true
    const writes = createAnalyticsOptOutWrites(async () => {
      if (failNext) throw new Error('disk full')
    })

    let reverted = false
    writes.enqueue(false, () => {
      reverted = true
    })
    await assert.rejects(writes.flush(), /analytics choice/)
    assert.equal(reverted, true)

    failNext = false
    writes.enqueue(false, () => {})
    await writes.flush()
  })
})
