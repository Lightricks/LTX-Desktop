import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { pickGenerationDetails, startAnalyticsSink } from './analytics-sink.ts'

describe('pickGenerationDetails', () => {
  it('drops prompts and filesystem paths', () => {
    assert.deepEqual(pickGenerationDetails({
      generation_id: 'generation-1',
      prompt: 'a fox',
      video_path: '/tmp/out.mp4',
      client: 'desktop',
    }), {
      generation_id: 'generation-1',
      client: 'desktop',
    })
  })
})

describe('analytics sink', () => {
  it('forwards allowed generation events and rejects a bad token', async () => {
    const forwarded: Array<{ eventName: string; extraDetails: Record<string, unknown> }> = []
    const sink = await startAnalyticsSink(async (eventName, extraDetails) => {
      forwarded.push({ eventName, extraDetails })
    })

    try {
      const rejected = await fetch(sink.url, {
        method: 'POST',
        headers: {
          Authorization: 'Bearer wrong',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          eventName: 'generate_started',
          extraDetails: { generation_id: 'generation-1' },
        }),
      })
      assert.equal(rejected.status, 401)
      assert.deepEqual(forwarded, [])

      const accepted = await fetch(sink.url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${sink.token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          eventName: 'generate_started',
          extraDetails: {
            generation_id: 'generation-1',
            prompt: 'a fox',
            video_path: '/tmp/out.mp4',
          },
        }),
      })
      assert.equal(accepted.status, 204)
      assert.deepEqual(forwarded, [{
        eventName: 'generate_started',
        extraDetails: { generation_id: 'generation-1' },
      }])
    } finally {
      await sink.close()
    }
  })
})
