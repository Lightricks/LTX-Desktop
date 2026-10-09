import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { describe, it } from 'node:test'

import {
  ENDED_DETAIL_KEYS,
  buildAnalyticsPayload,
  mergeAnalyticsDetails,
  shouldDeliverAnalytics,
} from './analytics-payload.ts'

const launchInput = {
  eventName: 'launched' as const,
  eventId: 'event-1',
  timestamp: 123,
  appVersion: '1.2.7',
  installationId: 'install-1',
  platform: 'darwin' as const,
}

describe('buildAnalyticsPayload', () => {
  it('does not send the retired app-launched subject', () => {
    const payload = buildAnalyticsPayload(launchInput)

    assert.equal(payload.events[0].subject, 'ltxdesktop_app_event')
    assert.equal(payload.events[0].event.event_name, 'launched')
  })

  it('serializes logical JSON details exactly once', () => {
    const payload = buildAnalyticsPayload({
      ...launchInput,
      platform: 'win32',
      extraDetails: { ram_gb: 32, gpu_name: 'GPU' },
    })

    assert.deepEqual(JSON.parse(payload.events[0].event.extra_details ?? ''), {
      ram_gb: 32,
      gpu_name: 'GPU',
    })
    assert.equal(payload.events[0].event.platform, 'windows')
  })
})

describe('mergeAnalyticsDetails', () => {
  it('drops unknown keys and records hardware once', () => {
    const merged = mergeAnalyticsDetails(
      {
        ram_gb: 1,
        gpu_name: 'caller',
        prompt: 'a fox',
        video_path: '/tmp/out.mp4',
        generation_id: 'generation-1',
      },
      { ramGb: 32, gpuName: 'GPU' },
    )

    assert.deepEqual(merged, {
      generation_id: 'generation-1',
      ram_gb: 32,
      gpu_name: 'GPU',
    })
  })

  it('keeps only the join key and the end-only fields', () => {
    const merged = mergeAnalyticsDetails(
      {
        generation_id: 'generation-1',
        attempt: 1,
        surface: 'home',
        feature: 'cozy-felt',
        prompt_provenance: 'raw',
        ram_gb: 48,
        outcome: 'cancelled',
        runtime_ms: 33892,
      },
      { ramGb: 48, gpuName: 'GPU' },
      'generate_ended',
    )

    assert.deepEqual(merged, {
      generation_id: 'generation-1',
      attempt: 1,
      outcome: 'cancelled',
      runtime_ms: 33892,
    })
  })
})

describe('ENDED_DETAIL_KEYS', () => {
  it('matches the shared contract', () => {
    const contract = JSON.parse(readFileSync(
      new URL('../shared/analytics-ended-details.json', import.meta.url),
      'utf8',
    ))
    assert.deepEqual(contract, [...ENDED_DETAIL_KEYS])
  })
})

describe('shouldDeliverAnalytics', () => {
  it('skips dev builds and opt-out', () => {
    assert.equal(shouldDeliverAnalytics({ dev: true, enabled: true }), false)
    assert.equal(shouldDeliverAnalytics({ dev: false, enabled: false }), false)
    assert.equal(shouldDeliverAnalytics({ dev: false, enabled: true }), true)
  })
})
