import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import {
  buildAnalyticsPayload,
  mergeAnalyticsDetails,
  normalizeOsVersion,
  parseOsRelease,
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

describe('shouldDeliverAnalytics', () => {
  it('skips dev builds and opt-out', () => {
    assert.equal(shouldDeliverAnalytics({ dev: true, enabled: true }), false)
    assert.equal(shouldDeliverAnalytics({ dev: false, enabled: false }), false)
    assert.equal(shouldDeliverAnalytics({ dev: false, enabled: true }), true)
  })
})

describe('telemetry hardware and mode fields', () => {
  it('keeps backend hardware snapshot keys on generate_started', () => {
    const merged = mergeAnalyticsDetails(
      {
        generation_id: 'generation-1',
        gpu_model: 'NVIDIA GeForce RTX 5090',
        total_vram_bytes: 34089730048,
        free_vram_bytes: 30000000000,
        drive_type: 'nvme',
      },
      { ramGb: 96, gpuName: 'GPU' },
      'generate_started',
    )

    assert.equal(merged['gpu_model'], 'NVIDIA GeForce RTX 5090')
    assert.equal(merged['total_vram_bytes'], 34089730048)
    assert.equal(merged['free_vram_bytes'], 30000000000)
    assert.equal(merged['drive_type'], 'nvme')
  })

  it('keeps load and offload modes on generate_ended', () => {
    const merged = mergeAnalyticsDetails(
      {
        generation_id: 'generation-1',
        attempt: 1,
        outcome: 'succeeded',
        runtime_ms: 1000,
        load_mode: 'streaming_models_loading',
        offload_mode: 'DISK',
      },
      { ramGb: 96, gpuName: 'GPU' },
      'generate_ended',
    )

    assert.deepEqual(merged, {
      generation_id: 'generation-1',
      attempt: 1,
      outcome: 'succeeded',
      runtime_ms: 1000,
      load_mode: 'streaming_models_loading',
      offload_mode: 'DISK',
    })
  })
})

describe('os_version', () => {
  it('keeps the numeric part of each platform version and drops suffixes', () => {
    assert.equal(normalizeOsVersion('26.6.2'), '26.6.2')
    assert.equal(normalizeOsVersion('14.8'), '14.8')
    assert.equal(normalizeOsVersion('10.0.26100'), '10.0.26100')
    assert.equal(normalizeOsVersion('6.8.0-45-generic'), '6.8.0')
    assert.equal(normalizeOsVersion('5.15.153.1-microsoft-standard-WSL2'), '5.15.153.1')
  })

  it('returns undefined when the value does not start with a number', () => {
    assert.equal(normalizeOsVersion(''), undefined)
    assert.equal(normalizeOsVersion('custom-kernel'), undefined)
  })

  it('is recorded on launched and generate_started, not on generate_ended', () => {
    const hardware = { ramGb: 48, gpuName: 'GPU', osVersion: '26.6.2' }

    assert.equal(mergeAnalyticsDetails(null, hardware, 'launched')['os_version'], '26.6.2')
    assert.equal(
      mergeAnalyticsDetails({ generation_id: 'generation-1' }, hardware, 'generate_started')['os_version'],
      '26.6.2',
    )
    assert.equal(
      mergeAnalyticsDetails(
        { generation_id: 'generation-1', outcome: 'succeeded', runtime_ms: 1 },
        hardware,
        'generate_ended',
      )['os_version'],
      undefined,
    )
  })

  it('uses the sender value over a caller-supplied os_version and omits the key when unknown', () => {
    const sender = mergeAnalyticsDetails(
      { os_version: 'caller', os_distro: 'caller' },
      { ramGb: 48, osVersion: '26.6.2', osDistro: 'ubuntu-24.04' },
      'launched',
    )
    const unknown = mergeAnalyticsDetails(
      { os_version: 'caller', os_distro: 'caller' },
      { ramGb: 48 },
      'launched',
    )

    assert.equal(sender['os_version'], '26.6.2')
    assert.equal(sender['os_distro'], 'ubuntu-24.04')
    assert.equal('os_version' in unknown, false)
    assert.equal('os_distro' in unknown, false)
  })
})

describe('os_distro', () => {
  it('reads ID and VERSION_ID, with or without quotes', () => {
    assert.equal(parseOsRelease('NAME="Ubuntu"\nID=ubuntu\nVERSION_ID="24.04"\n'), 'ubuntu-24.04')
    assert.equal(parseOsRelease('ID=fedora\nVERSION_ID=41\n'), 'fedora-41')
  })

  it('returns only the ID for a rolling release', () => {
    assert.equal(parseOsRelease('NAME="Arch Linux"\nID=arch\nBUILD_ID=rolling\n'), 'arch')
  })

  it('rejects free text and missing IDs', () => {
    assert.equal(parseOsRelease(''), undefined)
    assert.equal(parseOsRelease('NAME=Foo\n'), undefined)
    assert.equal(parseOsRelease('ID="my custom distro"\n'), undefined)
    assert.equal(parseOsRelease('ID=ubuntu\nVERSION_ID="24.04 LTS (custom)"\n'), 'ubuntu')
  })

  it('is recorded on launched and generate_started, not on generate_ended', () => {
    const hardware = { ramGb: 48, osDistro: 'ubuntu-24.04' }

    assert.equal(mergeAnalyticsDetails(null, hardware, 'launched')['os_distro'], 'ubuntu-24.04')
    assert.equal(
      mergeAnalyticsDetails(
        { generation_id: 'generation-1', outcome: 'succeeded', runtime_ms: 1 },
        hardware,
        'generate_ended',
      )['os_distro'],
      undefined,
    )
  })
})
