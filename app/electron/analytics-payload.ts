import endedDetailKeys from '../shared/analytics-ended-details.json' with { type: 'json' }
import generationDetailKeys from '../shared/generation-detail-keys.json' with { type: 'json' }

export const ANALYTICS_SUBJECT = 'ltxdesktop_app_event'

export const ANALYTICS_EVENT_NAMES = [
  'launched',
  'generate_started',
  'generate_ended',
] as const

export type AnalyticsEventName = (typeof ANALYTICS_EVENT_NAMES)[number]

/** Keys the backend may forward. Hardware keys are applied by the sender. Single source of truth: shared/generation-detail-keys.json. */
export const GENERATION_DETAIL_KEYS: readonly string[] = generationDetailKeys

export const ANALYTICS_DETAIL_KEYS = [
  ...GENERATION_DETAIL_KEYS,
  'ram_gb',
  'gpu_name',
] as const

export function shouldDeliverAnalytics(input: { dev: boolean; enabled: boolean }): boolean {
  return !input.dev && input.enabled
}

const ALLOWED_DETAIL_KEYS = new Set<string>(ANALYTICS_DETAIL_KEYS)

export interface AnalyticsHardware {
  ramGb: number
  gpuName?: string
  /** Operating system version, set by the sender only. See `normalizeOsVersion`. */
  osVersion?: string
  /** Linux distribution, set by the sender only. See `parseOsRelease`. */
  osDistro?: string
}

/**
 * Keep the leading numeric part of an OS version, for example "26.6.2" (macOS),
 * "10.0.26100" (Windows) or "6.8.0" from "6.8.0-45-generic" (Linux kernel).
 * Drop any suffix, so no host-specific text is sent. Return undefined when the
 * value does not start with a number.
 */
export function normalizeOsVersion(raw: string): string | undefined {
  return /^\d+(?:\.\d+){0,3}/.exec(raw.trim())?.[0]
}

const DISTRO_PART = /^[a-z0-9][a-z0-9._]{0,31}$/

/**
 * Read the distribution from the text of /etc/os-release. Return "ID-VERSION_ID"
 * (for example "ubuntu-24.04") or only "ID" for a rolling release such as "arch".
 * Accept only short lowercase tokens, so no free text is sent. Return undefined
 * when there is no valid ID.
 */
export function parseOsRelease(text: string): string | undefined {
  const fields = new Map<string, string>()
  for (const line of text.split('\n')) {
    const match = /^(ID|VERSION_ID)=(.*)$/.exec(line.trim())
    if (match) fields.set(match[1], match[2].replace(/^(["'])(.*)\1$/, '$2').toLowerCase())
  }
  const id = fields.get('ID')
  if (!id || !DISTRO_PART.test(id)) return undefined
  const versionId = fields.get('VERSION_ID')
  return versionId && DISTRO_PART.test(versionId) ? `${id}-${versionId}` : id
}

interface AnalyticsPayloadInput {
  eventName: AnalyticsEventName
  eventId: string
  timestamp: number
  appVersion: string
  installationId: string
  platform: NodeJS.Platform
  extraDetails?: Record<string, unknown> | null
}

export function analyticsPlatform(platform: NodeJS.Platform): string {
  const names: Partial<Record<NodeJS.Platform, string>> = {
    darwin: 'mac',
    win32: 'windows',
    linux: 'linux',
  }
  return names[platform] ?? platform
}

/** Single source of truth: shared/analytics-ended-details.json. */
export const ENDED_DETAIL_KEYS: readonly string[] = endedDetailKeys

/** Drop unknown keys. Launch and start events also record hardware and OS details once. */
export function mergeAnalyticsDetails(
  callerDetails: Record<string, unknown> | null | undefined,
  hardware: AnalyticsHardware,
  eventName: AnalyticsEventName = 'launched',
): Record<string, unknown> {
  const merged: Record<string, unknown> = {}
  if (callerDetails) {
    for (const [key, value] of Object.entries(callerDetails)) {
      if (ALLOWED_DETAIL_KEYS.has(key) && value !== undefined) merged[key] = value
    }
  }
  if (eventName === 'generate_ended') {
    const ended: Record<string, unknown> = {}
    for (const key of ENDED_DETAIL_KEYS) {
      if (merged[key] !== undefined) ended[key] = merged[key]
    }
    return ended
  }
  merged.ram_gb = hardware.ramGb
  if (hardware.gpuName) merged.gpu_name = hardware.gpuName
  if (hardware.osVersion) merged.os_version = hardware.osVersion
  if (hardware.osDistro) merged.os_distro = hardware.osDistro
  return merged
}

export function buildAnalyticsPayload(input: AnalyticsPayloadInput): {
  events: Array<{
    subject: string
    eventId: string
    eventTimestamp: number
    event: {
      app_version: string
      device_timestamp: number
      event_name: AnalyticsEventName
      installation_id: string
      platform: string
      extra_details: string | null
    }
  }>
} {
  return {
    events: [
      {
        subject: ANALYTICS_SUBJECT,
        eventId: input.eventId,
        eventTimestamp: input.timestamp,
        event: {
          app_version: input.appVersion,
          device_timestamp: input.timestamp,
          event_name: input.eventName,
          installation_id: input.installationId,
          platform: analyticsPlatform(input.platform),
          // The Avro field's underlying wire type is string. Its logicalType=json
          // makes this value a queryable JSON column after ingestion.
          extra_details: input.extraDetails ? JSON.stringify(input.extraDetails) : null,
        },
      },
    ],
  }
}
