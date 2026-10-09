export const ANALYTICS_SUBJECT = 'ltxdesktop_app_event'

export const ANALYTICS_EVENT_NAMES = [
  'launched',
  'generate_started',
  'generate_ended',
] as const

export type AnalyticsEventName = (typeof ANALYTICS_EVENT_NAMES)[number]

/** Keys the backend may forward. Hardware keys are applied by the sender. */
export const GENERATION_DETAIL_KEYS = [
  'generation_id',
  'client',
  'surface',
  'feature',
  'kind',
  'execution',
  'model',
  'resolution',
  'duration_sec',
  'fps',
  'lora_catalog_ids',
  'has_custom_lora',
  'outcome',
  'runtime_ms',
  'prompt_provenance',
  'error_code',
  'attempt',
  'output_count',
] as const

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

/** Keep equal to shared/analytics-ended-details.json. */
export const ENDED_DETAIL_KEYS = [
  'generation_id',
  'attempt',
  'outcome',
  'runtime_ms',
  'error_code',
] as const

/** Drop unknown keys. Launch and start events also record hardware once. */
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
