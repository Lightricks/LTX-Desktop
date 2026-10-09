import { randomUUID } from 'crypto';
import { app } from 'electron';
import os from 'os';
import {
  buildAnalyticsPayload,
  mergeAnalyticsDetails,
  shouldDeliverAnalytics,
  type AnalyticsEventName,
  type AnalyticsHardware,
} from './analytics-payload';
import { isDev } from './config';
import { readAppState, writeAppState } from './app-state';

const ANALYTICS_ENDPOINT = 'https://ltx-desktop.lightricks.com/v2/ingest';
const REQUEST_TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;
const RETRY_DELAYS_MS = [1000, 3000, 10000]
const MAX_INGEST_REQUESTS = 200
const INGEST_WINDOW_MS = 5 * 60 * 1000
const ingestTimestamps: number[] = []

function pruneIngestTimestamps(now: number): void {
  const cutoff = now - INGEST_WINDOW_MS
  while (ingestTimestamps.length > 0 && ingestTimestamps[0] <= cutoff) {
    ingestTimestamps.shift()
  }
}

function ingestAtCapacity(): boolean {
  pruneIngestTimestamps(Date.now())
  return ingestTimestamps.length >= MAX_INGEST_REQUESTS
}

function tryAcquireIngest(): boolean {
  const now = Date.now()
  pruneIngestTimestamps(now)
  if (ingestTimestamps.length >= MAX_INGEST_REQUESTS) return false
  ingestTimestamps.push(now)
  return true
}

export function getAnalyticsState(): { analyticsEnabled: boolean; installationId: string } {
  const state = readAppState()
  return {
    analyticsEnabled: state.analyticsEnabled !== false,
    installationId: state.installationId ?? '',
  }
}

export function setAnalyticsEnabled(enabled: boolean): void {
  const state = readAppState()
  state.analyticsEnabled = enabled
  // Generate installationId on first enable; persist forever after
  if (enabled && !state.installationId) {
    state.installationId = randomUUID()
  }
  writeAppState(state)
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function isRetryable(status: number): boolean {
  return status === 429 || status >= 500
}

async function sendWithRetry(
  url: string,
  options: RequestInit,
): Promise<void> {
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (!tryAcquireIngest()) return

    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
      const response = await fetch(url, { ...options, signal: controller.signal })
      clearTimeout(timeout)

      if (response.ok || !isRetryable(response.status)) return
    } catch (err) {
      console.warn('[analytics] request attempt failed:', err)
    }

    if (attempt < MAX_RETRIES) {
      await delay(RETRY_DELAYS_MS[attempt])
    }
  }
}

function readGpuName(info: unknown): string | undefined {
  if (typeof info !== 'object' || info === null || !('gpuDevice' in info)) return undefined
  const devices = (info as { gpuDevice?: unknown }).gpuDevice
  if (!Array.isArray(devices)) return undefined
  const named = devices.filter((device): device is { active?: boolean; deviceString?: string } => (
    typeof device === 'object' && device !== null
  ))
  const selected = named.find((device) => device.active) ?? named[0]
  return typeof selected?.deviceString === 'string' && selected.deviceString
    ? selected.deviceString
    : undefined
}

let hardwarePromise: Promise<AnalyticsHardware> | null = null

function currentHardware(): Promise<AnalyticsHardware> {
  hardwarePromise ??= loadHardware()
  return hardwarePromise
}

async function loadHardware(): Promise<AnalyticsHardware> {
  const hardware: AnalyticsHardware = {
    ramGb: Math.round(os.totalmem() / 1024 ** 3),
  }
  try {
    hardware.gpuName = readGpuName(await app.getGPUInfo('basic'))
  } catch {
    // Hardware metadata is optional and must never prevent an event.
  }
  return hardware
}

export async function sendAnalyticsEvent(
  eventName: AnalyticsEventName,
  extraDetails?: Record<string, unknown> | null,
): Promise<void> {
  try {
    const state = readAppState()
    if (!shouldDeliverAnalytics({
      dev: isDev,
      enabled: state.analyticsEnabled !== false,
    })) return
    if (ingestAtCapacity()) return

    // Generate installationId on first send
    if (!state.installationId) {
      state.installationId = randomUUID()
      writeAppState(state)
    }

    const now = Date.now()
    const payload = buildAnalyticsPayload({
      eventName,
      eventId: randomUUID(),
      timestamp: now,
      appVersion: app.getVersion(),
      installationId: state.installationId,
      platform: process.platform,
      extraDetails: mergeAnalyticsDetails(extraDetails, await currentHardware(), eventName),
    })

    // Fire-and-forget with retries — never throws
    void sendWithRetry(ANALYTICS_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
  } catch (err) {
    console.error('[analytics] failed to send event:', err)
  }
}
