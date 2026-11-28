import fs from 'fs'
import path from 'path'

// Kill-grace for the backend liveness monitor, driven by the generation
// proof-of-life file the backend refreshes while a local generation executes
// (backend/server_utils/generation_liveness.py). A GIL-saturated backend
// can't serve /health but is still making progress — killing it would destroy
// a healthy multi-minute generation. Every failure here resolves to null
// (no grace): a malformed file must fail closed to the kill, and must never
// throw out of the monitor tick.

export const GENERATION_HEARTBEAT_FILENAME = 'generation-heartbeat.json'

export interface LivenessHeartbeatState {
  pid: number
  updatedAtMs: number
}

export function readHeartbeatState(appDataDir: string): LivenessHeartbeatState | null {
  try {
    const raw = fs.readFileSync(path.join(appDataDir, GENERATION_HEARTBEAT_FILENAME), 'utf8')
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed !== 'object' || parsed === null) return null
    const { pid, updatedAtMs } = parsed as Record<string, unknown>
    if (typeof pid !== 'number' || typeof updatedAtMs !== 'number') return null
    if (!Number.isFinite(pid) || !Number.isFinite(updatedAtMs)) return null
    return { pid, updatedAtMs }
  } catch {
    return null
  }
}

export function isHeartbeatFresh(
  state: LivenessHeartbeatState | null,
  backendPid: number | undefined,
  nowMs: number,
  maxAgeMs: number,
): boolean {
  if (state === null || backendPid === undefined) return false
  // The pid pins the file to this backend instance: a crashed backend's
  // leftover file must not grace its hung replacement.
  if (state.pid !== backendPid) return false
  return nowMs - state.updatedAtMs < maxAgeMs
}

export function clearHeartbeatFile(appDataDir: string): void {
  try {
    fs.unlinkSync(path.join(appDataDir, GENERATION_HEARTBEAT_FILENAME))
  } catch {
    // Missing is the common case; anything else fails safe (a stale file
    // fails the pid/age check above).
  }
}

export type ThresholdTickOutcome =
  | { action: 'grace'; graceSinceMs: number }
  | { action: 'kill'; reason: 'grace-cap-exceeded' | 'no-fresh-heartbeat' }

export interface ThresholdTickInput {
  graceSinceMs: number | null
  heartbeat: LivenessHeartbeatState | null
  backendPid: number | undefined
  nowMs: number
  graceMaxHeartbeatAgeMs: number
  graceCapMs: number
}

export function decideThresholdTick(input: ThresholdTickInput): ThresholdTickOutcome {
  // Pure core of the monitor's threshold branch (python-backend.ts): a fresh
  // heartbeat from this backend instance extends the grace, otherwise the
  // backend is killed so the exit handler restarts it. The grace clock stamps
  // once, on the first graced tick — re-stamping every tick would let the cap
  // never bind and grace a deadlocked backend forever. Every failure resolves
  // to the kill, like the helpers above.
  if (!isHeartbeatFresh(input.heartbeat, input.backendPid, input.nowMs, input.graceMaxHeartbeatAgeMs)) {
    return { action: 'kill', reason: 'no-fresh-heartbeat' }
  }
  const since = input.graceSinceMs ?? input.nowMs
  if (input.nowMs - since < input.graceCapMs) {
    return { action: 'grace', graceSinceMs: since }
  }
  return { action: 'kill', reason: 'grace-cap-exceeded' }
}
