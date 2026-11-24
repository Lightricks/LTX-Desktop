import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, it } from 'node:test'
import {
  GENERATION_HEARTBEAT_FILENAME,
  clearHeartbeatFile,
  decideThresholdTick,
  isHeartbeatFresh,
  readHeartbeatState,
} from './liveness-grace.ts'

const MAX_AGE_MS = 180_000
const GRACE_CAP_MS = 600_000
const LIMITS = { graceMaxHeartbeatAgeMs: MAX_AGE_MS, graceCapMs: GRACE_CAP_MS }

function writeHeartbeat(dir: string, contents: string): void {
  fs.writeFileSync(path.join(dir, GENERATION_HEARTBEAT_FILENAME), contents)
}

function withTempDir(fn: (dir: string) => void): void {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'liveness-grace-'))
  try {
    fn(dir)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
}

describe('isHeartbeatFresh', () => {
  it('graces a fresh heartbeat from this backend instance', () => {
    assert.equal(isHeartbeatFresh({ pid: 42, updatedAtMs: 1_000 }, 42, 1_000 + MAX_AGE_MS - 1, MAX_AGE_MS), true)
  })

  it('withholds grace once the heartbeat goes stale, so a hung backend still restarts', () => {
    assert.equal(isHeartbeatFresh({ pid: 42, updatedAtMs: 1_000 }, 42, 1_000 + MAX_AGE_MS + 60_000, MAX_AGE_MS), false)
  })

  it('withholds grace at exactly max age, not one tick past it', () => {
    assert.equal(isHeartbeatFresh({ pid: 42, updatedAtMs: 1_000 }, 42, 1_000 + MAX_AGE_MS, MAX_AGE_MS), false)
  })

  it('withholds grace when the pid belongs to a dead backend, not this one', () => {
    assert.equal(isHeartbeatFresh({ pid: 41, updatedAtMs: 1_000 }, 42, 1_001, MAX_AGE_MS), false)
  })

  it('withholds grace when the backend pid is unknown', () => {
    assert.equal(isHeartbeatFresh({ pid: 42, updatedAtMs: 1_000 }, undefined, 1_001, MAX_AGE_MS), false)
  })

  it('withholds grace when no heartbeat was read', () => {
    assert.equal(isHeartbeatFresh(null, 42, 1_001, MAX_AGE_MS), false)
  })
})

describe('readHeartbeatState', () => {
  it('reads a well-formed heartbeat file', () => {
    withTempDir((dir) => {
      writeHeartbeat(dir, JSON.stringify({ pid: 42, updatedAtMs: 1_000 }))
      assert.deepEqual(readHeartbeatState(dir), { pid: 42, updatedAtMs: 1_000 })
    })
  })

  it('returns null instead of throwing on malformed json, so the monitor tick survives', () => {
    withTempDir((dir) => {
      writeHeartbeat(dir, '{"pid": 42, oops')
      assert.equal(readHeartbeatState(dir), null)
    })
  })

  it('returns null on wrong-shaped json instead of gracing garbage', () => {
    withTempDir((dir) => {
      writeHeartbeat(dir, JSON.stringify({ pid: '42', updatedAtMs: 1_000 }))
      assert.equal(readHeartbeatState(dir), null)
      writeHeartbeat(dir, JSON.stringify([42, 1_000]))
      assert.equal(readHeartbeatState(dir), null)
    })
  })

  it('returns null when the file is missing (idle backend)', () => {
    withTempDir((dir) => {
      assert.equal(readHeartbeatState(dir), null)
    })
  })
})

describe('clearHeartbeatFile', () => {
  it('removes the file and tolerates a missing one', () => {
    withTempDir((dir) => {
      writeHeartbeat(dir, JSON.stringify({ pid: 42, updatedAtMs: 1_000 }))
      clearHeartbeatFile(dir)
      assert.equal(fs.existsSync(path.join(dir, GENERATION_HEARTBEAT_FILENAME)), false)
      clearHeartbeatFile(dir)
    })
  })
})

describe('decideThresholdTick', () => {
  it('graces a fresh heartbeat and stamps the cap clock, or the cap never binds', () => {
    assert.deepEqual(
      decideThresholdTick(
        { graceSinceMs: null, heartbeat: { pid: 42, updatedAtMs: 1_000 }, backendPid: 42, nowMs: 1_001, ...LIMITS },
      ),
      { action: 'grace', graceSinceMs: 1_001 },
    )
  })

  it('keeps the original grace stamp across ticks, or re-stamping graces a deadlock forever', () => {
    assert.deepEqual(
      decideThresholdTick(
        { graceSinceMs: 1_000, heartbeat: { pid: 42, updatedAtMs: 1_000 + GRACE_CAP_MS - 2 }, backendPid: 42, nowMs: 1_000 + GRACE_CAP_MS - 1, ...LIMITS },
      ),
      { action: 'grace', graceSinceMs: 1_000 },
    )
  })

  it('kills at exactly the grace cap despite a fresh heartbeat, so a deadlocked writer still restarts', () => {
    assert.deepEqual(
      decideThresholdTick(
        { graceSinceMs: 1_000, heartbeat: { pid: 42, updatedAtMs: 1_000 + GRACE_CAP_MS }, backendPid: 42, nowMs: 1_000 + GRACE_CAP_MS, ...LIMITS },
      ),
      { action: 'kill', reason: 'grace-cap-exceeded' },
    )
  })

  it('kills on a stale heartbeat, or a hung backend never restarts', () => {
    assert.deepEqual(
      decideThresholdTick(
        { graceSinceMs: null, heartbeat: { pid: 42, updatedAtMs: 1_000 }, backendPid: 42, nowMs: 1_000 + MAX_AGE_MS + 1, ...LIMITS },
      ),
      { action: 'kill', reason: 'no-fresh-heartbeat' },
    )
  })

  it('kills when no heartbeat was read (idle hung backend)', () => {
    assert.deepEqual(
      decideThresholdTick(
        { graceSinceMs: null, heartbeat: null, backendPid: 42, nowMs: 1_001, ...LIMITS },
      ),
      { action: 'kill', reason: 'no-fresh-heartbeat' },
    )
  })
})
