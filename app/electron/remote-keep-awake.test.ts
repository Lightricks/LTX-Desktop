import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { createRemoteKeepAwake, shouldHoldRemoteSleepBlocker } from './remote-keep-awake.ts'

function harness(onBattery = false) {
  const started = new Set<number>()
  const acListeners: Array<() => void> = []
  const batteryListeners: Array<() => void> = []
  let nextId = 1
  const powerState = { onBattery }
  const keepAwake = createRemoteKeepAwake({
    power: {
      isOnBatteryPower: () => powerState.onBattery,
      onAc: (listener) => { acListeners.push(listener) },
      onBattery: (listener) => { batteryListeners.push(listener) },
    },
    blocker: {
      start: () => {
        const id = nextId
        nextId += 1
        started.add(id)
        return id
      },
      stop: (id) => { started.delete(id) },
      isStarted: (id) => started.has(id),
    },
  })
  return {
    keepAwake,
    started,
    powerState,
    plugIn: () => {
      powerState.onBattery = false
      for (const listener of acListeners) listener()
    },
    unplug: () => {
      powerState.onBattery = true
      for (const listener of batteryListeners) listener()
    },
  }
}

describe('shouldHoldRemoteSleepBlocker', () => {
  it('holds only while remote is on and the machine is on AC', () => {
    assert.equal(shouldHoldRemoteSleepBlocker(true, false), true)
    assert.equal(shouldHoldRemoteSleepBlocker(true, true), false)
    assert.equal(shouldHoldRemoteSleepBlocker(false, false), false)
    assert.equal(shouldHoldRemoteSleepBlocker(false, true), false)
  })
})

describe('createRemoteKeepAwake', () => {
  it('does not start a blocker while remote is off', () => {
    const { keepAwake, started } = harness()
    keepAwake.init()
    keepAwake.setRemoteExposureActive(false)
    assert.equal(started.size, 0)
  })

  it('starts prevent-app-suspension when remote turns on while on AC', () => {
    const { keepAwake, started } = harness(false)
    keepAwake.init()
    keepAwake.setRemoteExposureActive(true)
    assert.equal(started.size, 1)
  })

  it('does not start a second blocker if remote is already on', () => {
    const { keepAwake, started } = harness(false)
    keepAwake.setRemoteExposureActive(true)
    keepAwake.setRemoteExposureActive(true)
    assert.equal(started.size, 1)
  })

  it('does not start while on battery, then starts after AC returns', () => {
    const { keepAwake, started, unplug, plugIn } = harness(true)
    keepAwake.init()
    keepAwake.setRemoteExposureActive(true)
    assert.equal(started.size, 0)
    unplug()
    assert.equal(started.size, 0)
    plugIn()
    assert.equal(started.size, 1)
  })

  it('releases the blocker when the machine goes on battery', () => {
    const { keepAwake, started, unplug } = harness(false)
    keepAwake.init()
    keepAwake.setRemoteExposureActive(true)
    assert.equal(started.size, 1)
    unplug()
    assert.equal(started.size, 0)
  })

  it('releases the blocker when remote turns off', () => {
    const { keepAwake, started } = harness(false)
    keepAwake.setRemoteExposureActive(true)
    keepAwake.setRemoteExposureActive(false)
    assert.equal(started.size, 0)
  })

  it('stop() releases even if remote was still on', () => {
    const { keepAwake, started } = harness(false)
    keepAwake.setRemoteExposureActive(true)
    keepAwake.stop()
    assert.equal(started.size, 0)
  })

  it('restarts the blocker if the OS dropped it while remote is still on AC', () => {
    const { keepAwake, started } = harness(false)
    keepAwake.setRemoteExposureActive(true)
    const [id] = started
    assert.ok(id !== undefined)
    started.delete(id)
    keepAwake.setRemoteExposureActive(true)
    assert.equal(started.size, 1)
    assert.equal(started.has(id), false)
  })

  it('init is idempotent and still syncs current power state', () => {
    const { keepAwake, started } = harness(false)
    keepAwake.setRemoteExposureActive(true)
    keepAwake.init()
    keepAwake.init()
    assert.equal(started.size, 1)
  })
})
