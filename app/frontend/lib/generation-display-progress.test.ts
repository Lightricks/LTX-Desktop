import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { displayPolledProgress } from './generation-display-progress.ts'

describe('displayPolledProgress', () => {
  it('uses backend progress when inference reports a real step total', () => {
    assert.equal(
      displayPolledProgress({
        phase: 'inference',
        progress: 42,
        totalSteps: 11,
        elapsedInferenceS: 40,
        estimatedInferenceS: 45,
      }),
      42,
    )
  })

  it('still interpolates API inference that has no step total', () => {
    assert.equal(
      displayPolledProgress({
        phase: 'inference',
        progress: 15,
        totalSteps: null,
        elapsedInferenceS: 22.5,
        estimatedInferenceS: 45,
      }),
      55,
    )
  })

  it('interpolates when totalSteps is 0 (no denoise sink yet)', () => {
    assert.equal(
      displayPolledProgress({
        phase: 'inference',
        progress: 15,
        totalSteps: 0,
        elapsedInferenceS: 22.5,
        estimatedInferenceS: 45,
      }),
      55,
    )
  })

  it('interpolates placeholder totalSteps of 1', () => {
    assert.equal(
      displayPolledProgress({
        phase: 'inference',
        progress: 15,
        totalSteps: 1,
        elapsedInferenceS: 22.5,
        estimatedInferenceS: 45,
      }),
      55,
    )
  })

  it('caps a finished denoise pass at 95 while decode is still running', () => {
    assert.equal(
      displayPolledProgress({
        phase: 'inference',
        progress: 100,
        totalSteps: 11,
        elapsedInferenceS: 80,
        estimatedInferenceS: 45,
      }),
      95,
    )
  })

  it('holds at 95 while phase is complete', () => {
    assert.equal(
      displayPolledProgress({
        phase: 'complete',
        progress: 100,
        totalSteps: 11,
        elapsedInferenceS: 80,
        estimatedInferenceS: 45,
      }),
      95,
    )
  })
})
