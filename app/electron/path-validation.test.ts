import assert from 'node:assert/strict'
import path from 'node:path'
import { describe, it } from 'node:test'

import * as pathValidation from './path-validation.ts'

type PathValidationWithChildResolver = typeof pathValidation & {
  resolvePathWithinDirectory?: (directory: string, childPath: string) => string
  validateTempRecordingInput?: (suffix: unknown, data: unknown) => void
}

const { resolvePathWithinDirectory, validateTempRecordingInput } =
  pathValidation as PathValidationWithChildResolver

describe('resolvePathWithinDirectory', () => {
  it('accepts a file directly inside the directory', () => {
    assert.equal(typeof resolvePathWithinDirectory, 'function')
    assert.equal(
      resolvePathWithinDirectory?.('/tmp/ltx-desktop-recordings', 'recording.wav'),
      path.join('/tmp/ltx-desktop-recordings', 'recording.wav'),
    )
  })

  it('rejects child paths that escape the directory', () => {
    assert.equal(typeof resolvePathWithinDirectory, 'function')
    assert.throws(
      () =>
        resolvePathWithinDirectory?.(
          '/tmp/ltx-desktop-recordings',
          '.wav/../../../unwanted-file',
        ),
      /must remain within the directory/,
    )
  })
})

describe('validateTempRecordingInput', () => {
  it('rejects a suffix other than wav', () => {
    assert.equal(typeof validateTempRecordingInput, 'function')
    assert.throws(
      () =>
        validateTempRecordingInput?.(
          '.wav/../../../unwanted-file',
          new ArrayBuffer(1),
        ),
      /Only WAV recording files are supported/,
    )
  })

  it('rejects recording data over 100 MB', () => {
    assert.equal(typeof validateTempRecordingInput, 'function')
    assert.throws(
      () =>
        validateTempRecordingInput?.(
          '.wav',
          new ArrayBuffer(100 * 1024 * 1024 + 1),
        ),
      /Recording data exceeds the maximum supported size/,
    )
  })

  it('rejects non-binary recording data', () => {
    assert.equal(typeof validateTempRecordingInput, 'function')
    assert.throws(
      () => validateTempRecordingInput?.('.wav', 'not a recording'),
      /Recording data must be an ArrayBuffer/,
    )
  })
})
