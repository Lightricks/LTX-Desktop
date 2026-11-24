import assert from 'node:assert/strict'
import { describe, it } from 'node:test'

import { shouldSendLaunched } from './launched-event.ts'

describe('shouldSendLaunched', () => {
  it('sends at startup only when setup is already complete', () => {
    assert.equal(
      shouldSendLaunched({ alreadySent: false, setupComplete: true, trigger: 'startup' }),
      true,
    )
    assert.equal(
      shouldSendLaunched({ alreadySent: false, setupComplete: false, trigger: 'startup' }),
      false,
    )
  })

  it('sends when installation starts, before setup is complete', () => {
    assert.equal(
      shouldSendLaunched({
        alreadySent: false,
        setupComplete: false,
        trigger: 'installation-started',
      }),
      true,
    )
  })

  it('sends when setup completes without an Install click, unless this process already sent', () => {
    assert.equal(
      shouldSendLaunched({
        alreadySent: false,
        setupComplete: true,
        trigger: 'setup-completed',
      }),
      true,
    )
    assert.equal(
      shouldSendLaunched({
        alreadySent: true,
        setupComplete: true,
        trigger: 'setup-completed',
      }),
      false,
    )
  })

  it('does not send a second time in the same process', () => {
    assert.equal(
      shouldSendLaunched({ alreadySent: true, setupComplete: true, trigger: 'startup' }),
      false,
    )
    assert.equal(
      shouldSendLaunched({
        alreadySent: true,
        setupComplete: false,
        trigger: 'installation-started',
      }),
      false,
    )
  })
})
