import assert from 'node:assert/strict'
import path from 'path'
import { describe, it } from 'node:test'
import { registerDeepLinkProtocolClient } from './deep-link-protocol.ts'
import { DEEP_LINK_SCHEME } from '../shared/deep-link.ts'

describe('registerDeepLinkProtocolClient', () => {
  it('registers the packaged app as the protocol handler', () => {
    const calls: unknown[][] = []
    registerDeepLinkProtocolClient((...args) => {
      calls.push(args)
      return true
    }, {
      defaultApp: false,
      execPath: '/Applications/LTX Desktop.app/Contents/MacOS/LTX Desktop',
      argv: ['/Applications/LTX Desktop.app/Contents/MacOS/LTX Desktop'],
    })
    assert.deepEqual(calls, [[DEEP_LINK_SCHEME]])
  })

  it('points Electron at this repo during `pnpm dev`', () => {
    const calls: unknown[][] = []
    registerDeepLinkProtocolClient((...args) => {
      calls.push(args)
      return true
    }, {
      defaultApp: true,
      execPath: '/usr/local/bin/electron',
      argv: ['/usr/local/bin/electron', '.'],
    })
    assert.deepEqual(calls, [[
      DEEP_LINK_SCHEME,
      '/usr/local/bin/electron',
      [path.resolve('.')],
    ]])
  })
})
