import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { getViteDevServerUrl, resolveViteDevServerPort } from './vite-dev-server.ts'

const DEFAULT_PORT = 5173
const DEFAULT_URL = 'http://localhost:5173'

describe('resolveViteDevServerPort', () => {
  it('defaults to 5173 when the env var is missing', () => {
    assert.equal(resolveViteDevServerPort({}), DEFAULT_PORT)
  })

  it('defaults to 5173 when the env var is empty', () => {
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: '' }), DEFAULT_PORT)
  })

  it('uses a valid explicit port', () => {
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: '5174' }), 5174)
  })

  it('falls back to 5173 for non-integer values', () => {
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: '5174.5' }), DEFAULT_PORT)
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: 'abc' }), DEFAULT_PORT)
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: '1e4' }), DEFAULT_PORT)
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: ' 5174' }), DEFAULT_PORT)
  })

  it('falls back to 5173 for out-of-range values', () => {
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: '0' }), DEFAULT_PORT)
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: '-1' }), DEFAULT_PORT)
    assert.equal(resolveViteDevServerPort({ VITE_DEV_SERVER_PORT: '65536' }), DEFAULT_PORT)
  })
})

describe('getViteDevServerUrl', () => {
  it('defaults to http://localhost:5173', () => {
    assert.equal(getViteDevServerUrl({}), DEFAULT_URL)
  })

  it('uses the same resolved port as resolveViteDevServerPort', () => {
    const env = { VITE_DEV_SERVER_PORT: '5174' }
    assert.equal(getViteDevServerUrl(env), 'http://localhost:5174')
    assert.equal(
      getViteDevServerUrl(env),
      `http://localhost:${resolveViteDevServerPort(env)}`,
    )
  })

  it('does not produce a mismatched URL for malformed values', () => {
    const env = { VITE_DEV_SERVER_PORT: 'nope' }
    assert.equal(getViteDevServerUrl(env), DEFAULT_URL)
    assert.equal(
      getViteDevServerUrl(env),
      `http://localhost:${resolveViteDevServerPort(env)}`,
    )
  })

  it('reads VITE_DEV_SERVER_PORT from process.env by default', () => {
    const previous = process.env.VITE_DEV_SERVER_PORT
    process.env.VITE_DEV_SERVER_PORT = '5174'
    try {
      assert.equal(resolveViteDevServerPort(), 5174)
      assert.equal(getViteDevServerUrl(), 'http://localhost:5174')
    } finally {
      if (previous === undefined) {
        delete process.env.VITE_DEV_SERVER_PORT
      } else {
        process.env.VITE_DEV_SERVER_PORT = previous
      }
    }
  })
})
