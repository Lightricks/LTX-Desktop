import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { isAllowedExternalUrl } from './allowed-external-url.ts'

describe('isAllowedExternalUrl', () => {
  it('allows https on any host', () => {
    assert.equal(isAllowedExternalUrl('https://fal.ai/dashboard/keys'), true)
    assert.equal(isAllowedExternalUrl('https://example.com/docs'), true)
  })

  it('allows http pairing URLs on loopback and RFC1918', () => {
    assert.equal(isAllowedExternalUrl('http://127.0.0.1:41955/pairing?t=secret'), true)
    assert.equal(isAllowedExternalUrl('http://localhost:41955/?t=secret'), true)
    assert.equal(isAllowedExternalUrl('http://192.168.1.9:41955/pairing?t=grant'), true)
    assert.equal(isAllowedExternalUrl('http://10.0.0.5:41955/'), true)
    assert.equal(isAllowedExternalUrl('http://172.16.0.1:41955/'), true)
    assert.equal(isAllowedExternalUrl('http://[::1]:41955/?t=secret'), true)
  })

  it('rejects public http, metadata, and non-http schemes', () => {
    assert.equal(isAllowedExternalUrl('http://example.com/'), false)
    assert.equal(isAllowedExternalUrl('http://8.8.8.8/'), false)
    assert.equal(isAllowedExternalUrl('http://172.15.0.1/'), false)
    assert.equal(isAllowedExternalUrl('http://169.254.169.254/latest/meta-data'), false)
    assert.equal(isAllowedExternalUrl('http://0.0.0.0:41955/'), false)
    assert.equal(isAllowedExternalUrl('file:///etc/passwd'), false)
    assert.equal(isAllowedExternalUrl('javascript:alert(1)'), false)
    assert.equal(isAllowedExternalUrl('not-a-url'), false)
    assert.equal(isAllowedExternalUrl('http://user:pass@192.168.1.9:41955/'), false)
  })
})
