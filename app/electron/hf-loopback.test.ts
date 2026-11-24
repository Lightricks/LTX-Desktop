import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { test } from 'node:test'
import { listenForHuggingFaceCallback } from './hf-loopback.ts'

test('Hugging Face callbacks reach only the separate AMD backend and only for the pending login', async () => {
  const reserved = createServer()
  await new Promise<void>((ok) => reserved.listen(0, '127.0.0.1', ok))
  const address = reserved.address()
  assert.ok(address && typeof address !== 'string')
  const uri = 'http://127.0.0.1:' + address.port + '/api/auth/huggingface/callback'
  const target = 'http://127.0.0.1:41956'
  await assert.rejects(listenForHuggingFaceCallback(uri, target, 'pending'), { code: 'EADDRINUSE' })
  await new Promise<void>((ok) => reserved.close(() => ok()))
  const close = await listenForHuggingFaceCallback(uri, target, 'pending')
  try {
    assert.equal((await fetch(uri + '?state=unrelated&code=example', { redirect: 'manual' })).status, 404)
    const r = await fetch(uri + '?state=pending&code=a%2Bb', { redirect: 'manual' })
    assert.equal(r.status, 302)
    assert.equal(r.headers.get('location'), target + '/api/auth/huggingface/callback?state=pending&code=a%2Bb')
  } finally {
    close()
  }
})
