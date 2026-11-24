import { createServer, type Server } from 'node:http'

const callbackPath = '/api/auth/huggingface/callback'
let active: Server | undefined

/** Bridge the registered OAuth port to the isolated AMD backend for this sign-in only. */
export async function listenForHuggingFaceCallback(
  redirectUri: string, backendUrl: string, state: string,
): Promise<() => void> {
  const redirect = new URL(redirectUri)
  const backend = new URL(backendUrl)
  for (const url of [redirect, backend]) {
    if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port) {
      throw new Error('The Hugging Face callback requires a local LTX backend.')
    }
  }
  if (redirect.pathname !== callbackPath || !state) throw new Error('Invalid Hugging Face callback.')
  if (redirect.origin === backend.origin) return () => {}

  active?.close()
  const server = createServer((req, res) => {
    if (req.method !== 'GET' || !req.url?.startsWith(callbackPath + '?')) {
      res.writeHead(404).end()
      return
    }
    const incoming = new URL(req.url, redirect.origin)
    if (incoming.searchParams.get('state') !== state) {
      res.writeHead(404).end()
      return
    }
    const target = new URL(callbackPath, backend.origin)
    target.searchParams.set('state', state)
    if (incoming.searchParams.has('error')) target.searchParams.set('error', 'access_denied')
    else target.searchParams.set('code', incoming.searchParams.get('code') ?? '')
    res.writeHead(302, { Location: target.href, 'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer' }).end()
    server.close()
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(Number(redirect.port), '127.0.0.1', resolve)
  })
  active = server
  const timer = setTimeout(() => server.close(), 600_000)
  timer.unref()
  server.once('close', () => {
    clearTimeout(timer)
    if (active === server) active = undefined
  })
  return () => { server.close() }
}
