import { createHash, randomBytes, timingSafeEqual } from 'node:crypto'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'

import {
  ANALYTICS_EVENT_NAMES,
  GENERATION_DETAIL_KEYS,
  type AnalyticsEventName,
} from './analytics-payload.ts'

export type GenerationAnalyticsEventName = Exclude<AnalyticsEventName, 'launched'>

const GENERATION_EVENTS = new Set<GenerationAnalyticsEventName>(
  ANALYTICS_EVENT_NAMES.filter(
    (name): name is GenerationAnalyticsEventName => name !== 'launched',
  ),
)
const ALLOWED_DETAIL_KEYS = new Set<string>(GENERATION_DETAIL_KEYS)
const MAX_BODY_BYTES = 16 * 1024

export interface AnalyticsSink {
  url: string
  token: string
  close: () => Promise<void>
}

export function pickGenerationDetails(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return {}
  const picked: Record<string, unknown> = {}
  for (const [key, entry] of Object.entries(value)) {
    if (ALLOWED_DETAIL_KEYS.has(key)) picked[key] = entry
  }
  return picked
}

function authorized(header: string | undefined, token: string): boolean {
  const presented = header?.startsWith('Bearer ') ? header.slice('Bearer '.length) : ''
  const presentedHash = createHash('sha256').update(presented).digest()
  const tokenHash = createHash('sha256').update(token).digest()
  return timingSafeEqual(presentedHash, tokenHash)
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    let size = 0
    req.on('data', (chunk: Buffer) => {
      size += chunk.length
      if (size > MAX_BODY_BYTES) {
        reject(new Error('body too large'))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

function sendStatus(res: ServerResponse, status: number): void {
  res.writeHead(status)
  res.end()
}

export function startAnalyticsSink(
  send: (
    eventName: GenerationAnalyticsEventName,
    extraDetails: Record<string, unknown>,
  ) => Promise<void>,
): Promise<AnalyticsSink> {
  const token = randomBytes(32).toString('base64url')
  const server: Server = createServer((req, res) => {
    void handle(req, res, token, send)
  })

  return new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (address === null || typeof address === 'string') {
        reject(new Error('analytics sink did not bind'))
        return
      }
      resolve({
        url: `http://127.0.0.1:${address.port}/analytics`,
        token,
        close: () => new Promise((done, fail) => {
          server.close((err) => (err ? fail(err) : done()))
        }),
      })
    })
  })
}

async function handle(
  req: IncomingMessage,
  res: ServerResponse,
  token: string,
  send: (
    eventName: GenerationAnalyticsEventName,
    extraDetails: Record<string, unknown>,
  ) => Promise<void>,
): Promise<void> {
  if (req.method !== 'POST' || !authorized(req.headers.authorization, token)) {
    sendStatus(res, req.method === 'POST' ? 401 : 405)
    return
  }

  let body: unknown
  try {
    body = JSON.parse(await readBody(req))
  } catch {
    sendStatus(res, 400)
    return
  }
  if (typeof body !== 'object' || body === null || Array.isArray(body)) {
    sendStatus(res, 400)
    return
  }

  const eventName = (body as { eventName?: unknown }).eventName
  if (typeof eventName !== 'string' || !GENERATION_EVENTS.has(eventName as GenerationAnalyticsEventName)) {
    sendStatus(res, 400)
    return
  }

  const extraDetails = pickGenerationDetails((body as { extraDetails?: unknown }).extraDetails)
  await send(eventName as GenerationAnalyticsEventName, extraDetails)
  sendStatus(res, 204)
}
