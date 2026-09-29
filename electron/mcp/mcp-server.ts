import http from 'http'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { randomBytes } from 'crypto'
import { app } from 'electron'
import { isDev } from '../config'
import { logger } from '../logger'
import { listLibrary, ensureLibFolder } from '../ipc/library-handlers'
import { exportTimelineNative, type ExportNativeInput } from '../export/export-handler'
import { callEditor, registerEditorBridge } from './editor-bridge'
import { analyzeAudio, probeMedia, sampleFrames, type Frame } from './media-tools'

/**
 * RiX MCP server — lets an MCP client (Claude Code / Claude Desktop) assemble
 * videos in the RiX Video Editor from the user's assets.
 *
 * Transport: MCP Streamable HTTP, stateless, JSON responses only, at
 * http://127.0.0.1:<port>/mcp. Guarded by a bearer token stored in
 * <userData>/rix-mcp.json, loopback-only binding, a Host check (DNS rebinding)
 * and refusal of browser-originated requests. Only started in dev builds or
 * when RIX_MCP=1, so shipped installs never open a control port by default.
 */

const DEFAULT_PORT = 47821
const PROTOCOL_VERSIONS = ['2025-06-18', '2025-03-26', '2024-11-05']

type ToolContent = Array<{ type: 'text'; text: string } | { type: 'image'; data: string; mimeType: string }>
type Tool = {
  name: string
  description: string
  inputSchema: Record<string, unknown>
  run: (args: Record<string, unknown>) => Promise<ToolContent>
}

const text = (value: unknown): ToolContent => [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value, null, 2) }]
const withFrames = (summary: unknown, frames: Frame[]): ToolContent => [
  ...text(summary),
  ...frames.flatMap(f => [
    ...(f.time !== undefined ? [{ type: 'text' as const, text: `t=${f.time}s` }] : []),
    { type: 'image' as const, data: f.base64, mimeType: f.mimeType },
  ]),
]
const str = (v: unknown) => (typeof v === 'string' ? v : undefined)
const numArg = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)

let lastPreviewPath: string | null = null

const PROJECT_ID_PROP = { type: 'string', description: 'editor.project.id from rix_status — guards against editing the wrong project' }

async function renderTimeline(opts: { preview: boolean; width?: number; height?: number; fps?: number; quality?: number; outputPath?: string }) {
  const { payload, aspect, duration, timelineName } = await callEditor<{
    payload: Omit<ExportNativeInput, 'outputPath' | 'codec' | 'width' | 'height' | 'fps' | 'quality'>
    aspect: number
    duration: number
    timelineName: string
  }>('export_payload')
  const even = (n: number) => Math.max(2, Math.round(n / 2) * 2)
  const long = opts.preview ? 640 : 1920
  let width = opts.width ?? (aspect >= 1 ? long : even(long * aspect))
  let height = opts.height ?? (aspect >= 1 ? even(long / aspect) : long)
  if (opts.width && !opts.height) height = even(opts.width / aspect)
  if (opts.height && !opts.width) width = even(opts.height * aspect)
  const outputPath = opts.outputPath ?? path.join(os.tmpdir(), `rix-mcp-preview-${Date.now()}.mp4`)
  const r = await exportTimelineNative(
    { ...payload, outputPath, codec: 'h264', width: even(width), height: even(height), fps: opts.fps ?? 24, quality: opts.quality ?? 18 },
    { preview: opts.preview },
  )
  if (!r.success) throw new Error(`Render failed: ${r.error}`)
  return { outputPath, duration, width: even(width), height: even(height), timelineName }
}

const TOOLS: Tool[] = [
  {
    name: 'rix_status',
    description: 'Check that RiX is reachable and which project/timelines are open in the Video Editor. Call this first. '
      + 'Every editing tool requires the returned editor.project.id as "project_id" (edits are refused if the user has switched projects).',
    inputSchema: { type: 'object', properties: {} },
    run: async () => {
      let editor: unknown
      try { editor = await callEditor('status', {}, 5000) } catch (e) { editor = { error: (e as Error).message } }
      return text({ app: `RiX ${app.getVersion()}`, libraryRoot: listLibrary().root, editor })
    },
  },
  {
    name: 'list_library',
    description: 'Browse the Studio Assets library (the user\'s media folders on disk). With no folder, returns every folder with file counts. '
      + 'With a folder and/or query, returns matching files (newest first). File names are content-aware (e.g. storm-03.mp4), so query by subject.',
    inputSchema: {
      type: 'object',
      properties: {
        folder: { type: 'string', description: 'Folder name (case-insensitive), e.g. "Halloween"' },
        query: { type: 'string', description: 'Substring match on file name' },
        type: { type: 'string', enum: ['image', 'video', 'audio'] },
        limit: { type: 'number', description: 'Max files (default 100)' },
      },
    },
    run: async (args) => {
      const lib = listLibrary()
      const folder = str(args.folder)?.toLowerCase()
      const query = str(args.query)?.toLowerCase()
      const type = str(args.type)
      if (!folder && !query && !type) {
        const counts = new Map<string, number>()
        for (const f of lib.files) counts.set(f.folder, (counts.get(f.folder) ?? 0) + 1)
        return text({ root: lib.root, folders: lib.folders.map(name => ({ name, files: counts.get(name) ?? 0 })) })
      }
      const kind = (f: (typeof lib.files)[number]) => (f.isVideo ? 'video' : f.isAudio ? 'audio' : 'image')
      const files = lib.files
        .filter(f => !folder || f.folder.toLowerCase() === folder)
        .filter(f => !query || f.name.toLowerCase().includes(query))
        .filter(f => !type || kind(f) === type)
        .sort((a, b) => b.mtimeMs - a.mtimeMs)
      const limit = Math.max(1, Math.min(numArg(args.limit) ?? 100, 1000))
      return text({
        total: files.length,
        files: files.slice(0, limit).map(f => ({ folder: f.folder, name: f.name, type: kind(f), path: f.path, modified: new Date(f.mtimeMs).toISOString() })),
      })
    },
  },
  {
    name: 'list_project_assets',
    description: 'List assets registered in the open RiX project (generations + imports), newest first, with their generation prompts. '
      + 'Prompts describe content — use them to pick shots.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Substring match on prompt, path or folder/tag name' },
        type: { type: 'string', enum: ['image', 'video', 'audio'] },
        limit: { type: 'number', description: 'Max assets (default 50)' },
      },
    },
    run: async (args) => text(await callEditor('list_project_assets', args)),
  },
  {
    name: 'inspect_media',
    description: 'LOOK at a media file: returns metadata plus sampled frames as images (evenly spaced across a video, or the still itself). '
      + 'Use before choosing shots — prompts can lie; frames don\'t. Default 6 frames at 384px.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Absolute path to an image or video' },
        frames: { type: 'number', description: 'How many evenly spaced frames (1–16, default 6)' },
        times: { type: 'array', items: { type: 'number' }, description: 'Explicit timestamps in seconds (overrides frames)' },
        width: { type: 'number', description: 'Frame width in px (default 384)' },
      },
      required: ['path'],
    },
    run: async (args) => {
      const p = str(args.path)
      if (!p) throw new Error('"path" is required')
      const times = Array.isArray(args.times) ? args.times.filter((t): t is number => typeof t === 'number') : undefined
      const { info, frames } = sampleFrames(p, { count: numArg(args.frames), times, width: numArg(args.width) })
      return withFrames({ path: p, ...info }, frames)
    },
  },
  {
    name: 'analyze_audio',
    description: 'Hear a music/audio file: tempo (bpm), a beat grid, guessed downbeats (every 4th beat), the strongest hits, '
      + 'and loudness per second (0–1) to find intros, builds and drops. Use it to place cuts on the beat.',
    inputSchema: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] },
    run: async (args) => {
      const p = str(args.path)
      if (!p) throw new Error('"path" is required')
      return text(analyzeAudio(p))
    },
  },
  {
    name: 'get_timeline',
    description: 'Read the active timeline: tracks (index, kind) and every clip with id, track, start/end, source in-point, asset, text, transitions, audio settings.',
    inputSchema: { type: 'object', properties: {} },
    run: async () => text(await callEditor('get_timeline')),
  },
  {
    name: 'apply_edits',
    description: [
      'Edit the active timeline with a batch of ops. The batch is ATOMIC and is ONE undo step (if any op fails, nothing changes).',
      'Times are seconds. Returns per-op results (new clip ids) and the resulting timeline.',
      'Ops:',
      '  {"op":"import","path":"<abs path>","ref":"name"} — register a library file in the project (ref usable later as "$name")',
      '  {"op":"add_clip","asset":"<asset id | $ref | abs path>","track":0,"start":0,"in":1.5,"duration":2.0,"speed":1,"with_audio":true,"ref":"c1"}',
      '     start omitted = append at end of track; in = source in-point; duration capped to the source length. Placing over existing clips overwrites them.',
      '     Video clips bring linked audio on an audio track; with_audio:false drops it; audio_track:N routes it (e.g. dialogue to A2 so it',
      '     can\'t overwrite the music on A1). Images default to 5s unless duration is set.',
      '  {"op":"duck","clip":"<music clip>","ranges":[[t0,t1],...],"level":0.25,"attack":0.2,"release":0.4} — dip the music under',
      '     dialogue (ranges = TIMELINE seconds of speech; take phrase times from inspect_media soundSegments + the clip\'s start/in).',
      '  {"op":"add_text","text":"TITLE","start":0,"duration":3,"style":{"fontSize":96,"color":"#FFFFFF","positionX":50,"positionY":50},"fade_in":0.5,"fade_out":0.5}',
      '  {"op":"update_clip","clip":"<clip id | $ref>","set":{"start","duration","in","speed","volume"(0–2),"opacity"(0–100),"audio_fade_in","audio_fade_out",',
      '     "text_fade_in","text_fade_out","muted","reversed","color":{brightness,contrast,saturation,temperature,tint,exposure,highlights,shadows},"text","style",',
      '     "volume_keyframes":[{"t":clipSeconds,"value":0–2}]}}',
      '  {"op":"transition","clip":"<id>","in":{"type":"dissolve","duration":0.5},"out":{"type":"fade-to-black","duration":1}}',
      '     types: none, dissolve, fade-to-black, fade-to-white, wipe-left, wipe-right, wipe-up, wipe-down. A dissolve needs out on the left clip AND in on the right clip.',
      '  {"op":"delete","clips":["<id>"]}  (linked audio goes too)   {"op":"clear"}  (empty the active timeline)',
      '  {"op":"new_timeline","name":"Trailer v1"}  (becomes active)   {"op":"add_track","kind":"video"|"audio"}',
    ].join('\n'),
    inputSchema: {
      type: 'object',
      properties: {
        project_id: PROJECT_ID_PROP,
        ops: { type: 'array', items: { type: 'object' }, description: 'Ordered list of edit ops (see description)' },
      },
      required: ['project_id', 'ops'],
    },
    run: async (args) => text(await callEditor('apply_edits', args, 180000)),
  },
  {
    name: 'undo',
    description: 'Revert YOUR last apply_edits batch (one step). Refuses if the user has changed anything since, or if the last '
      + 'change was not yours — it never undoes the user\'s work. In that case fix things with a corrective apply_edits batch.',
    inputSchema: { type: 'object', properties: { project_id: PROJECT_ID_PROP }, required: ['project_id'] },
    run: async (args) => text(await callEditor('undo', args)),
  },
  {
    name: 'redo',
    description: 'Re-apply a batch you just reverted with undo (same safety rules).',
    inputSchema: { type: 'object', properties: { project_id: PROJECT_ID_PROP }, required: ['project_id'] },
    run: async (args) => text(await callEditor('redo', args)),
  },
  {
    name: 'render_frames',
    description: 'Render the active timeline to a fast low-res preview (with transitions, text, color) and return frames at the given times, '
      + 'so you can review your own cut. Default: 8 evenly spaced frames. Also returns the preview mp4 path (usable with inspect_media).',
    inputSchema: {
      type: 'object',
      properties: {
        times: { type: 'array', items: { type: 'number' }, description: 'Timeline seconds to grab (max 16)' },
        count: { type: 'number', description: 'Evenly spaced frame count when times is omitted (default 8)' },
      },
    },
    run: async (args) => {
      const render = await renderTimeline({ preview: true })
      if (lastPreviewPath && lastPreviewPath !== render.outputPath) { try { fs.unlinkSync(lastPreviewPath) } catch { /* temp */ } }
      lastPreviewPath = render.outputPath
      const times = Array.isArray(args.times) ? args.times.filter((t): t is number => typeof t === 'number') : undefined
      const { frames } = sampleFrames(render.outputPath, { times, count: numArg(args.count) ?? 8, width: 480 })
      return withFrames({ previewPath: render.outputPath, duration: render.duration, size: `${render.width}x${render.height}` }, frames)
    },
  },
  {
    name: 'export_video',
    description: 'Export the active timeline at full quality (H.264 MP4). Default output: Studio Assets/"RiX Edits"/<timeline>-<timestamp>.mp4, '
      + 'sized from the first clip\'s aspect at 1080p. Takes a while for long timelines.',
    inputSchema: {
      type: 'object',
      properties: {
        output_path: { type: 'string', description: 'Absolute .mp4 path (must be inside the library, Downloads, or project folders)' },
        width: { type: 'number' },
        height: { type: 'number' },
        fps: { type: 'number', description: 'Default 24' },
        quality: { type: 'number', description: 'H.264 CRF, lower = better (default 18)' },
      },
    },
    run: async (args) => {
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
      let outputPath = str(args.output_path)
      if (!outputPath) {
        const { timelineName } = await callEditor<{ timelineName: string }>('export_payload')
        const safe = timelineName.replace(/[^\w -]+/g, '').trim().replace(/\s+/g, '-') || 'timeline'
        outputPath = path.join(ensureLibFolder('RiX Edits'), `${safe}-${stamp}.mp4`)
      }
      const r = await renderTimeline({
        preview: false, outputPath, width: numArg(args.width), height: numArg(args.height), fps: numArg(args.fps), quality: numArg(args.quality),
      })
      return text({ exported: r.outputPath, duration: r.duration, size: `${r.width}x${r.height}`, media: probeMedia(r.outputPath) })
    },
  },
]

// ---------------------------------------------------------------- JSON-RPC

type RpcMessage = { jsonrpc: '2.0'; id?: string | number | null; method?: string; params?: Record<string, unknown> }
const rpcResult = (id: RpcMessage['id'], result: unknown) => ({ jsonrpc: '2.0', id, result })
const rpcError = (id: RpcMessage['id'], code: number, message: string) => ({ jsonrpc: '2.0', id: id ?? null, error: { code, message } })

async function handleMessage(msg: RpcMessage): Promise<object | null> {
  if (msg.id === undefined || msg.id === null) return null // notification (e.g. notifications/initialized)
  switch (msg.method) {
    case 'initialize': {
      const requested = str(msg.params?.protocolVersion)
      return rpcResult(msg.id, {
        protocolVersion: requested && PROTOCOL_VERSIONS.includes(requested) ? requested : PROTOCOL_VERSIONS[0],
        capabilities: { tools: {} },
        serverInfo: { name: 'rix-editor', version: app.getVersion() },
        instructions: 'Tools for assembling videos in the RiX Video Editor from the user\'s generated assets. '
          + 'Start with rix_status. Look at media with inspect_media before cutting; review your cut with render_frames before exporting.',
      })
    }
    case 'ping':
      return rpcResult(msg.id, {})
    case 'tools/list':
      return rpcResult(msg.id, { tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })) })
    case 'tools/call': {
      const name = str(msg.params?.name)
      const tool = TOOLS.find(t => t.name === name)
      if (!tool) return rpcError(msg.id, -32602, `Unknown tool: ${name}`)
      const args = (msg.params?.arguments ?? {}) as Record<string, unknown>
      const started = Date.now()
      try {
        const content = await tool.run(args)
        logger.info(`[MCP] ${name} ok (${Date.now() - started}ms)`)
        return rpcResult(msg.id, { content })
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e)
        logger.warn(`[MCP] ${name} failed: ${message}`)
        return rpcResult(msg.id, { content: text(`Error: ${message}`), isError: true })
      }
    }
    default:
      return rpcError(msg.id, -32601, `Method not found: ${msg.method}`)
  }
}

// ---------------------------------------------------------------- HTTP

type McpConfig = { port: number; token: string }

function loadConfig(): McpConfig {
  const file = path.join(app.getPath('userData'), 'rix-mcp.json')
  let cfg: Partial<McpConfig> = {}
  try { cfg = JSON.parse(fs.readFileSync(file, 'utf8')) } catch { /* first run */ }
  const envPort = Number(process.env.RIX_MCP_PORT)
  const next: McpConfig = {
    port: Number.isInteger(envPort) && envPort > 0 ? envPort : cfg.port ?? DEFAULT_PORT,
    token: cfg.token ?? randomBytes(24).toString('hex'),
  }
  if (next.token !== cfg.token || next.port !== cfg.port) {
    fs.writeFileSync(file, JSON.stringify(next, null, 2), { encoding: 'utf8', mode: 0o600 })
  }
  return next
}

function readBody(req: http.IncomingMessage, limit = 5 * 1024 * 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    let size = 0
    const chunks: Buffer[] = []
    req.on('data', (c: Buffer) => {
      size += c.length
      if (size > limit) { reject(new Error('Request too large')); req.destroy() } else chunks.push(c)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

export function startMcpServer(): void {
  if (!isDev && process.env.RIX_MCP !== '1') return
  registerEditorBridge()
  const { port, token } = loadConfig()
  const allowedHosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`])

  const server = http.createServer(async (req, res) => {
    const send = (status: number, body?: unknown) => {
      res.writeHead(status, body === undefined ? {} : { 'Content-Type': 'application/json' })
      res.end(body === undefined ? undefined : JSON.stringify(body))
    }
    // Loopback-only + Host check (DNS rebinding) + no browser-originated calls + token.
    if (!allowedHosts.has(req.headers.host ?? '')) return send(403, rpcError(null, -32000, 'Forbidden host'))
    if (req.headers.origin) return send(403, rpcError(null, -32000, 'Browser requests are not allowed'))
    if (req.headers.authorization !== `Bearer ${token}`) return send(401, rpcError(null, -32001, 'Missing or invalid bearer token'))
    if (req.url?.split('?')[0] !== '/mcp') return send(404, rpcError(null, -32000, 'Not found'))
    if (req.method === 'DELETE') return send(200) // stateless: no session to end
    if (req.method !== 'POST') return send(405, rpcError(null, -32000, 'Use POST'))

    let parsed: RpcMessage | RpcMessage[]
    try { parsed = JSON.parse(await readBody(req)) } catch { return send(400, rpcError(null, -32700, 'Parse error')) }
    const batch = Array.isArray(parsed)
    const replies = (await Promise.all((batch ? parsed : [parsed]).map(handleMessage))).filter((r): r is object => r !== null)
    if (replies.length === 0) return send(202)
    return send(200, batch ? replies : replies[0])
  })

  server.on('error', (err) => logger.warn(`[MCP] server not started: ${err.message}`))
  server.listen(port, '127.0.0.1', () => {
    logger.info(`[MCP] RiX editor MCP server listening on http://127.0.0.1:${port}/mcp (token in ${path.join(app.getPath('userData'), 'rix-mcp.json')})`)
  })
  app.on('before-quit', () => server.close())
}
