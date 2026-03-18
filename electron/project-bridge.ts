/**
 * Project Bridge — lightweight HTTP server (port 8100) that exposes
 * project data from the renderer's localStorage to external tools
 * (MCP Server, scripts, etc.).
 *
 * Endpoints:
 *   GET  /api/projects                — list all projects
 *   GET  /api/projects/:id            — get a single project
 *   PUT  /api/projects/:id            — update a single project
 *   POST /api/export                  — trigger FFmpeg export
 *   POST /api/import-asset            — copy a file into project assets
 */

import http from 'http'
import { getMainWindow } from './window'
import { logger } from './logger'

const BRIDGE_PORT = 8100

let bridgeServer: http.Server | null = null

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jsonResponse(res: http.ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, PUT, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  })
  res.end(payload)
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = []
    req.on('data', (chunk: Buffer) => chunks.push(chunk))
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

/** Execute JS in the renderer and return the result. */
async function rendererEval<T>(js: string): Promise<T> {
  const win = getMainWindow()
  if (!win) throw new Error('Main window not available')
  return win.webContents.executeJavaScript(js) as Promise<T>
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

async function handleListProjects(res: http.ServerResponse): Promise<void> {
  try {
    const projects = await rendererEval<unknown>(
      `JSON.parse(localStorage.getItem('ltx-projects') || '[]')`
    )
    jsonResponse(res, 200, projects)
  } catch (err) {
    jsonResponse(res, 500, { error: String(err) })
  }
}

async function handleGetProject(res: http.ServerResponse, projectId: string): Promise<void> {
  try {
    const projects = await rendererEval<Array<{ id: string }>>(
      `JSON.parse(localStorage.getItem('ltx-projects') || '[]')`
    )
    const project = projects.find((p) => p.id === projectId)
    if (!project) {
      jsonResponse(res, 404, { error: `Project not found: ${projectId}` })
      return
    }
    jsonResponse(res, 200, project)
  } catch (err) {
    jsonResponse(res, 500, { error: String(err) })
  }
}

async function handleUpdateProject(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  projectId: string
): Promise<void> {
  try {
    const body = await readBody(req)
    const updatedProject = JSON.parse(body)

    // Read current projects, replace the matching one
    const projects = await rendererEval<Array<{ id: string }>>(
      `JSON.parse(localStorage.getItem('ltx-projects') || '[]')`
    )
    const idx = projects.findIndex((p) => p.id === projectId)
    if (idx === -1) {
      jsonResponse(res, 404, { error: `Project not found: ${projectId}` })
      return
    }

    // Ensure the project ID is preserved
    updatedProject.id = projectId
    updatedProject.updatedAt = Date.now()
    projects[idx] = updatedProject

    // Write back to localStorage
    const escaped = JSON.stringify(JSON.stringify(projects))
    await rendererEval<void>(`localStorage.setItem('ltx-projects', ${escaped})`)

    // Trigger a storage event so React picks up the change
    await rendererEval<void>(
      `window.dispatchEvent(new StorageEvent('storage', { key: 'ltx-projects' }))`
    )

    // Force React re-render by dispatching a custom reload event
    await rendererEval<void>(
      `window.dispatchEvent(new CustomEvent('ltx-projects-reload'))`
    )

    jsonResponse(res, 200, { status: 'updated', id: projectId })
  } catch (err) {
    jsonResponse(res, 500, { error: String(err) })
  }
}

async function handleExport(
  req: http.IncomingMessage,
  res: http.ServerResponse
): Promise<void> {
  try {
    const body = await readBody(req)
    const params = JSON.parse(body)

    const { projectId, outputPath, width, height, fps, codec, quality } = params

    // Read the project and active timeline
    const projects = await rendererEval<Array<{ id: string; timelines?: Array<{ id: string; clips?: unknown[] }>; activeTimelineId?: string }>>(
      `JSON.parse(localStorage.getItem('ltx-projects') || '[]')`
    )
    const project = projects.find((p) => p.id === projectId)
    if (!project) {
      jsonResponse(res, 404, { error: `Project not found: ${projectId}` })
      return
    }

    const timeline = project.timelines?.find((t) => t.id === project.activeTimelineId) || project.timelines?.[0]
    if (!timeline || !timeline.clips?.length) {
      jsonResponse(res, 400, { error: 'No clips in timeline to export' })
      return
    }

    // Delegate to the existing export-native IPC handler by invoking it via renderer
    const exportData = {
      clips: timeline.clips,
      outputPath,
      codec: codec || 'h264',
      width: width || 1920,
      height: height || 1080,
      fps: fps || 24,
      quality: quality || 80,
    }

    const escaped = JSON.stringify(JSON.stringify(exportData))
    const result = await rendererEval<{ success?: boolean; error?: string }>(
      `window.electronAPI.exportNative(JSON.parse(${escaped}))`
    )
    jsonResponse(res, result?.success ? 200 : 500, result)
  } catch (err) {
    jsonResponse(res, 500, { error: String(err) })
  }
}

async function handleImportAsset(
  req: http.IncomingMessage,
  res: http.ServerResponse
): Promise<void> {
  try {
    const body = await readBody(req)
    const { projectId, filePath } = JSON.parse(body)

    if (!projectId || !filePath) {
      jsonResponse(res, 400, { error: 'projectId and filePath are required' })
      return
    }

    // Use the existing copyToProjectAssets IPC handler via renderer
    const escaped1 = JSON.stringify(filePath)
    const escaped2 = JSON.stringify(projectId)
    const result = await rendererEval<{ success: boolean; path?: string; url?: string; error?: string }>(
      `window.electronAPI.copyToProjectAssets(${escaped1}, ${escaped2})`
    )
    jsonResponse(res, result?.success ? 200 : 500, result)
  } catch (err) {
    jsonResponse(res, 500, { error: String(err) })
  }
}

// ---------------------------------------------------------------------------
// Request router
// ---------------------------------------------------------------------------

async function handleRequest(
  req: http.IncomingMessage,
  res: http.ServerResponse
): Promise<void> {
  const method = req.method?.toUpperCase() || 'GET'
  const url = req.url || '/'

  // CORS preflight
  if (method === 'OPTIONS') {
    jsonResponse(res, 204, '')
    return
  }

  // Route matching
  const projectMatch = url.match(/^\/api\/projects\/([^/]+)$/)

  if (url === '/api/projects' && method === 'GET') {
    await handleListProjects(res)
  } else if (projectMatch && method === 'GET') {
    await handleGetProject(res, projectMatch[1])
  } else if (projectMatch && method === 'PUT') {
    await handleUpdateProject(req, res, projectMatch[1])
  } else if (url === '/api/export' && method === 'POST') {
    await handleExport(req, res)
  } else if (url === '/api/import-asset' && method === 'POST') {
    await handleImportAsset(req, res)
  } else {
    jsonResponse(res, 404, { error: `Not found: ${method} ${url}` })
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function startProjectBridge(): void {
  if (bridgeServer) return

  bridgeServer = http.createServer((req, res) => {
    handleRequest(req, res).catch((err) => {
      logger.error(`[ProjectBridge] Unhandled error: ${err}`)
      jsonResponse(res, 500, { error: 'Internal server error' })
    })
  })

  bridgeServer.listen(BRIDGE_PORT, '127.0.0.1', () => {
    logger.info(`[ProjectBridge] Listening on http://127.0.0.1:${BRIDGE_PORT}`)
  })

  bridgeServer.on('error', (err) => {
    logger.error(`[ProjectBridge] Server error: ${err}`)
  })
}

export function stopProjectBridge(): void {
  if (bridgeServer) {
    bridgeServer.close()
    bridgeServer = null
    logger.info('[ProjectBridge] Stopped')
  }
}
