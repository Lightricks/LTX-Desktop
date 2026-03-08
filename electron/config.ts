import { app } from 'electron'
import path from 'path'
import os from 'os'
import { getProjectAssetsPath } from './app-state'

let pythonPort = parseInt(process.env.LTX_PORT || '8000', 10)
export const isDev = !app.isPackaged

export function getPythonPort(): number {
  return pythonPort
}

export function setPythonPort(port: number): void {
  pythonPort = port
}

export function getBackendBaseUrl(): string {
  return `http://localhost:${pythonPort}`
}

// Get directory - works in both CJS and ESM contexts
export function getCurrentDir(): string {
  // In bundled output, use app.getAppPath()
  if (!isDev) {
    return path.dirname(app.getPath('exe'))
  }
  // In development, use process.cwd() which is the project root
  return process.cwd()
}

export function getAllowedRoots(): string[] {
  const roots = [
    getCurrentDir(),
    app.getPath('userData'),
    app.getPath('downloads'),
    os.tmpdir(),
  ]
  if (!isDev && process.resourcesPath) {
    roots.push(process.resourcesPath)
  }
  roots.push(getProjectAssetsPath())
  return roots
}
