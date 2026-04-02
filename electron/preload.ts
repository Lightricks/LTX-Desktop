import { electronAPISchemas, type BackendHealthStatus } from '../shared/electron-api-schema'

const { contextBridge, ipcRenderer, webUtils } = require('electron')

const api: Record<string, unknown> = {}

for (const key of Object.keys(electronAPISchemas)) {
  api[key] = (input?: unknown) => ipcRenderer.invoke(key, input)
}

api.onPythonSetupProgress = (cb: (data: unknown) => void) => {
  ipcRenderer.on('python-setup-progress', (_: unknown, data: unknown) => cb(data))
}

api.removePythonSetupProgress = () => {
  ipcRenderer.removeAllListeners('python-setup-progress')
}

api.onBackendHealthStatus = (cb: (data: BackendHealthStatus) => void) => {
  const listener = (_: unknown, data: BackendHealthStatus) => cb(data)
  ipcRenderer.on('backend-health-status', listener)
  return () => {
    ipcRenderer.removeListener('backend-health-status', listener)
  }
}

api.getPathForFile = (file: File) => webUtils.getPathForFile(file)

api.platform = process.platform

contextBridge.exposeInMainWorld('electronAPI', api)

export {}
