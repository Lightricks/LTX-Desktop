import { electronAPISchemas, type BackendHealthStatus, type UpdateStatePayload, type ExportProgress, type McpEditorRequest } from '../shared/electron-api-schema'

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

api.onExportProgress = (cb: (data: ExportProgress) => void) => {
  const listener = (_: unknown, data: ExportProgress) => cb(data)
  ipcRenderer.on('export-progress', listener)
  return () => {
    ipcRenderer.removeListener('export-progress', listener)
  }
}

api.onMenuAction = (cb: (action: string) => void) => {
  const listener = (_: unknown, action: string) => cb(action)
  ipcRenderer.on('menu-action', listener)
  return () => {
    ipcRenderer.removeListener('menu-action', listener)
  }
}

api.onGpmLibChanged = (cb: () => void) => {
  const listener = () => cb()
  ipcRenderer.on('gpm-lib-changed', listener)
  return () => {
    ipcRenderer.removeListener('gpm-lib-changed', listener)
  }
}

api.onMcpEditorRequest = (cb: (req: McpEditorRequest) => void) => {
  const listener = (_: unknown, req: McpEditorRequest) => cb(req)
  ipcRenderer.on('mcp-editor-request', listener)
  return () => ipcRenderer.removeListener('mcp-editor-request', listener)
}

api.onUpdateEvent =(cb: (data: UpdateStatePayload) => void) => {
  const listener = (_: unknown, data: UpdateStatePayload) => cb(data)
  ipcRenderer.on('update-event', listener)
  return () => ipcRenderer.removeListener('update-event', listener)
}

api.getPathForFile = (file: File) => webUtils.getPathForFile(file)

api.platform = process.platform

contextBridge.exposeInMainWorld('electronAPI', api)

export {}
