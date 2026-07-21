import {
  electronAPISchemas,
  SELECTED_FILE_PATH_APPROVAL_CHANNEL,
  type BackendHealthStatus,
} from '../shared/electron-api-schema'
import { HF_GATING_ENABLED } from '../shared/feature-flags'

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

api.getPathForFile = (file: File) => {
  const filePath = webUtils.getPathForFile(file)
  if (!filePath) return ''
  const approved = ipcRenderer.sendSync(SELECTED_FILE_PATH_APPROVAL_CHANNEL, filePath)
  return approved === true ? filePath : ''
}

api.platform = process.platform

api.hfGatingEnabled = HF_GATING_ENABLED

contextBridge.exposeInMainWorld('electronAPI', api)

export {}
