import { app, ipcMain } from 'electron'
import path from 'path'
import fs from 'fs'
import { checkGPU } from '../gpu'
import { isPythonReady, downloadPythonEmbed } from '../python-setup'
import { getBackendHealthStatus, getBackendUrl, getAuthToken, startPythonBackend } from '../python-backend'
import { getMainWindow } from '../window'
import { getAnalyticsState, setAnalyticsEnabled, sendAnalyticsEvent } from '../analytics'

function getModelsPath(): string {
  const modelsPath = path.join(app.getPath('userData'), 'models')
  if (!fs.existsSync(modelsPath)) {
    fs.mkdirSync(modelsPath, { recursive: true })
  }
  return modelsPath
}

function getSetupStatus(settingsPath: string): { needsSetup: boolean; needsLicense: boolean } {
  if (!fs.existsSync(settingsPath)) {
    return { needsSetup: true, needsLicense: true }
  }
  try {
    const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    return {
      needsSetup: !settings.setupComplete,
      needsLicense: !settings.licenseAccepted,
    }
  } catch {
    return { needsSetup: true, needsLicense: true }
  }
}

function markSetupComplete(settingsPath: string): void {
  let settings: Record<string, unknown> = {}

  try {
    if (fs.existsSync(settingsPath)) {
      settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    }
  } catch {
    settings = {}
  }

  settings.setupComplete = true
  settings.licenseAccepted = true
  settings.licenseAcceptedDate = new Date().toISOString()
  settings.setupDate = new Date().toISOString()

  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2))
}

function markLicenseAccepted(settingsPath: string): void {
  let settings: Record<string, unknown> = {}

  try {
    if (fs.existsSync(settingsPath)) {
      settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    }
  } catch {
    settings = {}
  }

  settings.licenseAccepted = true
  settings.licenseAcceptedDate = new Date().toISOString()

  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2))
}

export function registerAppHandlers(): void {
  ipcMain.handle('get-backend', () => {
    return { url: getBackendUrl() ?? '', token: getAuthToken() ?? '' }
  })

  ipcMain.handle('get-models-path', () => {
    return getModelsPath()
  })

  ipcMain.handle('check-gpu', async () => {
    return await checkGPU()
  })

  ipcMain.handle('get-app-info', () => {
    return {
      version: app.getVersion(),
      isPackaged: app.isPackaged,
      modelsPath: getModelsPath(),
      userDataPath: app.getPath('userData'),
    }
  })

  ipcMain.handle('get-downloads-path', () => {
    return app.getPath('downloads')
  })

  ipcMain.handle('check-first-run', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    return getSetupStatus(settingsPath)
  })

  ipcMain.handle('accept-license', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markLicenseAccepted(settingsPath)
    return true
  })

  ipcMain.handle('complete-setup', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markSetupComplete(settingsPath)
    return true
  })

  ipcMain.handle('fetch-license-text', async () => {
    const resp = await fetch('https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE')
    if (!resp.ok) {
      throw new Error(`Failed to fetch license (HTTP ${resp.status})`)
    }
    return await resp.text()
  })

  ipcMain.handle('get-notices-text', async () => {
    const noticesPath = path.join(app.getAppPath(), 'NOTICES.md')
    return fs.readFileSync(noticesPath, 'utf-8')
  })

  ipcMain.handle('get-resource-path', () => {
    if (!app.isPackaged) {
      return null
    }
    return process.resourcesPath
  })

  ipcMain.handle('check-python-ready', () => {
    return isPythonReady()
  })

  ipcMain.handle('start-python-setup', async () => {
    await downloadPythonEmbed((progress) => {
      getMainWindow()?.webContents.send('python-setup-progress', progress)
    })
  })

  ipcMain.handle('start-python-backend', async () => {
    await startPythonBackend()
  })

  ipcMain.handle('get-backend-health-status', () => {
    return getBackendHealthStatus()
  })

  ipcMain.handle('get-analytics-state', () => {
    return getAnalyticsState()
  })

  ipcMain.handle('set-analytics-enabled', (_event, enabled: boolean) => {
    setAnalyticsEnabled(enabled)
  })

  ipcMain.handle('send-analytics-event', async (_event, eventName: string, extraDetails?: Record<string, unknown> | null) => {
    await sendAnalyticsEvent(eventName, extraDetails)
  })

}
