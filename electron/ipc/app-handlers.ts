import { app, dialog } from 'electron'
import path from 'path'
import fs from 'fs'
import { checkGPU } from '../gpu'
import { isPythonReady, downloadPythonEmbed } from '../python-setup'
import {
  configureBackendConnection,
  getBackendConnectionMode,
  getBackendConnectionRevision,
  getBackendHealthStatus,
  getBackendUrl,
  getAuthToken,
  getAdminToken,
  startBackendConnection,
  startPythonBackend,
  testExternalBackendConnection,
} from '../python-backend'
import { getMainWindow } from '../window'
import { getAnalyticsState, setAnalyticsEnabled, sendAnalyticsEvent } from '../analytics'
import { handle } from './typed-handle'
import { getBackendConnectionSummary } from '../app-state'
import { materializeBackendArtifact, uploadBackendMedia } from '../backend-media-transfer'

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
  handle('getBackend', () => {
    return {
      url: getBackendUrl() ?? '',
      token: getAuthToken() ?? '',
      mode: getBackendConnectionMode(),
      connectionRevision: getBackendConnectionRevision(),
    }
  })

  handle('getBackendConnectionConfig', () => {
    return getBackendConnectionSummary()
  })

  handle('testBackendConnection', async ({ url, authToken }) => {
    try {
      const serverInfo = await testExternalBackendConnection(url, authToken)
      return { success: true, serverInfo }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  })

  handle('setBackendConnection', async (config) => {
    try {
      await configureBackendConnection(config)
      return { success: true }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  })

  handle('startBackendConnection', async () => {
    await startBackendConnection()
  })

  handle('retryBackendConnection', async () => {
    await startBackendConnection()
  })

  handle('getModelsPath', () => {
    return getModelsPath()
  })

  handle('checkGpu', async () => {
    return await checkGPU()
  })

  handle('getAppInfo', () => {
    return {
      version: app.getVersion(),
      isPackaged: app.isPackaged,
      modelsPath: getModelsPath(),
      userDataPath: app.getPath('userData'),
    }
  })

  handle('getDownloadsPath', () => {
    return app.getPath('downloads')
  })

  handle('checkFirstRun', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    return getSetupStatus(settingsPath)
  })

  handle('acceptLicense', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markLicenseAccepted(settingsPath)
    return true
  })

  handle('completeSetup', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markSetupComplete(settingsPath)
    return true
  })

  handle('fetchLicenseText', async () => {
    const resp = await fetch('https://huggingface.co/Lightricks/LTX-2.3/raw/main/LICENSE')
    if (!resp.ok) {
      throw new Error(`Failed to fetch license (HTTP ${resp.status})`)
    }
    return await resp.text()
  })

  handle('getNoticesText', async () => {
    const noticesPath = path.join(app.getAppPath(), 'NOTICES.md')
    return fs.readFileSync(noticesPath, 'utf-8')
  })

  handle('getResourcePath', () => {
    if (!app.isPackaged) {
      return null
    }
    return process.resourcesPath
  })

  handle('checkPythonReady', () => {
    return isPythonReady()
  })

  handle('startPythonSetup', async () => {
    await downloadPythonEmbed((progress) => {
      getMainWindow()?.webContents.send('python-setup-progress', progress)
    })
  })

  handle('startPythonBackend', async () => {
    await startPythonBackend()
  })

  handle('getBackendHealthStatus', () => {
    return getBackendHealthStatus()
  })

  handle('uploadBackendMedia', async ({ filePath, mediaType }) => {
    try {
      const media = await uploadBackendMedia(filePath, mediaType)
      return { success: true, media }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  })

  handle('materializeBackendArtifact', async ({ artifact }) => {
    try {
      const materializedPath = await materializeBackendArtifact(artifact)
      return { success: true, path: materializedPath }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  })

  handle('getAnalyticsState', () => {
    return getAnalyticsState()
  })

  handle('setAnalyticsEnabled', ({ enabled }) => {
    setAnalyticsEnabled(enabled)
  })

  handle('sendAnalyticsEvent', async ({ eventName, extraDetails }) => {
    await sendAnalyticsEvent(eventName, extraDetails)
  })

  handle('openModelsDirChangeDialog', async () => {
    if (getBackendConnectionMode() === 'external') {
      return { success: false, error: 'Choose the models path on the Atom, not on this computer' }
    }
    const mainWindow = getMainWindow()
    if (!mainWindow) return { success: false, error: 'No window' }

    const result = await dialog.showOpenDialog(mainWindow, {
      title: 'Select Models Directory',
      properties: ['openDirectory', 'createDirectory'],
    })
    if (result.canceled || !result.filePaths.length) return { success: false, error: 'cancelled' }

    const newDir = result.filePaths[0]
    const url = getBackendUrl()
    const auth = getAuthToken()
    const admin = getAdminToken()
    if (!url || !auth || !admin) return { success: false, error: 'Backend not ready' }

    const resp = await fetch(`${url}/api/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${auth}`,
        'X-Admin-Token': admin,
      },
      body: JSON.stringify({ modelsDir: newDir }),
    })
    if (!resp.ok) return { success: false, error: await resp.text() }

    return { success: true, path: newDir }
  })

  handle('setBackendModelsDirectory', async ({ path: modelsDir }) => {
    const url = getBackendUrl()
    const auth = getAuthToken()
    const admin = getAdminToken()
    if (!url || !auth || !admin) {
      return { success: false, error: 'An admin token is required to change the backend models directory' }
    }

    try {
      const resp = await fetch(`${url}/api/settings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${auth}`,
          'X-Admin-Token': admin,
        },
        body: JSON.stringify({ modelsDir }),
      })
      if (!resp.ok) return { success: false, error: await resp.text() }
      return { success: true, path: modelsDir }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) }
    }
  })

}
