import { app, dialog, shell } from 'electron'
import path from 'path'
import fs from 'fs'
import { checkGPU } from '../gpu'
import { isPythonReady, downloadPythonEmbed } from '../python-setup'
import { getBackendHealthStatus, getBackendUrl, getAuthToken, getAdminToken, startPythonBackend, setGenerationActive } from '../python-backend'
import { getMainWindow, applyMainWindowMode, applyWindowAppearance } from '../window'
import { getAnalyticsState, setAnalyticsEnabled } from '../analytics'
import { setRemoteExposureActive } from '../remote-keep-awake'
import {
  getUpdateState, checkForUpdatesNow, startUpdateDownload, installUpdateAndRestart,
  skipUpdateVersion, setAutoCheckUpdatesEnabled,
} from '../updater'
import { getAutoCheckUpdates, setWindowTheme } from '../app-state'
import { freeDiskBytes } from '../free-disk-space'
import { handle } from './typed-handle'
import { takePendingDeepLink } from '../deep-link-inbox'
import { registerOptionalAppHandlers } from '@optional/app-handlers'

function getModelsPath(): string {
  const modelsPath = path.join(app.getPath('userData'), 'models')
  if (!fs.existsSync(modelsPath)) {
    fs.mkdirSync(modelsPath, { recursive: true })
  }
  return modelsPath
}

function getSetupStatus(settingsPath: string): { needsSetup: boolean } {
  if (!fs.existsSync(settingsPath)) {
    return { needsSetup: true }
  }
  try {
    const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'))
    return {
      needsSetup: !settings.setupComplete,
    }
  } catch {
    return { needsSetup: true }
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
  settings.setupDate = new Date().toISOString()

  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2))
}

export function registerAppHandlers(hooks?: {
  onInstallationStarted?: () => void
  onSetupCompleted?: () => void
}): void {
  handle('getBackend', () => {
    return { url: getBackendUrl() ?? '', token: getAuthToken() ?? '' }
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

  handle('getFreeDiskSpace', async ({ path: targetPath }) => {
    try {
      const bytes = await freeDiskBytes(targetPath)
      return { success: true, bytes }
    } catch (e) {
      return { success: false, error: e instanceof Error ? e.message : String(e) }
    }
  })

  handle('checkFirstRun', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    return getSetupStatus(settingsPath)
  })

  handle('completeSetup', () => {
    const settingsPath = path.join(app.getPath('userData'), 'app_state.json')
    markSetupComplete(settingsPath)
    // API-only first run never clicks Install. The once-per-process guard
    // drops this when the Install click already sent `launched`.
    hooks?.onSetupCompleted?.()
    applyMainWindowMode('app')
    return true
  })

  // First install only. The renderer calls this when Install is clicked, after
  // the opt-out preference has been written. Returning opens send from main.
  handle('notifyInstallStarted', () => {
    hooks?.onInstallationStarted?.()
  })

  handle('setMainWindowMode', ({ mode }) => {
    applyMainWindowMode(mode)
    return true
  })

  handle('setWindowAppearance', ({ theme, persist }) => {
    applyWindowAppearance(theme)
    if (persist) setWindowTheme(theme)
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

  handle('takePendingDeepLink', () => takePendingDeepLink())

  handle('notifyGenerationActive', ({ active }) => {
    setGenerationActive(active)
  })

  handle('notifyRemoteExposure', ({ active }) => {
    setRemoteExposureActive(active)
  })

  handle('getAnalyticsState', () => {
    return getAnalyticsState()
  })

  handle('setAnalyticsEnabled', ({ enabled }) => {
    setAnalyticsEnabled(enabled)
  })

  handle('openModelsDirChangeDialog', async () => {
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

  handle('openModelsFolder', async () => {
    // Resolve the models dir from the backend rather than trusting a renderer-supplied
    // path, so a compromised renderer can't open an arbitrary location.
    const url = getBackendUrl()
    const auth = getAuthToken()
    if (!url || !auth) return { success: false, error: 'Backend not ready' }

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 5000)
    let resp: Response
    try {
      resp = await fetch(`${url}/api/settings`, {
        headers: { 'Authorization': `Bearer ${auth}` },
        signal: controller.signal,
      })
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : String(err) }
    } finally {
      clearTimeout(timeout)
    }
    if (!resp.ok) return { success: false, error: await resp.text() }
    const settings = (await resp.json()) as { modelsDir?: string }
    if (!settings.modelsDir) return { success: false, error: 'No models directory configured' }

    const error = await shell.openPath(settings.modelsDir)
    if (error) return { success: false, error }
    return { success: true }
  })

  handle('getUpdateState', () => getUpdateState())

  handle('checkForUpdatesNow', async () => {
    try {
      await checkForUpdatesNow()
      return { success: true }
    } catch (e) {
      return { success: false, error: String(e) }
    }
  })

  handle('startUpdateDownload', async () => {
    try {
      await startUpdateDownload()
      return { success: true }
    } catch (e) {
      return { success: false, error: String(e) }
    }
  })

  handle('installUpdateAndRestart', () => {
    try {
      return installUpdateAndRestart()
    } catch (e) {
      return { success: false, error: String(e) }
    }
  })

  handle('skipUpdateVersion', ({ version }) => {
    try {
      skipUpdateVersion(version)
      return { success: true }
    } catch (e) {
      return { success: false, error: String(e) }
    }
  })

  handle('getAutoCheckUpdates', () => ({ enabled: getAutoCheckUpdates() }))

  handle('setAutoCheckUpdates', ({ enabled }) => {
    try {
      setAutoCheckUpdatesEnabled(enabled)
      return { success: true }
    } catch (e) {
      return { success: false, error: String(e) }
    }
  })

  registerOptionalAppHandlers()

}
