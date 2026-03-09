import './app-paths'
import { app, protocol, net } from 'electron'
import { setupCSP } from './csp'
import { registerExportHandlers } from './export/export-handler'
import { stopExportProcess } from './export/ffmpeg-utils'
import { registerAppHandlers } from './ipc/app-handlers'
import { registerFileHandlers } from './ipc/file-handlers'
import { registerLogHandlers } from './ipc/log-handlers'
import { registerVideoProcessingHandlers } from './ipc/video-processing-handlers'
import { initSessionLog } from './logging-management'
import { stopPythonBackend } from './python-backend'
import { initAutoUpdater } from './updater'
import { createWindow, getMainWindow } from './window'
import { sendAnalyticsEvent } from './analytics'
import { logger } from './logger'

// Register directorsdesktop:// protocol for auth callbacks
if (process.defaultApp) {
  // Dev mode: pass the app path so Electron can find us
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient('directorsdesktop', process.execPath, [
      path.resolve(process.argv[1]),
    ])
  }
} else {
  app.setAsDefaultProtocolClient('directorsdesktop')
}

import path from 'path'

/** Extract auth token from a directorsdesktop:// deep link URL. */
function handleDeepLink(url: string): void {
  // Redact query params (may contain auth tokens)
  const safeUrl = url.split('?')[0] + (url.includes('?') ? '?<redacted>' : '')
  logger.info(`[deep-link] Received: ${safeUrl}`)
  try {
    const parsed = new URL(url)
    // Expected: directorsdesktop://auth/callback?token=XXX
    if (parsed.hostname === 'auth' && parsed.pathname === '/callback') {
      const token = parsed.searchParams.get('token')
      if (token) {
        const mainWindow = getMainWindow()
        if (mainWindow) {
          mainWindow.webContents.send('palette-auth-callback', { token })
          logger.info('[deep-link] Auth token forwarded to renderer')
        }
      }
    }
  } catch (err) {
    logger.error(`[deep-link] Failed to parse URL: ${err}`)
  }
}

const gotLock = app.requestSingleInstanceLock()

if (!gotLock) {
  app.quit()
} else {
  initSessionLog()

  registerAppHandlers()
  registerFileHandlers()
  registerLogHandlers()
  registerExportHandlers()
  registerVideoProcessingHandlers()

  app.on('second-instance', (_event, commandLine) => {
    const mainWindow = getMainWindow()
    if (mainWindow) {
      if (mainWindow.isMinimized()) {
        mainWindow.restore()
      }
      if (!mainWindow.isVisible()) {
        mainWindow.show()
      }
      mainWindow.focus()
    } else if (app.isReady()) {
      createWindow()
    }

    // On Windows, the deep link URL is in the command line args
    const deepLinkUrl = commandLine.find((arg) => arg.startsWith('directorsdesktop://'))
    if (deepLinkUrl) {
      handleDeepLink(deepLinkUrl)
    }
  })

  // macOS: handle protocol URL via open-url event
  app.on('open-url', (_event, url) => {
    handleDeepLink(url)
  })

  app.whenReady().then(async () => {
    setupCSP()

    // Allow file:// URLs to load when the page is served from http://localhost (dev mode).
    // Without this, Chromium blocks file:// resources on http:// origins.
    protocol.handle('file', (request) => net.fetch(request))

    createWindow()
    initAutoUpdater()
    // Python setup + backend start are now driven by the renderer via IPC

    // Fire analytics event (no-op if user hasn't opted in)
    void sendAnalyticsEvent('ltxdesktop_app_launched')
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      stopPythonBackend()
      app.quit()
    }
  })

  app.on('activate', () => {
    if (getMainWindow() === null) {
      createWindow()
    }
  })

  app.on('before-quit', () => {
    stopExportProcess()
    stopPythonBackend()
  })
}
