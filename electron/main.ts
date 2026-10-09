import './win-dll-search'
import './app-paths'
import { randomUUID } from 'crypto'
import { app, powerMonitor, powerSaveBlocker } from 'electron'
import { readAppState } from './app-state'
import { setupCSP } from './csp'
import { installDeepLinkInbox } from './deep-link-inbox'
import { registerDeepLinkProtocolClient } from './deep-link-protocol'
import { registerExportHandlers } from './export/export-handler'
import { stopExportProcess } from './export/ffmpeg-utils'
import { registerAppHandlers } from './ipc/app-handlers'
import { registerFileHandlers } from './ipc/file-handlers'
import { registerMediaHandlers } from './ipc/media-handlers'
import { registerLogHandlers } from './ipc/log-handlers'
import { registerVideoProcessingHandlers } from './ipc/video-processing-handlers'
import { logger } from './logger'
import { initSessionLog } from './logging-management'
import { stopPythonBackend } from './python-backend'
import { isPythonReady } from './python-setup'
import {
  initRemoteKeepAwake,
  installRemoteKeepAwake,
  stopRemoteKeepAwake,
} from './remote-keep-awake'
import { initAutoUpdater } from './updater'
import { createWindow, getMainWindow } from './window'
import { sendAnalyticsEvent } from './analytics'
import { shouldSendLaunched, type LaunchedTrigger } from './launched-event'
import type { DeepLinkIntent } from '../shared/deep-link.ts'

function logAppVersion(): void {
  if (!app.isPackaged) {
    logger.info('[LTX Desktop] Running in development mode')
  } else {
    logger.info(`[LTX Desktop] Version ${app.getVersion()}`)
  }
}

function resolveWindowMode(): 'install' | 'app' {
  if (!isPythonReady().ready) return 'install'
  const state = readAppState()
  return !state.setupComplete ? 'install' : 'app'
}

function ensureWindow(): void {
  const mainWindow = getMainWindow()
  if (mainWindow) {
    if (mainWindow.isMinimized()) {
      mainWindow.restore()
    }
    if (!mainWindow.isVisible()) {
      mainWindow.show()
    }
    mainWindow.focus()
    return
  }
  if (app.isReady()) {
    createWindow(resolveWindowMode())
  }
}

function publishDeepLink(intent: DeepLinkIntent): void {
  getMainWindow()?.webContents.send('deep-link', intent)
}

const gotLock = app.requestSingleInstanceLock()

if (!gotLock) {
  app.quit()
} else {
  initSessionLog()
  logAppVersion()

  registerDeepLinkProtocolClient(
    (scheme, execPath, args) => app.setAsDefaultProtocolClient(scheme, execPath, args),
    {
      defaultApp: Boolean(process.defaultApp),
      execPath: process.execPath,
      argv: process.argv,
    },
  )

  installDeepLinkInbox({
    argv: process.argv,
    now: () => Date.now(),
    newId: () => randomUUID(),
    onOpenUrl: (listener) => {
      app.on('open-url', (event, url) => {
        event.preventDefault()
        listener(url)
      })
    },
    onSecondInstance: (listener) => {
      app.on('second-instance', (_event, commandLine) => {
        listener(commandLine)
      })
    },
    ensureWindow,
    publish: publishDeepLink,
    log: (message) => logger.info(message),
  })

  // `launched` is once per process, and only after the user's analytics choice is known.
  //
  // A returning open already stored that choice, so we send at startup.
  // A first install shows an opt-out on the setup screen. Collection stays on
  // unless the user checks it. We wait for the Install click, and the renderer
  // finishes writing that preference before it notifies us, so someone who opted
  // out is not recorded for this open. sendAnalyticsEvent then sends nothing
  // when analytics are disabled.
  //
  // An API-only first run never shows that screen. Setup completion sends the
  // event then, still only if this process has not sent it already.
  //
  // The flag below stops a second event in the same process: retrying Install,
  // or finishing setup after Install, must not send `launched` again.
  let launchedSent = false
  const sendLaunched = (trigger: LaunchedTrigger): void => {
    if (!shouldSendLaunched({
      alreadySent: launchedSent,
      setupComplete: readAppState().setupComplete === true,
      trigger,
    })) return
    launchedSent = true
    void sendAnalyticsEvent('launched')
  }

  registerAppHandlers({
    onInstallationStarted: () => sendLaunched('installation-started'),
    onSetupCompleted: () => sendLaunched('setup-completed'),
  })
  registerFileHandlers()
  registerMediaHandlers()
  registerLogHandlers()
  registerExportHandlers()
  registerVideoProcessingHandlers()

  app.whenReady().then(async () => {
    setupCSP()
    installRemoteKeepAwake({
      power: {
        isOnBatteryPower: () => powerMonitor.isOnBatteryPower(),
        onAc: (listener) => { powerMonitor.on('on-ac', listener) },
        onBattery: (listener) => { powerMonitor.on('on-battery', listener) },
      },
      blocker: {
        start: () => powerSaveBlocker.start('prevent-app-suspension'),
        stop: (id) => { powerSaveBlocker.stop(id) },
        isStarted: (id) => powerSaveBlocker.isStarted(id),
      },
      log: (message) => logger.info(message),
    })
    initRemoteKeepAwake()
    ensureWindow()
    initAutoUpdater()
    // Python setup + backend start are now driven by the renderer via IPC

    // Returning open only. A first install sends from onInstallationStarted.
    sendLaunched('startup')
  })

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      stopPythonBackend()
      app.quit()
    }
  })

  app.on('activate', () => {
    if (getMainWindow() === null) {
      createWindow(resolveWindowMode())
    }
  })

  app.on('before-quit', () => {
    stopRemoteKeepAwake()
    stopExportProcess()
    stopPythonBackend()
  })
}
