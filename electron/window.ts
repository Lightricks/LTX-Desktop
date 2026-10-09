import { app, BrowserWindow, Menu, nativeImage, nativeTheme, screen, shell } from 'electron'
import type { BrowserWindowConstructorOptions } from 'electron'
import path from 'path'
import fs from 'fs'
import { pathToFileURL } from 'url'
import { getViteDevServerUrl } from '../shared/vite-dev-server'
import { isDev, getCurrentDir } from './config'
import { getWindowTheme } from './app-state'
import { logger } from './logger'
import {
  isTrustedAppOrigin,
  mediaRequestOrigin,
  readIsMainFrame,
  runPermissionCheckHandler,
  runPermissionRequestHandler,
  shouldAllowPermissionCheck,
  shouldGrantPermissionRequest,
  shouldTrustMediaPermissionSource,
} from './media-permission'
import { isAllowedExternalUrl } from './allowed-external-url'
import { APP_WINDOW_BOUNDS, INSTALL_WINDOW_BOUNDS, centeredBounds, fitToWorkArea } from '../shared/window-bounds'
import {
  APP_THEME_PREFERENCE,
  GLASS_MATERIAL,
  MACOS_TRAFFIC_LIGHT_POSITION,
  WINDOW_BACKGROUND,
  WINDOW_BACKGROUND_CLEAR,
} from '../shared/window-chrome'

let mainWindow: BrowserWindow | null = null
let windowMode: 'install' | 'app' | null = null
/** Glass decision captured at construction — transparency prefs are live, the built window is not. */
let mainWindowIsGlass = false

/** Mac glass is window chrome (install and app), not an install-only look. */
function shouldUseMacGlass(): boolean {
  return process.platform === 'darwin' && !nativeTheme.prefersReducedTransparency
}

function applyGlass(win: BrowserWindow): void {
  if (!shouldUseMacGlass()) return
  win.setVibrancy(GLASS_MATERIAL)
}

export function applyWindowAppearance(
  // ThemeProvider notifies main of the in-app preference via setWindowAppearance;
  // launch restores the persisted value so glass matches the UI, not only the OS.
  theme: 'light' | 'dark' | 'system' = APP_THEME_PREFERENCE,
): void {
  nativeTheme.themeSource = theme
  // Single-window app: a glass window manages its own background via vibrancy.
  // Uses the construction-time flag — a mid-session reduced-transparency
  // toggle must not flip the branch for an already-built window.
  if (mainWindowIsGlass) return
  const win = mainWindow
  if (!win || win.isDestroyed()) return
  const color =
    theme === 'system'
      ? nativeTheme.shouldUseDarkColors
        ? WINDOW_BACKGROUND.dark
        : WINDOW_BACKGROUND.light
      : WINDOW_BACKGROUND[theme]
  win.setBackgroundColor(color)
}

function applyNativeMenuBar(mode: 'install' | 'app', win: BrowserWindow): void {
  if (process.platform === 'darwin') return
  win.setMenu(mode === 'install' ? null : Menu.getApplicationMenu())
}

export function applyMainWindowMode(mode: 'install' | 'app'): void {
  const win = mainWindow
  if (!win || win.isDestroyed()) return
  // Defer the whole transition (mode assignment included) until fullscreen
  // exits, so mode and bounds can never diverge.
  if (win.isFullScreen()) {
    win.once('leave-full-screen', () => applyMainWindowMode(mode))
    return
  }
  applyGlass(win)
  if (windowMode === mode) return
  windowMode = mode
  applyNativeMenuBar(mode, win)
  if (win.isMaximized()) win.unmaximize()

  const spec = mode === 'install' ? INSTALL_WINDOW_BOUNDS : APP_WINDOW_BOUNDS
  const display = screen.getDisplayMatching(win.getBounds())
  const target = centeredBounds(display.workArea, spec.width, spec.height)
  win.setMinimumSize(
    Math.min(spec.minWidth, target.width),
    Math.min(spec.minHeight, target.height),
  )
  win.setBounds(target, true)
}

function macChromeOptions(useGlass: boolean): Partial<BrowserWindowConstructorOptions> {
  if (process.platform !== 'darwin') {
    return {}
  }
  if (!useGlass) {
    return {
      titleBarStyle: 'hiddenInset',
      trafficLightPosition: MACOS_TRAFFIC_LIGHT_POSITION,
    }
  }
  return {
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: MACOS_TRAFFIC_LIGHT_POSITION,
    transparent: true,
    hasShadow: true,
    vibrancy: GLASS_MATERIAL,
    visualEffectState: 'active',
  }
}

export function createWindow(mode: 'install' | 'app'): BrowserWindow {
  // Get the path to preload script
  const preloadPath = isDev
    ? path.join(getCurrentDir(), 'dist-electron', 'preload.js')
    : path.join(app.getAppPath(), 'dist-electron', 'preload.js')

  // App icon — use .ico on Windows, .png elsewhere
  const iconExt = process.platform === 'win32' ? 'icon.ico' : 'icon.png'
  const iconPath = path.join(getCurrentDir(), 'resources', iconExt)
  logger.info(`[icon] Loading app icon from: ${iconPath} | exists: ${fs.existsSync(iconPath)}`)
  const appIcon = fs.existsSync(iconPath) ? nativeImage.createFromPath(iconPath) : undefined

  const install = mode === 'install'
  const spec = install ? INSTALL_WINDOW_BOUNDS : APP_WINDOW_BOUNDS
  const workArea = screen.getPrimaryDisplay().workArea
  const fitted = fitToWorkArea(spec.width, spec.height, workArea.width, workArea.height)
  windowMode = mode
  const useGlass = shouldUseMacGlass()

  mainWindow = new BrowserWindow({
    width: fitted.width,
    height: fitted.height,
    minWidth: Math.min(spec.minWidth, fitted.width),
    minHeight: Math.min(spec.minHeight, fitted.height),
    icon: appIcon,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: isDev ? false : true,
    },
    backgroundColor: useGlass ? WINDOW_BACKGROUND_CLEAR : WINDOW_BACKGROUND.dark,
    ...macChromeOptions(useGlass),
    autoHideMenuBar: process.platform !== 'darwin' && install,
    show: false,
  })
  mainWindowIsGlass = useGlass

  applyNativeMenuBar(windowMode, mainWindow)

  // Load the app
  if (isDev) {
    mainWindow.loadURL(getViteDevServerUrl())
    // DevTools can be opened manually with Ctrl+Shift+I or F12
  } else {
    mainWindow.loadFile(path.join(app.getAppPath(), 'dist', 'index.html'))
  }

  applyWindowAppearance(getWindowTheme() ?? APP_THEME_PREFERENCE)

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
  })

  const renderer = mainWindow.webContents
  const trustedRendererOrigin = (): string => {
    if (isDev) {
      return getViteDevServerUrl()
    }
    const loaded = renderer.getURL()
    if (loaded.startsWith('file:')) {
      return loaded
    }
    return pathToFileURL(path.join(app.getAppPath(), 'dist', 'index.html')).href
  }
  renderer.session.setPermissionRequestHandler(
    (contents, permission, callback, details) => {
      runPermissionRequestHandler(() => {
        if (permission !== 'media') {
          return true
        }
        const granted =
          shouldTrustMediaPermissionSource({
            contents,
            mainContents: renderer,
            requestingOrigin: mediaRequestOrigin(details),
            trustedOrigin: trustedRendererOrigin(),
            isMainFrame: readIsMainFrame(details),
          }) && shouldGrantPermissionRequest(permission, details)
        if (!granted) {
          logger.info('[media] denying media permission request')
        }
        return granted
      }, callback)
    },
  )
  renderer.session.setPermissionCheckHandler(
    (contents, permission, requestingOrigin, details) =>
      runPermissionCheckHandler(() => {
        if (permission !== 'media') {
          return true
        }
        return (
          shouldTrustMediaPermissionSource({
            contents,
            mainContents: renderer,
            requestingOrigin,
            trustedOrigin: trustedRendererOrigin(),
            allowNullContents: true,
            isMainFrame: readIsMainFrame(details),
          }) && shouldAllowPermissionCheck(permission, details)
        )
      }),
  )

  const allowRendererUrl = (url: string): boolean =>
    isTrustedAppOrigin(url, trustedRendererOrigin())
  renderer.on('will-navigate', (event, url) => {
    if (!allowRendererUrl(url)) {
      event.preventDefault()
    }
  })
  renderer.on('will-redirect', (event, url) => {
    if (!allowRendererUrl(url)) {
      event.preventDefault()
    }
  })
  renderer.on('will-frame-navigate', (details) => {
    if (!details.isMainFrame || !allowRendererUrl(details.url)) {
      details.preventDefault()
    }
  })
  renderer.on('will-attach-webview', (event) => {
    event.preventDefault()
  })
  renderer.setWindowOpenHandler(({ url }) => {
    if (isAllowedExternalUrl(url)) {
      void shell.openExternal(url)
    }
    return { action: 'deny' }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
    windowMode = null
    mainWindowIsGlass = false
  })

  return mainWindow
}

export function getMainWindow(): BrowserWindow | null {
  return mainWindow
}
