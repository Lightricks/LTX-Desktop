// Using require for Electron preload compatibility
const { contextBridge, ipcRenderer } = require('electron')

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // Get the backend URL
  getBackendUrl: (): Promise<string> => ipcRenderer.invoke('get-backend-url'),
  
  // Get the path where models are stored
  getModelsPath: (): Promise<string> => ipcRenderer.invoke('get-models-path'),
  
  // Read a local file and return as base64
  readLocalFile: (filePath: string): Promise<{ data: string; mimeType: string }> => 
    ipcRenderer.invoke('read-local-file', filePath),
  
  // Check GPU availability
  checkGpu: (): Promise<{ available: boolean; name?: string; vram?: number }> =>
    ipcRenderer.invoke('check-gpu'),
  
  // Get app info
  getAppInfo: (): Promise<{ version: string; isPackaged: boolean; modelsPath: string; userDataPath: string }> =>
    ipcRenderer.invoke('get-app-info'),
  
  // First-run setup
  checkFirstRun: (): Promise<{ needsSetup: boolean; needsLicense: boolean }> => ipcRenderer.invoke('check-first-run'),
  acceptLicense: (): Promise<boolean> => ipcRenderer.invoke('accept-license'),
  completeSetup: (): Promise<boolean> => ipcRenderer.invoke('complete-setup'),
  fetchLicenseText: (): Promise<string> => ipcRenderer.invoke('fetch-license-text'),
  getNoticesText: (): Promise<string> => ipcRenderer.invoke('get-notices-text'),
  
  // Open specific app pages / folders
  openLtxApiKeyPage: (): Promise<boolean> => ipcRenderer.invoke('open-ltx-api-key-page'),
  openReplicateApiKeyPage: (): Promise<boolean> => ipcRenderer.invoke('open-replicate-api-key-page'),
  openPaletteLoginPage: (): Promise<boolean> => ipcRenderer.invoke('open-palette-login-page'),
  openPaletteApiKeyPage: (): Promise<boolean> => ipcRenderer.invoke('open-palette-api-key-page'),
  openParentFolderOfFile: (filePath: string): Promise<void> => ipcRenderer.invoke('open-parent-folder-of-file', filePath),
  
  // Reveal a specific file in the OS file manager (Explorer/Finder)
  showItemInFolder: (filePath: string): Promise<void> => ipcRenderer.invoke('show-item-in-folder', filePath),
  
  // Log viewer
  getLogs: (): Promise<LogsResponse> => ipcRenderer.invoke('get-logs'),
  getLogPath: (): Promise<{ logPath: string; logDir: string }> => ipcRenderer.invoke('get-log-path'),
  openLogFolder: (): Promise<boolean> => ipcRenderer.invoke('open-log-folder'),
  
  // Get resources path (for video assets in production)
  getResourcePath: (): Promise<string | null> => ipcRenderer.invoke('get-resource-path'),
  
  // Paths
  getDownloadsPath: (): Promise<string> => ipcRenderer.invoke('get-downloads-path'),
  ensureDirectory: (dirPath: string): Promise<{ success: boolean; error?: string }> =>
    ipcRenderer.invoke('ensure-directory', dirPath),

  // File save/export
  showSaveDialog: (options: { title?: string; defaultPath?: string; filters?: { name: string; extensions: string[] }[] }): Promise<string | null> =>
    ipcRenderer.invoke('show-save-dialog', options),
  saveFile: (filePath: string, data: string, encoding?: string): Promise<{ success: boolean; path?: string; error?: string }> =>
    ipcRenderer.invoke('save-file', filePath, data, encoding),
  saveBinaryFile: (filePath: string, data: ArrayBuffer): Promise<{ success: boolean; path?: string; error?: string }> =>
    ipcRenderer.invoke('save-binary-file', filePath, data),
  showOpenDirectoryDialog: (options: { title?: string }): Promise<string | null> =>
    ipcRenderer.invoke('show-open-directory-dialog', options),
  searchDirectoryForFiles: (dir: string, filenames: string[]): Promise<Record<string, string>> =>
    ipcRenderer.invoke('search-directory-for-files', dir, filenames),
  copyFile: (src: string, dest: string): Promise<{ success: boolean; error?: string }> =>
    ipcRenderer.invoke('copy-file', src, dest),
  
  // Check multiple files at once
  checkFilesExist: (filePaths: string[]): Promise<Record<string, boolean>> =>
    ipcRenderer.invoke('check-files-exist', filePaths),
  
  // Show open file dialog
  showOpenFileDialog: (options: { title?: string; filters?: { name: string; extensions: string[] }[]; properties?: string[] }): Promise<string[] | null> =>
    ipcRenderer.invoke('show-open-file-dialog', options),
  
  // Video export via ffmpeg (native compositing — no canvas, no frame-by-frame)
  exportNative: (data: {
    clips: { url: string; type: string; startTime: number; duration: number; trimStart: number; speed: number; reversed: boolean; flipH: boolean; flipV: boolean; opacity: number; trackIndex: number; muted: boolean; volume: number }[];
    outputPath: string; codec: string; width: number; height: number; fps: number; quality: number;
    letterbox?: { ratio: number; color: string; opacity: number };
    subtitles?: { text: string; startTime: number; endTime: number; style: { fontSize: number; fontFamily: string; fontWeight: string; color: string; backgroundColor: string; position: string; italic: boolean } }[];
  }): Promise<{ success?: boolean; error?: string }> =>
    ipcRenderer.invoke('export-native', data),
  exportCancel: (sessionId: string): Promise<{ ok?: boolean }> =>
    ipcRenderer.invoke('export-cancel', sessionId),

  // Python setup (Windows first-launch download)
  checkPythonReady: (): Promise<{ ready: boolean }> => ipcRenderer.invoke('check-python-ready'),
  startPythonSetup: (): Promise<void> => ipcRenderer.invoke('start-python-setup'),
  startPythonBackend: (): Promise<void> => ipcRenderer.invoke('start-python-backend'),
  getBackendHealthStatus: (): Promise<BackendHealthStatus | null> => ipcRenderer.invoke('get-backend-health-status'),
  onPythonSetupProgress: (cb: (data: unknown) => void) => {
    ipcRenderer.on('python-setup-progress', (_: unknown, data: unknown) => cb(data))
  },
  removePythonSetupProgress: () => {
    ipcRenderer.removeAllListeners('python-setup-progress')
  },
  onBackendHealthStatus: (cb: (data: BackendHealthStatus) => void) => {
    const listener = (_: unknown, data: BackendHealthStatus) => cb(data)
    ipcRenderer.on('backend-health-status', listener)
    return () => {
      ipcRenderer.removeListener('backend-health-status', listener)
    }
  },

  // Extract a single video frame via ffmpeg (returns file path + file:// URL)
  extractVideoFrame: (videoUrl: string, seekTime: number, width?: number, quality?: number): Promise<{ path: string; url: string }> =>
    ipcRenderer.invoke('extract-video-frame', videoUrl, seekTime, width, quality),

  // Write a log line to the session log file
  writeLog: (level: string, message: string): Promise<void> =>
    ipcRenderer.invoke('write-log', level, message),

  // Analytics
  getAnalyticsState: (): Promise<{ analyticsEnabled: boolean; installationId: string }> =>
    ipcRenderer.invoke('get-analytics-state'),
  setAnalyticsEnabled: (enabled: boolean): Promise<void> =>
    ipcRenderer.invoke('set-analytics-enabled', enabled),
  sendAnalyticsEvent: (eventName: string, extraDetails?: Record<string, unknown> | null): Promise<void> =>
    ipcRenderer.invoke('send-analytics-event', eventName, extraDetails),

  // Deep link auth callback (from directorsdesktop:// protocol)
  onPaletteAuthCallback: (cb: (data: { token: string }) => void) => {
    const listener = (_: unknown, data: { token: string }) => cb(data)
    ipcRenderer.on('palette-auth-callback', listener)
    return () => {
      ipcRenderer.removeListener('palette-auth-callback', listener)
    }
  },

  // Platform info
  platform: process.platform,
})

interface LogsResponse {
  logPath: string
  lines: string[]
  error?: string
}

interface BackendHealthStatus {
  status: 'alive' | 'restarting' | 'dead'
  exitCode?: number | null
}

// Type definitions for the exposed API
declare global {
  interface Window {
    electronAPI: {
      getBackendUrl: () => Promise<string>
      getModelsPath: () => Promise<string>
      readLocalFile: (filePath: string) => Promise<{ data: string; mimeType: string }>
      checkGpu: () => Promise<{ available: boolean; name?: string; vram?: number }>
      getAppInfo: () => Promise<{ version: string; isPackaged: boolean; modelsPath: string; userDataPath: string }>
      checkFirstRun: () => Promise<{ needsSetup: boolean; needsLicense: boolean }>
      acceptLicense: () => Promise<boolean>
      completeSetup: () => Promise<boolean>
      fetchLicenseText: () => Promise<string>
      getNoticesText: () => Promise<string>
      openLtxApiKeyPage: () => Promise<boolean>
      openParentFolderOfFile: (filePath: string) => Promise<void>
      showItemInFolder: (filePath: string) => Promise<void>
      getLogs: () => Promise<LogsResponse>
      getLogPath: () => Promise<{ logPath: string; logDir: string }>
      openLogFolder: () => Promise<boolean>
      getResourcePath: () => Promise<string | null>
      getDownloadsPath: () => Promise<string>
      ensureDirectory: (dirPath: string) => Promise<{ success: boolean; error?: string }>
      showSaveDialog: (options: { title?: string; defaultPath?: string; filters?: { name: string; extensions: string[] }[] }) => Promise<string | null>
      saveFile: (filePath: string, data: string, encoding?: string) => Promise<{ success: boolean; path?: string; error?: string }>
      saveBinaryFile: (filePath: string, data: ArrayBuffer) => Promise<{ success: boolean; path?: string; error?: string }>
      showOpenDirectoryDialog: (options: { title?: string }) => Promise<string | null>
      searchDirectoryForFiles: (dir: string, filenames: string[]) => Promise<Record<string, string>>
      copyFile: (src: string, dest: string) => Promise<{ success: boolean; error?: string }>
      checkFilesExist: (filePaths: string[]) => Promise<Record<string, boolean>>
      showOpenFileDialog: (options: { title?: string; filters?: { name: string; extensions: string[] }[]; properties?: string[] }) => Promise<string[] | null>
      exportNative: (data: {
        clips: { url: string; type: string; startTime: number; duration: number; trimStart: number; speed: number; reversed: boolean; flipH: boolean; flipV: boolean; opacity: number; trackIndex: number; muted: boolean; volume: number }[];
        outputPath: string; codec: string; width: number; height: number; fps: number; quality: number;
        letterbox?: { ratio: number; color: string; opacity: number };
        subtitles?: { text: string; startTime: number; endTime: number; style: { fontSize: number; fontFamily: string; fontWeight: string; color: string; backgroundColor: string; position: string; italic: boolean } }[];
      }) => Promise<{ success?: boolean; error?: string }>
      exportCancel: (sessionId: string) => Promise<{ ok?: boolean }>
      checkPythonReady: () => Promise<{ ready: boolean }>
      startPythonSetup: () => Promise<void>
      startPythonBackend: () => Promise<void>
      getBackendHealthStatus: () => Promise<BackendHealthStatus | null>
      onPythonSetupProgress: (cb: (data: unknown) => void) => void
      removePythonSetupProgress: () => void
      onBackendHealthStatus: (cb: (data: BackendHealthStatus) => void) => (() => void)
      extractVideoFrame: (videoUrl: string, seekTime: number, width?: number, quality?: number) => Promise<{ path: string; url: string }>
      writeLog: (level: string, message: string) => Promise<void>
      getAnalyticsState: () => Promise<{ analyticsEnabled: boolean; installationId: string }>
      setAnalyticsEnabled: (enabled: boolean) => Promise<void>
      sendAnalyticsEvent: (eventName: string, extraDetails?: Record<string, unknown> | null) => Promise<void>
      onPaletteAuthCallback: (cb: (data: { token: string }) => void) => (() => void)
      platform: string
    }
  }
}

export {}
