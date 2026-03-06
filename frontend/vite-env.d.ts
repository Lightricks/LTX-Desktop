/// <reference types="vite/client" />

interface LogsResponse {
  logPath: string
  lines: string[]
  error?: string
}

interface BackendHealthStatus {
  status: 'alive' | 'restarting' | 'dead'
  exitCode?: number | null
}

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
    openFalApiKeyPage: () => Promise<boolean>
    openParentFolderOfFile: (filePath: string) => Promise<void>
    showItemInFolder: (filePath: string) => Promise<void>
    getLogs: () => Promise<LogsResponse>
    getLogPath: () => Promise<{ logPath: string; logDir: string }>
    openLogFolder: () => Promise<boolean>
    getResourcePath: () => Promise<string | null>
    getDownloadsPath: () => Promise<string>
    ensureDirectory: (dirPath: string) => Promise<{ success: boolean; error?: string }>
    getProjectsDir: () => Promise<string>
    loadAllProjects: () => Promise<unknown[]>
    saveProject: (project: unknown) => Promise<{ ok: true }>
    replaceAllProjects: (projects: unknown[]) => Promise<{ ok: true }>
    deleteProjectFile: (projectId: string) => Promise<{ ok: true }>
    showSaveDialog: (options: { title?: string; defaultPath?: string; filters?: { name: string; extensions: string[] }[] }) => Promise<string | null>
    saveFile: (filePath: string, data: string, encoding?: string) => Promise<{ success: boolean; path?: string; error?: string }>
    saveBinaryFile: (filePath: string, data: ArrayBuffer) => Promise<{ success: boolean; path?: string; error?: string }>
    showOpenDirectoryDialog: (options: { title?: string }) => Promise<string | null>
    copyFile: (src: string, dest: string) => Promise<{ success: boolean; error?: string }>
    checkFilesExist: (filePaths: string[]) => Promise<Record<string, boolean>>
    showOpenFileDialog: (options: { title?: string; filters?: { name: string; extensions: string[] }[]; properties?: string[] }) => Promise<string[] | null>
    searchDirectoryForFiles: (directory: string, filenames: string[]) => Promise<Record<string, string | null>>
    exportNative: (data: {
      clips: { url: string; type: string; startTime: number; duration: number; trimStart: number; speed: number; reversed: boolean; flipH: boolean; flipV: boolean; opacity: number; trackIndex: number; muted: boolean; volume: number }[]
      outputPath: string; codec: string; width: number; height: number; fps: number; quality: number
      letterbox?: { ratio: number; color: string; opacity: number }
      subtitles?: { text: string; startTime: number; endTime: number; style: { fontSize: number; fontFamily: string; fontWeight: string; color: string; backgroundColor: string; position: string; italic: boolean } }[]
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
    platform: string
  }
}
