import { z } from 'zod'

// Private preload-to-main channel. It is intentionally not part of
// electronAPISchemas, so renderer code cannot approve arbitrary path strings.
export const SELECTED_FILE_PATH_APPROVAL_CHANNEL = 'approve-selected-file-path'

const fileFilter = z.object({ name: z.string(), extensions: z.array(z.string()) })

function ipcResult<T extends z.ZodRawShape>(valueShape: T) {
  return z.discriminatedUnion('success', [
    z.object({ success: z.literal(true), ...valueShape }),
    z.object({ success: z.literal(false), error: z.string() }),
  ])
}

export type IpcResult<T extends z.ZodRawShape> = z.infer<ReturnType<typeof ipcResult<T>>>

const emptyResult = ipcResult({})

const exportClip = z.object({
  path: z.string(),
  type: z.string(),
  startTime: z.number(),
  duration: z.number(),
  trimStart: z.number(),
  speed: z.number(),
  reversed: z.boolean(),
  flipH: z.boolean(),
  flipV: z.boolean(),
  opacity: z.number(),
  trackIndex: z.number(),
  muted: z.boolean(),
  volume: z.number(),
})

const exportSubtitle = z.object({
  text: z.string(),
  startTime: z.number(),
  endTime: z.number(),
  style: z.object({
    fontSize: z.number(),
    fontFamily: z.string(),
    fontWeight: z.string(),
    color: z.string(),
    backgroundColor: z.string(),
    position: z.string(),
    italic: z.boolean(),
  }),
})

const logsResponse = z.object({
  logPath: z.string(),
  lines: z.array(z.string()),
  error: z.string().optional(),
})

const backendConnectionMode = z.enum(['managed-local', 'external'])

const backendHealthStatus = z.object({
  mode: backendConnectionMode,
  status: z.enum(['connecting', 'alive', 'restarting', 'unreachable', 'dead']),
  exitCode: z.number().nullable().optional(),
  message: z.string().optional(),
})

const externalServerInfo = z.object({
  api_version: z.number(),
  deployment_mode: z.enum(['managed_local', 'standalone']),
  capabilities: z.object({
    media_ids: z.boolean(),
    artifact_downloads: z.boolean(),
    legacy_path_inputs: z.boolean(),
    models_dir_editable: z.boolean(),
  }),
})

const backendConnectionInput = z.discriminatedUnion('mode', [
  z.object({ mode: z.literal('managed-local') }),
  z.object({
    mode: z.literal('external'),
    url: z.string(),
    authToken: z.string(),
  }),
])

const backendMediaRef = z.object({
  media_id: z.string(),
  media_type: z.enum(['image', 'audio', 'video']),
  filename: z.string(),
  content_type: z.string(),
  size_bytes: z.number(),
  sha256: z.string(),
  expires_at: z.string(),
})

const backendArtifactRef = z.object({
  artifact_id: z.string(),
  media_type: z.enum(['image', 'audio', 'video']),
  filename: z.string(),
  content_type: z.string(),
  size_bytes: z.number(),
  sha256: z.string(),
  expires_at: z.string(),
})

export type BackendHealthStatus = z.infer<typeof backendHealthStatus>

export const electronAPISchemas = {
  // App info
  getBackend: {
    input: z.object({}),
    output: z.object({
      url: z.string(),
      token: z.string(),
      mode: backendConnectionMode,
    }),
  },
  getBackendConnectionConfig: {
    input: z.object({}),
    output: z.object({
      mode: backendConnectionMode,
      url: z.string(),
      hasAuthToken: z.boolean(),
    }),
  },
  testBackendConnection: {
    input: z.object({ url: z.string(), authToken: z.string() }),
    output: ipcResult({ serverInfo: externalServerInfo }),
  },
  setBackendConnection: {
    input: backendConnectionInput,
    output: emptyResult,
  },
  startBackendConnection: {
    input: z.object({}),
    output: z.void(),
  },
  retryBackendConnection: {
    input: z.object({}),
    output: z.void(),
  },
  getModelsPath: {
    input: z.object({}),
    output: z.string(),
  },
  readLocalFile: {
    input: z.object({ filePath: z.string() }),
    output: z.object({ data: z.string(), mimeType: z.string() }),
  },
  checkGpu: {
    input: z.object({}),
    output: z.object({ available: z.boolean(), name: z.string().optional(), vram: z.number().optional() }),
  },
  getAppInfo: {
    input: z.object({}),
    output: z.object({ version: z.string(), isPackaged: z.boolean(), modelsPath: z.string(), userDataPath: z.string() }),
  },

  // First-run setup
  checkFirstRun: {
    input: z.object({}),
    output: z.object({ needsSetup: z.boolean(), needsLicense: z.boolean() }),
  },
  acceptLicense: {
    input: z.object({}),
    output: z.boolean(),
  },
  completeSetup: {
    input: z.object({}),
    output: z.boolean(),
  },
  fetchLicenseText: {
    input: z.object({}),
    output: z.string(),
  },
  getNoticesText: {
    input: z.object({}),
    output: z.string(),
  },

  // Open external pages / folders
  openLtxApiKeyPage: {
    input: z.object({}),
    output: z.boolean(),
  },
  openLtxBillingPage: {
    input: z.object({}),
    output: z.boolean(),
  },
  openFalApiKeyPage: {
    input: z.object({}),
    output: z.boolean(),
  },
  openHuggingFaceRepo: {
    input: z.object({ repoId: z.string() }),
    output: z.boolean(),
  },
  openHuggingFaceAuth: {
    input: z.object({
      clientId: z.string(),
      redirectUri: z.string(),
      scope: z.string(),
      state: z.string(),
      codeChallenge: z.string(),
      codeChallengeMethod: z.string(),
    }),
    output: z.boolean(),
  },
  openParentFolderOfFile: {
    input: z.object({ filePath: z.string() }),
    output: z.void(),
  },
  showItemInFolder: {
    input: z.object({ filePath: z.string() }),
    output: z.void(),
  },

  // Logs
  getLogs: {
    input: z.object({}),
    output: logsResponse,
  },
  getLogPath: {
    input: z.object({}),
    output: z.object({ logPath: z.string(), logDir: z.string() }),
  },
  openLogFolder: {
    input: z.object({}),
    output: z.boolean(),
  },

  // Paths
  getResourcePath: {
    input: z.object({}),
    output: z.string().nullable(),
  },
  getDownloadsPath: {
    input: z.object({}),
    output: z.string(),
  },

  // Project assets
  addVisualAssetToProject: {
    input: z.object({ srcPath: z.string(), projectId: z.string(), type: z.enum(['video', 'image']) }),
    output: ipcResult({
      path: z.string(),
      bigThumbnailPath: z.string(),
      smallThumbnailPath: z.string(),
      width: z.number(),
      height: z.number(),
    }),
  },
  addGenericAssetToProject: {
    input: z.object({ srcPath: z.string(), projectId: z.string() }),
    output: ipcResult({ path: z.string() }),
  },
  makeThumbnailsForProjectAsset: {
    input: z.object({ path: z.string(), type: z.enum(['video', 'image']) }),
    output: ipcResult({
      bigThumbnailPath: z.string(),
      smallThumbnailPath: z.string(),
    }),
  },
  makeDimensionsForProjectAsset: {
    input: z.object({ path: z.string(), type: z.enum(['video', 'image']) }),
    output: ipcResult({
      width: z.number(),
      height: z.number(),
    }),
  },
  getProjectAssetsPath: {
    input: z.object({}),
    output: z.string(),
  },
  openProjectAssetsPathChangeDialog: {
    input: z.object({}),
    output: ipcResult({ path: z.string() }),
  },

  // File dialogs & save
  showSaveDialog: {
    input: z.object({
      title: z.string().optional(),
      defaultPath: z.string().optional(),
      filters: z.array(fileFilter).optional(),
    }),
    output: z.string().nullable(),
  },
  saveFile: {
    input: z.object({ filePath: z.string(), data: z.string(), encoding: z.string().optional() }),
    output: ipcResult({ path: z.string() }),
  },
  saveBinaryFile: {
    input: z.object({ filePath: z.string(), data: z.instanceof(ArrayBuffer) }),
    output: ipcResult({ path: z.string() }),
  },
  showOpenDirectoryDialog: {
    input: z.object({ title: z.string().optional() }),
    output: z.string().nullable(),
  },
  searchDirectoryForFiles: {
    input: z.object({ directory: z.string(), filenames: z.array(z.string()) }),
    output: z.record(z.string(), z.string()),
  },
  checkFilesExist: {
    input: z.object({ filePaths: z.array(z.string()) }),
    output: z.record(z.string(), z.boolean()),
  },
  showOpenFileDialog: {
    input: z.object({
      title: z.string().optional(),
      filters: z.array(fileFilter).optional(),
      properties: z.array(z.string()).optional(),
    }),
    output: z.array(z.string()).nullable(),
  },

  // Video export
  exportNative: {
    input: z.object({
      clips: z.array(exportClip),
      outputPath: z.string(),
      codec: z.string(),
      width: z.number(),
      height: z.number(),
      fps: z.number(),
      quality: z.number(),
      letterbox: z.object({ ratio: z.number(), color: z.string(), opacity: z.number() }).optional(),
      subtitles: z.array(exportSubtitle).optional(),
    }),
    output: emptyResult,
  },
  exportCancel: {
    input: z.object({ sessionId: z.string() }),
    output: emptyResult,
  },

  // Python setup
  checkPythonReady: {
    input: z.object({}),
    output: z.object({ ready: z.boolean() }),
  },
  startPythonSetup: {
    input: z.object({}),
    output: z.void(),
  },
  startPythonBackend: {
    input: z.object({}),
    output: z.void(),
  },
  getBackendHealthStatus: {
    input: z.object({}),
    output: backendHealthStatus.nullable(),
  },

  // Remote backend media transfer
  uploadBackendMedia: {
    input: z.object({
      filePath: z.string(),
      mediaType: z.enum(['image', 'audio', 'video']),
    }),
    output: ipcResult({ media: backendMediaRef }),
  },
  materializeBackendArtifact: {
    input: z.object({ artifact: backendArtifactRef }),
    output: ipcResult({ path: z.string() }),
  },

  // Video processing
  extractVideoFrame: {
    input: z.object({ videoPath: z.string(), seekTime: z.number(), width: z.number().optional(), quality: z.number().optional() }),
    output: z.object({ path: z.string() }),
  },

  // Logging
  writeLog: {
    input: z.object({ level: z.string(), message: z.string() }),
    output: z.void(),
  },

  // Models
  openModelsDirChangeDialog: {
    input: z.object({}),
    output: ipcResult({ path: z.string() }),
  },

  // Analytics
  getAnalyticsState: {
    input: z.object({}),
    output: z.object({ analyticsEnabled: z.boolean(), installationId: z.string() }),
  },
  setAnalyticsEnabled: {
    input: z.object({ enabled: z.boolean() }),
    output: z.void(),
  },
  sendAnalyticsEvent: {
    input: z.object({ eventName: z.string(), extraDetails: z.record(z.string(), z.unknown()).nullable().optional() }),
    output: z.void(),
  },
} as const

type Schemas = typeof electronAPISchemas

type InvokeAPI = {
  [K in keyof Schemas]: z.infer<Schemas[K]['input']> extends Record<string, never>
    ? () => Promise<z.infer<Schemas[K]['output']>>
    : (input: z.infer<Schemas[K]['input']>) => Promise<z.infer<Schemas[K]['output']>>
}

export type ElectronAPI = InvokeAPI & {
  onPythonSetupProgress: (cb: (data: unknown) => void) => void
  removePythonSetupProgress: () => void
  onBackendHealthStatus: (cb: (data: BackendHealthStatus) => void) => (() => void)
  getPathForFile: (file: File) => string
  platform: string
  hfGatingEnabled: boolean
}
