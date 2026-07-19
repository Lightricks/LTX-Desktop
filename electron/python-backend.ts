import { ChildProcess, spawn } from 'child_process'
import crypto from 'crypto'
import fs from 'fs'
import path from 'path'
import { getAppDataDir } from './app-paths'
import { getCurrentDir, isDev } from './config'
import { logger, writeLog } from './logger'
import { getCurrentLogFilename } from './logging-management'
import { getPythonDir } from './python-setup'
import { getMainWindow } from './window'

let pythonProcess: ChildProcess | null = null
let isIntentionalShutdown = false
let lastCrashTime = 0
const CRASH_DEBOUNCE_MS = 10_000
let startPromise: Promise<void> | null = null
let takeoverInFlight: Promise<void> | null = null

// HTTP liveness monitoring: once the backend has answered /health after
// startup, poll it periodically. On sustained failure, SIGTERM the process so
// the exit handler runs the normal restart/dead flow.
const STARTUP_PROBE_TIMEOUT_MS = 30_000
const STARTUP_PROBE_INTERVAL_MS = 500
const LIVENESS_POLL_INTERVAL_MS = 10_000
const LIVENESS_FAILURE_THRESHOLD = 5
let livenessMonitorTimer: NodeJS.Timeout | null = null
let livenessFailureCount = 0

let backendUrl: string | null = null
let authToken: string | null = null
let adminToken: string | null = null

export function getBackendUrl(): string | null { return backendUrl }
export function getAuthToken(): string | null { return authToken }
export function getAdminToken(): string | null { return adminToken }

type BackendOwnership = 'managed' | 'adopted' | null

let backendOwnership: BackendOwnership = null

export interface BackendHealthStatus {
  status: 'alive' | 'restarting' | 'dead'
  exitCode?: number | null
}

let latestBackendHealthStatus: BackendHealthStatus | null = null

function publishBackendHealthStatus(status: BackendHealthStatus): void {
  latestBackendHealthStatus = status
  getMainWindow()?.webContents.send('backend-health-status', status)
}

export function getBackendHealthStatus(): BackendHealthStatus | null {
  return latestBackendHealthStatus
}

function getBackendPath(): string {
  if (isDev) {
    return path.join(getCurrentDir(), 'backend')
  }
  return path.join(process.resourcesPath, 'backend')
}

function isPortConflictOutput(output: string): boolean {
  const normalizedOutput = output.toLowerCase()
  return (
    normalizedOutput.includes('address already in use') ||
    normalizedOutput.includes('eaddrinuse') ||
    normalizedOutput.includes('errno 48')
  )
}

async function probeBackendHealth(timeoutMs = 1500, probeUrl?: string): Promise<boolean> {
  const url = probeUrl || backendUrl
  if (!url) return false
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const headers: Record<string, string> = {}
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`
    const response = await fetch(`${url}/health`, {
      signal: controller.signal,
      headers,
    })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

async function requestAdoptedBackendShutdown(timeoutMs = 2000): Promise<boolean> {
  if (!backendUrl) return false
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const headers: Record<string, string> = {}
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`
    const response = await fetch(`${backendUrl}/api/system/shutdown`, {
      method: 'POST',
      signal: controller.signal,
      headers,
    })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

async function waitUntilBackendDown(timeoutMs = 8000): Promise<boolean> {
  const startedAt = Date.now()
  while (Date.now() - startedAt < timeoutMs) {
    const healthy = await probeBackendHealth(800)
    if (!healthy) {
      return true
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  return false
}

function stopLivenessMonitor(): void {
  if (livenessMonitorTimer) {
    clearInterval(livenessMonitorTimer)
    livenessMonitorTimer = null
  }
  livenessFailureCount = 0
}

function startLivenessMonitor(): void {
  stopLivenessMonitor()
  livenessMonitorTimer = setInterval(() => {
    void (async () => {
      if (!pythonProcess || backendOwnership !== 'managed' || isIntentionalShutdown) {
        return
      }
      const healthy = await probeBackendHealth(2000)
      if (healthy) {
        livenessFailureCount = 0
        return
      }
      livenessFailureCount += 1
      logger.warn(`Backend liveness probe failed (${livenessFailureCount}/${LIVENESS_FAILURE_THRESHOLD})`)
      if (livenessFailureCount >= LIVENESS_FAILURE_THRESHOLD) {
        logger.error('Backend liveness probe failed repeatedly — killing process to trigger restart')
        stopLivenessMonitor()
        try {
          pythonProcess?.kill('SIGTERM')
        } catch {
          // Process may already be dead; exit handler will run.
        }
      }
    })()
  }, LIVENESS_POLL_INTERVAL_MS)
}

function startOwnershipTakeover(): void {
  if (takeoverInFlight || backendOwnership !== 'adopted') {
    return
  }

  takeoverInFlight = (async () => {
    try {
      const shutdownRequested = await requestAdoptedBackendShutdown()
      if (!shutdownRequested) {
        throw new Error('Failed to request shutdown for adopted backend')
      }

      const backendStopped = await waitUntilBackendDown()
      if (!backendStopped) {
        throw new Error('Timed out waiting for adopted backend shutdown')
      }

      backendOwnership = null
      await startPythonBackend()
    } catch (error) {
      logger.error(`Failed to reclaim backend process ownership: ${error}`)
      backendOwnership = null
      publishBackendHealthStatus({ status: 'dead' })
    } finally {
      takeoverInFlight = null
    }
  })()
}

export function getPythonPath(): string {
  // In production, use bundled/downloaded Python first
  if (!isDev) {
    const pythonDir = getPythonDir()
    const bundledPython = process.platform === 'win32'
      ? path.join(pythonDir, 'python.exe')
      : path.join(pythonDir, 'bin', 'python3')
    if (fs.existsSync(bundledPython)) {
      logger.info(`Using bundled Python: ${bundledPython}`)
      return bundledPython
    }
  }

  // Check for venv in backend directory
  const backendPath = getBackendPath()
  const isWindows = process.platform === 'win32'
  const venvPython = isWindows
    ? path.join(backendPath, '.venv', 'Scripts', 'python.exe')
    : path.join(backendPath, '.venv', 'bin', 'python')

  if (fs.existsSync(venvPython)) {
    logger.info(`Using venv Python: ${venvPython}`)
    return venvPython
  }

  if (isDev) {
    // In development, try common Python paths
    const pythonPaths = isWindows
      ? [
          'python',
          'python3',
          path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python311', 'python.exe'),
          path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python312', 'python.exe'),
        ]
      : [
          'python3',
          'python',
        ]

    for (const p of pythonPaths) {
      try {
        if (fs.existsSync(p)) {
          return p
        }
      } catch {
        continue
      }
    }
    return isWindows ? 'python' : 'python3'
  }

  // Fallback
  return 'python'
}

export async function startPythonBackend(): Promise<void> {
  if (startPromise) {
    return startPromise
  }

  if (pythonProcess && backendOwnership === 'managed') {
    publishBackendHealthStatus({ status: 'alive' })
    return
  }

  if (backendOwnership === 'adopted') {
    const adoptedHealthy = await probeBackendHealth()
    if (adoptedHealthy) {
      publishBackendHealthStatus({ status: 'alive' })
      return
    }
    backendOwnership = null
  }

  isIntentionalShutdown = false

  startPromise = new Promise((resolve, reject) => {
    const pythonPath = getPythonPath()
    const backendPath = getBackendPath()
    const mainPy = path.join(backendPath, 'ltx2_server.py')

    logger.info(`Starting Python backend: ${pythonPath} ${mainPy}`)

    // Windows embedded Python's ._pth file suppresses normal sys.path setup —
    // the script's directory isn't added, so sibling packages (e.g. state/)
    // can't be found. Use a -c wrapper to fix sys.path before running the server.
    let pythonArgs: string[]
    if (!isDev && process.platform === 'win32') {
      const preamble = `import sys; sys.path.insert(0, r"${backendPath}"); import runpy; runpy.run_path(r"${mainPy}", run_name="__main__")`
      pythonArgs = ['-u', '-c', preamble]
    } else {
      pythonArgs = isDev ? ['-Xfrozen_modules=off', '-u', mainPy] : ['-u', mainPy]
    }

    // Generate auth token and admin token for this backend session
    authToken = crypto.randomBytes(32).toString('base64url')
    adminToken = crypto.randomBytes(32).toString('base64url')

    pythonProcess = spawn(pythonPath, pythonArgs, {
      cwd: backendPath,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONNOUSERSITE: '1',
        // Put the interpreter's bin/ on PATH so torch's C++ extension loader can find
        // `ninja` — required to load mps-sdpa's zero-copy `mpsgraph_zc` attention
        // backend on Apple Silicon (even a prebuilt cache needs ninja to load; without
        // it, mps-sdpa falls back to a Metal-memory-leaking backend). macOS-only so it
        // can't shadow PATH entries for backend subprocesses on Windows/Linux.
        ...(process.platform === 'darwin' ? {
          PATH: `${path.dirname(pythonPath)}${path.delimiter}${process.env.PATH ?? ''}`,
        } : {}),
        // Only pass LTX_PORT when the developer explicitly set it
        ...(process.env.LTX_PORT ? { LTX_PORT: process.env.LTX_PORT } : {}),
        LTX_AUTH_TOKEN: authToken,
        LTX_ADMIN_TOKEN: adminToken,
        LTX_LOG_FILE: getCurrentLogFilename(),
        LTX_APP_DATA_DIR: getAppDataDir(),
        LTX_DEV_MODE: isDev ? '1' : '0',
        // Bundled prebuilt mps-sdpa zero-copy extension cache (macOS). Lives inside
        // python-embed (→ resources/python) so it rides the CI python-embed cache. The
        // backend direct-imports the .so from here (mps_prebuilt_ext.py), no copy step;
        // ignored if absent (dev, where torch JIT-builds it instead).
        LTX_MPS_EXT_PREBUILT_DIR: path.join(getPythonDir(), 'mps-ext-prebuilt', 'mps_sdpa_zc_ext'),
        PYTORCH_ENABLE_MPS_FALLBACK: '1',
        // Set PYTHONHOME for bundled Python on macOS so it finds its stdlib
        ...(!isDev && process.platform !== 'win32' ? {
          PYTHONHOME: getPythonDir(),
        } : {}),
      },
      stdio: ['ignore', 'pipe', 'pipe']
    })

    let started = false
    let startupSettled = false
    let sawPortConflict = false
    let probeGateStarted = false

    const settleResolve = () => {
      if (startupSettled) return
      startupSettled = true
      resolve()
    }

    const settleReject = (error: Error) => {
      if (startupSettled) return
      startupSettled = true
      reject(error)
    }

    const gateAliveOnProbe = async () => {
      const deadline = Date.now() + STARTUP_PROBE_TIMEOUT_MS
      while (Date.now() < deadline) {
        if (!pythonProcess || isIntentionalShutdown) {
          return
        }
        if (await probeBackendHealth(1500)) {
          started = true
          backendOwnership = 'managed'
          publishBackendHealthStatus({ status: 'alive' })
          settleResolve()
          startLivenessMonitor()
          return
        }
        await new Promise((resolveSleep) => setTimeout(resolveSleep, STARTUP_PROBE_INTERVAL_MS))
      }
      logger.error('Backend HTTP probe never succeeded after ready signal — killing process')
      try {
        pythonProcess?.kill('SIGTERM')
      } catch {
        // Exit handler will run and fail startup with dead.
      }
    }

    const checkStarted = (output: string) => {
      if (isPortConflictOutput(output)) {
        sawPortConflict = true
      }

      if (started || probeGateStarted) return

      const readyMatch = output.match(/Server running on (http:\/\/\S+)/)
      if (readyMatch) {
        backendUrl = readyMatch[1]
        probeGateStarted = true
        void gateAliveOnProbe()
      }
    }

    pythonProcess.stdout?.on('data', (data: Buffer) => {
      const output = data.toString()
      console.log(`[Python] ${output}`)
      for (const line of output.split('\n')) {
        const trimmed = line.trimEnd()
        if (trimmed) writeLog('INFO', 'Backend', trimmed)
      }
      checkStarted(output)
    })

    pythonProcess.stderr?.on('data', (data: Buffer) => {
      const output = data.toString()
      console.error(`[Python Error] ${output}`)
      for (const line of output.split('\n')) {
        const trimmed = line.trimEnd()
        if (trimmed) writeLog('ERROR', 'Backend', trimmed)
      }
      checkStarted(output)
    })

    pythonProcess.on('error', (error) => {
      logger.error(`Failed to start Python backend: ${error}`)
      if (!started) {
        backendOwnership = null
        publishBackendHealthStatus({ status: 'dead' })
        settleReject(error)
      }
    })

    pythonProcess.on('exit', async (code) => {
      logger.info(`Python backend exited with code ${code}`)
      stopLivenessMonitor()
      pythonProcess = null
      backendUrl = null
      authToken = null
      adminToken = null

      if (!started) {
        if (isIntentionalShutdown) {
          isIntentionalShutdown = false
          backendOwnership = null
          settleReject(new Error('Python backend stopped during startup'))
          return
        }

        if (sawPortConflict && process.env.LTX_PORT) {
          const explicitUrl = `http://127.0.0.1:${process.env.LTX_PORT}`
          const healthyExistingBackend = await probeBackendHealth(1500, explicitUrl)
          if (healthyExistingBackend) {
            backendUrl = explicitUrl
            backendOwnership = 'adopted'
            publishBackendHealthStatus({ status: 'alive' })
            settleResolve()
            startOwnershipTakeover()
            return
          }
        }

        backendOwnership = null
        publishBackendHealthStatus({ status: 'dead', exitCode: code })
        settleReject(new Error(`Python backend exited during startup with code ${code}`))
        return
      }

      if (isIntentionalShutdown) {
        isIntentionalShutdown = false
        backendOwnership = null
        return
      }

      backendOwnership = 'managed'
      const now = Date.now()
      if (now - lastCrashTime < CRASH_DEBOUNCE_MS) {
        publishBackendHealthStatus({ status: 'dead', exitCode: code })
        return
      }

      lastCrashTime = now
      publishBackendHealthStatus({ status: 'restarting', exitCode: code })
      try {
        await startPythonBackend()
      } catch {
        publishBackendHealthStatus({ status: 'dead', exitCode: code })
      }
    })

    // Timeout after 5 minutes (model loading can take a while on first run)
    setTimeout(() => {
      if (startupSettled || started) {
        return
      }

      try {
        pythonProcess?.kill('SIGTERM')
      } catch {
        // Process may already be dead.
      }
      backendOwnership = null
      publishBackendHealthStatus({ status: 'dead' })
      settleReject(new Error('Python backend failed to start within 5 minutes'))
    }, 300000)
  })

  try {
    await startPromise
  } finally {
    startPromise = null
  }
}

export function stopPythonBackend(): void {
  if (pythonProcess) {
    isIntentionalShutdown = true
    stopLivenessMonitor()
    logger.info('Stopping Python backend...')
    const pid = pythonProcess.pid
    pythonProcess.kill('SIGTERM')
    pythonProcess = null
    // Force kill after 5 seconds if SIGTERM didn't work (PyTorch/uvicorn threads)
    if (pid) {
      setTimeout(() => {
        try {
          process.kill(pid, 0) // Check if still alive (throws if dead)
          process.kill(pid, 'SIGKILL')
        } catch {
          // Already dead
        }
      }, 5000)
    }
    return
  }

  if (backendOwnership === 'adopted') {
    backendOwnership = null
    latestBackendHealthStatus = null
  }
}
