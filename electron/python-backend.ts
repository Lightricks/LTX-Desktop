import { ChildProcess, spawn } from 'child_process'
import crypto from 'crypto'
import fs from 'fs'
import path from 'path'
import { sendAnalyticsEvent } from './analytics'
import { startAnalyticsSink, type AnalyticsSink } from './analytics-sink'
import { getAppDataDir } from './app-paths'
import { getAssetsDir, isCustomAssetsDir } from './app-state'
import { ensureAssetsDirReady } from './assets-startup'
import { getCurrentDir, isDev } from './config'
import { clearHeartbeatFile, decideThresholdTick, readHeartbeatState } from './liveness-grace'
import { logger, writeLog } from './logger'
import { getCurrentLogFilename } from './logging-management'
import { getPythonDir } from './python-setup'
import { getMainWindow } from './window'
import { PY_REMOVE_CWD_FROM_DLL_SEARCH } from './win-dll-search'

let pythonProcess: ChildProcess | null = null
let analyticsSink: AnalyticsSink | null = null
let isIntentionalShutdown = false
let lastCrashTime = 0
const CRASH_DEBOUNCE_MS = 10_000
let startPromise: Promise<void> | null = null
let takeoverInFlight: Promise<void> | null = null
let startEpoch = 0

function closeAnalyticsSink(): void {
  const sink = analyticsSink
  analyticsSink = null
  if (sink) void sink.close()
}

// HTTP liveness monitoring: once the backend has answered /health after
// startup, poll it periodically. On sustained failure, SIGTERM the process so
// the exit handler runs the normal restart/dead flow.
const STARTUP_PROBE_TIMEOUT_MS = 30_000
const STARTUP_PROBE_INTERVAL_MS = 500
const LIVENESS_POLL_INTERVAL_MS = 10_000
const LIVENESS_FAILURE_THRESHOLD = 5
// Kill-grace from the backend's generation proof-of-life file: a GIL-saturated
// backend can't serve /health but is still progressing, so a fresh heartbeat
// (refreshed every 15s while any local generation executes) skips the kill.
// 3 minutes covers the worst observed stall (~70s of MPS model build) with
// margin. The grace itself is capped: a backend deadlocked with a live writer
// thread still fails closed to the kill, just late.
const LIVENESS_GRACE_MAX_HEARTBEAT_AGE_MS = 180_000
const LIVENESS_GRACE_CAP_MS = 10 * 60_000
let livenessMonitorTimer: NodeJS.Timeout | null = null
let livenessFailureCount = 0
let livenessGraceSince: number | null = null

// A local generation can hold the Python process's GIL for long, uninterrupted stretches (MPS
// PyTorch ops release it far less eagerly than CUDA), which starves the asyncio event loop that
// would otherwise accept and dispatch the /health request — the backend isn't hung, it's just
// busy, but the liveness probe can't tell the difference and would otherwise kill it mid-
// generation. The renderer tells us when one is in flight so we can suspend the kill-on-failure
// behavior for its duration. Bounded by MAX_SUPPRESSION_MS so a renderer crash/reload that never
// clears the flag can't permanently disable the safety net for a genuinely hung backend.
const MAX_SUPPRESSION_MS = 20 * 60_000
let generationActiveSince: number | null = null
// Ref-counted: withGenerationActive scopes can overlap (e.g. a second click that 409s fast
// while the first generation is still running) — a plain boolean would let the loser's exit
// clear suppression mid-generation. Depth also means only the 0->1 transition stamps
// generationActiveSince, so an overlapping/looping notification can't keep resetting the
// MAX_SUPPRESSION_MS clock the comment above promises.
let activeGenerationCount = 0

export function setGenerationActive(active: boolean): void {
  if (active) {
    activeGenerationCount += 1
    if (generationActiveSince == null) generationActiveSince = Date.now()
    livenessFailureCount = 0
    return
  }
  activeGenerationCount = Math.max(0, activeGenerationCount - 1)
  if (activeGenerationCount === 0) generationActiveSince = null
}

export function isGenerationActive(): boolean {
  return activeGenerationCount > 0
}

function isLivenessSuppressed(): boolean {
  return generationActiveSince != null && Date.now() - generationActiveSince < MAX_SUPPRESSION_MS
}

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
  livenessGraceSince = null
}

function startLivenessMonitor(): void {
  stopLivenessMonitor()
  // A crashed backend's leftover file must not grace this instance (the pid
  // check would reject it anyway; this is hygiene for pid reuse).
  clearHeartbeatFile(getAppDataDir())
  livenessMonitorTimer = setInterval(() => {
    void (async () => {
      if (!pythonProcess || backendOwnership !== 'managed' || isIntentionalShutdown) {
        return
      }
      if (isLivenessSuppressed()) {
        // Suppression means the renderer attests liveness, so the grace cap
        // clock must not accrue across it — otherwise a long suppressed run
        // could exhaust the cap before the first grace check ever runs.
        livenessGraceSince = null
        return
      }
      const healthy = await probeBackendHealth(2000)
      if (healthy) {
        livenessFailureCount = 0
        livenessGraceSince = null
        return
      }
      livenessFailureCount += 1
      logger.warn(`Backend liveness probe failed (${livenessFailureCount}/${LIVENESS_FAILURE_THRESHOLD})`)
      if (livenessFailureCount >= LIVENESS_FAILURE_THRESHOLD) {
        const outcome = decideThresholdTick({
          graceSinceMs: livenessGraceSince,
          heartbeat: readHeartbeatState(getAppDataDir()),
          backendPid: pythonProcess?.pid,
          nowMs: Date.now(),
          graceMaxHeartbeatAgeMs: LIVENESS_GRACE_MAX_HEARTBEAT_AGE_MS,
          graceCapMs: LIVENESS_GRACE_CAP_MS,
        })
        if (outcome.action === 'grace') {
          livenessGraceSince = outcome.graceSinceMs
          logger.warn('Backend liveness probe failing but generation heartbeat is fresh — giving it more time')
          livenessFailureCount = 0
          return
        }
        if (outcome.reason === 'grace-cap-exceeded') {
          logger.error('Backend liveness probe failing past grace cap despite fresh heartbeat — killing process to trigger restart')
        } else {
          logger.error('Backend liveness probe failed repeatedly — killing process to trigger restart')
        }
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

// While an assets move runs, the backend must stay stopped. A renderer reload calls
// `startPythonBackend`, and that start would use the old folder while files are copied.
let startHold: Promise<void> | null = null

/** Makes `startPythonBackend` wait until the returned function is called. */
export function holdBackendStart(): () => void {
  let release = () => {}
  startHold = new Promise<void>((resolve) => {
    release = () => {
      startHold = null
      resolve()
    }
  })
  return release
}

/**
 * `askUser: false` is for the assets move. A folder that cannot be used then fails the start, so the
 * move can roll back by itself. The recovery dialog could change the setting behind the move.
 */
export async function startPythonBackend({ askUser = true }: { askUser?: boolean } = {}): Promise<void> {
  while (startHold) await startHold
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
  const epoch = startEpoch

  // The check may show a native dialog, so it runs inside the start promise: a second
  // caller waits on `startPromise` and does not spawn a second backend.
  const openSink: Promise<AnalyticsSink | null> = ensureAssetsDirReady({ askUser }).then(() =>
    analyticsSink
      ? analyticsSink
      : startAnalyticsSink(sendAnalyticsEvent).catch((error: unknown) => {
          const message = error instanceof Error ? error.message : String(error)
          logger.warn(`Analytics sink failed to start; continuing without it: ${message}`)
          return null
        }),
  )

  startPromise = openSink.then((sink) => new Promise<void>((resolve, reject) => {
    if (epoch !== startEpoch) {
      if (sink && sink !== analyticsSink) void sink.close()
      reject(new Error('Python backend stopped during startup'))
      return
    }
    if (sink) {
      analyticsSink = sink
    }
    const pythonPath = getPythonPath()
    const backendPath = getBackendPath()
    const mainPy = path.join(backendPath, 'ltx2_server.py')

    logger.info(`Starting Python backend: ${pythonPath} ${mainPy}`)

    // Windows embedded Python's ._pth file suppresses normal sys.path setup —
    // the script's directory isn't added, so sibling packages (e.g. state/)
    // can't be found. Use a -c wrapper to fix sys.path before running the server.
    let pythonArgs: string[]
    if (!isDev && process.platform === 'win32') {
      const preamble = `${PY_REMOVE_CWD_FROM_DLL_SEARCH}import sys; sys.path.insert(0, r"${backendPath}"); import runpy; runpy.run_path(r"${mainPy}", run_name="__main__")`
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
          // Skip mps-sdpa's import-time microbench: it can cache fused_min_bytes=None
          // ("always stock"), which routes video-length attention through stock MPS
          // SDPA and OOMs (https://github.com/Lightricks/LTX-Desktop/issues/161).
          // Honor an explicit parent value (matches setdefault in ltx2_server.py).
          MPS_SDPA_SKIP_CALIBRATION: process.env.MPS_SDPA_SKIP_CALIBRATION ?? '1',
        } : {}),
        // Only pass LTX_PORT when the developer explicitly set it
        ...(process.env.LTX_PORT ? { LTX_PORT: process.env.LTX_PORT } : {}),
        ...(process.env.LTX_REMOTE_PORT ? { LTX_REMOTE_PORT: process.env.LTX_REMOTE_PORT } : {}),
        ...(process.env.LTX_REMOTE_CLIENT_DIR ? { LTX_REMOTE_CLIENT_DIR: process.env.LTX_REMOTE_CLIENT_DIR } : {}),
        ...(!isDev ? {
          LTX_REMOTE_CLIENT_DIR: process.env.LTX_REMOTE_CLIENT_DIR
            ?? path.join(process.resourcesPath, 'dist-remote'),
        } : {}),
        LTX_AUTH_TOKEN: authToken,
        LTX_ADMIN_TOKEN: adminToken,
        LTX_LOG_FILE: getCurrentLogFilename(),
        LTX_APP_DATA_DIR: getAppDataDir(),
        // Always set: an empty value stops a LTX_ASSETS_DIR from the parent environment.
        LTX_ASSETS_DIR: isCustomAssetsDir() ? getAssetsDir() : '',
        LTX_DEV_MODE: isDev ? '1' : '0',
        LTX_ANALYTICS_SINK_URL: sink?.url ?? '',
        LTX_ANALYTICS_TOKEN: sink?.token ?? '',
        // Explore UI is gated on local_viable; designers on low-RAM Macs still need
        // to iterate in pnpm dev without closing every other app first.
        // Honor an explicit parent value: LTX_DEV_FORCE_LOCAL_VIABLE=0 turns the
        // override off, so a developer can see the real "unsupported" state.
        ...(isDev ? { LTX_DEV_FORCE_LOCAL_VIABLE: process.env.LTX_DEV_FORCE_LOCAL_VIABLE ?? '1' } : {}),
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

    pythonProcess.on('exit', async (code, signal) => {
      logger.info(`Python backend exited with code ${code} signal ${signal}`)
      stopLivenessMonitor()
      pythonProcess = null
      backendUrl = null
      authToken = null
      adminToken = null

      if (!started) {
        if (isIntentionalShutdown) {
          isIntentionalShutdown = false
          backendOwnership = null
          closeAnalyticsSink()
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
        closeAnalyticsSink()
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
  }))

  try {
    await startPromise
  } finally {
    startPromise = null
  }
}

/** True when the backend was started outside this app (dev). This app cannot restart it. */
export function isBackendAdopted(): boolean {
  return backendOwnership === 'adopted'
}

/**
 * Stops the backend and waits until the process exits. A new start must not overlap the old
 * exit. It throws when the process is still alive after the timeout.
 */
export async function stopPythonBackendAndWait(timeoutMs = 15000): Promise<void> {
  const proc = pythonProcess
  // A start that is still running ends with "stopped during startup". Wait for that end, so
  // the next start does not join the old attempt.
  const pendingStart = startPromise
  stopPythonBackend()
  if (proc && proc.exitCode === null && proc.signalCode === null) {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('timeout')), timeoutMs)
      proc.once('exit', () => {
        clearTimeout(timer)
        resolve()
      })
    })
  }
  await pendingStart?.catch(() => undefined)
}

export function stopPythonBackend(): void {
  startEpoch += 1

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

  closeAnalyticsSink()

  if (backendOwnership === 'adopted') {
    backendOwnership = null
    latestBackendHealthStatus = null
  }
}
