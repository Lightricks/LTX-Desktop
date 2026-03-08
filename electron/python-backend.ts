import { ChildProcess, spawn } from 'child_process'
import fs from 'fs'
import net from 'net'
import path from 'path'
import { getAppDataDir } from './app-paths'
import { getBackendBaseUrl, getCurrentDir, getPythonPort, isDev, setPythonPort } from './config'
import { logger, writeLog } from './logger'
import { getPythonDir } from './python-setup'
import { getMainWindow } from './window'

let pythonProcess: ChildProcess | null = null
let isIntentionalShutdown = false
let lastCrashTime = 0
const CRASH_DEBOUNCE_MS = 10_000
let startPromise: Promise<void> | null = null
let takeoverInFlight: Promise<void> | null = null

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
    normalizedOutput.includes('errno 48') ||
    normalizedOutput.includes('winerror 10013') ||
    normalizedOutput.includes('errno 10013') ||
    normalizedOutput.includes('errno 10048')
  )
}

async function canBindToPort(port: number): Promise<boolean> {
  return await new Promise((resolve) => {
    const server = net.createServer()

    server.once('error', () => {
      resolve(false)
    })

    server.once('listening', () => {
      server.close(() => resolve(true))
    })

    server.listen(port, '127.0.0.1')
  })
}

async function selectAvailableBackendPort(startPort: number, maxOffsets = 25): Promise<number> {
  for (let offset = 0; offset <= maxOffsets; offset += 1) {
    const candidate = startPort + offset
    if (candidate > 65535) {
      break
    }
    if (await canBindToPort(candidate)) {
      return candidate
    }
  }

  return startPort
}

async function probeBackendHealth(timeoutMs = 1500): Promise<boolean> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${getBackendBaseUrl()}/health`, {
      signal: controller.signal,
    })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

async function requestAdoptedBackendShutdown(timeoutMs = 2000): Promise<boolean> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${getBackendBaseUrl()}/api/system/shutdown`, {
      method: 'POST',
      signal: controller.signal,
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

  const preferredPort = getPythonPort()
  const selectedPort = await selectAvailableBackendPort(preferredPort)
  if (selectedPort !== preferredPort) {
    logger.warn(`Backend port ${preferredPort} unavailable, switching to ${selectedPort}`)
    setPythonPort(selectedPort)
  }
  logger.info(`Backend base URL: ${getBackendBaseUrl()}`)

  startPromise = new Promise((resolve, reject) => {
    const pythonPath = getPythonPath()
    const backendPath = getBackendPath()
    const mainPy = path.join(backendPath, 'ltx2_server.py')

    logger.info(`Starting Python backend: ${pythonPath} ${mainPy} (port=${getPythonPort()})`)

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

    pythonProcess = spawn(pythonPath, pythonArgs, {
      cwd: backendPath,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        PYTHONNOUSERSITE: '1',
        LTX_PORT: String(getPythonPort()),
        LTX_APP_DATA_DIR: getAppDataDir(),
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
    const recentStartupOutput: string[] = []

    const pushStartupOutput = (stream: 'stdout' | 'stderr', line: string) => {
      const trimmed = line.trim()
      if (!trimmed) {
        return
      }
      recentStartupOutput.push(`[${stream}] ${trimmed}`)
      if (recentStartupOutput.length > 60) {
        recentStartupOutput.shift()
      }
    }

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

    const checkStarted = (output: string) => {
      if (isPortConflictOutput(output)) {
        sawPortConflict = true
      }

      // Check if server has started
      if (!started && (output.includes('Server running on') || output.includes('Uvicorn running'))) {
        started = true
        backendOwnership = 'managed'
        publishBackendHealthStatus({ status: 'alive' })
        settleResolve()
      }
    }

    pythonProcess.stdout?.on('data', (data: Buffer) => {
      const output = data.toString()
      console.log(`[Python] ${output}`)
      for (const line of output.split('\n')) {
        const trimmed = line.trimEnd()
        if (trimmed) writeLog('INFO', 'Backend', trimmed)
        pushStartupOutput('stdout', line)
      }
      checkStarted(output)
    })

    pythonProcess.stderr?.on('data', (data: Buffer) => {
      const output = data.toString()
      console.error(`[Python Error] ${output}`)
      for (const line of output.split('\n')) {
        const trimmed = line.trimEnd()
        if (trimmed) writeLog('ERROR', 'Backend', trimmed)
        pushStartupOutput('stderr', line)
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
      pythonProcess = null

      if (!started) {
        if (isIntentionalShutdown) {
          isIntentionalShutdown = false
          backendOwnership = null
          settleReject(new Error('Python backend stopped during startup'))
          return
        }

        if (sawPortConflict) {
          const healthyExistingBackend = await probeBackendHealth()
          if (healthyExistingBackend) {
            backendOwnership = 'adopted'
            publishBackendHealthStatus({ status: 'alive' })
            settleResolve()
            startOwnershipTakeover()
            return
          }
        }

        backendOwnership = null
        publishBackendHealthStatus({ status: 'dead', exitCode: code })
        const recentOutput = recentStartupOutput.slice(-12).join(' | ')
        const detail = recentOutput ? `; recent output: ${recentOutput}` : ''
        settleReject(new Error(`Python backend exited during startup with code ${code}${detail}`))
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
