#!/usr/bin/env node
/**
 * perf:dev — one command to run the performance runner against a HEADLESS backend.
 *
 * Starts the Python backend directly (no Electron, no frontend) with the same
 * environment the app uses, generates one auth token and injects it into BOTH the
 * backend and the dashboard (so there is nothing to paste), then launches the
 * dashboard which opens a browser.
 *
 * Why headless: with no Electron/frontend, the backend is the only GPU consumer,
 * so perf measurements aren't contended by the app's own rendering/decoding.
 *
 * Note: headless can't download or pick a model. It reuses whatever the app has
 * already downloaded + activated (via LTX_APP_DATA_DIR). If no model is loaded,
 * open the app once to fetch one, then re-run this.
 *
 *   pnpm perf:dev
 */
import { spawn, spawnSync } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const BACKEND_DIR = path.join(ROOT, 'backend')
const PERF_DIR = path.join(BACKEND_DIR, 'performance_runner')
const APP_FOLDER_NAME = 'LTXDesktop' // must match electron/app-paths.ts
const PORT = process.env.LTX_PORT || '41954'
const TOKEN = crypto.randomBytes(32).toString('base64url')

// Same resolution as electron/app-paths.ts resolveUserDataPath(), so the headless
// backend reuses the app's downloaded models + settings.json.
function appDataDir() {
  if (process.platform === 'win32') {
    const base = process.env.LOCALAPPDATA || path.join(os.homedir(), 'AppData', 'Local')
    return path.join(base, APP_FOLDER_NAME)
  }
  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support', APP_FOLDER_NAME)
  }
  const xdg = process.env.XDG_DATA_HOME || path.join(os.homedir(), '.local', 'share')
  return path.join(xdg, APP_FOLDER_NAME)
}

const HAS_UV = spawnSync('uv', ['--version'], { stdio: 'ignore' }).status === 0
const runPy = (script) => (HAS_UV ? ['uv', ['run', 'python', script]] : ['python', [script]])

function tee(child, tag) {
  const write = (buf) => {
    for (const line of buf.toString().split('\n')) {
      if (line) process.stdout.write(`[${tag}] ${line}\n`)
    }
  }
  child.stdout.on('data', write)
  child.stderr.on('data', write)
}

const appData = appDataDir()
const logsDir = path.join(appData, 'logs')
fs.mkdirSync(logsDir, { recursive: true })
const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
const backendLogPath = path.join(logsDir, `backend_perfdev_${ts}.log`)
const backendLog = fs.createWriteStream(backendLogPath, { flags: 'a' })

console.log('\nperf:dev — headless backend + performance runner')
console.log(`  app data    : ${appData}`)
console.log(`  backend     : http://127.0.0.1:${PORT}`)
console.log(`  backend log : ${backendLogPath}`)
console.log('  token       : auto-injected into backend + dashboard (no paste needed)')
console.log('  Ctrl+C to stop both.\n')

let dashboard = null
let backendReady = false

const [bcmd, bargs] = runPy('ltx2_server.py')
const backend = spawn(bcmd, bargs, {
  cwd: BACKEND_DIR,
  env: {
    ...process.env,
    LTX_APP_DATA_DIR: appData,
    LTX_AUTH_TOKEN: TOKEN,
    LTX_DEV_MODE: '1',
    LTX_PORT: PORT,
    PYTHONUNBUFFERED: '1',
    // Mirror the flag Electron sets when it spawns the backend (electron/python-backend.ts):
    // on Apple Silicon, let unsupported MPS ops fall back to CPU instead of erroring, so the
    // headless generation path matches the app's. (The packaged app additionally loads a
    // prebuilt mps-sdpa extension via LTX_MPS_EXT_PREBUILT_DIR; a dev run has no prebuilt .so
    // and torch JIT-builds it, so that key is intentionally omitted here.)
    PYTORCH_ENABLE_MPS_FALLBACK: '1',
  },
})

// Backend output -> its own log file (viewable via the dashboard's "backend" log
// toggle). Mirror to the console only during startup, so once it's up the console
// stays clean instead of interleaving backend + dashboard chatter.
function teeBackend(buf) {
  const text = buf.toString()
  backendLog.write(text)
  if (!backendReady) process.stdout.write(text)
  if (!backendReady && text.includes('Server running on')) {
    backendReady = true
    console.log('\n[perf:dev] backend up — its logs now go to the file above')
    console.log("[perf:dev] (view them live via the 'backend' toggle in the dashboard).\n")
    setTimeout(startDashboard, 500)
  }
}
backend.stdout.on('data', teeBackend)
backend.stderr.on('data', teeBackend)

async function startDashboard() {
  const auth = { headers: { Authorization: `Bearer ${TOKEN}` } }

  // Local-generation preflight. The backend decides local-vs-API at startup from the
  // hardware available AT THAT MOMENT — on Apple Silicon, the FREE (not total) unified
  // RAM, since the GPU shares system memory. The perf runner measures LOCAL generation,
  // so if the backend came up in API-only mode there's nothing for it to measure. Tell
  // the user how to fix it and stop, rather than opening a dashboard that can't run.
  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/api/runtime-policy`, auth)
    const policy = await r.json()
    if (policy && policy.force_api_generations) {
      console.log('\n[perf:dev] Local generation is UNAVAILABLE — the backend started in API-only mode.')
      if (process.platform === 'darwin') {
        console.log('[perf:dev] On Apple Silicon this is gated on FREE RAM at the moment the server starts')
        console.log('[perf:dev] (roughly 15 GB+ needed). Close memory-heavy apps — browsers are usually the')
        console.log('[perf:dev] biggest offender — then re-run `pnpm perf:dev` to re-check.')
      } else {
        console.log('[perf:dev] This machine has no supported local-generation GPU (needs a CUDA GPU with')
        console.log('[perf:dev] >=16 GB VRAM, or Apple Silicon with enough free RAM).')
      }
      console.log('[perf:dev] The performance runner measures local generation, so stopping here.\n')
      shutdown()
      return
    }
  } catch { /* policy check is best-effort — fall through to the dashboard */ }

  try {
    const r = await fetch(`http://127.0.0.1:${PORT}/health`, auth)
    const h = await r.json()
    if (h && h.models_loaded === false) {
      console.log('\n[perf:dev] NOTE: no model loaded. Open the app once to download/activate a')
      console.log('           model, then re-run `pnpm perf:dev`.\n')
    }
  } catch { /* health check is best-effort */ }

  const [dcmd, dargs] = runPy('run_dashboard.py')
  dashboard = spawn(dcmd, dargs, {
    cwd: PERF_DIR,
    env: { ...process.env, PERF_AUTH_TOKEN: TOKEN, LTX_PORT: PORT },
  })
  tee(dashboard, 'dashboard')
}

function shutdown() {
  for (const child of [dashboard, backend]) {
    try { child?.kill() } catch { /* already gone */ }
  }
  process.exit(0)
}
process.on('SIGINT', shutdown)
process.on('SIGTERM', shutdown)
backend.on('exit', (code) => { console.log(`[backend] exited (code ${code})`); shutdown() })
