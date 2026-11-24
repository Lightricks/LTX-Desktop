import { useEffect, useMemo, useSyncExternalStore } from 'react'
import type { ExploreApi } from '../ltx-io/runtime/ExploreRuntime'

import { H3_VARIANTS, isH3Variant, type H3Variant } from './h3-models.ts'

export type ManagedModelKind = 'comfy-int8' | 'kandinsky' | H3Variant
type Phase = 'requested' | 'preparing' | 'starting' | 'weights' | 'nodes' | 'restarting' | 'ready' | 'error'
type Setup = { id: string; phase: Phase; message: string; restarted?: boolean }
const kinds: ManagedModelKind[] = ['comfy-int8', 'kandinsky', ...H3_VARIANTS]

export function isManagedModelKind(value: unknown): value is ManagedModelKind {
  return value === 'comfy-int8' || value === 'kandinsky' || isH3Variant(value)
}
const phases: Phase[] = ['requested', 'preparing', 'starting', 'weights', 'nodes', 'restarting', 'ready', 'error']
const changed = 'ltx-managed-model-setup'
const key = (kind: ManagedModelKind) => `ltx.amd.model-setup.v1.${kind}`

function parse(raw: string | null): Setup | null {
  if (!raw) return null
  try {
    const data = JSON.parse(raw)
    return typeof data.id === 'string' && phases.includes(data.phase) && typeof data.message === 'string' ? data : null
  } catch { return null }
}

function store(kind: ManagedModelKind, setup: Setup) {
  window.localStorage.setItem(key(kind), JSON.stringify(setup))
  window.dispatchEvent(new Event(changed))
}

function subscribe(listener: () => void) {
  window.addEventListener(changed, listener)
  window.addEventListener('storage', listener)
  return () => {
    window.removeEventListener(changed, listener)
    window.removeEventListener('storage', listener)
  }
}

export function useManagedModelSetup(kind: ManagedModelKind) {
  const raw = useSyncExternalStore(subscribe, () => window.localStorage.getItem(key(kind)), () => null)
  return useMemo(() => parse(raw), [raw])
}

export function beginManagedModelSetup(kind: ManagedModelKind) {
  store(kind, { id: crypto.randomUUID(), phase: 'requested', message: 'Checking the local video engine…' })
}

class BackendUnavailable extends Error {}

function accepted<T>(response: { ok: true; data: T } | { ok: false; error: { code?: string; message: string } }): T {
  if (!response.ok) {
    if (['NETWORK_ERROR', 'HTTP_502', 'HTTP_503'].includes(response.error.code ?? '')) {
      throw new BackendUnavailable(response.error.message)
    }
    throw new Error(response.error.message)
  }
  return response.data
}

async function advance(api: ExploreApi, kind: ManagedModelKind, setup: Setup) {
  const update = (phase: Phase, message: string, restarted = setup.restarted) => {
    const current = parse(window.localStorage.getItem(key(kind)))
    if (current?.id === setup.id && (current.phase !== phase || current.message !== message || current.restarted !== restarted)) {
      store(kind, { ...setup, phase, message, restarted })
    }
  }
  try {
    const engine = accepted(await api.getComfyEngine())
    if (engine.status === 'unsupported') throw new Error(engine.message)
    if (engine.status === 'installing' || engine.status === 'starting') {
      update(engine.status === 'installing' ? 'preparing' : 'starting', engine.message)
      return
    }
    if (engine.status === 'error' && setup.phase !== 'requested') throw new Error(engine.message)
    if (engine.status === 'not_installed' || engine.status === 'stopped' || engine.status === 'error' || engine.status === 'external') {
      const action = engine.canStart ? 'start' : engine.canPrepare ? 'prepare' : null
      if (!action) throw new Error(engine.message)
      const result = await window.electronAPI.manageComfyEngine({ action })
      if (result.status === 'error' || result.status === 'unsupported') throw new Error(result.message)
      update(action === 'start' ? 'starting' : 'preparing', result.message)
      return
    }

    if (isH3Variant(kind)) {
      const catalog = accepted(await api.getH3Models({ fresh: true }))
      if (!catalog.licenseAcknowledged) throw new Error('Confirm that you have rights to use the MiniMax H3 weights on this PC.')
      const model = catalog.variants[kind]
      if (model.status === 'ready') { update('ready', model.message); return }
      if (model.status === 'offline') { update('starting', engine.message); return }
      if (model.status === 'error' && setup.phase === 'weights') throw new Error(model.message)
      if (model.status === 'downloading') { update('weights', model.message); return }
      const started = accepted(await api.downloadH3Models({ variant: kind, acceptedLicense: true })).variants[kind]
      if (started.status === 'error') throw new Error(started.message)
      update('weights', started.message)
      return
    }

    const k6 = kind === 'kandinsky'
    const models = accepted(await api.getComfyModels({ fresh: true }))
    const status = k6 ? models.kandinskyStatus : models.status
    const message = k6 ? models.kandinskyMessage : models.message
    if (status === 'offline') { update('starting', engine.message); return }
    if (status === 'error' && setup.phase === 'weights') throw new Error(message)
    if (status === 'downloading') { update('weights', message); return }
    const weightsReady = k6 ? models.kandinskyWeightsReady : status === 'ready'
    if (!weightsReady) {
      const result = accepted(k6 ? await api.downloadKandinskyModels() : await api.downloadComfyModels())
      if ((k6 ? result.kandinskyStatus : result.status) === 'error') throw new Error(k6 ? result.kandinskyMessage : result.message)
      update('weights', k6 ? result.kandinskyMessage : result.message)
      return
    }
    if (k6) {
      const node = accepted(await api.getKandinskyNode())
      if (node.status === 'ready' && models.kandinskyReady) { update('ready', 'Kandinsky is ready.'); return }
      if (node.status === 'busy' || node.status === 'installing' || node.status === 'restarting' || node.status === 'offline' || node.status === 'ready') {
        update(node.status === 'restarting' ? 'restarting' : 'nodes', node.message)
        return
      }
      const action = node.canInstall ? 'install' : node.canRestart ? 'restart' : null
      if (!action) throw new Error(node.message)
      // A manual retry must be explicit if the restart finished but the node still cannot load.
      if (action === 'restart' && setup.restarted) throw new Error(node.message)
      const result = await window.electronAPI.manageKandinskyNode({ action, managedOnly: true })
      if (result.status === 'manual' || result.status === 'offline') throw new Error(result.message)
      update(result.status === 'restarting' ? 'restarting' : 'nodes', result.message, setup.restarted || result.status === 'restarting')
      return
    }
    update('ready', 'LTX INT8 is ready.')
  } catch (error) {
    if (error instanceof BackendUnavailable) update(setup.phase, 'Waiting to reconnect to the LTX backend…')
    else update('error', error instanceof Error ? error.message : String(error))
  }
}

// One runner carries explicit model setup requests across panel closure, navigation and
// application restarts. It never provisions a fresh installation until the user requests it.
let running = false

export function useManagedSetupRunner(api: ExploreApi, enabled: boolean) {
  useEffect(() => {
    if (!enabled) return
    let live = true
    let timer: ReturnType<typeof setTimeout>
    const run = async () => {
      clearTimeout(timer)
      if (!live) return
      if (running) { timer = setTimeout(run, 2000); return }
      running = true
      try {
        for (const kind of kinds) {
          const setup = parse(window.localStorage.getItem(key(kind)))
          if (!live) break
          if (setup && setup.phase !== 'ready' && setup.phase !== 'error') await advance(api, kind, setup)
        }
      } finally {
        running = false
        if (live && kinds.some((kind) => {
          const s = parse(window.localStorage.getItem(key(kind)))
          return s && s.phase !== 'error' && s.phase !== 'ready'
        })) timer = setTimeout(run, 2000)
      }
    }
    const onIntent = () => {
      if (!running) void run()
    }
    window.addEventListener(changed, onIntent)
    void run()
    return () => {
      live = false
      clearTimeout(timer)
      window.removeEventListener(changed, onIntent)
    }
  }, [api, enabled])
}
