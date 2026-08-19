import { useCallback, useEffect, useState } from 'react'
import type { UpdateStatePayload } from '../../shared/electron-api-schema'
import { useIsGenerationActive } from '../lib/generation-active'

export type AppUpdate = {
  state: UpdateStatePayload
  checkForUpdates: () => Promise<void>
  startDownload: () => Promise<void>
  installAndRestart: () => Promise<{ success: true } | { success: false; error: string }>
  skipVersion: (version: string) => Promise<void>
}

const INITIAL: UpdateStatePayload = { status: 'idle', currentVersion: '' }

const MODAL_STATUSES: ReadonlySet<UpdateStatePayload['status']> = new Set([
  'available',
  'downloading',
  'downloaded',
])

export function useAppUpdate(): AppUpdate {
  const [state, setState] = useState<UpdateStatePayload>(INITIAL)

  useEffect(() => {
    let alive = true
    let fromEvent = false
    // Subscribe first so a check that starts during getUpdateState cannot be missed,
    // then ignore the snapshot if an event already applied a newer value.
    const unsubscribe = window.electronAPI.onUpdateEvent((data) => {
      fromEvent = true
      if (alive) setState(data)
    })
    void window.electronAPI.getUpdateState()
      .then((s) => { if (alive && !fromEvent) setState(s) })
      .catch(() => {})
    return () => { alive = false; unsubscribe() }
  }, [])

  const checkForUpdates = useCallback(async () => {
    await window.electronAPI.checkForUpdatesNow()
  }, [])
  const startDownload = useCallback(async () => {
    await window.electronAPI.startUpdateDownload()
  }, [])
  const installAndRestart = useCallback(async () => {
    return window.electronAPI.installUpdateAndRestart()
  }, [])
  const skipVersion = useCallback(async (version: string) => {
    await window.electronAPI.skipUpdateVersion({ version })
  }, [])

  return { state, checkForUpdates, startDownload, installAndRestart, skipVersion }
}

/** Session Later / skip / manual-check intent. App only mounts the modal. */
export function useAppUpdateModal() {
  const update = useAppUpdate()
  const isGenerationActive = useIsGenerationActive()
  const [modalOpen, setModalOpen] = useState(false)
  const [manualCheckPending, setManualCheckPending] = useState(false)
  const [laterVersion, setLaterVersion] = useState<string | null>(null)

  useEffect(() => {
    const s = update.state
    if (s.status === 'available') {
      if (manualCheckPending || s.version !== laterVersion) {
        setModalOpen(true)
      }
      if (manualCheckPending) setManualCheckPending(false)
    } else if (s.status === 'downloaded') {
      // Bits are on disk: the modal is the only install path and must stay up,
      // including after Hide-during-download.
      setModalOpen(true)
    } else if (s.status === 'not-available') {
      if (manualCheckPending) setManualCheckPending(false)
    }
  }, [update.state.status, update.state.version, manualCheckPending, laterVersion])

  const requestCheck = update.checkForUpdates
  const skipVersion = update.skipVersion
  const version = update.state.version
  const status = update.state.status

  const checkForUpdates = useCallback(() => {
    setManualCheckPending(true)
    void requestCheck()
  }, [requestCheck])

  const closeModal = useCallback((skipThisVersion: boolean) => {
    if (status === 'downloaded') return
    if (skipThisVersion && version) void skipVersion(version)
    // Hide during download is not Later — keep the session prompt so a failed
    // download can reopen the modal. Later/skip only apply when dismissing the offer.
    else if (version && status !== 'downloading') setLaterVersion(version)
    setModalOpen(false)
  }, [skipVersion, version, status])

  const openModal = useCallback(() => setModalOpen(true), [])

  return {
    update,
    isGenerationActive,
    // Keep the modal mounted across a periodic re-check (`checking`) so it does not
    // unmount/remount. Do not treat `available` as busy in main — that would hide a
    // newer version after the user clicked Later.
    isModalOpen: modalOpen && (MODAL_STATUSES.has(status) || status === 'checking'),
    openModal,
    closeModal,
    checkForUpdates,
  }
}
