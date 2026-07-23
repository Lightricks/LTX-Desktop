import { AlertCircle, Check, Download, Folder, HardDrive, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiClient, type ApiSuccessOf } from '../../lib/api-client'
import { formatBytes } from '../../lib/format'
import { logger } from '../../lib/logger'
import { Button } from '../ui/button'

type LtxModelVersionItem = ApiSuccessOf<'getLtxVersions'>['versions'][number]

const DOWNLOAD_POLL_INTERVAL_MS = 1000

// One row in the base-model version list. Owns its own download polling so a
// download on one version doesn't block interaction with the others.
function VersionRow({
  version,
  onChanged,
  resumeSessionId,
}: {
  version: LtxModelVersionItem
  onChanged: () => Promise<void>
  // Set when a download for this version is already running in the backend (e.g. the modal
  // was reopened mid-download); the row reattaches to it instead of showing "Download".
  resumeSessionId: string | null
}) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadPercent, setDownloadPercent] = useState(0)
  const mountedRef = useRef(true)
  useEffect(() => () => { mountedRef.current = false }, [])

  // Reattach to an already-running download. The ref guard prevents re-adopting the same
  // session after it completes (when downloadSessionId is cleared but the prop lingers a tick).
  const adoptedRef = useRef<string | null>(null)
  useEffect(() => {
    if (resumeSessionId && resumeSessionId !== adoptedRef.current && !downloadSessionId) {
      adoptedRef.current = resumeSessionId
      setDownloadSessionId(resumeSessionId)
      setBusy(true)
    }
  }, [resumeSessionId, downloadSessionId])

  // Poll an in-flight download until it completes or errors, then refetch.
  useEffect(() => {
    if (!downloadSessionId) return
    let cancelled = false

    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (cancelled) return
      if (!result.ok) {
        logger.error(`Progress poll error: ${result.error.message}`)
        return
      }
      const progress = result.data
      if (progress.status === 'downloading') {
        setDownloadPercent(Math.round(progress.total_progress))
        return
      }
      if (progress.status === 'error') {
        setDownloadSessionId(null)
        setBusy(false)
        setError(progress.error || 'Download failed.')
        return
      }
      if (progress.status === 'complete') {
        setDownloadSessionId(null)
        setDownloadPercent(100)
        await onChanged()
        if (!cancelled) setBusy(false)
      }
    }

    void poll()
    const interval = setInterval(() => void poll(), DOWNLOAD_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [downloadSessionId, onChanged])

  const handleSetActive = useCallback(async () => {
    setError(null)
    setBusy(true)
    const result = await ApiClient.setActiveLtxModel({ model_id: version.model_id })
    if (!result.ok) {
      setBusy(false)
      setError(result.error.message || 'Failed to set active model.')
      return
    }
    await onChanged()
    if (mountedRef.current) setBusy(false)
  }, [version.model_id, onChanged])

  const handleDownload = useCallback(async () => {
    setError(null)
    setBusy(true)
    setDownloadPercent(0)
    const result = await ApiClient.startModelDownload({ type: 'download', cp_ids: version.cps_to_download })
    if (!result.ok) {
      setBusy(false)
      setError(result.error.message || 'Failed to start download.')
      return
    }
    if (result.data.status !== 'started') {
      setBusy(false)
      setError('Unexpected response while starting download.')
      return
    }
    setDownloadSessionId(result.data.sessionId)
  }, [version.cps_to_download])

  const handleDelete = useCallback(async () => {
    setError(null)
    setBusy(true)
    const result = await ApiClient.deleteModels({ cp_ids: [version.model_cp] })
    if (!result.ok) {
      setBusy(false)
      // The backend blocks deleting the active version (409 DELETE_PROTECTED_CHECKPOINT);
      // surface it inline.
      setError(result.error.message || 'Failed to delete model.')
      return
    }
    await onChanged()
    if (mountedRef.current) setBusy(false)
  }, [version.model_cp, onChanged])

  const isDownloading = downloadSessionId !== null

  return (
    <div className="bg-zinc-800/50 rounded-lg p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          {/* Active state shows a disabled checked radio; others can be selected. */}
          <input
            type="radio"
            name="active-ltx-model"
            checked={version.active}
            disabled={!version.installed || version.active || busy}
            onChange={() => void handleSetActive()}
            className="h-4 w-4 accent-blue-500 flex-shrink-0 disabled:cursor-not-allowed"
          />
          <span className="text-sm text-white truncate">{version.label}</span>
          {version.active && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/10 text-green-400 inline-flex items-center gap-1 flex-shrink-0">
              <Check className="h-3 w-3" />
              Active
            </span>
          )}
          {version.is_newest && !version.installed && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 flex-shrink-0">
              New
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {!version.installed && (
            <Button
              size="sm"
              onClick={() => void handleDownload()}
              disabled={busy}
              className="bg-blue-600 hover:bg-blue-500 text-white text-xs"
            >
              <Download className="h-3.5 w-3.5" />
              {isDownloading ? `Downloading… ${downloadPercent}%` : `Download ${formatBytes(version.size_bytes)}`}
            </Button>
          )}
          {version.installed && !version.active && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => void handleDelete()}
              disabled={busy}
              className="border-zinc-700 text-zinc-300 hover:text-red-400 text-xs"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-2 text-xs text-red-400 inline-flex items-center gap-1.5">
          <AlertCircle className="h-3 w-3 flex-shrink-0" />
          {error}
        </div>
      )}
    </div>
  )
}

export function BaseModelSection() {
  const [versions, setVersions] = useState<LtxModelVersionItem[]>([])
  const [modelsDir, setModelsDir] = useState('')
  // A download keeps running in the backend even if this modal is closed. Track the active
  // session so a row can reattach to its progress on remount instead of resetting to "Download".
  const [activeDownload, setActiveDownload] = useState<{ sessionId: string; cpIds: string[] } | null>(null)

  const refreshVersions = useCallback(async () => {
    const [versionsResult, activeResult] = await Promise.all([
      ApiClient.getLtxVersions(),
      ApiClient.getActiveDownload(),
    ])
    if (!versionsResult.ok) {
      logger.error(`Failed to fetch LTX versions: ${versionsResult.error.message}`)
      return
    }
    setVersions(versionsResult.data.versions)
    if (activeResult.ok) {
      setActiveDownload(
        activeResult.data.session_id
          ? { sessionId: activeResult.data.session_id, cpIds: activeResult.data.cp_ids ?? [] }
          : null,
      )
    }
  }, [])

  useEffect(() => {
    void refreshVersions()
    void (async () => {
      const result = await ApiClient.getSettings()
      if (!result.ok) {
        logger.error(`Failed to fetch settings: ${result.error.message}`)
        return
      }
      setModelsDir(result.data.modelsDir ?? '')
    })()
  }, [refreshVersions])

  return (
    <>
      {/* Models Folder */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Folder className="h-4 w-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-white">Models Folder</h3>
        </div>
        <p className="text-xs text-zinc-500 leading-relaxed">
          Where model checkpoints are stored. Changing the location requires restarting the app.
        </p>
        <div className="flex gap-2">
          <div className="flex-1 px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-zinc-300 text-sm truncate select-text">
            {modelsDir || <span className="text-zinc-600">Not set</span>}
          </div>
          <Button
            variant="outline"
            className="border-zinc-700 flex-shrink-0"
            onClick={async () => {
              const result = await window.electronAPI.openModelsDirChangeDialog()
              if (result.success) {
                setModelsDir(result.path)
              }
            }}
          >
            Change…
          </Button>
          <Button
            variant="outline"
            className="border-zinc-700 flex-shrink-0"
            disabled={!modelsDir}
            onClick={() => {
              void window.electronAPI.openModelsFolder()
            }}
          >
            Open folder
          </Button>
        </div>
      </div>

      {/* Base Model Versions */}
      <div className="space-y-3 pt-4 border-t border-zinc-800">
        <div className="flex items-center gap-2">
          <HardDrive className="h-4 w-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-white">Base Model</h3>
        </div>
        <p className="text-xs text-zinc-500 leading-relaxed">
          The active version is used for new generations. Download a version to make it available,
          then set it active.
        </p>
        <div className="space-y-2">
          {versions.length === 0 ? (
            <div className="text-xs text-zinc-600">No versions available.</div>
          ) : (
            versions.map((version) => (
              <VersionRow
                key={version.model_id}
                version={version}
                onChanged={refreshVersions}
                resumeSessionId={
                  activeDownload && activeDownload.cpIds.includes(version.model_cp)
                    ? activeDownload.sessionId
                    : null
                }
              />
            ))
          )}
        </div>
      </div>
    </>
  )
}
