import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertCircle, Download, RefreshCw, X } from 'lucide-react'
import { Button } from './ui/button'
import type { AppUpdate } from '../hooks/use-app-update'
import './UpdateAvailableModal.css'

interface Props {
  update: AppUpdate
  isGenerationActive: boolean
  // onClose receives whether the user ticked "Skip this version".
  onClose: (skipThisVersion: boolean) => void
}

function titleForStatus(status: AppUpdate['state']['status']): string {
  if (status === 'downloaded') return 'Ready to install'
  if (status === 'downloading') return 'Downloading update'
  if (status === 'checking') return 'Checking for updates'
  return 'Update available'
}

export function UpdateAvailableModal({ update, isGenerationActive, onClose }: Props) {
  const { state, startDownload, installAndRestart } = update
  const [skipChecked, setSkipChecked] = useState(false)
  const [installError, setInstallError] = useState<string | null>(null)
  const dialogRef = useRef<HTMLDivElement>(null)

  const downloading = state.status === 'downloading'
  const downloaded = state.status === 'downloaded'
  const canDownload = state.status === 'available'
  const canDismiss = !downloaded

  const handleClose = useCallback(() => {
    if (!canDismiss) return
    onClose(skipChecked && state.status === 'available')
  }, [canDismiss, onClose, skipChecked, state.status])

  useEffect(() => {
    dialogRef.current?.focus()
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && canDismiss) handleClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [canDismiss, handleClose])

  const handleInstall = async () => {
    try {
      const res = await installAndRestart()
      if (!res.success) setInstallError(res.error ?? 'Could not install the update.')
    } catch (e) {
      setInstallError(e instanceof Error ? e.message : 'Could not install the update.')
    }
  }

  return (
    <div
      className="update-modal-backdrop"
      onClick={canDismiss ? handleClose : undefined}
    >
      <div
        ref={dialogRef}
        className="update-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="update-modal-title"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        {canDismiss && (
          <button className="update-modal-close" onClick={handleClose} aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        )}

        <h2 id="update-modal-title">{titleForStatus(state.status)}</h2>
        <p className="update-modal-version">
          {state.currentVersion} → <strong>{state.version}</strong>
        </p>

        {state.releaseNotes && <div className="update-modal-notes">{state.releaseNotes}</div>}

        {state.message && (
          <p className="update-modal-error"><AlertCircle className="h-4 w-4" /> {state.message}</p>
        )}
        {installError && (
          <p className="update-modal-error"><AlertCircle className="h-4 w-4" /> {installError}</p>
        )}

        {downloading && (
          <div className="update-modal-progress">
            <div className="update-modal-bar" style={{ width: `${state.percent ?? 0}%` }} />
            <span>{state.percent ?? 0}%</span>
          </div>
        )}

        {downloaded && isGenerationActive && (
          <p className="update-modal-warning">
            A generation is running. Installing will restart the app — it will be enabled when the
            generation finishes.
          </p>
        )}

        {state.status === 'available' && (
          <label className="update-modal-skip">
            <input type="checkbox" checked={skipChecked} onChange={(e) => setSkipChecked(e.target.checked)} />
            Skip this version
          </label>
        )}

        <div className="update-modal-actions">
          {downloaded ? (
            <Button onClick={handleInstall} disabled={isGenerationActive}>
              <RefreshCw className="h-4 w-4" /> Restart to update
            </Button>
          ) : downloading ? (
            <Button variant="ghost" onClick={handleClose}>Hide</Button>
          ) : (
            <>
              <Button variant="ghost" onClick={handleClose}>Later</Button>
              <Button onClick={startDownload} disabled={!canDownload}>
                {state.message ? (<><Download className="h-4 w-4" /> Try again</>)
                               : (<><Download className="h-4 w-4" /> Update now</>)}
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
