import { useEffect, useState } from 'react'
import { ActivityBar } from '@/ds/ActivityBar/ActivityBar'
import { Button as DsButton } from '@/ds/Button/Button'
import { formatBytes } from '@/lib/format'
import type { AssetsMoveState } from '../../../shared/electron-api-schema'
import { AppProductDialogShell } from '../dialog/AppProductDialogShell'

const PHASE_LABEL: Record<AssetsMoveState['phase'], string> = {
  checking: 'Checking the folder…',
  stopping: 'Stopping the app server…',
  copying: 'Copying and checking files…',
  restarting: 'Restarting the app server…',
  cleaning: 'Deleting the old files…',
  done: '',
}

/**
 * Blocks the app while the main process moves the assets folder. The dialog shell makes the
 * rest of the app inert and keeps the focus inside. It restores its state after a reload.
 */
export function AssetsMoveOverlay() {
  const [move, setMove] = useState<AssetsMoveState | null>(null)

  useEffect(() => {
    let received = false
    const unsubscribe = window.electronAPI.onAssetsMoveProgress((next) => {
      received = true
      setMove(next)
    })
    void window.electronAPI.getAssetsMoveState().then((current) => {
      if (!received) setMove(current)
    })
    return unsubscribe
  }, [])

  if (!move) return null

  const result = move.phase === 'done' ? move.result : undefined
  const percent = move.bytesTotal > 0 ? Math.min(100, (move.bytesDone / move.bytesTotal) * 100) : 0

  const footer = result ? (
    <DsButton
      appearance="neutral"
      hierarchy="secondary"
      size="md"
      label="OK"
      // The backend health event already gives the page the new token. After a move, the asset
      // paths in the page are old, so only a finished move reloads. A failed or cancelled move keeps the page.
      onClick={() => (result.ok ? window.location.reload() : setMove(null))}
    />
  ) : move.canCancel ? (
    <DsButton
      appearance="neutral"
      hierarchy="secondary"
      size="md"
      label="Cancel"
      onClick={() => void window.electronAPI.cancelAssetsMove()}
    />
  ) : undefined

  return (
    <>
      {/* The server is stopped during a move, so the pages behind show errors or "Coming soon". Hide them. */}
      <div className="fixed inset-0 z-[69] bg-surface-primary" aria-hidden />
      <AppProductDialogShell
        // A new shell for the result: it moves the focus to the new "OK" button.
        key={result ? 'result' : 'progress'}
        aria-labelledby="assets-move-title"
        title={result ? (result.ok ? 'Done' : 'The assets folder was not changed') : 'Moving your assets'}
        footer={footer}
      >
        {result ? (
          <p className="text-sm text-fg-secondary" role="status">{result.message}</p>
        ) : (
          <>
            <p className="text-sm text-fg-secondary" role="status">{PHASE_LABEL[move.phase]}</p>
            {move.phase === 'copying' && move.filesTotal > 0 ? (
              <div>
                <ActivityBar progress={percent} size="md" />
                <p className="mt-2 text-xs text-fg-secondary">
                  {move.filesDone} of {move.filesTotal} files · {formatBytes(move.bytesDone)} of {formatBytes(move.bytesTotal)}
                </p>
              </div>
            ) : null}
            <p className="text-xs text-fg-secondary">Do not close the app.</p>
          </>
        )}
      </AppProductDialogShell>
    </>
  )
}
