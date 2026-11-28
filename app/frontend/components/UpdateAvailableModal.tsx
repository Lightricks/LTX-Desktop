import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, X } from 'lucide-react'
import { Button } from '@ds/Button/Button'
import type { UpdateStatePayload } from '../../shared/electron-api-schema'
import type { AppUpdate } from '../hooks/use-app-update'
import {
  AppProductDialogShell,
  ProductDialogBlock,
  ProductDialogIconButton,
  ProductDialogTextButton,
} from './dialog/AppProductDialogShell'

/** Frozen state for local UI previews. Production leaves this unset. */
export type UpdateAvailableModalPreview = {
  state: UpdateStatePayload
  isGenerationActive?: boolean
}

interface Props {
  update: AppUpdate
  isGenerationActive: boolean
  // onClose(true) when the user skips this version; false when dismissing with X / backdrop.
  onClose: (skipThisVersion: boolean) => void
  preview?: UpdateAvailableModalPreview
}

function titleForStatus(status: UpdateStatePayload['status']): string {
  if (status === 'downloaded') return 'Restart to update'
  if (status === 'downloading') return 'Downloading update'
  if (status === 'checking') return 'Checking for updates'
  return 'Update available'
}

export function UpdateAvailableModal({ update, isGenerationActive, onClose, preview }: Props) {
  // Vite replaces this with `false` in a production build, so the preview branches drop out.
  const devPreview = import.meta.env.DEV ? preview : undefined
  const { startDownload, installAndRestart } = update
  const state = devPreview?.state ?? update.state
  const generationActive = devPreview?.isGenerationActive ?? isGenerationActive

  const [installError, setInstallError] = useState<string | null>(null)

  const downloading = state.status === 'downloading'
  const downloaded = state.status === 'downloaded'
  const canDownload = state.status === 'available'
  const canDismiss = !downloaded

  const handleClose = useCallback(() => {
    if (!canDismiss) return
    onClose(false)
  }, [canDismiss, onClose])

  const handleSkipVersion = useCallback(() => {
    if (state.status !== 'available') return
    onClose(true)
  }, [onClose, state.status])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && canDismiss) handleClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [canDismiss, handleClose])

  const handleInstall = async () => {
    if (devPreview) return
    try {
      const res = await installAndRestart()
      if (!res.success) setInstallError(res.error ?? 'Could not install the update.')
    } catch (e) {
      setInstallError(e instanceof Error ? e.message : 'Could not install the update.')
    }
  }

  const handleDownload = () => {
    if (devPreview) return
    void startDownload()
  }

  const footer = downloaded ? (
    <Button
      appearance="brand"
      hierarchy="primary"
      size="xl"
      className="w-fit"
      label="Restart"
      disabled={generationActive}
      onClick={() => void handleInstall()}
    />
  ) : downloading ? (
    <div className="flex flex-col items-center gap-2">
      <Button
        appearance="brand"
        hierarchy="primary"
        size="xl"
        className="min-w-[5.5rem] w-fit tabular-nums"
        label={`${Math.round(state.percent ?? 0)}%`}
        isLoading
        disabled
      />
      <p className="text-center text-xs text-fg-tertiary">Downloading update</p>
    </div>
  ) : (
    <div className="flex flex-col items-center gap-2">
      <Button
        appearance="brand"
        hierarchy="primary"
        size="xl"
        className="w-fit"
        label={state.message ? 'Retry' : 'Update'}
        disabled={!canDownload}
        onClick={handleDownload}
      />
      {state.status === 'available' ? (
        <ProductDialogTextButton type="button" onClick={handleSkipVersion}>
          Skip this version
        </ProductDialogTextButton>
      ) : null}
    </div>
  )

  return (
    <AppProductDialogShell
      aria-labelledby="app-update-dialog-title"
      title={titleForStatus(state.status)}
      subtitle={
        state.version ? (
          <>
            Version {state.currentVersion} → {state.version}
          </>
        ) : (
          <>Version {state.currentVersion}</>
        )
      }
      onBackdropClick={canDismiss ? handleClose : undefined}
      headerActions={
        canDismiss ? (
          <ProductDialogIconButton type="button" onClick={handleClose} aria-label="Close app update dialog">
            <X className="h-4 w-4" />
          </ProductDialogIconButton>
        ) : undefined
      }
      footer={footer}
    >
      {state.releaseNotes ? (
          <ProductDialogBlock title="What's new" tone="highlight">
            <div className="whitespace-pre-wrap text-sm leading-relaxed text-fg-secondary">
              {state.releaseNotes}
            </div>
          </ProductDialogBlock>
        ) : null}

        {downloaded && generationActive ? (
          <p className="text-sm text-fg-tertiary">
            A generation is running. Restart will be available when it finishes.
          </p>
        ) : null}

        {state.message ? (
          <p className="flex items-start gap-2 text-sm text-red-600 dark:text-red-300">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{state.message}</span>
          </p>
        ) : null}

        {installError ? (
          <p className="flex items-start gap-2 text-sm text-red-600 dark:text-red-300">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{installError}</span>
          </p>
        ) : null}
    </AppProductDialogShell>
  )
}
