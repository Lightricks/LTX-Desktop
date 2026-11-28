import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { ApiClient } from '../../lib/api-client'
import { useGlobalGenerationLock } from '../../hooks/use-global-generation-lock'
import { fetchGenerationQueue, generationQueryKeys } from '../../ltx-io/hooks/generationQueryKeys'
import { Button as DsButton } from '@/ds/Button/Button'
import { Tooltip } from '@/ds/Tooltip/Tooltip'
import {
  SettingsFieldRow,
  SettingsReadonlyField,
  SettingsSubsection,
} from './SettingsContent'
import contentStyles from './SettingsContent.module.scss'

const QUEUE_POLL_MS = 3000
const RUNNING_TOOLTIP =
  'Changing the assets folder restarts the app server. Wait for the running generation to finish.'
const CHECKING_TOOLTIP = 'Checking for running generations…'
const UNKNOWN_TOOLTIP =
  'The app cannot check for running generations now. Changing the assets folder restarts the app server.'

/**
 * Why the folder cannot change now, or `null`. A generation that runs or waits blocks it.
 * A queue that is not known also blocks it, with its own text: the restart must not hit a generation.
 */
function useChangeBlockedReason(): string | null {
  const { isRunning } = useGlobalGenerationLock()
  const queue = useQuery({
    queryKey: generationQueryKeys.queue,
    queryFn: () => fetchGenerationQueue(ApiClient),
    refetchInterval: QUEUE_POLL_MS,
  })
  // After a failed poll the query keeps its last data, so the error is checked first.
  if (isRunning || (!queue.isError && queue.data && queue.data.active.length + queue.data.queued.length > 0)) {
    return RUNNING_TOOLTIP
  }
  if (queue.isError) return UNKNOWN_TOOLTIP
  return queue.data ? null : CHECKING_TOOLTIP
}

export function AssetsFolderSection() {
  const [assetsDir, setAssetsDir] = useState('')
  const blockedReason = useChangeBlockedReason()

  const load = () =>
    window.electronAPI
      .getAssetsDir()
      .then((dir) => setAssetsDir(dir))
      .catch(() => {})

  useEffect(() => {
    void load()
  }, [])

  // The main process shows the dialogs and runs the move. The page reloads when it ends.
  const change = async (reset: boolean) => {
    const result = reset ? await window.electronAPI.resetAssetsDir() : await window.electronAPI.changeAssetsDir()
    if (result.success) setAssetsDir(result.path)
  }

  const control = (label: string, reset: boolean) => (
    <span>
      <DsButton
        appearance="neutral"
        hierarchy="secondary"
        size="md"
        className={contentStyles.settingsControlButton}
        label={label}
        disabled={blockedReason !== null}
        onClick={() => void change(reset)}
      />
    </span>
  )

  const wrap = (node: React.ReactNode) =>
    blockedReason ? (
      <Tooltip content={blockedReason} side="top" align="center" maxWidth={280} portalContainer={document.body}>
        {node}
      </Tooltip>
    ) : (
      node
    )

  return (
    <SettingsSubsection
      showDivider
      title="Assets folder"
      description="Where your generated videos and images are saved. When you change the folder, the app moves your existing assets to it."
      descriptionClassName={contentStyles.apiKeysDescriptionLine}
      descriptionSize="sm"
    >
      <SettingsFieldRow>
        <SettingsReadonlyField value={assetsDir} />
        {wrap(control('Change…', false))}
        {wrap(control('Reset to default', true))}
      </SettingsFieldRow>
    </SettingsSubsection>
  )
}
