import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'

import { InstallProgress } from '@/components/install/InstallProgress'
import type { ApiSuccessOf } from '@/lib/api-client'
import type { DownloadProgressSnapshot } from '@/lib/download-progress'

import styles from '../FirstRunSetup.module.scss'

type DownloadProgress = ApiSuccessOf<'getModelDownloadProgress'>
type RunningDownloadProgress = Extract<DownloadProgress, { status: 'downloading' }>

interface SetupInstallingStepProps {
  percent: number
  runningProgress: RunningDownloadProgress | null
  installFinished: boolean
  downloadError: string | null
  onBack: () => void
  onRetry: () => void
}

export function SetupInstallingStep({
  percent,
  runningProgress,
  installFinished,
  downloadError,
  onBack,
  onRetry,
}: SetupInstallingStepProps) {
  const snapshot: DownloadProgressSnapshot = {
    statusLabel: installFinished ? 'Installed' : percent > 85 ? 'Installing...' : 'Downloading...',
    percent,
    downloadedBytes: runningProgress?.total_downloaded_bytes,
    totalBytes: runningProgress?.expected_total_bytes,
    speedBytesPerSec: runningProgress?.speed_bytes_per_sec,
  }
  return (
    <InstallProgress
      snapshot={snapshot}
      error={
        downloadError ? (
          <div className={styles.centered}>
            <Text as="span" variant="body" size="md" align="center">
              {downloadError === 'DOWNLOAD_ALREADY_RUNNING'
                ? 'A download is already running.'
                : downloadError}
            </Text>
            <div className={styles.actions}>
              <Button
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                label="Back"
                onClick={onBack}
              />
              <Button appearance="neutral" hierarchy="primary" size="md" label="Retry" onClick={onRetry} />
            </div>
          </div>
        ) : undefined
      }
    />
  )
}
