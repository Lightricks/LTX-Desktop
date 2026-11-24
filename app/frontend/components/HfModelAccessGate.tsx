import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'
import type { ApiSuccessOf } from '../lib/api-client'
import { HfStatusToast } from './HfStatusToast'
import styles from './HfModelAccessGate.module.scss'

type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']
type ModelAccessMap = ApiSuccessOf<'checkModelAccess'>['access']

interface HfModelAccessGateProps {
  accessMap: ModelAccessMap
  allAuthorized: boolean
  hfAuthStatus: HfAuthStatus
  hfAuthPolling: boolean
  startHuggingFaceLogin: () => void
  /** When the access check itself failed (network/backend), distinct from unauthorized. */
  checkError?: string | null
  onRetryCheck?: () => void
  className?: string
  /** Hide the license sentence when the parent already explains the next step. */
  hideMessage?: boolean
}

export function HfModelAccessGate({
  accessMap,
  allAuthorized,
  hfAuthStatus,
  hfAuthPolling,
  startHuggingFaceLogin,
  checkError = null,
  onRetryCheck,
  className,
  hideMessage = false,
}: HfModelAccessGateProps) {
  if (allAuthorized) return null

  if (checkError) {
    return (
      <div className={className ?? styles.gate}>
        <HfStatusToast
          message="Couldn't verify Hugging Face access."
          detail={checkError}
        >
          {onRetryCheck && (
            <Button
              appearance="neutral"
              hierarchy="primary"
              size="md"
              label="Retry"
              onClick={onRetryCheck}
            />
          )}
        </HfStatusToast>
      </div>
    )
  }

  const unauthorizedRepos = Object.entries(accessMap).filter(([, status]) => status === 'not_authorized')
  if (unauthorizedRepos.length === 0) return null

  if (hfAuthStatus !== 'authenticated') {
    return (
      <div className={className ?? styles.gate}>
        {!hideMessage && (
          <Text as="p" variant="body" size="lg" className={styles.message}>
            This model is gated on Hugging Face. Sign in, then accept the license to download.
          </Text>
        )}
        <Button
          appearance="brand"
          hierarchy="primary"
          size="sm"
          label={hfAuthPolling ? 'Waiting for sign in…' : 'Sign in with Hugging Face'}
          disabled={hfAuthPolling}
          onClick={startHuggingFaceLogin}
        />
      </div>
    )
  }

  return (
    <div className={className ?? styles.gate}>
      {unauthorizedRepos.map(([repoId]) => (
        <HfStatusToast key={repoId} message="Accept the license on Hugging Face.">
          <Button
            appearance="neutral"
            hierarchy="primary"
            size="md"
            label="Request access"
            onClick={() => {
              void window.electronAPI.openHuggingFaceRepo({ repoId })
            }}
          />
        </HfStatusToast>
      ))}
    </div>
  )
}
