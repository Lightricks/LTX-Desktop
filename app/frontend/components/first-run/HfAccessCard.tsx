import AcceptFillIcon from '@ds/assets/Icons/Accept/Fill.svg?react'
import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'

import { HfModelAccessGate } from '@/components/HfModelAccessGate'
import type { ApiSuccessOf } from '@/lib/api-client'

import styles from '../FirstRunSetup.module.scss'

type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']
type ModelAccessMap = ApiSuccessOf<'checkModelAccess'>['access']

interface HfAccessCardProps {
  complete: boolean
  showGate: boolean
  accessMap: ModelAccessMap
  allAuthorized: boolean
  hfAuthStatus: HfAuthStatus
  hfAuthPolling: boolean
  unsigned: boolean
  checkError: string | null
  onSignIn: () => void
  onSignOut: () => void
  onRetryCheck: () => void
}

export function HfAccessCard({
  complete,
  showGate,
  accessMap,
  allAuthorized,
  hfAuthStatus,
  hfAuthPolling,
  unsigned,
  checkError,
  onSignIn,
  onSignOut,
  onRetryCheck,
}: HfAccessCardProps) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <Text as="h3" variant="heading" size="md">
          Hugging Face access
        </Text>
        {complete && <AcceptFillIcon className={styles.cardDone} aria-label="Hugging Face connected" />}
      </div>
      <Text as="p" variant="body" size="lg" className={styles.hfSubtitle}>
        Sign in and accept the license to download gated models.
      </Text>
      <div className={styles.hfActions}>
        {showGate && (
          <HfModelAccessGate
            accessMap={accessMap}
            allAuthorized={allAuthorized}
            hfAuthStatus={hfAuthStatus}
            hfAuthPolling={hfAuthPolling}
            startHuggingFaceLogin={() => {
              void onSignIn()
            }}
            checkError={checkError}
            onRetryCheck={onRetryCheck}
            hideMessage
          />
        )}
        {unsigned ? (
          <Button
            appearance="neutral"
            hierarchy="primary"
            size="md"
            label="Sign in"
            disabled={hfAuthPolling}
            onClick={() => {
              void onSignIn()
            }}
          />
        ) : (
          <Button
            appearance="neutral"
            hierarchy="primary"
            size="md"
            label="Sign out"
            onClick={() => {
              void onSignOut()
            }}
          />
        )}
      </div>
    </div>
  )
}
