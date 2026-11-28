import AcceptFillIcon from '@ds/assets/Icons/Accept/Fill.svg?react'
import { Button } from '@ds/Button/Button'
import { Text } from '@ds/Text/Text'

import styles from '../FirstRunSetup.module.scss'

interface LtxApiKeyCardProps {
  collapsed: boolean
  hasSavedKey: boolean
  apiKey: string
  apiKeyError: string | null
  onApiKeyChange: (value: string) => void
  onChangeKey: () => void
}

// The typed key is saved (unverified) when Install runs; a bad key surfaces at
// the first generation, not here. Presence gating lives in the parent: skipping
// the local text model without any key blocks Install.
export function LtxApiKeyCard({
  collapsed,
  hasSavedKey,
  apiKey,
  apiKeyError,
  onApiKeyChange,
  onChangeKey,
}: LtxApiKeyCardProps) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <Text as="h3" variant="heading" size="md">
          LTX API Key
        </Text>
        {collapsed && <AcceptFillIcon className={styles.cardDone} aria-label="API key saved" />}
      </div>
      <Text as="p" variant="body" size="lg" className={styles.hfSubtitle}>
        Required if you skip the local text model. Used to enhance your prompts.{' '}
        <button
          type="button"
          className={styles.inlineLink}
          onClick={() => {
            void window.electronAPI.openLtxApiKeyPage()
          }}
        >
          Create a free LTX API key
        </button>
      </Text>
      <div className={styles.fieldWrap}>
        {collapsed ? (
          <Button
            appearance="neutral"
            hierarchy="primary"
            size="md"
            label="Change API key"
            onClick={onChangeKey}
          />
        ) : (
          <>
            <div className={styles.fieldRow}>
              <div className={`${styles.fieldInner} ${apiKeyError ? styles.fieldInnerInvalid : ''}`}>
                <input
                  type="password"
                  value={apiKey}
                  // Keys never contain whitespace, so a messy paste loses it instead
                  // of being rejected.
                  onChange={(e) => onApiKeyChange(e.target.value.replace(/\s/g, ''))}
                  placeholder={hasSavedKey ? 'Enter new key…' : 'Enter API key…'}
                  autoComplete="off"
                  aria-invalid={apiKeyError != null}
                  aria-describedby={apiKeyError ? 'ltx-api-key-error' : undefined}
                  aria-label="LTX API key"
                  className={`${styles.field} ${styles.fieldWide}`}
                />
              </div>
            </div>
            {apiKeyError && (
              <Text as="p" id="ltx-api-key-error" variant="body" size="sm" className={styles.fieldError}>
                {apiKeyError}
              </Text>
            )}
          </>
        )}
      </div>
    </div>
  )
}
