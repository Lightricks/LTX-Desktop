import React, { forwardRef } from 'react'

import contentStyles from './settings/SettingsContent.module.scss'

interface LtxApiKeyInputProps {
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  placeholder?: string
  id?: string
  stopPropagation?: boolean
  className?: string
}

export const LtxApiKeyInput = forwardRef<HTMLInputElement, LtxApiKeyInputProps>(
  ({ value, onChange, placeholder = 'Paste your API key', id, stopPropagation, className }, ref) => {
    return (
      <input
        ref={ref}
        id={id}
        type="password"
        value={value}
        onChange={onChange}
        onClick={stopPropagation ? (e) => e.stopPropagation() : undefined}
        onKeyDown={stopPropagation ? (e) => e.stopPropagation() : undefined}
        placeholder={placeholder}
        autoComplete="off"
        className={[contentStyles.select, className].filter(Boolean).join(' ')}
      />
    )
  },
)
LtxApiKeyInput.displayName = 'LtxApiKeyInput'

interface ApiKeyHelperRowProps {
  stopPropagation?: boolean
  label?: string
  onOpenKey?: () => void
}

export function ApiKeyHelperRow({
  stopPropagation,
  label = 'Get API key',
  onOpenKey,
}: ApiKeyHelperRowProps) {
  return (
    <div className={contentStyles.apiKeyHelperRow}>
      <span className={contentStyles.apiKeyHelperNote}>
        Your key stays in your local app settings.
      </span>
      <button
        type="button"
        onClick={(e) => {
          if (stopPropagation) e.stopPropagation()
          onOpenKey?.()
        }}
        className={contentStyles.linkButton}
      >
        {label}
      </button>
    </div>
  )
}

interface LtxApiKeyHelperRowProps {
  stopPropagation?: boolean
}

export function LtxApiKeyHelperRow({ stopPropagation }: LtxApiKeyHelperRowProps) {
  return (
    <ApiKeyHelperRow
      stopPropagation={stopPropagation}
      label="Get API key"
      onOpenKey={() => window.electronAPI.openLtxApiKeyPage()}
    />
  )
}
