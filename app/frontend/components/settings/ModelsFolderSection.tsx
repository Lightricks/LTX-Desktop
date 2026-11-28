import { useEffect, useState } from 'react'
import { ApiClient } from '../../lib/api-client'
import { logger } from '../../lib/logger'
import { Button as DsButton } from '@/ds/Button/Button'
import {
  SettingsFieldRow,
  SettingsReadonlyField,
  SettingsSubsection,
} from './SettingsContent'
import contentStyles from './SettingsContent.module.scss'

export function ModelsFolderSection({ nested = false }: { nested?: boolean }) {
  const [modelsDir, setModelsDir] = useState('')

  useEffect(() => {
    void (async () => {
      const result = await ApiClient.getSettings()
      if (!result.ok) {
        logger.error(`Failed to fetch settings: ${result.error.message}`)
        return
      }
      setModelsDir(result.data.modelsDir ?? '')
    })()
  }, [])

  return (
    <SettingsSubsection
      bare={nested}
      className={nested ? contentStyles.modelsManagementNestedSubsection : undefined}
      title="Models folder"
    >
      <SettingsFieldRow>
        <SettingsReadonlyField value={modelsDir} />
        <DsButton
          appearance="neutral"
          hierarchy="secondary"
          size="md"
          className={contentStyles.settingsControlButton}
          label="Change…"
          onClick={async () => {
            const result = await window.electronAPI.openModelsDirChangeDialog()
            if (result.success) {
              setModelsDir(result.path)
            }
          }}
        />
        <DsButton
          appearance="neutral"
          hierarchy="secondary"
          size="md"
          className={contentStyles.settingsControlButton}
          label="Open folder"
          disabled={!modelsDir}
          onClick={() => {
            void window.electronAPI.openModelsFolder()
          }}
        />
      </SettingsFieldRow>
    </SettingsSubsection>
  )
}
