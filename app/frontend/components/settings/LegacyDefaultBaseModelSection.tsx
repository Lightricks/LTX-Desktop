import { useCallback, useEffect, useMemo, useState } from 'react'

import { useAppSettings } from '../../contexts/AppSettingsContext'
import { ApiClient, type ApiSuccessOf } from '../../lib/api-client'
import { formatLtxBaseModelLabel } from '../../lib/format'
import { logger } from '../../lib/logger'
import { Text } from '@/ds/Text/Text'

import {
  SettingsFieldRow,
  SettingsSelect,
  SettingsStack,
  SettingsSubsection,
} from './SettingsContent'
import contentStyles from './SettingsContent.module.scss'

type LtxModelVersionItem = ApiSuccessOf<'getLtxVersions'>['versions'][number]

function formatSetActiveError(message: string, code?: string): string {
  if (code === 'LTX_MODEL_NOT_INSTALLED') {
    return 'This version is not fully downloaded yet. Finish the download, then try again.'
  }
  if (code === 'LOCAL_MODEL_RECOMMENDATIONS_DISABLED_IN_FORCE_API_MODE') {
    return 'Local model selection is unavailable while remote-only generation is enabled.'
  }
  return message || 'Failed to set active model.'
}

export function LegacyDefaultBaseModelSection() {
  const { notifyModelsChanged } = useAppSettings()
  const [versions, setVersions] = useState<LtxModelVersionItem[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshVersions = useCallback(async () => {
    const result = await ApiClient.getLtxVersions()
    if (!result.ok) {
      logger.error(`Failed to fetch LTX versions: ${result.error.message}`)
      return
    }
    setVersions(result.data.versions)
  }, [])

  useEffect(() => {
    void refreshVersions()
  }, [refreshVersions])

  const installedVersions = useMemo(
    () => versions.filter((version) => version.installed),
    [versions],
  )

  const activeModelId =
    versions.find((version) => version.active)?.model_id ??
    installedVersions[0]?.model_id ??
    ''

  const handleChange = useCallback(
    async (modelId: string) => {
      if (!modelId || modelId === activeModelId) return
      setError(null)
      setBusy(true)
      try {
        const result = await ApiClient.setActiveLtxModel({
          model_id: modelId as LtxModelVersionItem['model_id'],
        })
        if (!result.ok) {
          setError(formatSetActiveError(result.error.message, result.error.code))
          return
        }
        await refreshVersions()
        notifyModelsChanged()
      } finally {
        setBusy(false)
      }
    },
    [activeModelId, notifyModelsChanged, refreshVersions],
  )

  return (
    <SettingsSubsection
      showDivider
      title="Default base model"
      description="Installed LTX version used for local generation in Gen Space and Video Editor."
      descriptionClassName={contentStyles.apiKeysDescriptionLine}
      descriptionSize="sm"
    >
      <SettingsStack>
        {installedVersions.length === 0 ? (
          <Text as="p" variant="body" size="sm" className={contentStyles.apiKeysDescriptionLine}>
            Download a base model on the Models tab to set a default here.
          </Text>
        ) : (
          <SettingsFieldRow>
            <SettingsSelect
              value={activeModelId}
              disabled={busy}
              aria-label="Default base model for local generation"
              onChange={(event) => void handleChange(event.target.value)}
            >
              {installedVersions.map((version) => (
                <option key={version.model_id} value={version.model_id}>
                  {formatLtxBaseModelLabel(version.label)}
                </option>
              ))}
            </SettingsSelect>
          </SettingsFieldRow>
        )}
        {error ? (
          <Text as="p" variant="body" size="xs" className="text-fg-danger">
            {error}
          </Text>
        ) : null}
      </SettingsStack>
    </SettingsSubsection>
  )
}
