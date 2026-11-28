import { AlertCircle } from 'lucide-react'
import { Fragment, useCallback, useMemo, useState } from 'react'
import { Button as DsButton } from '@/ds/Button/Button'
import { Text } from '@/ds/Text/Text'
import type { components } from '@/generated/backend-openapi'
// eslint-disable-next-line no-restricted-imports
import {
  useCustomLoraImport,
  useCustomLorasWith,
  useDeleteCustomLoraWeightsWith,
  useRemoveCustomLoraWith,
} from '@/ltx-io/hooks/useCustomLoras'
// eslint-disable-next-line no-restricted-imports
import { ApiResultError } from '@/ltx-io/lib/unwrapApiResult'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { ApiClient } from '../../lib/api-client'
import { formatBytes } from '../../lib/format'
import {
  ModelCatalogActionGroup,
  ModelCatalogCellText,
  ModelCatalogDetailRow,
  ModelCatalogInstalledCheck,
  ModelCatalogNameCell,
  ModelCatalogSizeCell,
  ModelCatalogSizeText,
  ModelCatalogTable,
  confirmCatalogModelDelete,
  modelCatalogActionButtonWideClassName,
  modelCatalogDownloadButtonLabel,
  modelCatalogTableStyles,
} from './ModelCatalogTable'
import { SettingsStack, SettingsSubsection } from './SettingsContent'

type CustomLoraItem = components['schemas']['CustomLoraItem']

// The backend refuses a delete with a message the user can act on (a queued run uses the LoRA).
// Any other failure shows the fallback.
function errorMessage(error: unknown, fallback: string): string {
  return error instanceof ApiResultError && error.code === 'CUSTOM_LORA_IN_USE'
    ? error.message
    : fallback
}

function pickWeightsFile(): Promise<string | null> {
  return window.electronAPI
    .showOpenFileDialog({
      title: 'Choose LoRA weights',
      filters: [{ name: 'LoRA weights', extensions: ['safetensors'] }],
      properties: ['openFile'],
    })
    .then((paths) => paths?.[0] ?? null)
}

/**
 * The LoRAs the user added. Each one is a row like a catalog LoRA: Download and Delete work on the
 * weights. Delete keeps the entry, so the LoRA stays in the sidebar and asks for its weights
 * again. A LoRA with no weights also has Remove, which drops the entry.
 */
export function CustomLoraSettingsBlock({ searchQuery = '' }: { searchQuery?: string }) {
  const { notifyModelsChanged } = useAppSettings()
  const lorasQuery = useCustomLorasWith(ApiClient)
  const deleteWeights = useDeleteCustomLoraWeightsWith(ApiClient)
  const removeLora = useRemoveCustomLoraWith(ApiClient)
  const importer = useCustomLoraImport(ApiClient, () => notifyModelsChanged())
  const [errorById, setErrorById] = useState<Readonly<Record<string, string>>>({})

  const setError = useCallback((id: string, message: string | null) => {
    setErrorById((current) => {
      const { [id]: _removed, ...rest } = current
      return message === null ? rest : { ...rest, [id]: message }
    })
  }, [])

  const rows = useMemo(() => {
    const all = lorasQuery.data ?? []
    return searchQuery
      ? all.filter((lora) => lora.name.toLowerCase().includes(searchQuery))
      : all
  }, [lorasQuery.data, searchQuery])

  if (rows.length === 0) return null

  const busy = importer.state.status === 'running'
  const progress = importer.state.status === 'running' ? importer.state.progress : 0

  const handleDelete = async (lora: CustomLoraItem) => {
    if (!confirmCatalogModelDelete(lora.name)) return
    setError(lora.id, null)
    try {
      await deleteWeights.mutateAsync(lora.id)
      notifyModelsChanged()
    } catch (error) {
      setError(lora.id, errorMessage(error, 'Could not remove this file.'))
    }
  }

  const handleRemove = async (lora: CustomLoraItem) => {
    if (!confirmCatalogModelDelete(lora.name)) return
    setError(lora.id, null)
    try {
      await removeLora.mutateAsync(lora.id)
      notifyModelsChanged()
    } catch (error) {
      setError(lora.id, errorMessage(error, 'Could not remove this LoRA.'))
    }
  }

  const handleDownload = (lora: CustomLoraItem) => {
    setError(lora.id, null)
    importer.dismissError()
    void importer.restore(lora.id, {})
  }

  const handleChooseFile = async (lora: CustomLoraItem) => {
    setError(lora.id, null)
    const picked = await pickWeightsFile()
    if (picked === null) return
    importer.dismissError()
    void importer.restore(lora.id, { source_path: picked })
  }

  return (
    <SettingsSubsection title="Custom LoRAs">
      <SettingsStack>
        <ModelCatalogTable>
          {rows.map((lora) => {
            const isRestoring =
              importer.state.status === 'running' &&
              importer.state.customLoraId === lora.id
            const restoreError =
              importer.state.status === 'error' &&
              importer.state.customLoraId === lora.id
                ? importer.state.message
                : null
            const detail = errorById[lora.id] ?? restoreError
            return (
              <Fragment key={lora.id}>
                <tr className={modelCatalogTableStyles.dataRow}>
                  <ModelCatalogNameCell>
                    <ModelCatalogCellText>{lora.name}</ModelCatalogCellText>
                    {lora.downloaded ? <ModelCatalogInstalledCheck /> : null}
                  </ModelCatalogNameCell>
                  <ModelCatalogSizeCell>
                    <ModelCatalogSizeText>
                      {lora.size_bytes ? formatBytes(lora.size_bytes) : '—'}
                    </ModelCatalogSizeText>
                  </ModelCatalogSizeCell>
                  <ModelCatalogActionGroup
                    leading={
                      lora.downloaded ? undefined : (
                        <DsButton
                          appearance="neutral"
                          hierarchy="secondary"
                          size="md"
                          label="Remove"
                          disabled={busy || removeLora.isPending}
                          onClick={() => void handleRemove(lora)}
                        />
                      )
                    }
                    action={
                      lora.downloaded ? (
                        <DsButton
                          appearance="neutral"
                          hierarchy="secondary"
                          size="md"
                          className={modelCatalogActionButtonWideClassName()}
                          label="Delete"
                          disabled={busy || deleteWeights.isPending}
                          onClick={() => void handleDelete(lora)}
                        />
                      ) : lora.can_download ? (
                        <DsButton
                          appearance="neutral"
                          hierarchy="secondary"
                          size="md"
                          className={modelCatalogActionButtonWideClassName()}
                          label={modelCatalogDownloadButtonLabel(isRestoring, progress)}
                          isLoading={isRestoring}
                          disabled={busy && !isRestoring}
                          onClick={() => handleDownload(lora)}
                        />
                      ) : (
                        <DsButton
                          appearance="neutral"
                          hierarchy="secondary"
                          size="md"
                          className={modelCatalogActionButtonWideClassName()}
                          label={isRestoring ? modelCatalogDownloadButtonLabel(true, progress) : 'Choose file'}
                          isLoading={isRestoring}
                          disabled={busy && !isRestoring}
                          onClick={() => void handleChooseFile(lora)}
                        />
                      )
                    }
                  />
                </tr>
                {detail ? (
                  <ModelCatalogDetailRow>
                    <Text
                      as="p"
                      variant="body"
                      size="xs"
                      className="inline-flex items-center gap-1.5 text-fg-danger"
                    >
                      <AlertCircle className="h-3 w-3 flex-shrink-0" aria-hidden />
                      {detail}
                    </Text>
                  </ModelCatalogDetailRow>
                ) : null}
              </Fragment>
            )
          })}
        </ModelCatalogTable>
      </SettingsStack>
    </SettingsSubsection>
  )
}
