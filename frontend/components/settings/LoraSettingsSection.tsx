import { AlertCircle } from 'lucide-react'
import { Fragment, useCallback, useLayoutEffect, useMemo, useState } from 'react'
import { ApiClient } from '../../lib/api-client'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { useHfAuth } from '../../hooks/use-hf-auth'
import {
  useIcLoras,
  useLoraCatalog,
  type IcLoraListItem,
  type LoraCatalogListItem,
} from '../../hooks/use-catalog'
import {
  catalogItemToEntry,
  catalogVariantKey,
  preferredVariantId,
  type LibraryEntry,
} from '../../lib/lora-library'
import { formatBytes } from '../../lib/format'
import { isHomeFeatureEnabled } from '../../lib/home-features'
import { Button as DsButton } from '@/ds/Button/Button'
import { Text } from '@/ds/Text/Text'
import {
  ModelCatalogActionGroup,
  ModelCatalogDetailRow,
  ModelCatalogEmpty,
  ModelCatalogCellText,
  ModelCatalogNameCell,
  ModelCatalogSizeCell,
  ModelCatalogSizeText,
  ModelCatalogInstalledCheck,
  ModelCatalogTable,
  confirmCatalogModelDelete,
  modelCatalogActionButtonWideClassName,
  modelCatalogDownloadButtonLabel,
  modelCatalogTableStyles,
} from './ModelCatalogTable'
import {
  SettingsStack,
  SettingsSubsection,
} from './SettingsContent'
import contentStyles from './SettingsContent.module.scss'

function isGatedRepoError(message: string): boolean {
  return message.includes('gated repo') || message.includes('authorized list')
}

function loraListItemToEntry(item: LoraCatalogListItem): LibraryEntry {
  const base = catalogItemToEntry(item.lora)
  return {
    ...base,
    downloaded: item.downloaded,
    downloadedVariantIds: item.downloaded_variant_ids ?? [],
  }
}

function icListItemToEntry(item: IcLoraListItem): LibraryEntry {
  const base = catalogItemToEntry(item.ic_lora)
  return {
    ...base,
    downloaded: item.downloaded,
    downloadedVariantIds: item.downloaded_variant_ids ?? [],
  }
}

function LoraTableRows({
  entry,
  downloadingKey,
  progress,
  downloadError,
  onDownload,
  onDelete,
  hfAuthStatus,
  startHuggingFaceLogin,
}: {
  entry: LibraryEntry
  downloadingKey: string | null
  progress: number
  downloadError: { key: string; message: string } | null
  onDownload: (id: string, variantId?: string) => void
  onDelete: (id: string, variantId?: string) => Promise<boolean>
  hfAuthStatus: ReturnType<typeof useHfAuth>['hfAuthStatus']
  startHuggingFaceLogin: () => void
}) {
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const hasVariants = (entry.variants?.length ?? 0) > 1
  const downloadedVariantIds = entry.downloadedVariantIds ?? []
  const fallbackVariantId = entry.defaultVariantId ?? entry.variants?.[0]?.id
  const [variantId, setVariantId] = useState(
    () =>
      preferredVariantId(entry.variants, entry.defaultVariantId, downloadedVariantIds) ??
      fallbackVariantId,
  )

  const selectedVariant = entry.variants?.find((v) => v.id === variantId)
  const accessRepoId = selectedVariant?.repoId ?? entry.repoId
  const activeKey = catalogVariantKey(entry.id, hasVariants ? variantId : undefined)
  const isDownloading = downloadingKey === activeKey
  const variantOnDisk =
    hasVariants && variantId
      ? downloadedVariantIds.includes(variantId)
      : entry.downloaded
  const sizeBytes = selectedVariant?.sizeBytes ?? entry.sizeBytes
  const activeError = downloadError?.key === activeKey ? downloadError.message : null
  const showHfLogin =
    Boolean(entry.requiresHfLogin) &&
    hfAuthStatus !== 'authenticated' &&
    !variantOnDisk
  const showDetails = showHfLogin || Boolean(activeError) || Boolean(deleteError)

  const handleDelete = useCallback(async () => {
    if (!confirmCatalogModelDelete(entry.name)) return
    setDeleteError(null)
    setDeleteBusy(true)
    const ok = await onDelete(entry.id, hasVariants ? variantId : undefined)
    if (!ok) {
      setDeleteError('Could not remove this file.')
    }
    setDeleteBusy(false)
  }, [entry.id, entry.name, hasVariants, onDelete, variantId])

  return (
    <Fragment>
      <tr className={modelCatalogTableStyles.dataRow}>
        <ModelCatalogNameCell>
          <ModelCatalogCellText>{entry.name}</ModelCatalogCellText>
          {variantOnDisk ? <ModelCatalogInstalledCheck /> : null}
        </ModelCatalogNameCell>
        <ModelCatalogSizeCell>
          <ModelCatalogSizeText>
            {sizeBytes ? formatBytes(sizeBytes) : '—'}
          </ModelCatalogSizeText>
        </ModelCatalogSizeCell>
        <ModelCatalogActionGroup
          leading={
            hasVariants ? (
              <select
                className={modelCatalogTableStyles.variantSelect}
                value={variantId ?? ''}
                aria-label={`Checkpoint for ${entry.name}`}
                onChange={(event) => setVariantId(event.target.value)}
              >
                {(entry.variants ?? []).map((variant) => (
                  <option key={variant.id} value={variant.id}>
                    {variant.label}
                    {downloadedVariantIds.includes(variant.id) ? ' ✓' : ''}
                  </option>
                ))}
              </select>
            ) : undefined
          }
          action={
            !variantOnDisk ? (
              <DsButton
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                className={modelCatalogActionButtonWideClassName()}
                label={modelCatalogDownloadButtonLabel(isDownloading, progress)}
                isLoading={isDownloading}
                disabled={deleteBusy && !isDownloading}
                onClick={() => onDownload(entry.id, hasVariants ? variantId : undefined)}
              />
            ) : (
              <DsButton
                appearance="neutral"
                hierarchy="secondary"
                size="md"
                className={modelCatalogActionButtonWideClassName()}
                label="Delete"
                disabled={deleteBusy || isDownloading}
                onClick={() => void handleDelete()}
              />
            )
          }
        />
      </tr>
      {showDetails ? (
        <ModelCatalogDetailRow>
          <SettingsStack>
            {showHfLogin ? (
              <Text as="p" variant="body" size="xs" className={contentStyles.textTertiary}>
                Gated on Hugging Face —{' '}
                <button
                  type="button"
                  className="text-fg-brand underline hover:text-fg-brand-hover"
                  onClick={startHuggingFaceLogin}
                >
                  sign in
                </button>{' '}
                to download.
              </Text>
            ) : null}
            {activeError ? (
              isGatedRepoError(activeError) ? (
                <Text as="p" variant="body" size="xs" className={contentStyles.textTertiary}>
                  This checkpoint is gated on Hugging Face.{' '}
                  {accessRepoId ? (
                    <button
                      type="button"
                      className="text-fg-brand underline hover:text-fg-brand-hover"
                      onClick={() => {
                        void window.electronAPI.openHuggingFaceRepo({ repoId: accessRepoId })
                      }}
                    >
                      Request access
                    </button>
                  ) : null}
                </Text>
              ) : (
                <Text as="p" variant="body" size="xs" className="inline-flex items-center gap-1.5 text-fg-danger">
                  <AlertCircle className="h-3 w-3 flex-shrink-0" aria-hidden />
                  {activeError}
                </Text>
              )
            ) : null}
            {deleteError ? (
              <Text as="p" variant="body" size="xs" className="inline-flex items-center gap-1.5 text-fg-danger">
                <AlertCircle className="h-3 w-3 flex-shrink-0" aria-hidden />
                {deleteError}
              </Text>
            ) : null}
          </SettingsStack>
        </ModelCatalogDetailRow>
      ) : null}
    </Fragment>
  )
}

function LoraCatalogBlock({
  title,
  entries,
  searchQuery = '',
  catalogStatus,
  onRetry,
  downloadingKey,
  progress,
  downloadError,
  onDownload,
  onDelete,
  hfAuthStatus,
  startHuggingFaceLogin,
}: {
  title: string
  entries: LibraryEntry[]
  searchQuery?: string
  catalogStatus: 'loading' | 'error' | 'loaded'
  onRetry: () => void
  downloadingKey: string | null
  progress: number
  downloadError: { key: string; message: string } | null
  onDownload: (id: string, variantId?: string) => void
  onDelete: (id: string, variantId?: string) => Promise<boolean>
  hfAuthStatus: ReturnType<typeof useHfAuth>['hfAuthStatus']
  startHuggingFaceLogin: () => void
}) {
  const tableEntries = useMemo(() => {
    let filtered = entries
    if (searchQuery) {
      filtered = entries.filter((entry) => entry.name.toLowerCase().includes(searchQuery))
    }
    const installed = filtered.filter((entry) => entry.downloaded)
    const notInstalled = filtered.filter((entry) => !entry.downloaded)
    return [...installed, ...notInstalled]
  }, [entries, searchQuery])

  const rowProps = {
    downloadingKey,
    progress,
    downloadError,
    onDownload,
    onDelete,
    hfAuthStatus,
    startHuggingFaceLogin,
  }

  if (searchQuery && catalogStatus === 'loaded' && tableEntries.length === 0) {
    return null
  }

  return (
    <SettingsSubsection title={title}>
      <SettingsStack>
        {catalogStatus === 'loading' ? (
          <Text as="p" variant="body" size="xs" className={contentStyles.textTertiary}>
            Loading catalog…
          </Text>
        ) : null}
        {catalogStatus === 'error' ? (
          <div className="flex flex-wrap items-center gap-2">
            <Text as="p" variant="body" size="xs" className="text-fg-danger">
              Could not load the catalog.
            </Text>
            <DsButton
              appearance="neutral"
              hierarchy="secondary"
              size="sm"
              label="Retry"
              onClick={onRetry}
            />
          </div>
        ) : null}
        {catalogStatus === 'loaded' ? (
          tableEntries.length === 0 ? (
            <ModelCatalogEmpty message="No items in this catalog." />
          ) : (
            <ModelCatalogTable>
              {tableEntries.map((entry) => (
                <LoraTableRows key={entry.id} entry={entry} {...rowProps} />
              ))}
            </ModelCatalogTable>
          )
        ) : null}
      </SettingsStack>
    </SettingsSubsection>
  )
}

export function LoraSettingsSection({
  searchQuery = '',
  onSearchHasMatches,
}: {
  searchQuery?: string
  onSearchHasMatches?: (hasMatches: boolean) => void
}) {
  const { hfAuthStatus, startHuggingFaceLogin } = useHfAuth(true)
  const { notifyModelsChanged } = useAppSettings()
  const {
    loras,
    catalogStatus: loraCatalogStatus,
    refresh: refreshLoras,
    downloadLora,
    downloadingKey: loraDownloadingKey,
    progress: loraProgress,
    downloadError: loraDownloadError,
  } = useLoraCatalog(true)
  const {
    icLoras,
    catalogStatus: icCatalogStatus,
    refresh: refreshIcLoras,
    downloadIcLora,
    downloadingKey: icDownloadingKey,
    progress: icProgress,
    downloadError: icDownloadError,
  } = useIcLoras(true)

  const styleEntries = useMemo(() => loras.map(loraListItemToEntry), [loras])
  const icEntries = useMemo(
    () =>
      icLoras
        .filter((item) => isHomeFeatureEnabled(item.ic_lora.id))
        .map(icListItemToEntry),
    [icLoras],
  )

  const searchMatchCount = useMemo(() => {
    if (!searchQuery) return 1
    const matches = (entries: LibraryEntry[]) =>
      entries.filter((entry) => entry.name.toLowerCase().includes(searchQuery)).length
    return matches(styleEntries) + matches(icEntries)
  }, [icEntries, searchQuery, styleEntries])

  useLayoutEffect(() => {
    if (!onSearchHasMatches) return
    if (!searchQuery) {
      onSearchHasMatches(true)
      return
    }
    if (loraCatalogStatus === 'loading' || icCatalogStatus === 'loading') {
      return
    }
    onSearchHasMatches(searchMatchCount > 0)
  }, [icCatalogStatus, loraCatalogStatus, onSearchHasMatches, searchMatchCount, searchQuery])

  const deleteStyleLora = useCallback(async (id: string, variantId?: string) => {
    const result = await ApiClient.deleteLoraInstallation({
      lora_id: id,
      variant_id: variantId,
    })
    if (result.ok) {
      // Bump modelsVersion too: the generation model specs cache keys off it, so without
      // this Gen Space keeps offering a LoRA whose weights are already gone.
      notifyModelsChanged()
      await refreshLoras()
      return true
    }
    return false
  }, [notifyModelsChanged, refreshLoras])

  const deleteIcLora = useCallback(async (id: string, variantId?: string) => {
    const result = await ApiClient.deleteIcLoraInstallation({
      ic_lora_id: id,
      variant_id: variantId,
    })
    if (result.ok) {
      notifyModelsChanged()
      await refreshIcLoras()
      return true
    }
    return false
  }, [notifyModelsChanged, refreshIcLoras])

  return (
    // A fragment, not a stack: Style LoRAs and IC-LoRAs are two catalog blocks like any
    // other, so they join the Models tab's own rhythm instead of being a nested pair with
    // spacing of their own. Each block returns null when search filters it out.
    <>
      <LoraCatalogBlock
        title="Style LoRAs"
        entries={styleEntries}
        searchQuery={searchQuery}
        catalogStatus={loraCatalogStatus}
        onRetry={() => void refreshLoras()}
        downloadingKey={loraDownloadingKey}
        progress={loraProgress}
        downloadError={loraDownloadError}
        onDownload={(id, variantId) => void downloadLora(id, variantId)}
        onDelete={deleteStyleLora}
        hfAuthStatus={hfAuthStatus}
        startHuggingFaceLogin={() => void startHuggingFaceLogin()}
      />

      <LoraCatalogBlock
        title="IC-LoRAs"
        entries={icEntries}
        searchQuery={searchQuery}
        catalogStatus={icCatalogStatus}
        onRetry={() => void refreshIcLoras()}
        downloadingKey={icDownloadingKey}
        progress={icProgress}
        downloadError={icDownloadError}
        onDownload={(id, variantId) => void downloadIcLora(id, variantId)}
        onDelete={deleteIcLora}
        hfAuthStatus={hfAuthStatus}
        startHuggingFaceLogin={() => void startHuggingFaceLogin()}
      />
    </>
  )
}
