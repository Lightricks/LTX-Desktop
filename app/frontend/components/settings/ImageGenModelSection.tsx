import { AlertCircle } from 'lucide-react'
import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { useAppSettings } from '../../contexts/AppSettingsContext'
import { useHfAuth } from '../../hooks/use-hf-auth'
import { useHfModelAccess } from '../../hooks/use-hf-model-access'
import { ApiClient, type ApiRequestBodyOf } from '../../lib/api-client'
import { formatBytes } from '../../lib/format'
import { logger } from '../../lib/logger'
import { IMAGE_MODEL_SEARCH_HAYSTACK, matchesCatalogSearch } from '../../lib/model-catalog'
import { Button as DsButton } from '@/ds/Button/Button'
import { Text } from '@/ds/Text/Text'

import { HfModelAccessGate } from '../HfModelAccessGate'
import {
  ModelCatalogActionGroup,
  ModelCatalogCellText,
  ModelCatalogDetailRow,
  ModelCatalogNameCell,
  ModelCatalogSizeCell,
  ModelCatalogSizeText,
  ModelCatalogInstalledCheck,
  ModelCatalogTable,
  confirmCatalogModelDelete,
  modelCatalogActionButtonWideClassName,
  modelCatalogDownloadButtonLabel,
  modelCatalogRowClassNames,
} from './ModelCatalogTable'
import { SettingsStack, SettingsSubsection } from './SettingsContent'
import contentStyles from './SettingsContent.module.scss'

const Z_IMAGE_TURBO_CP = 'z-image-turbo' satisfies NonNullable<
  ApiRequestBodyOf<'deleteModels'>['cp_ids']
>[number]

const Z_IMAGE_TURBO_LABEL = 'Z Image Turbo'
const Z_IMAGE_TURBO_SIZE_BYTES = 31_000_000_000

const DOWNLOAD_POLL_INTERVAL_MS = 1000

export function ImageGenModelSection({
  searchQuery = '',
  onSearchHasMatches,
}: {
  searchQuery?: string
  onSearchHasMatches?: (hasMatches: boolean) => void
} = {}) {
  const { notifyModelsChanged } = useAppSettings()
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(true)
  const [needsDownload, setNeedsDownload] = useState<boolean | null>(null)
  const [downloading, setDownloading] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadPercent, setDownloadPercent] = useState(0)
  const mountedRef = useRef(true)

  // StrictMode runs setup→cleanup→setup, so the ref has to be re-armed on every setup;
  // leaving it false would keep the post-delete Download button stuck disabled.
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const installed = needsDownload === false

  const cpsToDownload = useMemo((): readonly typeof Z_IMAGE_TURBO_CP[] => {
    return installed ? [] : [Z_IMAGE_TURBO_CP]
  }, [installed])
  const { accessMap, allAuthorized, checking: checkingAccess, checkError, recheckAccess } =
    useHfModelAccess(cpsToDownload, hfAuthStatus)

  const refreshRecommendation = useCallback(async () => {
    const [recResult, activeResult] = await Promise.all([
      ApiClient.getImgGenRecommendation(),
      ApiClient.getActiveDownload(),
    ])
    if (!recResult.ok) {
      logger.error(`Failed to fetch image model recommendation: ${recResult.error.message}`)
      return
    }
    setNeedsDownload(recResult.data.cp_to_download !== null)
    if (activeResult.ok && activeResult.data.session_id) {
      const cpIds = activeResult.data.cp_ids ?? []
      if (cpIds.includes(Z_IMAGE_TURBO_CP)) {
        setDownloadSessionId(activeResult.data.session_id)
        setDownloading(true)
      }
    }
  }, [])

  useEffect(() => {
    void refreshRecommendation()
  }, [refreshRecommendation])

  useEffect(() => {
    if (!downloadSessionId) return
    let cancelled = false

    const poll = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (cancelled) return
      if (!result.ok) {
        logger.error(`Progress poll error: ${result.error.message}`)
        return
      }
      const progress = result.data
      if (progress.status === 'downloading') {
        setDownloadPercent(Math.round(progress.total_progress))
        return
      }
      if (progress.status === 'error') {
        setDownloadSessionId(null)
        setDownloading(false)
        setError(progress.error || 'Download failed.')
        return
      }
      if (progress.status === 'complete') {
        setDownloadSessionId(null)
        setDownloadPercent(100)
        notifyModelsChanged()
        await refreshRecommendation()
        if (!cancelled) setDownloading(false)
      }
    }

    void poll()
    const interval = setInterval(() => void poll(), DOWNLOAD_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [downloadSessionId, notifyModelsChanged, refreshRecommendation])

  const handleDownload = useCallback(async () => {
    setError(null)
    setDownloading(true)
    setDownloadPercent(0)
    const result = await ApiClient.startModelDownload({
      type: 'download',
      cp_ids: [Z_IMAGE_TURBO_CP],
    })
    if (!result.ok) {
      setDownloading(false)
      setError(result.error.message || 'Failed to start download.')
      return
    }
    if (result.data.status !== 'started') {
      setDownloading(false)
      setError('Unexpected response while starting download.')
      return
    }
    setDownloadSessionId(result.data.sessionId)
  }, [])

  const handleDelete = useCallback(async () => {
    if (!confirmCatalogModelDelete(Z_IMAGE_TURBO_LABEL)) return
    setError(null)
    setDeleting(true)
    const result = await ApiClient.deleteModels({ cp_ids: [Z_IMAGE_TURBO_CP] })
    if (!result.ok) {
      setDeleting(false)
      setError(result.error.message || 'Failed to delete model.')
      return
    }
    notifyModelsChanged()
    await refreshRecommendation()
    if (mountedRef.current) setDeleting(false)
  }, [notifyModelsChanged, refreshRecommendation])

  useLayoutEffect(() => {
    onSearchHasMatches?.(matchesCatalogSearch(IMAGE_MODEL_SEARCH_HAYSTACK, searchQuery))
  }, [onSearchHasMatches, searchQuery])

  if (needsDownload === null) {
    return null
  }

  if (searchQuery && !matchesCatalogSearch(IMAGE_MODEL_SEARCH_HAYSTACK, searchQuery)) {
    return null
  }

  const canDownload = !installed && (allAuthorized && !checkingAccess)
  const downloadButtonActive = !installed && downloading
  const showHfGate =
    !installed &&
    !allAuthorized &&
    (Boolean(checkError) ||
      Object.values(accessMap).some((status) => status === 'not_authorized'))
  const showDetails = showHfGate || Boolean(error)

  return (
    <SettingsSubsection
      title="Image models"
      description="Local text-to-image generation in Gen Space and legacy image tools."
      descriptionClassName={contentStyles.apiKeysDescriptionLine}
      descriptionSize="sm"
    >
      <ModelCatalogTable>
        <Fragment>
          <tr className={modelCatalogRowClassNames()}>
            <ModelCatalogNameCell>
              <ModelCatalogCellText>{Z_IMAGE_TURBO_LABEL}</ModelCatalogCellText>
              {installed ? <ModelCatalogInstalledCheck /> : null}
            </ModelCatalogNameCell>
            <ModelCatalogSizeCell>
              <ModelCatalogSizeText>{formatBytes(Z_IMAGE_TURBO_SIZE_BYTES)}</ModelCatalogSizeText>
            </ModelCatalogSizeCell>
            <ModelCatalogActionGroup
              action={
                !installed ? (
                  <DsButton
                    appearance="neutral"
                    hierarchy="secondary"
                    size="md"
                    className={modelCatalogActionButtonWideClassName()}
                    label={modelCatalogDownloadButtonLabel(downloadButtonActive, downloadPercent)}
                    isLoading={downloadButtonActive}
                    disabled={!downloadButtonActive && (!canDownload || deleting)}
                    onClick={(event) => {
                      event.stopPropagation()
                      void handleDownload()
                    }}
                  />
                ) : (
                  <DsButton
                    appearance="neutral"
                    hierarchy="secondary"
                    size="md"
                    className={modelCatalogActionButtonWideClassName()}
                    label="Delete"
                    disabled={downloading || deleting}
                    onClick={(event) => {
                      event.stopPropagation()
                      void handleDelete()
                    }}
                  />
                )
              }
            />
          </tr>
          {showDetails ? (
            <ModelCatalogDetailRow>
              <SettingsStack>
                {showHfGate ? (
                  <HfModelAccessGate
                    accessMap={accessMap}
                    allAuthorized={allAuthorized}
                    hfAuthStatus={hfAuthStatus}
                    hfAuthPolling={hfAuthPolling}
                    startHuggingFaceLogin={() => {
                      void startHuggingFaceLogin()
                    }}
                    checkError={checkError}
                    onRetryCheck={recheckAccess}
                  />
                ) : null}
                {error ? (
                  <Text
                    as="p"
                    variant="body"
                    size="xs"
                    className="inline-flex items-center gap-1.5 text-fg-danger"
                  >
                    <AlertCircle className="h-3 w-3 flex-shrink-0" aria-hidden />
                    {error}
                  </Text>
                ) : null}
              </SettingsStack>
            </ModelCatalogDetailRow>
          ) : null}
        </Fragment>
      </ModelCatalogTable>
    </SettingsSubsection>
  )
}
