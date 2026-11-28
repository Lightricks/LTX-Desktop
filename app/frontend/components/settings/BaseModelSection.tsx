import { AlertCircle } from 'lucide-react'
import { Fragment, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { useHfAuth } from '../../hooks/use-hf-auth'
import { useHfModelAccess } from '../../hooks/use-hf-model-access'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../../lib/api-client'
import { formatBytes, formatLtxBaseModelLabel } from '../../lib/format'
import { logger } from '../../lib/logger'
import { Badge } from '@/ds/Badge/Badge'
import { Button as DsButton } from '@/ds/Button/Button'
import { Text } from '@/ds/Text/Text'
import { Tooltip, TooltipProvider } from '@/ds/Tooltip/Tooltip'

import { HfModelAccessGate } from '../HfModelAccessGate'
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
  modelCatalogRowClassNames,
  modelCatalogSettingsTooltipPortal,
  modelCatalogTableStyles,
  LTX_CANNOT_DELETE_DEFAULT_MODEL_TOOLTIP,
} from './ModelCatalogTable'
import { SettingsStack, SettingsSubsection } from './SettingsContent'
import contentStyles from './SettingsContent.module.scss'

type LtxModelVersionItem = ApiSuccessOf<'getLtxVersions'>['versions'][number]
type ModelCheckpointID = NonNullable<
  NonNullable<ApiRequestBodyOf<'checkModelAccess'>>['cp_ids']
>[number]
type HfAuthStatus = ApiSuccessOf<'getHuggingFaceAuthStatus'>['status']

const DOWNLOAD_POLL_INTERVAL_MS = 1000

function ModelVersionRows({
  version,
  onChanged,
  resumeSessionId,
  hfAuthStatus,
  hfAuthPolling,
  startHuggingFaceLogin,
}: {
  version: LtxModelVersionItem
  onChanged: () => Promise<void>
  resumeSessionId: string | null
  hfAuthStatus: HfAuthStatus
  hfAuthPolling: boolean
  startHuggingFaceLogin: () => void
}) {
  const [downloading, setDownloading] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [downloadPercent, setDownloadPercent] = useState(0)
  const mountedRef = useRef(true)
  useEffect(() => () => { mountedRef.current = false }, [])

  const cpsToDownload = useMemo(
    () => (version.installed ? [] : (version.cps_to_download as ModelCheckpointID[])),
    [version.installed, version.cps_to_download],
  )
  const { accessMap, allAuthorized, checking: checkingAccess, checkError, recheckAccess } = useHfModelAccess(
    cpsToDownload,
    hfAuthStatus,
  )
  const canDownload = version.installed || (allAuthorized && !checkingAccess)
  const adoptedRef = useRef<string | null>(null)
  useEffect(() => {
    if (resumeSessionId && resumeSessionId !== adoptedRef.current && !downloadSessionId) {
      adoptedRef.current = resumeSessionId
      setDownloadSessionId(resumeSessionId)
      setDownloading(true)
    }
  }, [resumeSessionId, downloadSessionId])

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
        await onChanged()
        if (!cancelled) setDownloading(false)
      }
    }

    void poll()
    const interval = setInterval(() => void poll(), DOWNLOAD_POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [downloadSessionId, onChanged])

  const handleDownload = useCallback(async () => {
    setError(null)
    setDownloading(true)
    setDownloadPercent(0)
    const result = await ApiClient.startModelDownload({ type: 'download', cp_ids: version.cps_to_download })
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
  }, [version.cps_to_download])

  const handleDelete = useCallback(async () => {
    if (!confirmCatalogModelDelete(formatLtxBaseModelLabel(version.label))) return
    setError(null)
    setDeleting(true)
    const result = await ApiClient.deleteModels({ cp_ids: [version.model_cp] })
    if (!result.ok) {
      setDeleting(false)
      setError(result.error.message || 'Failed to delete model.')
      return
    }
    await onChanged()
    if (mountedRef.current) setDeleting(false)
  }, [version.model_cp, version.label, onChanged])

  const busy = downloading || deleting
  const downloadButtonActive = !version.installed && downloading
  const showHfGate =
    !version.installed &&
    !allAuthorized &&
    (Boolean(checkError) ||
      Object.values(accessMap).some((status) => status === 'not_authorized'))
  const showDetails = showHfGate || Boolean(error)

  const deleteControl = (
    <span className={modelCatalogTableStyles.actionButtonTooltipWrap}>
      <DsButton
        appearance="neutral"
        hierarchy="secondary"
        size="md"
        className={modelCatalogActionButtonWideClassName()}
        label="Delete"
        disabled={busy || version.active}
        onClick={(event) => {
          event.stopPropagation()
          if (version.active) return
          void handleDelete()
        }}
      />
    </span>
  )

  return (
    <Fragment>
      <tr
        className={modelCatalogRowClassNames()}
      >
        <ModelCatalogNameCell>
          <ModelCatalogCellText>{formatLtxBaseModelLabel(version.label)}</ModelCatalogCellText>
          {version.installed ? <ModelCatalogInstalledCheck /> : null}
          {!version.active && version.is_newest && !version.installed ? (
            <Badge appearance="brand" size="xs" text="Recommended" textVariant="label" />
          ) : null}
        </ModelCatalogNameCell>
        <ModelCatalogSizeCell>
          <ModelCatalogSizeText>{formatBytes(version.size_bytes)}</ModelCatalogSizeText>
        </ModelCatalogSizeCell>
        <ModelCatalogActionGroup
          action={
            !version.installed ? (
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
            ) : version.active ? (
              <Tooltip
                content={LTX_CANNOT_DELETE_DEFAULT_MODEL_TOOLTIP}
                side="top"
                align="center"
                maxWidth={280}
                portalContainer={modelCatalogSettingsTooltipPortal}
              >
                {deleteControl}
              </Tooltip>
            ) : (
              deleteControl
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
                startHuggingFaceLogin={startHuggingFaceLogin}
                checkError={checkError}
                onRetryCheck={recheckAccess}
              />
            ) : null}
            {error ? (
              <Text as="p" variant="body" size="xs" className="inline-flex items-center gap-1.5 text-fg-danger">
                <AlertCircle className="h-3 w-3 flex-shrink-0" aria-hidden />
                {error}
              </Text>
            ) : null}
          </SettingsStack>
        </ModelCatalogDetailRow>
      ) : null}
    </Fragment>
  )
}

export function BaseModelSection({
  searchQuery = '',
  onSearchHasMatches,
}: {
  searchQuery?: string
  onSearchHasMatches?: (hasMatches: boolean) => void
}) {
  const [versions, setVersions] = useState<LtxModelVersionItem[]>([])
  const [activeDownload, setActiveDownload] = useState<{ sessionId: string; cpIds: string[] } | null>(null)
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(true)
  const { notifyModelsChanged } = useAppSettings()
  const knownActiveRef = useRef<string | null>(null)

  const refreshVersions = useCallback(async () => {
    const [versionsResult, activeResult] = await Promise.all([
      ApiClient.getLtxVersions(),
      ApiClient.getActiveDownload(),
    ])
    if (!versionsResult.ok) {
      logger.error(`Failed to fetch LTX versions: ${versionsResult.error.message}`)
      return
    }
    setVersions(versionsResult.data.versions)
    const nextActive = versionsResult.data.versions.find((item) => item.active)?.model_id ?? null
    const nextKey = `${nextActive}|${versionsResult.data.versions.filter((item) => item.installed).map((item) => item.model_id).join(',')}`
    if (knownActiveRef.current !== null && knownActiveRef.current !== nextKey) {
      notifyModelsChanged()
    }
    knownActiveRef.current = nextKey
    if (activeResult.ok) {
      setActiveDownload(
        activeResult.data.session_id
          ? { sessionId: activeResult.data.session_id, cpIds: activeResult.data.cp_ids ?? [] }
          : null,
      )
    }
  }, [notifyModelsChanged])

  useEffect(() => {
    void refreshVersions()
  }, [refreshVersions])

  const tableVersions = useMemo(() => {
    const installed = versions.filter((version) => version.installed)
    const notInstalled = versions.filter((version) => !version.installed)
    let list = [...installed, ...notInstalled]
    if (searchQuery) {
      list = list.filter((version) =>
        formatLtxBaseModelLabel(version.label).toLowerCase().includes(searchQuery),
      )
    }
    return list
  }, [versions, searchQuery])

  useLayoutEffect(() => {
    if (!onSearchHasMatches) return
    onSearchHasMatches(!searchQuery || tableVersions.length > 0)
  }, [onSearchHasMatches, searchQuery, tableVersions.length])

  if (searchQuery && tableVersions.length === 0) {
    return null
  }

  const rowProps = {
    onChanged: refreshVersions,
    hfAuthStatus,
    hfAuthPolling,
    startHuggingFaceLogin: () => {
      void startHuggingFaceLogin()
    },
  }

  const resumeFor = (version: LtxModelVersionItem) =>
    activeDownload && activeDownload.cpIds.includes(version.model_cp)
      ? activeDownload.sessionId
      : null

  return (
    <SettingsSubsection title="Base models">
      <SettingsStack>
        {tableVersions.length === 0 ? (
          <ModelCatalogEmpty message="No models available." />
        ) : (
          <TooltipProvider delay={300}>
            <ModelCatalogTable>
              {tableVersions.map((version) => (
                <ModelVersionRows
                  key={version.model_id}
                  version={version}
                  resumeSessionId={resumeFor(version)}
                  {...rowProps}
                />
              ))}
            </ModelCatalogTable>
          </TooltipProvider>
        )}

        {versions.length === 0 ? (
          <Text as="p" variant="body" size="xs" className={contentStyles.textTertiary}>
            No versions available.
          </Text>
        ) : null}
      </SettingsStack>
    </SettingsSubsection>
  )
}
