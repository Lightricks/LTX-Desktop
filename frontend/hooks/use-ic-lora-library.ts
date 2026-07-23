import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ApiSuccessOf } from '../lib/api-client'
import { useIcLoras, type IcLoraListItem } from './use-catalog'
import { catalogItemToEntry, preferredVariantId, resolveDownloadedVariantIds } from '../lib/lora-library'

// Owns the IC-LoRA library wiring: the catalog hook (fetch + download), the modal open state,
// the selected recipe id (+ optional variant), and the catalog→entry mapping for the modal.
// Selection *side effects* (seeding the generation controls/settings from the picked recipe)
// stay with the caller via onSelect — they touch GenSpace gen state.
export function useIcLoraLibrary(
  enabled: boolean,
  onSelect: (item: IcLoraListItem | null) => void,
  installed: ApiSuccessOf<'listModels'>['models'] = [],
  refreshInstalled: () => void = () => {},
) {
  const { icLoras, downloadIcLora, downloadingKey, progress, downloadError } = useIcLoras(enabled)
  const [selectedIcLoraId, setSelectedIcLoraId] = useState<string | null>(null)
  const [selectedIcLoraVariantId, setSelectedIcLoraVariantId] = useState<string | null>(null)
  const [modalOpen, setModalOpen] = useState(false)
  const installedPaths = useMemo(() => installed.map(m => m.path), [installed])
  const items = useMemo(
    () => icLoras.map(r => {
      const base = catalogItemToEntry(r.ic_lora)
      const downloadedVariantIds = resolveDownloadedVariantIds(
        base.variants,
        r.downloaded_variant_ids,
        installedPaths,
      )
      return {
        ...base,
        downloaded: r.downloaded || downloadedVariantIds.length > 0,
        downloadedVariantIds,
      }
    }),
    [icLoras, installedPaths],
  )

  // Re-list on-disk IC-LoRAs once a download finishes so per-variant ✓ / Use stay accurate
  // even if the catalog list payload lags or omits downloaded_variant_ids.
  const prevDownloadingKey = useRef<string | null>(null)
  useEffect(() => {
    if (prevDownloadingKey.current && !downloadingKey) refreshInstalled()
    prevDownloadingKey.current = downloadingKey
  }, [downloadingKey, refreshInstalled])

  const selectIcLora = useCallback(
    (item: IcLoraListItem | null, variantId?: string | null) => {
      setSelectedIcLoraId(item?.ic_lora.id ?? null)
      if (!item) {
        setSelectedIcLoraVariantId(null)
      } else {
        const entry = items.find(e => e.id === item.ic_lora.id)
        const preferred = preferredVariantId(
          entry?.variants,
          entry?.defaultVariantId,
          entry?.downloadedVariantIds,
        )
        // Explicit variant wins; otherwise only an *installed* preferred id (never undownloaded default).
        setSelectedIcLoraVariantId(variantId ?? preferred ?? null)
      }
      onSelect(item)
    },
    [onSelect, items],
  )
  return {
    icLoras,
    items,
    downloadIcLora,
    downloadingKey,
    progress,
    downloadError,
    modalOpen,
    setModalOpen,
    selectedIcLoraId,
    selectedIcLoraVariantId,
    selectIcLora,
  }
}
